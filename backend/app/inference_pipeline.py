"""
Fraud Intelligence Inference Pipeline
========================================
Orchestrates the full scoring flow for one incoming transaction:

  1. Update customer's behavioral history with the new transaction
  2. Recompute behavioral features (Behavioral Fraud DNA) using history only
  3. LSTM -> Risk Score (0-100), a temporal risk signal from the customer's previous 10 transactions
  4. DNN  -> Fraud Score (0-100, a model score, not a calibrated probability), from current features + risk_score
  5. SHAP -> top reasons behind the Fraud Score
  6. Behavioral Similarity Score vs the customer's own historical profile

KNOWN SIMPLIFICATION (documented honestly for the paper's limitations
section): behavioral features are recomputed from a customer's FULL history
on every incoming transaction, which is O(n) per call. This is fine at demo
scale (a few hundred transactions per customer) but a production system
would maintain incremental rolling statistics (updated in O(1) per
transaction) instead of recomputing from scratch each time.

MODEL SETS: which models are loaded is chosen by MODEL_SET (see
app/model_sets.py; default "production" = models/saved/, unchanged). For a
DNN-only model set (v2_dnn_only) there is no LSTM: step 3 is skipped, the
DNN scores the 9 features alone, and the response's risk_score field (kept
for API compatibility) carries the DNN's own score, i.e. the same value as
fraud_probability.

HISTORY ORDER (4C-2f-1): a customer's history is kept in time order. A
transaction may be back-dated (its timestamp earlier than the customer's
newest transaction): it is inserted at its place in time, its features,
LSTM window and similarity use only the transactions before it, and the
score returned is the score of that transaction's own row (located by its
transaction_id, never assumed to be the last row). is_new_device /
is_new_location mean "not used in any EARLIER transaction", so the flags of
transactions after an inserted one are recomputed. Restoring from the
database replays the same insertion, so a restart rebuilds the same history.

COLD START (4C-2f-1): every model was trained only on transactions with at
least SEQUENCE_LENGTH (10) earlier transactions. Below that:
  * the LSTM is not run (a padded or empty window never occurred in
    training, and its Masking layer masks nothing); the DNN's risk_score
    input is the training mean of risk_score (scaled 0), and the response's
    risk_score reports that value;
  * for a customer's first transaction, the baseline-relative features
    (BASELINE_RELATIVE_FEATURES) have no baseline -- the feature builder
    emits placeholders (every device/city/category "new", z-score 0) -- so
    they are treated as missing: scaled 0, the training mean;
  * baseline-free features are used as observed, so a suspicious first
    transaction can still score high; imputed inputs are never reported as
    SHAP reasons. result["history_context"] records the state (it is not
    part of the API response).

HOME DEVICE (4C-2f-1): see get_customer_profile -- most frequent device in
the 90 days up to the customer's newest transaction.

PERSISTENCE: the in-memory histories start from the seed CSV. Every scored
transaction is also saved to the database, and on application startup
restore_scored_history() replays those saved rows on top of the seed data,
so a restart does not lose what the models know about each customer.
"""

import functools
import threading
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
from .config import FEATURES_CSV, SEQUENCE_LENGTH
from .features.feature_engineering import build_point_features, FEATURE_COLUMNS
from .model_metadata import model_metadata
from .model_sets import load_model_set
from .models.dnn_model import alert_level_from_probability
from .models.lstm_model import risk_probability_to_score
from .models.shap_explainer import FraudExplainer
from .models.similarity import compute_similarity

# every model was trained on transactions with at least this many earlier ones
MIN_PRIOR_TRANSACTIONS = SEQUENCE_LENGTH

# features that compare a transaction with the customer's own earlier
# behaviour; undefined (placeholder values) for a customer's first transaction
BASELINE_RELATIVE_FEATURES = (
    "amount_zscore", "hour_is_unusual", "is_new_device", "is_new_location",
    "category_is_unusual", "amount_pct_of_avg",
)

HOME_DEVICE_WINDOW = pd.Timedelta(days=90)
HOME_DEVICE_MIN_TRANSACTIONS = 5

_MAX_ID_ATTEMPTS = 5

RAW_COLUMNS_FOR_FEATURES = [
    "customer_id", "transaction_id", "timestamp", "amount", "merchant_category",
    "device_id", "location", "failed_logins_24h", "is_new_device", "is_new_location", "is_fraud",
]


def insert_chronologically(prior_raw: pd.DataFrame, new_raw: pd.DataFrame) -> pd.DataFrame:
    """One customer's raw history with new transactions inserted at their
    place in time. The existing rows keep their order; a new row goes after
    existing rows with the same timestamp, and new rows keep the given order
    among themselves (stable sort). is_new_device / is_new_location are
    recomputed ("not used by any earlier row") for every row from the first
    inserted position on; earlier rows cannot be affected by the insertion.
    For the seed data this is exactly how the stored flags are defined."""
    combined = pd.concat([prior_raw, new_raw], ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"])
    is_new = np.zeros(len(combined), dtype=bool)
    is_new[len(prior_raw):] = True
    order = np.argsort(combined["timestamp"].to_numpy(), kind="stable")
    combined = combined.iloc[order].reset_index(drop=True)
    is_new = is_new[order]
    if is_new.any():
        start = int(np.flatnonzero(is_new)[0])
        first_device = (~combined["device_id"].duplicated()).astype(int).to_numpy()
        first_location = (~combined["location"].duplicated()).astype(int).to_numpy()
        combined["is_new_device"] = combined["is_new_device"].astype(int)
        combined["is_new_location"] = combined["is_new_location"].astype(int)
        combined.loc[start:, "is_new_device"] = first_device[start:]
        combined.loc[start:, "is_new_location"] = first_location[start:]
    return combined


def home_device_window(history: pd.DataFrame) -> pd.DataFrame:
    """The customer's transactions in (anchor - 90 days, anchor], anchor = newest."""
    ts = pd.to_datetime(history["timestamp"])
    return history[ts > ts.max() - HOME_DEVICE_WINDOW]


def _synchronized(method):
    """Serialize access to the shared in-memory customer histories. Requests
    are handled on worker threads, and scoring is a read-modify-write of a
    customer's history, so two concurrent scorings for the same customer
    could otherwise lose one of the transactions."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapper


class FraudIntelligencePipeline:
    def __init__(self, model_set=None, candidates_root=None):
        """model_set: a name (else $MODEL_SET, else "production"); see app/model_sets.py.
        candidates_root: only for tests (a copy of models/candidates/<dataset>/)."""
        print("[pipeline] Loading trained models...")
        self.model_set = load_model_set(model_set, candidates_root=candidates_root)
        print(f"[pipeline] {self.model_set.describe()}")
        self.lstm_model = self.model_set.lstm_model        # None for a DNN-only model set
        self.dnn_model = self.model_set.dnn_model

        self.lstm_mean = self.model_set.lstm_mean
        self.lstm_std = self.model_set.lstm_std
        self.dnn_mean = self.model_set.dnn_mean
        self.dnn_std = self.model_set.dnn_std

        self.explainer = FraudExplainer(self.dnn_model, self.model_set.shap_background,
                                        feature_names=self.model_set.dnn_input_columns)
        # provenance / observability (4C-2f-2): recorded with every scored transaction
        self.model_metadata = model_metadata(self.model_set)
        self.model_version = self.model_metadata["model_version"]

        print("[pipeline] Loading seed customer history for behavioral profiles...")
        if FEATURES_CSV.exists():
            seed_df = pd.read_csv(FEATURES_CSV, parse_dates=["timestamp"])
            self.customer_histories = {
                cid: group.sort_values("timestamp").reset_index(drop=True)
                for cid, group in seed_df.groupby("customer_id")
            }
            print(f"[pipeline] Loaded history for {len(self.customer_histories)} customers.")
        else:
            print("[pipeline] WARNING: no seed feature file found, starting with empty histories.")
            self.customer_histories = {}

        # the untouched seed histories, kept so restore_scored_history() can
        # rebuild from "seed + persisted scored transactions" at any time
        self._seed_histories = dict(self.customer_histories)
        self._lock = threading.RLock()

    def _get_history(self, customer_id: str) -> pd.DataFrame:
        if customer_id not in self.customer_histories:
            empty_cols = RAW_COLUMNS_FOR_FEATURES + FEATURE_COLUMNS
            self.customer_histories[customer_id] = pd.DataFrame(columns=empty_cols)
        return self.customer_histories[customer_id]

    @_synchronized
    def restore_scored_history(self, scored_rows: list[dict]) -> int:
        """Rebuild every customer's in-memory history as seed history plus the
        given previously-scored transactions (the rows persisted in the
        database). Idempotent: it always starts again from the seed data, and
        skips any transaction_id already present, so calling it repeatedly
        never duplicates transactions. Rows are inserted in time order exactly as score_transaction inserts
        them (insert_chronologically), including rows dated before seed rows.
        Returns the number of scored transactions restored."""
        rows_by_customer: dict[str, list[dict]] = {}
        for row in scored_rows:
            rows_by_customer.setdefault(row["customer_id"], []).append(row)

        self.customer_histories.clear()
        self.customer_histories.update(self._seed_histories)

        restored = 0
        for customer_id, rows in rows_by_customer.items():
            seed = self._seed_histories.get(customer_id)
            if seed is not None and len(seed):
                prior_raw = seed[RAW_COLUMNS_FOR_FEATURES]
            else:
                prior_raw = pd.DataFrame(columns=RAW_COLUMNS_FOR_FEATURES)
            seen_ids = set(prior_raw["transaction_id"])

            new_rows = []
            for row in sorted(rows, key=lambda r: (pd.Timestamp(r["timestamp"]), r["transaction_id"])):
                if row["transaction_id"] in seen_ids:
                    continue
                seen_ids.add(row["transaction_id"])
                new_rows.append({
                    "customer_id": customer_id,
                    "transaction_id": row["transaction_id"],
                    "timestamp": row["timestamp"],
                    "amount": row["amount"],
                    "merchant_category": row["merchant_category"],
                    "device_id": row["device_id"],
                    "location": row["location"],
                    "failed_logins_24h": row["failed_logins_24h"] or 0,
                    "is_new_device": 0,     # recomputed in time order by insert_chronologically
                    "is_new_location": 0,
                    "is_fraud": 0,  # unknown, same placeholder score_transaction uses
                })

            if not new_rows:
                continue
            combined_raw = insert_chronologically(prior_raw, pd.DataFrame(new_rows)[RAW_COLUMNS_FOR_FEATURES])
            self.customer_histories[customer_id] = build_point_features(combined_raw)
            restored += len(new_rows)
        return restored

    @_synchronized
    def known_customer_ids(self):
        return sorted(self.customer_histories.keys())

    @_synchronized
    def get_customer_profile(self, customer_id: str):
        """The customer's usual ("home") device and city, derived from their
        own transaction history. Returns None for a customer with no history.
        The most frequent value wins, ties broken alphabetically.

        home_device (4C-2 design, docs/step4c2e-retraining-design.md section 5):
        the most frequent device in (anchor - 90 days, anchor], anchor = the
        customer's newest transaction (not the wall clock), so a phone
        upgrade takes over once the new phone dominates the last 90 days.
        Falls back to the whole history when that window holds fewer than
        HOME_DEVICE_MIN_TRANSACTIONS transactions or the history spans less
        than 90 days. home_location keeps the whole-history rule.
        In the v1 seed data both rules give every customer's primary device."""
        history = self.customer_histories.get(customer_id)
        if history is None or len(history) == 0:
            return None

        def most_common(frame: pd.DataFrame, column: str) -> str:
            counts = frame[column].dropna().astype(str).value_counts()
            top = counts.max()
            return sorted(counts[counts == top].index)[0]

        ts = pd.to_datetime(history["timestamp"])
        window = home_device_window(history)
        if ts.max() - ts.min() >= HOME_DEVICE_WINDOW and len(window) >= HOME_DEVICE_MIN_TRANSACTIONS:
            device_source = window
        else:
            device_source = history

        return {
            "customer_id": customer_id,
            "home_device": most_common(device_source, "device_id"),
            "home_location": most_common(history, "location"),
            "n_transactions": int(len(history)),
        }

    @_synchronized
    def detect_fraud_rings(self, min_customers: int = 2) -> list:
        """Finds devices used by more than one distinct customer. A single
        legitimate customer's own devices are namespaced to them, so a
        device appearing across multiple customer_ids is a strong signal of
        an organized fraud ring (e.g. a stolen or shared device used to hit
        several accounts) rather than one customer behaving oddly alone."""
        device_to_customers: dict[str, set] = {}
        for customer_id, history in self.customer_histories.items():
            if len(history) == 0 or "device_id" not in history.columns:
                continue
            for device in history["device_id"].dropna().unique():
                device_to_customers.setdefault(device, set()).add(customer_id)

        rings = []
        for device, customers in device_to_customers.items():
            if len(customers) >= min_customers:
                txn_count = sum(
                    int((self.customer_histories[c]["device_id"] == device).sum())
                    for c in customers
                )
                rings.append({
                    "ring_type": "shared_device",
                    "identifier": device,
                    "customer_ids": sorted(customers),
                    "transaction_count": txn_count,
                })

        rings.sort(key=lambda r: (len(r["customer_ids"]), r["transaction_count"]), reverse=True)
        return rings

    def _new_transaction_id(self, history: pd.DataFrame) -> str:
        """A transaction id not already in this customer's history."""
        taken = set(history["transaction_id"]) if len(history) else set()
        for _ in range(_MAX_ID_ATTEMPTS):
            transaction_id = f"TXN_{uuid.uuid4().hex[:10].upper()}"
            if transaction_id not in taken:
                return transaction_id
        raise RuntimeError("could not generate a unique transaction id for this customer")

    @_synchronized
    def score_transaction(self, txn: dict) -> dict:
        customer_id = txn["customer_id"]
        history = self._get_history(customer_id)

        timestamp = txn.get("timestamp") or datetime.now()
        transaction_id = self._new_transaction_id(history)

        new_raw_row = {
            "customer_id": customer_id,
            "transaction_id": transaction_id,
            "timestamp": timestamp,
            "amount": txn["amount"],
            "merchant_category": txn["merchant_category"],
            "device_id": txn["device_id"],
            "location": txn["location"],
            "failed_logins_24h": txn.get("failed_logins_24h", 0),
            "is_new_device": 0,     # set by insert_chronologically: not used by any earlier transaction
            "is_new_location": 0,
            "is_fraud": 0,  # unknown at inference time; placeholder only
        }

        prior_raw = history[RAW_COLUMNS_FOR_FEATURES] if len(history) else pd.DataFrame(columns=RAW_COLUMNS_FOR_FEATURES)
        combined_raw = insert_chronologically(prior_raw, pd.DataFrame([new_raw_row])[RAW_COLUMNS_FOR_FEATURES])

        recomputed = build_point_features(combined_raw)
        # the row of THIS transaction -- not necessarily the last one (back-dated input)
        matches = np.flatnonzero(recomputed["transaction_id"].to_numpy() == transaction_id)
        if len(matches) != 1:
            raise RuntimeError(f"transaction {transaction_id} appears {len(matches)} times in the rebuilt history")
        position = int(matches[0])
        current_point_features = recomputed.iloc[position][FEATURE_COLUMNS].values.astype(np.float32)

        # the customer's transactions strictly before this one, in time order
        n_features = len(FEATURE_COLUMNS)
        prior_features = recomputed.iloc[:position][FEATURE_COLUMNS].values.astype(np.float32)
        n_prior = len(prior_features)
        cold_start = n_prior < MIN_PRIOR_TRANSACTIONS
        input_columns = self.model_set.dnn_input_columns
        imputed = []

        if self.model_set.uses_lstm:
            risk_index = input_columns.index("risk_score")
            if not cold_start:
                # ---- LSTM: risk score from the 10 transactions before this one ----
                seq = prior_features[-SEQUENCE_LENGTH:]
                seq_norm = (seq - self.lstm_mean) / self.lstm_std
                risk_prob = float(self.lstm_model.predict(seq_norm[np.newaxis, ...], verbose=0)[0][0])
                risk_score = risk_probability_to_score(risk_prob)
            else:
                # fewer than 10 earlier transactions: no LSTM window exists (padding
                # was never seen in training); use the training mean of risk_score
                risk_score = round(float(self.dnn_mean[risk_index]), 2)
                imputed.append("risk_score")

            # ---- DNN: fraud probability from current point features + risk_score ----
            dnn_input_raw = np.concatenate([current_point_features, [risk_score]]).astype(np.float32)
        else:
            # DNN-only model set: no LSTM, the DNN scores the 9 point features
            dnn_input_raw = current_point_features.astype(np.float32)
        dnn_input_norm = (dnn_input_raw - self.dnn_mean) / self.dnn_std
        # clip to the range of normalized inputs the model was actually
        # trained on -- an unclipped outlier feature (e.g. an extreme
        # amount_zscore) can otherwise push a raw input far outside anything
        # seen in training and saturate the sigmoid to a flat, not-credible
        # 100.00%. This keeps predictions inside the model's learned range.
        dnn_input_norm = np.clip(dnn_input_norm, -6.0, 6.0)
        if n_prior == 0:
            # first transaction: no baseline, so the baseline-relative features are missing
            imputed = list(BASELINE_RELATIVE_FEATURES) + imputed
        for column in imputed:
            dnn_input_norm[input_columns.index(column)] = 0.0      # the training mean, scaled
        fraud_prob = float(self.dnn_model.predict(dnn_input_norm[np.newaxis, :], verbose=0)[0][0])
        # the output is the model's fraud score (class-weighted training, so it is
        # not a calibrated probability); cap it so it is never shown as 100% certain
        fraud_prob = min(fraud_prob, 0.999)
        alert_level = alert_level_from_probability(fraud_prob)
        if not self.model_set.uses_lstm:
            # no behavioral LSTM score exists; the response keeps the risk_score
            # field (same name, float, 0-100) and fills it with the DNN's score
            risk_score = round(fraud_prob * 100, 2)

        # ---- SHAP explanation ----
        # Only surface reasons when the fraud probability is meaningfully
        # elevated. Below this, any "positive" SHAP contributions are noise
        # relative to the overall near-zero probability and showing them
        # would misleadingly suggest concern about a clearly normal transaction.
        REASON_DISPLAY_THRESHOLD = 0.05  # 5% fraud probability
        if fraud_prob >= REASON_DISPLAY_THRESHOLD:
            # imputed (unobserved) inputs are never offered as reasons
            reasons = self.explainer.explain(dnn_input_norm[np.newaxis, :], exclude=imputed)
        else:
            reasons = []

        # ---- Behavioral similarity vs this customer's own historical profile ----
        if len(prior_features) > 0:
            hist_mean = prior_features.mean(axis=0)
            hist_std = prior_features.std(axis=0) if len(prior_features) > 1 else np.ones(n_features)
        else:
            hist_mean = np.zeros(n_features)
            hist_std = np.ones(n_features)
        similarity = compute_similarity(current_point_features, hist_mean, hist_std)

        # persist updated history in memory so the NEXT transaction for this
        # customer builds on top of it
        self.customer_histories[customer_id] = recomputed

        return {
            "transaction_id": transaction_id,
            "customer_id": customer_id,
            "timestamp": timestamp,
            "amount": txn["amount"],
            "merchant_category": txn["merchant_category"],
            "device_id": txn["device_id"],
            "location": txn["location"],
            "failed_logins_24h": txn.get("failed_logins_24h", 0),
            "risk_score": risk_score,
            "fraud_probability": round(fraud_prob * 100, 2),
            "alert_level": alert_level,
            "similarity_pct": similarity["similarity_pct"],
            "deviation_pct": similarity["deviation_pct"],
            "reasons": reasons,
            # provenance, persisted with the transaction (not part of the /predict response)
            "model_set": self.model_set.name,
            "model_version": self.model_version,
            # internal (not part of the API response): how much history backed this score
            "history_context": {
                "prior_transactions": n_prior,
                "min_prior_transactions": MIN_PRIOR_TRANSACTIONS,
                "cold_start": cold_start,
                "lstm_used": bool(self.model_set.uses_lstm and not cold_start),
                "imputed_inputs": imputed,
            },
        }


# a single shared pipeline instance, loaded once at API startup
_pipeline_instance = None


def get_pipeline() -> FraudIntelligencePipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = FraudIntelligencePipeline()
    return _pipeline_instance
