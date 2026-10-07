"""
Fraud Intelligence Inference Pipeline
========================================
Orchestrates the full scoring flow for one incoming transaction:

  1. Validate the request (TransactionInput) -- nothing invalid gets further
  2. Add the transaction to a COPY of the customer's behavioral history
  3. Recompute behavioral features (Behavioral Fraud DNA) using history only
  4. LSTM -> Risk Score (0-100), from the customer's PRIOR transactions
  5. DNN  -> Fraud Risk Score (0-100), from current features + risk_score
  6. SHAP -> top reasons behind the DNN score (labels checked against the
            actual feature state, see shap_explainer.py)
  7. Behavioral Similarity Score vs the customer's own historical profile
  8. Only now -- after everything succeeded (and after the optional ``persist``
     callback, e.g. the DB commit, succeeded) -- is the new history committed.

A request that fails validation, fails during inference, or fails to persist
therefore leaves the customer history exactly as it was.

TIME ORDERING: a transaction is always scored against the history that
PRECEDES its own timestamp. A back-dated timestamp is inserted at its correct
chronological position (its row is located by transaction_id, never assumed
to be the last row), and the derived "first time this device / location was
seen" flags of the WHOLE history are re-derived chronologically on every
insert, so a back-dated row can never leave later rows with stale flags.

COLD START (customers with little or no history) -- see ESTABLISHED_HISTORY_MIN:
  none         0 prior transactions. History-based inputs (amount z-score,
               unusual hour/category, new device/location, amount-vs-average)
               and the LSTM temporal risk do not exist, so they are NOT
               manufactured: the DNN receives each such input at its training
               mean (= "no information") and the score rests only on the
               genuinely available transaction-level signals (foreign location,
               failed logins). Similarity and temporal risk are reported as null.
  limited      1..9 prior transactions. Features come from the real (short)
               history, but the LSTM was only trained on full 10-transaction
               windows, so a padded window is never fed to it: temporal risk
               and similarity are null and the DNN gets the training mean for
               risk_score.
  established  >= 10 prior transactions: everything is computed as trained.

CONCURRENCY: every customer has its own lock (same-customer requests stay
strictly sequential, so history updates can't be lost), a batch holds only the
locks of the customers it touches, and a separate short-lived inference lock
serialises just the Keras/SHAP calls (~0.4 s per transaction, not per batch).
Unrelated customers are therefore never blocked for the duration of a batch.

KNOWN SIMPLIFICATION (documented honestly for the paper's limitations
section): behavioral features are recomputed from a customer's FULL history
on every incoming transaction, which is O(n) per call. This is fine at demo
scale (a few hundred transactions per customer) but a production system
would maintain incremental rolling statistics (updated in O(1) per
transaction) instead of recomputing from scratch each time.
"""

import threading
import uuid
from collections import OrderedDict
from contextlib import ExitStack
from datetime import datetime
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
from tensorflow import keras

from .config import (
    FEATURES_CSV, LSTM_MODEL_PATH, LSTM_FEATURE_MEAN_PATH, LSTM_FEATURE_STD_PATH,
    DNN_MODEL_PATH, DNN_FEATURE_MEAN_PATH, DNN_FEATURE_STD_PATH, SHAP_BACKGROUND_PATH,
    SEQUENCE_LENGTH, RECENT_RESULTS_MAX,
)
from .features.feature_engineering import build_point_features, FEATURE_COLUMNS
from .models.dnn_model import alert_level_from_probability, DNN_INPUT_COLUMNS
from .models.lstm_model import risk_probability_to_score
from .models.shap_explainer import FraudExplainer
from .models.similarity import compute_similarity
from .profile import build_customer_profile, history_status_for
from .schemas import TransactionInput

RAW_COLUMNS_FOR_FEATURES = [
    "customer_id", "transaction_id", "timestamp", "amount", "merchant_category",
    "device_id", "location", "failed_logins_24h", "is_new_device", "is_new_location", "is_fraud",
]

# DNN inputs that are computed FROM the customer's history (or from the LSTM,
# which is also history-based). With zero prior transactions they carry no
# real information, so they are neutral-imputed (training mean), never invented.
HISTORY_BASED_FEATURES = frozenset({
    "amount_zscore", "hour_is_unusual", "is_new_device", "is_new_location",
    "category_is_unusual", "amount_pct_of_avg", "risk_score",
})


def with_chronological_flags(raw: pd.DataFrame) -> pd.DataFrame:
    """Sort one customer's rows chronologically (stable: on equal timestamps the
    existing row stays before the newly added one) and re-derive is_new_device /
    is_new_location as 'first occurrence in time order'. This is exactly how the
    dataset generator defined them, and it keeps every row's flag correct even
    after a back-dated transaction is inserted into the middle of the history."""
    ordered = raw.sort_values("timestamp", kind="stable").reset_index(drop=True)
    ordered["is_new_device"] = (~ordered.duplicated(subset=["device_id"], keep="first")).astype(int)
    ordered["is_new_location"] = (~ordered.duplicated(subset=["location"], keep="first")).astype(int)
    return ordered


class FraudIntelligencePipeline:
    def __init__(self):
        print("[pipeline] Loading trained models...")
        self.lstm_model = keras.models.load_model(LSTM_MODEL_PATH)
        self.dnn_model = keras.models.load_model(DNN_MODEL_PATH)

        self.lstm_mean = np.load(LSTM_FEATURE_MEAN_PATH)
        self.lstm_std = np.load(LSTM_FEATURE_STD_PATH)
        self.dnn_mean = np.load(DNN_FEATURE_MEAN_PATH)
        self.dnn_std = np.load(DNN_FEATURE_STD_PATH)

        shap_background = np.load(SHAP_BACKGROUND_PATH)
        self.explainer = FraudExplainer(self.dnn_model, shap_background)

        # Locking (see module docstring). Histories are only ever REPLACED, never
        # mutated in place, and every read-modify-write happens while holding the
        # customer's own lock.
        self._state_lock = threading.Lock()          # guards the dicts below (held only for dict operations)
        self._customer_locks: dict = {}
        self._infer_lock = threading.Lock()          # Keras/SHAP calls only
        self._recent: "OrderedDict[str, dict]" = OrderedDict()   # server-side record of recent predictions (for /report/pdf)

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

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _empty_history() -> pd.DataFrame:
        return pd.DataFrame(columns=RAW_COLUMNS_FOR_FEATURES + FEATURE_COLUMNS)

    def _history_or_empty(self, customer_id: str) -> pd.DataFrame:
        """Read-only lookup. Deliberately does NOT register unknown customers."""
        history = self.customer_histories.get(customer_id)
        return history if history is not None else self._empty_history()

    def _lock_for(self, customer_id: str) -> threading.RLock:
        with self._state_lock:
            lock = self._customer_locks.get(customer_id)
            if lock is None:
                lock = self._customer_locks[customer_id] = threading.RLock()
            return lock

    def _commit(self, histories: dict, results: List[dict]) -> None:
        """Publish new histories + remember results. Caller holds the customers' locks."""
        with self._state_lock:
            self.customer_histories.update(histories)
            for result in results:
                self._recent[result["transaction_id"]] = result
                self._recent.move_to_end(result["transaction_id"])
            while len(self._recent) > RECENT_RESULTS_MAX:
                self._recent.popitem(last=False)

    def known_customer_ids(self):
        with self._state_lock:
            return sorted(self.customer_histories.keys())

    def customer_profile(self, customer_id: str) -> Optional[dict]:
        """Behavioral context from the stored history; None if the customer is unknown."""
        history = self.customer_histories.get(customer_id)
        if history is None:
            return None
        return build_customer_profile(customer_id, history)

    def get_recent_result(self, transaction_id: str) -> Optional[dict]:
        """The server's own record of a recent prediction (None if unknown/expired)."""
        with self._state_lock:
            result = self._recent.get(transaction_id)
        return dict(result) if result is not None else None

    def detect_fraud_rings(self, min_customers: int = 2) -> list:
        """Finds devices used by more than one distinct customer. A single
        legitimate customer's own devices are namespaced to them, so a
        device appearing across multiple customer_ids is a strong signal of
        an organized fraud ring (e.g. a stolen or shared device used to hit
        several accounts) rather than one customer behaving oddly alone."""
        with self._state_lock:
            snapshot = dict(self.customer_histories)
        device_to_customers: dict[str, set] = {}
        for customer_id, history in snapshot.items():
            if len(history) == 0 or "device_id" not in history.columns:
                continue
            for device in history["device_id"].dropna().unique():
                device_to_customers.setdefault(device, set()).add(customer_id)

        rings = []
        for device, customers in device_to_customers.items():
            if len(customers) >= min_customers:
                txn_count = sum(
                    int((snapshot[c]["device_id"] == device).sum())
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

    # --------------------------------------------------------------- scoring core
    def _compute(self, txn: dict, history: pd.DataFrame):
        """Pure scoring step. Reads `history`, never modifies it, and never
        touches self.customer_histories. Returns (result_dict, new_history)."""
        customer_id = txn["customer_id"]

        timestamp = txn.get("timestamp") or datetime.now()
        transaction_id = f"TXN_{uuid.uuid4().hex[:10].upper()}"

        new_raw_row = {
            "customer_id": customer_id,
            "transaction_id": transaction_id,
            "timestamp": timestamp,
            "amount": txn["amount"],
            "merchant_category": txn["merchant_category"],
            "device_id": txn["device_id"],
            "location": txn["location"],
            "failed_logins_24h": txn.get("failed_logins_24h", 0),
            "is_new_device": 0,    # re-derived chronologically below
            "is_new_location": 0,  # re-derived chronologically below
            "is_fraud": 0,         # unknown at inference time; placeholder only
        }

        prior_raw = history[RAW_COLUMNS_FOR_FEATURES] if len(history) else pd.DataFrame(columns=RAW_COLUMNS_FOR_FEATURES)
        combined_raw = pd.concat([prior_raw, pd.DataFrame([new_raw_row])], ignore_index=True)
        # "new" device / location = first occurrence in TIME order (also fixes the
        # flags of later rows when this transaction is back-dated)
        combined_raw = with_chronological_flags(combined_raw)

        # build_point_features sorts chronologically; locate the NEW row by id
        # instead of assuming it is the last row
        recomputed = build_point_features(combined_raw)
        pos = int(np.flatnonzero(recomputed["transaction_id"].values == transaction_id)[0])
        current_point_features = recomputed.loc[pos, FEATURE_COLUMNS].values.astype(np.float32)

        n_features = len(FEATURE_COLUMNS)
        n_prior = pos                                   # transactions chronologically BEFORE this one
        status = history_status_for(n_prior)
        established = status == "established"
        prior_features = (
            recomputed.iloc[:pos][FEATURE_COLUMNS].values.astype(np.float32)
            if n_prior > 0 else np.zeros((0, n_features), dtype=np.float32)
        )

        # ---- DNN input: current point features (+ LSTM risk when it can honestly be computed) ----
        point = current_point_features.copy()
        unavailable = set()
        if status == "none":
            # no history -> history-based inputs don't exist; use the training mean
            # ("no information") instead of the artefacts an empty history produces
            for name in HISTORY_BASED_FEATURES - {"risk_score"}:
                idx = FEATURE_COLUMNS.index(name)
                point[idx] = self.dnn_mean[DNN_INPUT_COLUMNS.index(name)]
            unavailable |= HISTORY_BASED_FEATURES - {"risk_score"}
        if not established:
            unavailable.add("risk_score")

        # Keras / SHAP calls are serialised by a short-lived lock (one transaction at a time),
        # NOT by the customer/batch locks, so unrelated requests interleave with a running batch.
        with self._infer_lock:
            # ---- LSTM: risk score from the PRIOR seq_len transactions (current excluded) ----
            if established:
                seq = prior_features[-SEQUENCE_LENGTH:]
                seq_norm = (seq - self.lstm_mean) / self.lstm_std
                risk_prob = float(self.lstm_model.predict(seq_norm[np.newaxis, ...], verbose=0)[0][0])
                risk_score = risk_probability_to_score(risk_prob)
                risk_input = risk_score
            else:
                risk_score = None                        # never run the LSTM on a padded/zero window
                risk_input = float(self.dnn_mean[-1])    # training mean -> 0 after normalisation

            dnn_input_raw = np.concatenate([point, [risk_input]]).astype(np.float32)
            dnn_input_norm = (dnn_input_raw - self.dnn_mean) / self.dnn_std
            # clip to +/-6 sigma so an extreme feature can't push the sigmoid to a
            # flat, not-credible 100.00%. (Verified in the Phase 1 audit that this
            # clip is not what mutes single-signal inputs: removing it lowers, not
            # raises, those scores.)
            dnn_input_norm = np.clip(dnn_input_norm, -6.0, 6.0)
            fraud_prob = float(self.dnn_model.predict(dnn_input_norm[np.newaxis, :], verbose=0)[0][0])
            # the output is a model score, never "absolute certainty"
            fraud_prob = min(fraud_prob, 0.999)
            alert_level = alert_level_from_probability(fraud_prob)

            # ---- SHAP explanation ----
            # Only surface reasons when the fraud risk score is meaningfully
            # elevated. Below this, any "positive" SHAP contributions are noise
            # relative to the overall near-zero score and showing them
            # would misleadingly suggest concern about a clearly normal transaction.
            REASON_DISPLAY_THRESHOLD = 0.05  # 5% model score
            if fraud_prob >= REASON_DISPLAY_THRESHOLD:
                # raw values let the explainer verify each factor's actual state
                reasons = self.explainer.explain(
                    dnn_input_norm[np.newaxis, :], raw_values=dnn_input_raw, skip_features=frozenset(unavailable),
                )
            else:
                reasons = []

        # ---- Behavioral similarity vs this customer's own historical profile ----
        # Only with an established history: before that there is no meaningful mean/std.
        if established:
            hist_mean = prior_features.mean(axis=0)
            hist_std = prior_features.std(axis=0)
            similarity = compute_similarity(current_point_features, hist_mean, hist_std)
            similarity_pct, deviation_pct = similarity["similarity_pct"], similarity["deviation_pct"]
        else:
            similarity_pct = deviation_pct = None

        result = {
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
            "similarity_pct": similarity_pct,
            "deviation_pct": deviation_pct,
            "history_status": status,
            "history_transactions": n_prior,
            "reasons": reasons,
        }
        return result, recomputed

    # ------------------------------------------------------------- public scoring
    def score_transaction(self, txn: dict, persist: Optional[Callable[[dict], None]] = None) -> dict:
        """Score one transaction.

        `txn` is re-validated here (single source of truth: TransactionInput) so
        no caller can bypass validation. `persist(result)`, if given, is called
        BEFORE the in-memory history is updated; if it raises, the history is
        left untouched and the exception propagates.
        """
        validated = TransactionInput.model_validate(txn).model_dump()
        customer_id = validated["customer_id"]
        with self._lock_for(customer_id):
            result, new_history = self._compute(validated, self._history_or_empty(customer_id))
            if persist is not None:
                persist(result)
            self._commit({customer_id: new_history}, [result])
        return result

    def score_batch(self, txns: List[dict], persist: Optional[Callable[[List[dict]], None]] = None) -> List[dict]:
        """Score many transactions ALL-OR-NOTHING. Every row is validated first;
        scoring happens against staged copies of the affected histories, which
        are committed together only after every row scored and `persist(results)`
        succeeded. Any failure leaves every history unchanged.

        Only the locks of the customers in this batch are held (acquired in sorted
        order, so batches can't deadlock each other); other customers' requests
        proceed normally while the batch runs."""
        validated = [TransactionInput.model_validate(t).model_dump() for t in txns]
        involved = sorted({t["customer_id"] for t in validated})
        results: List[dict] = []
        with ExitStack() as stack:
            for cid in involved:
                stack.enter_context(self._lock_for(cid))
            staged: dict = {}
            for txn in validated:
                cid = txn["customer_id"]
                history = staged[cid] if cid in staged else self._history_or_empty(cid)
                result, staged[cid] = self._compute(txn, history)
                results.append(result)
            if persist is not None:
                persist(results)
            self._commit(staged, results)
        return results


# a single shared pipeline instance, loaded once at API startup
_pipeline_instance = None


def get_pipeline() -> FraudIntelligencePipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = FraudIntelligencePipeline()
    return _pipeline_instance
