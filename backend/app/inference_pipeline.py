"""
Fraud Intelligence Inference Pipeline
========================================
Orchestrates the full scoring flow for one incoming transaction:

  1. Update customer's behavioral history with the new transaction
  2. Recompute behavioral features (Behavioral Fraud DNA) using history only
  3. LSTM -> Risk Score (0-100), from the customer's PRIOR trajectory
  4. DNN  -> Fraud Probability (0-100%), from current features + risk_score
  5. SHAP -> top reasons behind the fraud probability
  6. Behavioral Similarity Score vs the customer's own historical profile

KNOWN SIMPLIFICATION (documented honestly for the paper's limitations
section): behavioral features are recomputed from a customer's FULL history
on every incoming transaction, which is O(n) per call. This is fine at demo
scale (a few hundred transactions per customer) but a production system
would maintain incremental rolling statistics (updated in O(1) per
transaction) instead of recomputing from scratch each time.
"""

import uuid
from datetime import datetime

import numpy as np
import pandas as pd
from tensorflow import keras

from .config import (
    FEATURES_CSV, LSTM_MODEL_PATH, LSTM_FEATURE_MEAN_PATH, LSTM_FEATURE_STD_PATH,
    DNN_MODEL_PATH, DNN_FEATURE_MEAN_PATH, DNN_FEATURE_STD_PATH, SHAP_BACKGROUND_PATH,
    SEQUENCE_LENGTH,
)
from .features.feature_engineering import build_point_features, FEATURE_COLUMNS
from .models.dnn_model import alert_level_from_probability
from .models.lstm_model import risk_probability_to_score
from .models.shap_explainer import FraudExplainer
from .models.similarity import compute_similarity

RAW_COLUMNS_FOR_FEATURES = [
    "customer_id", "transaction_id", "timestamp", "amount", "merchant_category",
    "device_id", "location", "failed_logins_24h", "is_new_device", "is_new_location", "is_fraud",
]


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

    def _get_history(self, customer_id: str) -> pd.DataFrame:
        if customer_id not in self.customer_histories:
            empty_cols = RAW_COLUMNS_FOR_FEATURES + FEATURE_COLUMNS
            self.customer_histories[customer_id] = pd.DataFrame(columns=empty_cols)
        return self.customer_histories[customer_id]

    def known_customer_ids(self):
        return sorted(self.customer_histories.keys())

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

    def score_transaction(self, txn: dict) -> dict:
        customer_id = txn["customer_id"]
        history = self._get_history(customer_id)

        timestamp = txn.get("timestamp") or datetime.now()
        transaction_id = f"TXN_{uuid.uuid4().hex[:10].upper()}"

        seen_devices = set(history["device_id"]) if len(history) else set()
        seen_locations = set(history["location"]) if len(history) else set()
        is_new_device = int(txn["device_id"] not in seen_devices)
        is_new_location = int(txn["location"] not in seen_locations)

        new_raw_row = {
            "customer_id": customer_id,
            "transaction_id": transaction_id,
            "timestamp": timestamp,
            "amount": txn["amount"],
            "merchant_category": txn["merchant_category"],
            "device_id": txn["device_id"],
            "location": txn["location"],
            "failed_logins_24h": txn.get("failed_logins_24h", 0),
            "is_new_device": is_new_device,
            "is_new_location": is_new_location,
            "is_fraud": 0,  # unknown at inference time; placeholder only
        }

        prior_raw = history[RAW_COLUMNS_FOR_FEATURES] if len(history) else pd.DataFrame(columns=RAW_COLUMNS_FOR_FEATURES)
        combined_raw = pd.concat([prior_raw, pd.DataFrame([new_raw_row])], ignore_index=True)

        recomputed = build_point_features(combined_raw)
        current_point_features = recomputed.iloc[-1][FEATURE_COLUMNS].values.astype(np.float32)

        # ---- LSTM: risk score derived from the PRIOR seq_len transactions (current excluded) ----
        n_features = len(FEATURE_COLUMNS)
        if len(recomputed) > 1:
            prior_features = recomputed.iloc[:-1][FEATURE_COLUMNS].values.astype(np.float32)
        else:
            prior_features = np.zeros((0, n_features), dtype=np.float32)

        if len(prior_features) == 0:
            seq = np.zeros((SEQUENCE_LENGTH, n_features), dtype=np.float32)
        elif len(prior_features) < SEQUENCE_LENGTH:
            pad_needed = SEQUENCE_LENGTH - len(prior_features)
            pad = np.tile(prior_features[0], (pad_needed, 1))
            seq = np.vstack([pad, prior_features])
        else:
            seq = prior_features[-SEQUENCE_LENGTH:]

        seq_norm = (seq - self.lstm_mean) / self.lstm_std
        risk_prob = float(self.lstm_model.predict(seq_norm[np.newaxis, ...], verbose=0)[0][0])
        risk_score = risk_probability_to_score(risk_prob)

        # ---- DNN: fraud probability from current point features + risk_score ----
        dnn_input_raw = np.concatenate([current_point_features, [risk_score]]).astype(np.float32)
        dnn_input_norm = (dnn_input_raw - self.dnn_mean) / self.dnn_std
        # clip to the range of normalized inputs the model was actually
        # trained on -- an unclipped outlier feature (e.g. an extreme
        # amount_zscore) can otherwise push a raw input far outside anything
        # seen in training and saturate the sigmoid to a flat, not-credible
        # 100.00%. This keeps predictions inside the model's learned range.
        dnn_input_norm = np.clip(dnn_input_norm, -6.0, 6.0)
        fraud_prob = float(self.dnn_model.predict(dnn_input_norm[np.newaxis, :], verbose=0)[0][0])
        # a calibrated model should never claim absolute certainty
        fraud_prob = min(fraud_prob, 0.999)
        alert_level = alert_level_from_probability(fraud_prob)

        # ---- SHAP explanation ----
        # Only surface reasons when the fraud probability is meaningfully
        # elevated. Below this, any "positive" SHAP contributions are noise
        # relative to the overall near-zero probability and showing them
        # would misleadingly suggest concern about a clearly normal transaction.
        REASON_DISPLAY_THRESHOLD = 0.05  # 5% fraud probability
        if fraud_prob >= REASON_DISPLAY_THRESHOLD:
            reasons = self.explainer.explain(dnn_input_norm[np.newaxis, :])
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
        }


# a single shared pipeline instance, loaded once at API startup
_pipeline_instance = None


def get_pipeline() -> FraudIntelligencePipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = FraudIntelligencePipeline()
    return _pipeline_instance
