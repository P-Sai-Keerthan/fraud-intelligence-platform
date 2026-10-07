"""
Explainable AI Module (SHAP)
==============================
Wraps the trained DNN in a SHAP explainer and converts raw SHAP values into
human-readable reasons like "New Device", "Foreign Location", etc.
"""

import numpy as np
import shap

from .dnn_model import DNN_INPUT_COLUMNS
from ..config import DNN_MODEL_PATH, SHAP_BACKGROUND_PATH, DNN_FEATURE_MEAN_PATH, DNN_FEATURE_STD_PATH

# Human-readable label + the condition under which a feature "fires" as a reason.
# threshold_direction: "above" means high raw value = suspicious, "below" means
# low raw value = suspicious (only used for informational display, SHAP already
# tells us direction and magnitude of contribution).
FEATURE_DISPLAY_NAMES = {
    "amount_zscore": "Unusual Transaction Amount",
    "hour_is_unusual": "Unusual Transaction Time",
    "is_new_device": "New Device",
    "is_new_location": "New Location",
    "is_foreign_location": "Foreign Location",
    "failed_logins_24h": "Multiple Failed Logins",
    "category_is_unusual": "Unusual Merchant Category",
    "txn_velocity_1h": "High Transaction Velocity",
    "amount_pct_of_avg": "Amount Far Above Average",
    "risk_score": "Elevated Behavioral Risk Score",
}

# ---------------------------------------------------------------------------
# Feature STATE vs model CONTRIBUTION
# ---------------------------------------------------------------------------
# A positive SHAP value only says "the model's output is higher than the
# background average because of this input". It does NOT mean the feature is in
# a suspicious state (e.g. transaction velocity can get a positive SHAP value
# while the velocity is 0). A natural-language label such as "High Transaction
# Velocity" is therefore only shown when the underlying feature value actually
# meets the project's definition of that condition below. Otherwise the factor
# is either dropped or, if nothing anomalous is left, shown with neutral wording.
HIGH_VELOCITY_MIN = 2            # >= 2 other transactions in the previous hour
UNUSUAL_AMOUNT_ZSCORE = 2.0      # > 2 std-devs above the customer's own average
FAR_ABOVE_AVG_PCT = 200.0        # amount >= 2x the customer's own average
MULTIPLE_FAILED_LOGINS_MIN = 2   # "multiple" = at least 2 failed logins
ELEVATED_RISK_SCORE_MIN = 25.0   # LSTM score at/above the Medium band (25) of the app's own bands

FEATURE_IS_FRAUD_LIKE = {
    "amount_zscore": lambda v: v >= UNUSUAL_AMOUNT_ZSCORE,
    "hour_is_unusual": lambda v: v >= 0.5,
    "is_new_device": lambda v: v >= 0.5,
    "is_new_location": lambda v: v >= 0.5,
    "is_foreign_location": lambda v: v >= 0.5,
    "failed_logins_24h": lambda v: v >= MULTIPLE_FAILED_LOGINS_MIN,
    "category_is_unusual": lambda v: v >= 0.5,
    "txn_velocity_1h": lambda v: v >= HIGH_VELOCITY_MIN,
    "amount_pct_of_avg": lambda v: v >= FAR_ABOVE_AVG_PCT,
    "risk_score": lambda v: v >= ELEVATED_RISK_SCORE_MIN,
}

# wording used when the model weights a feature that is NOT in a fraud-like state
NEUTRAL_DISPLAY_NAMES = {
    "amount_zscore": "Transaction amount (not unusually high)",
    "hour_is_unusual": "Transaction time (within usual hours)",
    "is_new_device": "Device (previously used)",
    "is_new_location": "Location (previously used)",
    "is_foreign_location": "Location (within home-city list)",
    "failed_logins_24h": "Failed logins (fewer than 'multiple')",
    "category_is_unusual": "Merchant category (usual for customer)",
    "txn_velocity_1h": "Transaction velocity (normal)",
    "amount_pct_of_avg": "Amount vs average (not far above)",
    "risk_score": "LSTM temporal risk score (below elevated level)",
}
NEUTRAL_FALLBACK_COUNT = 2


class FraudExplainer:
    def __init__(self, model, background_data: np.ndarray):
        """
        model: trained keras DNN model
        background_data: normalized background sample (n_samples, n_features)
                          used as the SHAP reference distribution
        """
        self.model = model
        # KernelExplainer works model-agnostically but is slow; for a Keras
        # dense model, GradientExplainer is much faster and accurate enough.
        self.explainer = shap.GradientExplainer(model, background_data)

    def explain(self, x_normalized: np.ndarray, raw_values: np.ndarray = None, top_k: int = 4):
        """
        x_normalized: single normalized feature vector, shape (1, n_features)
        raw_values:   the same vector BEFORE normalization (DNN_INPUT_COLUMNS
                      order). Used to check each factor's actual state so a
                      label is never shown for a feature that is not in a
                      fraud-like state. If omitted, every positive SHAP
                      contribution is labelled as before (self-test only).
        Returns: list of {feature, display_name, shap_value}, strongest first,
        restricted to POSITIVE contributions (pushing toward "fraud"):
          * factors whose feature is actually in a fraud-like state get their
            descriptive label ("New Device", ...);
          * positive-SHAP factors whose feature is NOT in a fraud-like state are
            dropped; if that leaves nothing, the top few are returned with
            neutral wording ("Device (previously used)") instead of an alarming one.
        """
        shap_values = self.explainer.shap_values(x_normalized)
        # shap_values shape can be (1, n_features, 1) for single-output models
        values = np.array(shap_values)
        values = values.reshape(-1)  # flatten to (n_features,)

        positives = []
        for idx, (feat_name, shap_val) in enumerate(zip(DNN_INPUT_COLUMNS, values)):
            if shap_val > 0:
                positives.append((idx, feat_name, float(shap_val)))
        # sort by how much they pushed TOWARD fraud (positive SHAP value = more fraud-like)
        positives.sort(key=lambda p: p[2], reverse=True)

        if not positives:
            # no feature pushed toward fraud - transaction looks clean
            return []

        if raw_values is None:
            return [
                {"feature": f, "display_name": FEATURE_DISPLAY_NAMES.get(f, f), "shap_value": s}
                for _, f, s in positives[:top_k]
            ]

        raw = np.asarray(raw_values, dtype=np.float64).reshape(-1)
        active = [p for p in positives if FEATURE_IS_FRAUD_LIKE[p[1]](raw[p[0]])]
        if active:
            return [
                {"feature": f, "display_name": FEATURE_DISPLAY_NAMES.get(f, f), "shap_value": s}
                for _, f, s in active[:top_k]
            ]
        return [
            {"feature": f, "display_name": NEUTRAL_DISPLAY_NAMES.get(f, f), "shap_value": s}
            for _, f, s in positives[:NEUTRAL_FALLBACK_COUNT]
        ]


if __name__ == "__main__":
    # Run this with:  cd backend && python -m app.models.shap_explainer
    from tensorflow import keras

    model = keras.models.load_model(DNN_MODEL_PATH)
    background = np.load(SHAP_BACKGROUND_PATH)

    explainer = FraudExplainer(model, background)

    # test on a synthetic "obviously fraudulent" vector vs a "normal" one
    dnn_mean = np.load(DNN_FEATURE_MEAN_PATH)
    dnn_std = np.load(DNN_FEATURE_STD_PATH)

    fraud_like_raw = np.array([[8.0, 1, 1, 1, 1, 5, 1, 3, 450, 92]], dtype=np.float32)
    fraud_like_norm = (fraud_like_raw - dnn_mean) / dnn_std

    normal_like_raw = np.array([[0.1, 0, 0, 0, 0, 0, 0, 1, 100, 5]], dtype=np.float32)
    normal_like_norm = (normal_like_raw - dnn_mean) / dnn_std

    print("=== Explaining a suspicious-looking transaction ===")
    prob = model.predict(fraud_like_norm, verbose=0)[0][0]
    print(f"Fraud risk score: {prob*100:.2f}")
    reasons = explainer.explain(fraud_like_norm)
    for r in reasons:
        print(f"  - {r['display_name']} (contribution: {r['shap_value']:.4f})")

    print("\n=== Explaining a normal-looking transaction ===")
    prob2 = model.predict(normal_like_norm, verbose=0)[0][0]
    print(f"Fraud risk score: {prob2*100:.2f}")
    reasons2 = explainer.explain(normal_like_norm)
    for r in reasons2:
        print(f"  - {r['display_name']} (contribution: {r['shap_value']:.4f})")
    if not reasons2:
        print("  (no significant fraud-indicating factors found)")
