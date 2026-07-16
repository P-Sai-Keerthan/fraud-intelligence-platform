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

    def explain(self, x_normalized: np.ndarray, top_k: int = 4):
        """
        x_normalized: single normalized feature vector, shape (1, n_features)
        Returns: list of {feature, display_name, shap_value, contribution} sorted
        by absolute contribution, top_k only, restricted to POSITIVE
        contributions (i.e. features pushing toward "fraud").
        """
        shap_values = self.explainer.shap_values(x_normalized)
        # shap_values shape can be (1, n_features, 1) for single-output models
        values = np.array(shap_values)
        values = values.reshape(-1)  # flatten to (n_features,)

        reasons = []
        for feat_name, shap_val in zip(DNN_INPUT_COLUMNS, values):
            reasons.append({
                "feature": feat_name,
                "display_name": FEATURE_DISPLAY_NAMES.get(feat_name, feat_name),
                "shap_value": float(shap_val),
            })

        # sort by how much they pushed TOWARD fraud (positive SHAP value = more fraud-like)
        reasons.sort(key=lambda r: r["shap_value"], reverse=True)
        top_reasons = [r for r in reasons if r["shap_value"] > 0][:top_k]

        if not top_reasons:
            # no feature pushed toward fraud - transaction looks clean
            return []
        return top_reasons


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
    print(f"Fraud probability: {prob*100:.2f}%")
    reasons = explainer.explain(fraud_like_norm)
    for r in reasons:
        print(f"  - {r['display_name']} (contribution: {r['shap_value']:.4f})")

    print("\n=== Explaining a normal-looking transaction ===")
    prob2 = model.predict(normal_like_norm, verbose=0)[0][0]
    print(f"Fraud probability: {prob2*100:.2f}%")
    reasons2 = explainer.explain(normal_like_norm)
    for r in reasons2:
        print(f"  - {r['display_name']} (contribution: {r['shap_value']:.4f})")
    if not reasons2:
        print("  (no significant fraud-indicating factors found)")
