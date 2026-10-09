"""
Explainable AI Module (SHAP)
==============================
Wraps the loaded downstream classifier in a SHAP explainer and converts raw
SHAP values into human-readable reasons like "New Device", "Foreign Location".

* Random forest / gradient boosting (Step 4D): shap.TreeExplainer, exact
  interventional Tree SHAP on the predicted score, with the model set's
  background rows as the reference. The values add up to
  (score - average score over the background).
* Logistic regression: shap.LinearExplainer (exact), log-odds space.
* Keras DNN (production / v2 candidates): shap.GradientExplainer (expected
  gradients, sampled; approximate).
The explanation always comes from the same model object that produced the
score.
"""

import numpy as np
import shap

from .dnn_model import DNN_INPUT_COLUMNS
from .downstream_classifier import shap_explainer_for
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
    def __init__(self, model, background_data: np.ndarray, feature_names=None):
        """
        model: trained keras DNN model
        background_data: normalized background sample (n_samples, n_features)
                          used as the SHAP reference distribution
        feature_names: the model's input columns, in input order (from the
                       loaded model set). Defaults to the production DNN's
                       DNN_INPUT_COLUMNS. The names must match the model's
                       input width and the background width, otherwise a
                       reason would be attributed to the wrong feature, so a
                       mismatch raises ValueError instead.
        """
        self.feature_names = list(DNN_INPUT_COLUMNS if feature_names is None else feature_names)
        n = len(self.feature_names)
        model_width = model.input_shape[-1]
        if model_width != n:
            raise ValueError(f"SHAP: model takes {model_width} inputs but {n} feature names were given")
        if background_data.ndim != 2 or background_data.shape[1] != n:
            raise ValueError(f"SHAP: background shape {background_data.shape} does not match {n} feature names")
        self.model = model
        self.explainer, self.output_space = shap_explainer_for(model, background_data)
        if self.explainer is None:
            # Keras DNN: KernelExplainer works model-agnostically but is slow; for a
            # dense network GradientExplainer is much faster and accurate enough.
            self.explainer = shap.GradientExplainer(model, background_data)
            self.output_space = "dnn output (expected gradients, sampled)"
            self.method = "GradientExplainer"
        else:
            self.method = type(self.explainer).__name__

    def explain(self, x_normalized: np.ndarray, top_k: int = 4, exclude=()):
        """
        x_normalized: single normalized feature vector, shape (1, n_features)
        Returns: list of {feature, display_name, shap_value, contribution} sorted
        by absolute contribution, top_k only, restricted to POSITIVE
        contributions (i.e. features pushing toward "fraud").
        exclude: feature names never reported (e.g. inputs the pipeline
        imputed because they were not observed).
        """
        shap_values = self.explainer.shap_values(x_normalized)
        # shap_values shape can be (1, n_features, 1) for single-output models,
        # or (1, n_features, 2) for a two-class scikit-learn classifier
        values = np.array(shap_values)
        if values.ndim == 3 and values.shape[-1] == 2:
            values = values[..., 1]            # contributions to the fraud class
        values = values.reshape(-1)  # flatten to (n_features,)
        if len(values) != len(self.feature_names):
            raise ValueError(f"SHAP returned {len(values)} values for {len(self.feature_names)} features")

        reasons = []
        for feat_name, shap_val in zip(self.feature_names, values):
            if feat_name in exclude:
                continue
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
