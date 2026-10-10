"""
Downstream classifiers behind the LSTM (Step 4D).

A fitted scikit-learn classifier exposed through the small interface the
pipeline and the evaluation code already use for the Keras DNN:

* predict(X, verbose=0) / predict_on_batch(X) -> (n, 1) array, the
  classifier's positive-class output (predict_proba[:, 1]);
* input_shape -> (None, n_inputs).

The output is a model score. Whether it may be read as a probability is a
question of calibration, answered in docs/model_selection_report.md.
"""

import numpy as np

TREE_FAMILIES = ("random_forest", "hist_gradient_boosting")
LINEAR_FAMILIES = ("logistic_regression",)
FAMILY_LABELS = {
    "dnn": "DNN",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "Histogram Gradient Boosting",
}


class ClassifierModel:
    def __init__(self, estimator, family: str, n_inputs: int):
        self.estimator, self.family = estimator, family
        self.input_shape = (None, int(n_inputs))

    def _proba(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        return self.estimator.predict_proba(X)[:, 1].astype(np.float32).reshape(-1, 1)

    def predict_on_batch(self, X):
        return self._proba(X)

    def predict(self, X, verbose=0, batch_size=None):
        return self._proba(X)


def shap_explainer_for(model, background: np.ndarray):
    """(explainer, output space) for exact SHAP values of `model`:

    * tree families: shap.TreeExplainer, interventional, on the predicted
      probability, with the whole background as reference (exact Tree SHAP);
      the values plus the expected value add up to the model's output;
    * logistic regression: shap.LinearExplainer (exact), log-odds space;
    * anything else (the Keras DNN): None (the caller keeps GradientExplainer)."""
    import shap
    if not isinstance(model, ClassifierModel):
        return None, None
    if model.family in TREE_FAMILIES:
        masker = shap.maskers.Independent(background, max_samples=len(background))
        return shap.TreeExplainer(model.estimator, data=masker, feature_perturbation="interventional",
                                  model_output="probability"), "probability"
    if model.family in LINEAR_FAMILIES:
        return shap.LinearExplainer(model.estimator, background), "log-odds"
    raise ValueError(f"no SHAP explainer for classifier family {model.family!r}")
