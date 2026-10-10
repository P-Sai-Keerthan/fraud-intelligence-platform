"""
Baselines, evaluated on exactly the same rows and split as the stacked model.

* amount_hour_rule: score = amount_pct_of_avg if hour_is_unusual else 0,
  i.e. "unusual hour AND amount at least X% of the customer's average".
  X is not hand-picked: it is the threshold chosen on validation by the same
  F1 rule as every other model (metrics.select_threshold).
* logistic_regression: standardized 9 behavioral features, class-balanced
  logistic regression, fitted on the training split.
* dnn_without_risk_score: the production DNN architecture on the 9
  behavioral features only (no LSTM input), same recipe as the stacked DNN.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..features.feature_engineering import FEATURE_COLUMNS

AMOUNT = FEATURE_COLUMNS.index("amount_pct_of_avg")
HOUR = FEATURE_COLUMNS.index("hour_is_unusual")


def amount_hour_rule_scores(F: np.ndarray) -> np.ndarray:
    """F: [n, 9] behavioral features of the scored transaction."""
    return np.where(F[:, HOUR] >= 0.5, F[:, AMOUNT], 0.0)


def fit_logistic_regression(F_train, y_train, seed: int = 42):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed),
    )
    model.fit(F_train, y_train)
    return model


def logistic_regression_scores(model, F) -> np.ndarray:
    return model.predict_proba(F)[:, 1]
