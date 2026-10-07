"""
Model Performance Evaluation
==============================
Evaluates the trained LSTM risk predictor and DNN fraud classifier against
a held-out test split (same random_state and test_size as lstm_model.py /
dnn_model.py, so this is a genuine held-out evaluation, not a re-fit).

The intermediate lstm_X.npy / lstm_y.npy / lstm_meta.csv training artifacts
aren't shipped with the repo (only the final features CSV and trained
weights are, so the app runs without retraining) -- so sequences are
rebuilt on the fly from the tracked features CSV via build_sequences(),
which is deterministic and produces the identical arrays that would have
been saved during training.

Results are cached in-process after the first call, since re-scoring the
full test set through both models takes a few seconds.
"""

import threading

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix,
)
from tensorflow import keras

from ..config import (
    FEATURES_CSV,
    LSTM_MODEL_PATH, LSTM_FEATURE_MEAN_PATH, LSTM_FEATURE_STD_PATH,
    DNN_MODEL_PATH, DNN_FEATURE_MEAN_PATH, DNN_FEATURE_STD_PATH,
)
from ..features.feature_engineering import build_sequences
from .lstm_model import risk_probability_to_score
from .dnn_model import DNN_INPUT_COLUMNS

THRESHOLD = 0.5


def _metrics_at_threshold(y_true, y_prob, threshold=THRESHOLD):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    has_both_classes = len(np.unique(y_true)) > 1
    return {
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "auc_roc": round(float(roc_auc_score(y_true, y_prob)), 4) if has_both_classes else None,
        # computed from the same held-out predictions (not hard-coded):
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4) if has_both_classes else None,
        "false_positive_rate": round(float(fp / (fp + tn)), 4) if (fp + tn) > 0 else None,
        "confusion_matrix": {
            "true_negative": int(tn), "false_positive": int(fp),
            "false_negative": int(fn), "true_positive": int(tp),
        },
        "test_set_size": int(len(y_true)),
        "fraud_rate_pct": round(float(y_true.mean() * 100), 3),
        "threshold": threshold,
    }


def _load_sequences():
    feat_df = pd.read_csv(FEATURES_CSV, parse_dates=["timestamp"])
    X_seq, y_seq, meta = build_sequences(feat_df)
    return feat_df, X_seq, y_seq, meta


def _evaluate_lstm(X_seq, y_seq):
    _, X_test, _, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
    )

    model = keras.models.load_model(LSTM_MODEL_PATH)
    mean = np.load(LSTM_FEATURE_MEAN_PATH)
    std = np.load(LSTM_FEATURE_STD_PATH)
    X_test_norm = (X_test - mean) / std

    y_prob = model.predict(X_test_norm, batch_size=512, verbose=0).flatten()
    return _metrics_at_threshold(y_test, y_prob)


def _evaluate_dnn(feat_df, X_seq, meta):
    lstm_model = keras.models.load_model(LSTM_MODEL_PATH)
    lstm_mean = np.load(LSTM_FEATURE_MEAN_PATH)
    lstm_std = np.load(LSTM_FEATURE_STD_PATH)
    X_seq_norm = (X_seq - lstm_mean) / lstm_std

    risk_probs = lstm_model.predict(X_seq_norm, batch_size=512, verbose=0).flatten()
    lstm_meta = meta.copy()
    lstm_meta["risk_score"] = [risk_probability_to_score(p) for p in risk_probs]

    merged = feat_df.merge(lstm_meta[["transaction_id", "risk_score"]], on="transaction_id", how="inner")

    X = merged[DNN_INPUT_COLUMNS].values.astype(np.float32)
    y = merged["is_fraud"].values.astype(np.float32)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    dnn_model = keras.models.load_model(DNN_MODEL_PATH)
    dnn_mean = np.load(DNN_FEATURE_MEAN_PATH)
    dnn_std = np.load(DNN_FEATURE_STD_PATH)
    X_test_norm = np.clip((X_test - dnn_mean) / dnn_std, -6.0, 6.0)

    y_prob = dnn_model.predict(X_test_norm, batch_size=512, verbose=0).flatten()
    return _metrics_at_threshold(y_test, y_prob)


# Plain-language framing shipped WITH the numbers so they cannot be quoted
# without their caveats. These describe how the metrics were produced; they are
# not metric values.
EVALUATION_CONTEXT = {
    "dataset": "Synthetic behavioral dataset (data/generate_synthetic_data.py); no real banking data.",
    "split": "Random stratified 80/20 hold-out over sequence/transaction rows; the same customers appear in train and test.",
    "scores_are_calibrated_probabilities": False,
    "interpretation": (
        "Strong results on this synthetic data (generated with strongly separable fraud patterns) show the "
        "pipeline works end-to-end on the supplied dataset. They are not evidence of real-world banking "
        "fraud performance, generalization, or calibration."
    ),
}

_cache = None
# the full evaluation re-scores the whole test set (4-10 s of CPU): never let several
# requests (e.g. the dev server's double fetch, or ?refresh=true spam) all do it at once
_eval_lock = threading.Lock()
_generation = 0   # bumped after every completed evaluation


def evaluate_all(force_refresh: bool = False) -> dict:
    global _cache, _generation
    if _cache is not None and not force_refresh:
        return _cache
    seen_generation = _generation
    with _eval_lock:
        # a request that waited on the lock re-uses what the winner just computed
        # (even a forced refresh: _generation changed while it was waiting)
        if _cache is None or (force_refresh and _generation == seen_generation):
            feat_df, X_seq, y_seq, meta = _load_sequences()
            _cache = {
                "lstm_risk_predictor": _evaluate_lstm(X_seq, y_seq),
                "dnn_fraud_classifier": _evaluate_dnn(feat_df, X_seq, meta),
                "evaluation_context": EVALUATION_CONTEXT,
            }
            _generation += 1
    return _cache


if __name__ == "__main__":
    # Run this with:  cd backend && python -m app.models.evaluate
    import json
    print(json.dumps(evaluate_all(), indent=2))
