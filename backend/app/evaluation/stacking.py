"""
Model fitting for the corrected evaluation, and the leakage-safe LSTM -> DNN handoff.

Recipe (architectures and hyperparameters unchanged from lstm_model.py /
dnn_model.py): same networks, epochs (30 / 40), batch sizes (64 / 128),
class weights (n_neg / n_pos over the training rows), early stopping on
val_auc with patience 5 and best-weight restore, and the scaler fitted on
the training rows only. Changes:

* Early stopping monitors the chronologically LAST 15% of the model's own
  training rows, instead of Keras's validation_split (the last 15% of a
  randomly shuffled array). The evaluation's validation split stays untouched
  for threshold selection and model comparison.
* Seeds are fixed (keras.utils.set_random_seed) and TensorFlow op determinism
  is enabled, so a run is reproducible on the same machine and library versions.

Leakage-safe stacking (stacked_risk_scores):
* DNN training rows get OUT-OF-FOLD LSTM scores: the training rows are
  divided into N_FOLDS customer-grouped folds (StratifiedGroupKFold, so each
  fold gets a share of the fraud customers); for each fold an LSTM is
  trained on the other folds and scores only the held-out fold. No training
  row is ever scored by an LSTM that saw its label, its customer, or an
  overlapping window.
* The final LSTM is trained on all training rows and scores the validation
  and test rows, whose labels it never saw.
"""

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

EVAL_SEED = 42
N_FOLDS = 3
EARLY_STOP_FRACTION = 0.15
CLIP = 6.0   # same +-6 clip production inference applies to normalized DNN inputs


def chronological_holdout(timestamps, fraction: float = EARLY_STOP_FRACTION):
    """(fit_idx, early_stop_idx): the latest `fraction` of rows by time are held out."""
    order = np.argsort(np.asarray(timestamps), kind="mergesort")
    n_hold = max(1, int(round(len(order) * fraction)))
    return np.sort(order[:-n_hold]), np.sort(order[-n_hold:])


def _set_seeds(seed: int):
    import tensorflow as tf
    from tensorflow import keras
    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def _class_weight(y):
    n_pos = float(np.sum(y))
    return {0: 1.0, 1: max((len(y) - n_pos) / max(n_pos, 1.0), 1.0)}


def _early_stopping(y_es):
    from tensorflow import keras
    # val_auc is undefined without positives in the early-stopping slice
    monitor, mode = ("val_auc", "max") if np.sum(y_es) > 0 else ("val_loss", "min")
    return keras.callbacks.EarlyStopping(monitor=monitor, mode=mode, patience=5, restore_best_weights=True), monitor


class LSTMScorer:
    """Scaler + trained LSTM; predict() returns fraud probabilities."""

    def __init__(self, model, mean, std, info):
        self.model, self.mean, self.std, self.info = model, mean, std, info

    def predict(self, X):
        return self.model.predict((X - self.mean) / self.std, batch_size=2048, verbose=0).ravel()


def fit_lstm(X, y, timestamps, seed: int = EVAL_SEED) -> LSTMScorer:
    from ..models.lstm_model import build_lstm_model
    _set_seeds(seed)
    flat = X.reshape(-1, X.shape[2])
    mean, std = flat.mean(axis=0), flat.std(axis=0)
    std[std == 0] = 1.0
    Xn = (X - mean) / std
    fit_idx, es_idx = chronological_holdout(timestamps)
    stop, monitor = _early_stopping(y[es_idx])
    model = build_lstm_model(X.shape[1], X.shape[2])
    hist = model.fit(
        Xn[fit_idx], y[fit_idx], validation_data=(Xn[es_idx], y[es_idx]),
        epochs=30, batch_size=64, class_weight=_class_weight(y), callbacks=[stop], verbose=0,
    )
    info = {"rows": int(len(y)), "fraud": int(y.sum()), "early_stop_rows": int(len(es_idx)),
            "early_stop_fraud": int(y[es_idx].sum()), "monitor": monitor, "epochs_run": len(hist.history["loss"])}
    return LSTMScorer(model, mean, std, info)


DNN_PREDICT_BATCH = 4096


def predict_fixed_batch(model, X, batch_size: int = DNN_PREDICT_BATCH) -> np.ndarray:
    """model outputs for X, computed in batches of exactly `batch_size` rows (the
    last batch is zero-padded and the padding discarded).

    A float32 forward pass is not batch-invariant: the CPU matmul kernels that
    Keras/TensorFlow pick depend on the number of rows in the batch, so the same
    row can come out a few ulp different depending on which other rows are
    scored with it (e.g. all rows at once vs. a subset, or one row alone), and
    how much depends on the platform (oneDNN on/off, instruction set, OS build).
    With every batch the same shape, each row's result depends only on the row
    itself, so scoring a subset reproduces scoring everything, bit for bit."""
    X = np.asarray(X, dtype=np.float32)
    out = []
    for start in range(0, len(X), batch_size):
        chunk = X[start:start + batch_size]
        n = len(chunk)
        if n < batch_size:
            chunk = np.concatenate([chunk, np.zeros((batch_size - n,) + chunk.shape[1:], dtype=np.float32)])
        out.append(np.asarray(model.predict_on_batch(chunk)).reshape(batch_size, -1)[:n, 0])
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


class DNNScorer:
    def __init__(self, model, mean, std, info):
        self.model, self.mean, self.std, self.info = model, mean, std, info

    def predict(self, X):
        Xn = np.clip((X - self.mean) / self.std, -CLIP, CLIP)
        return predict_fixed_batch(self.model, Xn)


def fit_dnn(X, y, timestamps, seed: int = EVAL_SEED) -> DNNScorer:
    from ..models.dnn_model import build_dnn_model
    _set_seeds(seed)
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std == 0] = 1.0
    Xn = (X - mean) / std
    fit_idx, es_idx = chronological_holdout(timestamps)
    stop, monitor = _early_stopping(y[es_idx])
    model = build_dnn_model(X.shape[1])
    hist = model.fit(
        Xn[fit_idx], y[fit_idx], validation_data=(Xn[es_idx], y[es_idx]),
        epochs=40, batch_size=128, class_weight=_class_weight(y), callbacks=[stop], verbose=0,
    )
    info = {"rows": int(len(y)), "fraud": int(y.sum()), "early_stop_rows": int(len(es_idx)),
            "early_stop_fraud": int(y[es_idx].sum()), "monitor": monitor, "epochs_run": len(hist.history["loss"])}
    return DNNScorer(model, mean, std, info)


def stacked_risk_scores(X, y, timestamps, groups, train_mask, fit_fn=fit_lstm, n_folds: int = N_FOLDS, seed: int = EVAL_SEED):
    """Returns (probabilities for every row, final LSTM, provenance).

    Rows in train_mask get out-of-fold probabilities (customer-grouped folds);
    every other row is scored by the final LSTM trained on all train_mask rows.
    provenance["fold_of_row"] is the fold index for training rows and -1 for
    rows scored by the final model; provenance["fold_train_groups"][k] lists
    the customers fold model k was trained on (used by the tests)."""
    train_idx = np.flatnonzero(train_mask)
    probs = np.full(len(y), np.nan)
    fold_of_row = np.full(len(y), -1)
    fold_train_groups, fold_info = [], []
    folds = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (tr, ho) in enumerate(folds.split(train_idx, y[train_idx], groups=groups[train_idx])):
        fit_rows, held_rows = train_idx[tr], train_idx[ho]
        scorer = fit_fn(X[fit_rows], y[fit_rows], timestamps[fit_rows], seed=seed + 1 + k)
        probs[held_rows] = scorer.predict(X[held_rows])
        fold_of_row[held_rows] = k
        fold_train_groups.append(sorted(set(groups[fit_rows])))
        fold_info.append(getattr(scorer, "info", {}))
    final = fit_fn(X[train_idx], y[train_idx], timestamps[train_idx], seed=seed)
    other = np.flatnonzero(~train_mask)
    if len(other):
        probs[other] = final.predict(X[other])
    provenance = {"fold_of_row": fold_of_row, "fold_train_groups": fold_train_groups, "fold_info": fold_info}
    return probs, final, provenance
