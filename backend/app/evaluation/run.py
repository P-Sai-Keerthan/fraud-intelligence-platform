"""
Run the corrected evaluation end to end and write the report.

    cd backend
    python -m app.evaluation.run                 # v1, both splits (~11 min on a 2-core CPU)
    python -m app.evaluation.run --skip-customer # time-based split only
    python -m app.evaluation.run --dataset v2    # the generated v2 dataset (evaluation only)
    python -m app.evaluation.run --output-dir /tmp/eval   # write somewhere else

Writes, for the default dataset v1 (models/evaluation/):
    split_time.json, split_customer.json   saved split definitions
    evaluation_report.json                  served by GET /metrics and /metrics/report
    time_split/                             the evaluation LSTM + DNN (+ scalers)
and the same files under models/evaluation/v2/ for --dataset v2, which
/metrics never reads. Dataset paths come from datasets.resolve_dataset.

Nothing here touches the production models in models/saved/.
"""

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..config import FEATURES_CSV
from ..features.feature_engineering import FEATURE_COLUMNS
from ..features.ground_truth import LABEL_COLUMN, assert_no_ground_truth
from . import baselines
from .datasets import DATASETS, DEFAULT_DATASET_VERSION, dataset_info, load_evaluation_data, resolve_dataset
from .metrics import episode_metrics, evaluate_scores, ranking_metrics
from .split import build_customer_split, build_time_split, canonical_order, save_definition
from .stacking import EVAL_SEED, N_FOLDS, fit_dnn, fit_lstm, stacked_risk_scores
from .windows import assert_past_only, build_windows


def _log(msg):
    print(f"[evaluation {datetime.now():%H:%M:%S}] {msg}", flush=True)


def first_fraud_flags(df: pd.DataFrame, episode_id: pd.Series) -> np.ndarray:
    """True for the earliest fraud transaction of each episode (df in canonical order)."""
    flags = np.zeros(len(df), dtype=bool)
    fraud = df.assign(episode_id=episode_id)[episode_id > 0]
    firsts = fraud.sort_values(["timestamp", "transaction_id"]).groupby("episode_id").head(1).index
    flags[firsts] = True
    return flags


def evaluate_split(df, split_labels, episode_id, windows, fit_lstm_fn=fit_lstm, fit_dnn_fn=fit_dnn,
                   n_folds=N_FOLDS, seed=EVAL_SEED, log=_log):
    """Leakage-safe evaluation of the stacked LSTM -> DNN and the baselines
    on one split. `windows` = (X, y, target_index) from build_windows."""
    # the frame must hold identifiers, the target and the production features only:
    # ground-truth / metadata columns (v2) are kept out of every feature matrix
    assert_no_ground_truth([c for c in df.columns if c != LABEL_COLUMN], "columns of the evaluation frame")
    X, y, target = windows
    split = np.asarray(split_labels)[target]
    tr, va, te = split == "train", split == "validation", split == "test"
    ts = df["timestamp"].to_numpy()[target]
    groups = df["customer_id"].to_numpy()[target]
    F = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)[target]    # the scored transaction's own features

    log(f"LSTM: {n_folds} out-of-fold models + final model on {int(tr.sum())} training windows")
    lstm_p, final_lstm, prov = stacked_risk_scores(X, y, ts, groups, tr, fit_fn=fit_lstm_fn, n_folds=n_folds, seed=seed)
    risk = np.round(lstm_p * 100, 2)                               # production risk_score scale

    log("DNN with LSTM risk_score (trained on out-of-fold risk scores)")
    dnn = fit_dnn_fn(np.column_stack([F, risk]).astype(np.float32)[tr], y[tr], ts[tr], seed=seed)
    dnn_p = dnn.predict(np.column_stack([F, risk]).astype(np.float32))

    log("baselines: DNN without risk_score, logistic regression, amount/hour rule")
    dnn_nr = fit_dnn_fn(F[tr], y[tr], ts[tr], seed=seed)
    dnn_nr_p = dnn_nr.predict(F)
    lr = baselines.fit_logistic_regression(F[tr], y[tr], seed=seed)
    lr_p = baselines.logistic_regression_scores(lr, F)
    rule_p = baselines.amount_hour_rule_scores(F)

    firsts = first_fraud_flags(df, episode_id)[target]
    ep_frame = pd.DataFrame({
        "customer_id": groups[te], "timestamp": ts[te], "is_fraud": y[te].astype(int),
        "episode_id": np.asarray(episode_id)[target][te], "is_first_fraud": firsts[te],
    })

    def full(scores):
        m = evaluate_scores(y[va], scores[va], y[te], scores[te])
        m["episodes"] = episode_metrics(ep_frame, scores[te] >= m["threshold"])
        return m

    result = {
        "lstm_risk_predictor": full(lstm_p),
        "dnn_fraud_classifier": full(dnn_p),
        "baselines": {
            "amount_hour_rule": full(rule_p),
            "logistic_regression": full(lr_p),
            "dnn_without_risk_score": full(dnn_nr_p),
        },
        "stacking": {
            "n_folds": n_folds,
            "fold_models": prov["fold_info"],
            "final_lstm": getattr(final_lstm, "info", {}),
            "dnn": getattr(dnn, "info", {}),
            "train_out_of_fold_lstm": ranking_metrics(y[tr], lstm_p[tr]),
        },
        "evaluated_rows": {
            name: {"windows": int(mask.sum()), "fraud": int(y[mask].sum())}
            for name, mask in (("train", tr), ("validation", va), ("test", te))
        },
    }
    result["lstm_risk_predictor"]["task"] = "fraud of the NEXT transaction, from the 10 transactions before it"
    result["dnn_fraud_classifier"]["task"] = "fraud of the current transaction, from its 9 features + LSTM risk_score"
    # per-window scores for later analysis (by fraud type, warning period, ...). On
    # training rows the LSTM scores are out-of-fold, the other models' are in-sample.
    scores = pd.DataFrame({
        "transaction_id": df["transaction_id"].to_numpy()[target],
        "split": split,
        "is_fraud": y.astype(int),
        "lstm_risk_predictor": lstm_p,
        "dnn_fraud_classifier": dnn_p,
        "dnn_without_risk_score": dnn_nr_p,
        "logistic_regression": lr_p,
        "amount_hour_rule": rule_p,
    })
    models = {"lstm": final_lstm, "dnn": dnn, "scores": scores}
    return result, models


def legacy_random_split_metrics() -> dict:
    """The OLD methodology (random stratified 80/20 split, threshold 0.5)
    applied to the production models, reproduced for comparison only."""
    from sklearn.model_selection import train_test_split
    from tensorflow import keras
    from ..config import (DNN_FEATURE_MEAN_PATH, DNN_FEATURE_STD_PATH, DNN_MODEL_PATH,
                          LSTM_FEATURE_MEAN_PATH, LSTM_FEATURE_STD_PATH, LSTM_MODEL_PATH)
    from ..features.feature_engineering import build_sequences
    from ..models.dnn_model import DNN_INPUT_COLUMNS
    from .metrics import classification_metrics

    feat = pd.read_csv(FEATURES_CSV, parse_dates=["timestamp"])
    X, y, meta = build_sequences(feat)
    _, X_te, _, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    lstm = keras.models.load_model(LSTM_MODEL_PATH)
    mean, std = np.load(LSTM_FEATURE_MEAN_PATH), np.load(LSTM_FEATURE_STD_PATH)
    p_lstm = lstm.predict((X_te - mean) / std, batch_size=2048, verbose=0).ravel()
    p_all = lstm.predict((X - mean) / std, batch_size=2048, verbose=0).ravel()
    merged = feat.merge(meta.assign(risk_score=[round(float(p) * 100, 2) for p in p_all])[["transaction_id", "risk_score"]],
                        on="transaction_id", how="inner")
    Xd, yd = merged[DNN_INPUT_COLUMNS].values.astype(np.float32), merged["is_fraud"].values.astype(np.float32)
    _, Xd_te, _, yd_te = train_test_split(Xd, yd, test_size=0.2, random_state=42, stratify=yd)
    dnn = keras.models.load_model(DNN_MODEL_PATH)
    dm, ds = np.load(DNN_FEATURE_MEAN_PATH), np.load(DNN_FEATURE_STD_PATH)
    p_dnn = dnn.predict(np.clip((Xd_te - dm) / ds, -6, 6), batch_size=4096, verbose=0).ravel()
    out = {}
    for name, yy, pp in (("lstm_risk_predictor", y_te, p_lstm), ("dnn_fraud_classifier", yd_te, p_dnn)):
        out[name] = {**classification_metrics(yy, pp, 0.5), **ranking_metrics(yy, pp), "threshold": 0.5}
    return out


def _save_models(models, directory):
    directory.mkdir(parents=True, exist_ok=True)
    models["lstm"].model.save(directory / "lstm_risk_model.keras")
    np.save(directory / "lstm_feature_mean.npy", models["lstm"].mean)
    np.save(directory / "lstm_feature_std.npy", models["lstm"].std)
    models["dnn"].model.save(directory / "dnn_fraud_model.keras")
    np.save(directory / "dnn_feature_mean.npy", models["dnn"].mean)
    np.save(directory / "dnn_feature_std.npy", models["dnn"].std)


SCORES_FILES = {"time": "scores_time_split.csv.gz", "customer": "scores_customer_split.csv.gz"}


def _save_scores(scores: pd.DataFrame, path) -> None:
    """Per-window scores (datasets with metadata only, i.e. v2), for app.evaluation.analysis."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # no file name and mtime=0 in the gzip header, so reruns are byte-identical
    import gzip
    with open(path, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as fh:
        fh.write(scores.to_csv(index=False, float_format="%.17g", lineterminator="\n").encode())


def _display_path(path) -> str:
    """Path relative to backend/ when possible (the report has no machine-specific paths)."""
    from ..config import BACKEND_DIR
    try:
        return path.relative_to(BACKEND_DIR).as_posix()
    except ValueError:
        return path.as_posix()


def feature_version() -> dict:
    return {"columns": list(FEATURE_COLUMNS), "count": len(FEATURE_COLUMNS),
            "sha256": hashlib.sha256(json.dumps(list(FEATURE_COLUMNS)).encode()).hexdigest()}


def main(skip_customer: bool = False, dataset_version: str | None = None, output_dir=None):
    spec = resolve_dataset(dataset_version)             # ValueError for an unknown version
    if output_dir is not None:
        spec = spec.with_output_dir(output_dir)
    import sklearn
    import tensorflow as tf
    started = time.time()
    _log(f"dataset {spec.version}: loading {spec.features_csv}")
    data = load_evaluation_data(spec)                   # FileNotFoundError if missing; no fallback
    df = data.frame                                     # identifiers + target + FEATURE_COLUMNS only

    labels, episode_id, time_def = build_time_split(df, data.grouping)
    save_definition(spec.time_split_path, time_def)
    cust_labels, cust_def = build_customer_split(df, episode_id, data.grouping)
    save_definition(spec.customer_split_path, cust_def)
    _log(f"time split: train < {time_def['train_before']} <= validation < {time_def['validation_before']} <= test")

    windows = build_windows(df)
    assert_past_only(df, windows[2])
    _log(f"{len(windows[1])} past-only windows built and verified")

    _log("=== primary: time-based split ===")
    primary, models = evaluate_split(df, labels, episode_id, windows)
    _save_models(models, spec.models_dir)
    if spec.has_metadata:
        _save_scores(models["scores"], spec.output_dir / SCORES_FILES["time"])

    secondary = None
    if not skip_customer:
        _log("=== secondary: customer-grouped split ===")
        secondary, secondary_models = evaluate_split(df, cust_labels, episode_id, windows)
        if spec.has_metadata:
            _save_scores(secondary_models["scores"], spec.output_dir / SCORES_FILES["customer"])

    if spec.legacy_comparison:
        _log("legacy random-split metrics of the production models (for comparison)")
        legacy = {
            "note": "OLD methodology: random stratified 80/20 split, threshold 0.5, production models. Kept for comparison only.",
            **legacy_random_split_metrics(),
        }
    else:
        legacy = None

    report = {
        "report_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runtime_seconds": round(time.time() - started, 1),
        "dataset": dataset_info(data),
        "feature_version": feature_version(),
        "model_version": {
            "recipe": "app/evaluation/stacking.py (architectures and hyperparameters of lstm_model.py / dnn_model.py)",
            "trained_by": "this run, on the training split only",
            "saved_to": _display_path(spec.models_dir),
        },
        "reproducibility": {
            "seed": EVAL_SEED, "out_of_fold_folds": N_FOLDS, "tensorflow_op_determinism": True,
            "versions": {"python": platform.python_version(), "tensorflow": tf.__version__,
                         "scikit_learn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        },
        "methodology": {
            "primary_split": "time-based: train on the earliest transactions, choose thresholds on the next period, test on the latest",
            "secondary_split": "customer-grouped: test customers never seen in training",
            "windows": "the 10 transactions immediately before each target (strictly earlier in (timestamp, transaction_id) order)",
            "stacking": f"DNN training rows get out-of-fold LSTM scores ({N_FOLDS} customer-grouped folds); "
                        "validation/test rows are scored by an LSTM trained on the training split only",
            "threshold": "chosen on validation (F1-maximizing); recall-at-FPR operating points also chosen on validation",
            "early_stopping": "chronologically last 15% of each model's own training rows",
            "evaluated_rows": "transactions with at least 10 earlier transactions (each customer's first 10 have no window)",
            "models_evaluated": f"evaluation copies trained by this run ({_display_path(spec.models_dir)}); "
                                "the production models used by /predict were trained with the legacy method and are not evaluated here",
        },
        "splits": {"time": time_def, "customer": cust_def},
        "primary": primary,
        "secondary_customer_grouped": secondary,
        "legacy_random_split": legacy,
    }
    spec.report_path.parent.mkdir(parents=True, exist_ok=True)
    spec.report_path.write_text(json.dumps(report, indent=2) + "\n")
    _log(f"wrote {spec.report_path} in {report['runtime_seconds']}s")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-customer", action="store_true", help="only run the primary time-based evaluation")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default=DEFAULT_DATASET_VERSION,
                        help=f"dataset version to evaluate (default: {DEFAULT_DATASET_VERSION})")
    parser.add_argument("--output-dir", default=None,
                        help="write the splits, report and evaluation models here instead of the dataset's default")
    args = parser.parse_args()
    main(skip_customer=args.skip_customer, dataset_version=args.dataset, output_dir=args.output_dir)
