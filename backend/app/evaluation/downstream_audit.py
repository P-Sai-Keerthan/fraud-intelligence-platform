"""
Step 4D-1: audit of the current DNN's near-perfect results.

Answers, with numbers computed from the data in this repository, why the DNN
(production recipe, LSTM risk score -> DNN) reaches accuracy close to 100%:

  A. dataset separability      -> single features and a two-condition rule on the same split
  B. leakage                   -> ground-truth columns, split overlap, duplicates, train/test vector overlap
  C. class imbalance           -> accuracy of "always legitimate"
  D. evaluation methodology    -> accuracy against fraud-specific metrics
  E. an easy synthetic dataset -> the same checks on v1 (production data) and v2 (harder data)

Nothing is trained here and no model file is written. The DNN numbers on v1
are read from models/evaluation/evaluation_report.json (the corrected
time-split evaluation, docs/EVALUATION.md); everything else is computed.

    cd backend
    python -m app.evaluation.downstream_audit       # writes models/evaluation/downstream/audit.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .. import config
from ..features.feature_engineering import FEATURE_COLUMNS
from ..features.ground_truth import FORBIDDEN_FEATURE_COLUMNS
from .datasets import load_evaluation_data, resolve_dataset
from .split import build_time_split, build_customer_split
from .windows import build_windows

OUTPUT_DIR = config.EVALUATION_DIR / "downstream"
AUDIT_NAME = "audit.json"


def _r(x, d=6):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), d)


def _accuracy_from_confusion(cm: dict) -> float:
    total = sum(cm.values())
    return (cm["true_negative"] + cm["true_positive"]) / total


def separability(frame: pd.DataFrame, rows: np.ndarray) -> dict:
    """Per-feature ROC-AUC (oriented so that > 0.5 means 'higher = more fraud') and
    the best single-feature threshold's precision/recall on these rows."""
    y = frame["is_fraud"].to_numpy()[rows]
    out = {}
    for c in FEATURE_COLUMNS:
        x = frame[c].to_numpy(dtype=np.float64)[rows]
        auc = roc_auc_score(y, x) if 0 < y.sum() < len(y) else float("nan")
        orient = 1.0 if auc >= 0.5 else -1.0
        xs = orient * x
        # alert if xs >= t, for every distinct t: counts via a descending sort (ties alerted together)
        order = np.argsort(-xs, kind="mergesort")
        s, ys = xs[order], y[order]
        ends = np.flatnonzero(np.r_[s[1:] != s[:-1], True])
        tp = np.cumsum(ys == 1)[ends]
        fp = np.cumsum(ys == 0)[ends]
        prec = tp / np.maximum(tp + fp, 1)
        rec = tp / max(y.sum(), 1)
        f1 = np.where(tp > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
        i = int(np.argmax(f1))
        best = {"threshold": float(orient * s[ends[i]]), "direction": ">=" if orient > 0 else "<=",
                "precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i])}
        out[c] = {"roc_auc_oriented": _r(max(auc, 1 - auc)), "higher_means_fraud": bool(auc >= 0.5),
                  "best_single_threshold": {k: (_r(v, 4) if isinstance(v, float) else v) for k, v in (best or {}).items()}}
    return out


def label_conflicts(frame: pd.DataFrame, rows: np.ndarray) -> dict:
    """How many rows share an identical 9-feature vector with a row of the other class."""
    f = frame.iloc[rows][FEATURE_COLUMNS + ["is_fraud"]].round(6)
    key = pd.util.hash_pandas_object(f[FEATURE_COLUMNS], index=False)
    g = pd.DataFrame({"key": key.to_numpy(), "y": f["is_fraud"].to_numpy()})
    classes = g.groupby("key")["y"].nunique()
    mixed = set(classes[classes > 1].index)
    in_mixed = g["key"].isin(mixed)
    return {"rows": int(len(g)), "distinct_feature_vectors": int(g["key"].nunique()),
            "fraud_rows_with_an_identical_legitimate_vector": int((in_mixed & (g["y"] == 1)).sum()),
            "fraud_rows": int((g["y"] == 1).sum())}


def train_test_vector_overlap(frame: pd.DataFrame, train_rows, test_rows) -> dict:
    f = frame[FEATURE_COLUMNS].round(6)
    key = pd.util.hash_pandas_object(f, index=False).to_numpy()
    train_keys = set(key[train_rows])
    y = frame["is_fraud"].to_numpy()
    test_fraud = test_rows[y[test_rows] == 1]
    return {"test_rows": int(len(test_rows)),
            "test_rows_whose_vector_occurs_in_training": int(np.isin(key[test_rows], list(train_keys)).sum()),
            "test_fraud_rows": int(len(test_fraud)),
            "test_fraud_rows_whose_vector_occurs_in_training": int(np.isin(key[test_fraud], list(train_keys)).sum())}


def split_integrity(frame: pd.DataFrame, labels: pd.Series, episode_id: pd.Series, definition: dict,
                    customer_labels=None) -> dict:
    ts = frame["timestamp"]
    lab = np.asarray(labels)
    out = {}
    for s in ("train", "validation", "test"):
        m = lab == s
        out[s] = {"rows": int(m.sum()), "fraud": int(frame["is_fraud"].to_numpy()[m].sum()),
                  "first": str(ts[m].min()), "last": str(ts[m].max())}
    ep = pd.DataFrame({"episode": np.asarray(episode_id), "split": lab})
    ep = ep[ep["episode"].astype(str).isin(["0", "-1", ""]) == False]   # noqa: E712
    crossing = int((ep.groupby("episode")["split"].nunique() > 1).sum())
    out["fraud_episodes_crossing_splits"] = crossing
    # straddling episodes are moved whole to the LATER split, so validation/test may start
    # before the boundary; training must hold nothing at or after it
    b1, b2 = pd.Timestamp(definition["train_before"]), pd.Timestamp(definition["validation_before"])
    out["boundaries"] = {"train_before": str(b1), "validation_before": str(b2)}
    out["training_rows_at_or_after_train_boundary"] = int(((lab == "train") & (ts >= b1).to_numpy()).sum())
    out["validation_rows_at_or_after_validation_boundary"] = int(((lab == "validation") & (ts >= b2).to_numpy()).sum())
    if customer_labels is not None:
        cl = np.asarray(customer_labels)
        cust = frame["customer_id"].to_numpy()
        sets = {s: set(cust[cl == s]) for s in ("train", "validation", "test")}
        out["customer_split_customers_shared"] = {
            "train&test": len(sets["train"] & sets["test"]), "train&validation": len(sets["train"] & sets["validation"]),
            "validation&test": len(sets["validation"] & sets["test"])}
    return out


def audit_dataset(version: str) -> dict:
    spec = resolve_dataset(version)
    data = load_evaluation_data(spec)
    f = data.frame
    labels, episode_id, definition = build_time_split(f, data.grouping)
    cust_labels, _ = build_customer_split(f, episode_id, data.grouping)
    _, y_w, target = build_windows(f)
    lab = np.asarray(labels)
    train_rows = target[lab[target] == "train"]
    test_rows = target[lab[target] == "test"]
    y_test = f["is_fraud"].to_numpy()[test_rows]
    gt_in_frame = sorted(set(f.columns) & (FORBIDDEN_FEATURE_COLUMNS - {"is_fraud"}))
    duplicated_ids = int(f["transaction_id"].duplicated().sum())
    exact_dupes = int(f.drop(columns=["transaction_id"]).duplicated().sum())
    return {
        "dataset": {"version": version, "sha256": data.sha256, "transactions": int(len(f)),
                    "fraud_transactions": int(f["is_fraud"].sum()),
                    "fraud_rate": _r(f["is_fraud"].mean()),
                    "scored_rows_with_10_earlier": int(len(target))},
        "class_imbalance": {
            "test_rows": int(len(test_rows)), "test_fraud": int(y_test.sum()),
            "accuracy_of_always_predicting_legitimate": _r(1 - y_test.mean()),
        },
        "leakage_checks": {
            "ground_truth_columns_in_model_frame": gt_in_frame,
            "model_input_columns": list(FEATURE_COLUMNS) + ["risk_score (LSTM output)"],
            "forbidden_columns_among_model_inputs": sorted(set(FEATURE_COLUMNS) & FORBIDDEN_FEATURE_COLUMNS),
            "duplicated_transaction_ids": duplicated_ids,
            "exact_duplicate_rows_ignoring_transaction_id": exact_dupes,
            "time_split": split_integrity(f, labels, episode_id, definition, cust_labels),
            "test_vs_training_feature_vector_overlap": train_test_vector_overlap(f, train_rows, test_rows),
            "lstm_windows_strictly_before_target": "checked by windows.assert_past_only in every training run",
        },
        "separability_on_test_period": separability(f, test_rows),
        "identical_feature_vectors_with_both_labels_test_period": label_conflicts(f, test_rows),
    }


def v1_dnn_from_report() -> dict:
    report = json.loads((config.EVALUATION_DIR / "evaluation_report.json").read_text())
    p = report["primary"]
    out = {}
    for name, block in (("dnn_with_risk_score", p["dnn_fraud_classifier"]),
                        ("lstm_alone", p["lstm_risk_predictor"])):
        cm = block["confusion_matrix"]
        out[name] = {"accuracy": _r(_accuracy_from_confusion(cm)), "precision": block["precision"],
                     "recall": block["recall"], "pr_auc": block["pr_auc"], "roc_auc": block["auc_roc"],
                     "confusion_matrix": cm, "threshold": block["threshold"],
                     "threshold_source": block["threshold_source"]}
    out["baselines_same_split"] = {k: {m: v.get(m) for m in ("pr_auc", "auc_roc", "precision", "recall")}
                                   for k, v in p["baselines"].items()}
    out["source"] = "models/evaluation/evaluation_report.json (primary = time split, test period)"
    return out


def run(output_dir=None) -> Path:
    out = {
        "step": "4D-1 audit of the current DNN",
        "v1_dnn_evaluation": v1_dnn_from_report(),
        "v1": audit_dataset("v1"),
        "v2": audit_dataset("v2"),
    }
    d = Path(output_dir) if output_dir else OUTPUT_DIR
    d.mkdir(parents=True, exist_ok=True)
    path = d / AUDIT_NAME
    path.write_text(json.dumps(out, indent=2) + "\n")
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args(argv)
    print(run(args.output_dir))


if __name__ == "__main__":
    main()
