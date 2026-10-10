"""
Evaluation metrics with validation-only threshold selection.

* The decision threshold is chosen on the VALIDATION split only: the cut that
  maximizes F1 there (ties: the middle of the tied cuts), placed halfway
  between the two neighbouring validation scores. Test labels never
  influence it.
* "Recall at FPR x" operating points are also chosen on validation: the
  lowest threshold whose validation false-positive rate is <= x. Test
  recall and test FPR are then reported at that threshold.
* Episode metrics: an episode counts as detected if any of its fraud
  transactions is alerted. The first fraud of an episode is its earliest
  fraud transaction; for the LSTM that prediction is made from the window of
  transactions BEFORE it, so first-fraud recall is its early-warning rate.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

FPR_TARGETS = (0.001, 0.01)


def select_threshold(y_val, p_val) -> float:
    """F1-maximizing threshold on validation data (predict fraud if p >= threshold)."""
    y_val = np.asarray(y_val).astype(int)
    p_val = np.asarray(p_val, dtype=float)
    if y_val.sum() == 0:
        raise ValueError("validation split has no fraud; cannot select a threshold")
    uniq = np.unique(p_val)                              # ascending
    # candidate cut k: predict positive for scores >= uniq[k]
    all_sorted = np.sort(p_val)
    pos_sorted = np.sort(p_val[y_val == 1])
    all_ge = len(all_sorted) - np.searchsorted(all_sorted, uniq, side="left")
    pos_ge = len(pos_sorted) - np.searchsorted(pos_sorted, uniq, side="left")
    n_pos = y_val.sum()
    precision = np.divide(pos_ge, all_ge, out=np.zeros(len(uniq)), where=all_ge > 0)
    recall = pos_ge / n_pos
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(len(uniq)), where=(precision + recall) > 0)
    best = np.flatnonzero(np.isclose(f1, f1.max()))
    k = int(best[len(best) // 2])
    return float(uniq[k]) if k == 0 else float((uniq[k - 1] + uniq[k]) / 2)


def threshold_for_fpr(y_val, p_val, target_fpr: float) -> float:
    """Lowest threshold with validation FPR <= target_fpr."""
    y_val = np.asarray(y_val).astype(int)
    neg = np.sort(np.asarray(p_val, dtype=float)[y_val == 0])[::-1]
    allowed_fp = int(np.floor(target_fpr * len(neg)))
    if allowed_fp >= len(neg):
        return float(neg[-1])
    return float(np.nextafter(neg[allowed_fp], np.inf))


def classification_metrics(y, p, threshold: float) -> dict:
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "false_positive_rate": round(float(fp / (fp + tn)) if fp + tn else 0.0, 6),
        "alerts_per_1000": round(float((tp + fp) / len(y) * 1000), 2),
        "confusion_matrix": {
            "true_negative": int(tn), "false_positive": int(fp),
            "false_negative": int(fn), "true_positive": int(tp),
        },
        "test_set_size": int(len(y)),
        "fraud_rate_pct": round(float(y.mean() * 100), 3),
    }


def ranking_metrics(y, p) -> dict:
    y = np.asarray(y).astype(int)
    both = len(np.unique(y)) > 1
    return {
        "pr_auc": round(float(average_precision_score(y, p)), 4) if both else None,
        "auc_roc": round(float(roc_auc_score(y, p)), 4) if both else None,
    }


def evaluate_scores(y_val, p_val, y_test, p_test) -> dict:
    """Threshold chosen on validation, applied to test. Keeps the legacy
    /metrics field names (precision, recall, f1_score, auc_roc,
    confusion_matrix, test_set_size, fraud_rate_pct, threshold)."""
    thr = select_threshold(y_val, p_val)
    out = {**classification_metrics(y_test, p_test, thr), **ranking_metrics(y_test, p_test)}
    out["threshold"] = round(thr, 6)
    out["threshold_source"] = "validation (F1-maximizing)"
    out["recall_at_fpr"] = {}
    for target in FPR_TARGETS:
        t = threshold_for_fpr(y_val, p_val, target)
        m = classification_metrics(y_test, p_test, t)
        out["recall_at_fpr"][f"{target:g}"] = {
            "threshold": round(t, 6), "test_recall": m["recall"], "test_false_positive_rate": m["false_positive_rate"],
        }
    val = classification_metrics(y_val, p_val, thr)
    out["validation"] = {**ranking_metrics(y_val, p_val), **{k: val[k] for k in ("precision", "recall", "f1_score")}}
    return out


def episode_metrics(frame: pd.DataFrame, alert) -> dict:
    """frame: the evaluated rows (one per scored transaction) with columns
    customer_id, timestamp, is_fraud, episode_id (0 = not fraud) and
    is_first_fraud; alert: boolean per row."""
    f = frame.assign(alert=np.asarray(alert, dtype=bool))
    fraud = f[f["is_fraud"] == 1]
    first = fraud[fraud["is_first_fraud"]]
    episodes = []
    for ep_id, rows in fraud.sort_values("timestamp").groupby("episode_id"):
        hit = rows[rows["alert"]]
        episodes.append({
            "episode_id": int(ep_id),
            "fraud_transactions": int(len(rows)),
            "detected": bool(len(hit)),
            "missed_before_first_alert": int(rows["alert"].values.argmax()) if len(hit) else None,
            "hours_to_first_alert": round((hit["timestamp"].iloc[0] - rows["timestamp"].iloc[0]).total_seconds() / 3600, 2) if len(hit) else None,
        })
    detected = [e for e in episodes if e["detected"]]
    delays_tx = [e["missed_before_first_alert"] for e in detected]
    delays_h = [e["hours_to_first_alert"] for e in detected]
    return {
        "fraud_episodes": len(episodes),
        "first_fraud_transactions": int(len(first)),
        "first_fraud_recall": round(float(first["alert"].mean()), 4) if len(first) else None,
        "overall_fraud_recall": round(float(fraud["alert"].mean()), 4) if len(fraud) else None,
        "episodes_detected": len(detected),
        "episode_detection_rate": round(len(detected) / len(episodes), 4) if episodes else None,
        "detection_delay_fraud_transactions": {
            "median": float(np.median(delays_tx)) if delays_tx else None,
            "mean": round(float(np.mean(delays_tx)), 2) if delays_tx else None,
            "max": int(max(delays_tx)) if delays_tx else None,
        },
        "detection_delay_hours": {
            "median": float(np.median(delays_h)) if delays_h else None,
            "max": float(max(delays_h)) if delays_h else None,
        },
    }
