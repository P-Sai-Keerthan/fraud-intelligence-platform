"""
Analysis of a finished evaluation run on a dataset with ground-truth metadata (v2).

    cd backend
    python -m app.evaluation.run --dataset v2         # trains the evaluation models, writes scores
    python -m app.evaluation.analysis --dataset v2    # this module -> models/evaluation/v2/analysis.json

It reads the saved report, the per-window scores (scores_*.csv.gz) and the
dataset's metadata, and never trains or changes a model. The metadata is
used only here, to slice results; it was never a model input.

Thresholds are recomputed from the saved VALIDATION scores with the same
rule as the report (metrics.select_threshold), so every alert below is the
report's alert. Test labels are never used for any choice.

Out-of-sample scope:
* test rows: every model is out-of-sample.
* validation rows: out-of-sample, but thresholds were selected on them, so
  alert-based numbers on validation rows are optimistic (marked as such).
* training rows: LSTM scores are out-of-fold (out-of-sample); the DNN,
  logistic regression and rule scores are in-sample. Only LSTM scores of
  training rows are used (warning-period analysis).

Uncertainty: customer-cluster bootstrap on the test split (resampling whole
customers keeps each customer's transactions and episodes together),
thresholds held fixed at their validation values; percentile intervals.
Episode-level proportions also get Wilson score intervals.
"""

import argparse
import json
import math

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .datasets import load_evaluation_data, resolve_dataset
from .metrics import select_threshold
from .run import SCORES_FILES

MODELS = {
    "dnn_fraud_classifier": "DNN + LSTM risk score",
    "dnn_without_risk_score": "DNN without LSTM risk score",
    "lstm_risk_predictor": "LSTM alone",
    "logistic_regression": "Logistic regression",
    "amount_hour_rule": "Amount/hour rule",
}
MAJOR = list(MODELS)
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 42
BASELINE_DAYS = 30        # "normal" window before a warning period


def _r(x, n=4):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), n)


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def _prf(y, alert):
    tp = int((alert & (y == 1)).sum()); fp = int((alert & (y == 0)).sum())
    fn = int((~alert & (y == 1)).sum()); tn = int((~alert & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1,
            "fpr": fp / (fp + tn) if fp + tn else float("nan"), "alerts_per_1000": (tp + fp) / len(y) * 1000}


# ---- assembly ----------------------------------------------------------------------------

def load_inputs(dataset_version: str = "v2", output_dir=None):
    spec = resolve_dataset(dataset_version)
    if output_dir is not None:
        spec = spec.with_output_dir(output_dir)
    if not spec.has_metadata:
        raise ValueError(f"dataset {spec.version} has no ground-truth metadata; the analysis needs it (use v2)")
    data = load_evaluation_data(spec)
    report = json.loads(spec.report_path.read_text())
    if report["dataset"]["sha256"] != data.sha256:
        raise ValueError("the saved report was produced on different data than the current dataset file")
    scores = pd.read_csv(spec.output_dir / SCORES_FILES["time"], float_precision="round_trip")
    cust_path = spec.output_dir / SCORES_FILES["customer"]
    customer_scores = pd.read_csv(cust_path, float_precision="round_trip") if cust_path.exists() else None
    return spec, data, report, scores, customer_scores


def build_table(data, scores: pd.DataFrame) -> pd.DataFrame:
    """One row per scored window: scores + model frame identifiers + metadata."""
    frame = data.frame[["customer_id", "transaction_id", "timestamp", "device_id", "is_fraud"] +
                       ["hour_is_unusual", "amount_pct_of_avg"]]
    table = scores.merge(frame, on="transaction_id", how="left", validate="one_to_one", suffixes=("", "_frame"))
    table = table.merge(data.metadata, on="transaction_id", how="left", validate="one_to_one")
    if not (table["is_fraud"] == table["is_fraud_frame"]).all():
        raise ValueError("scores and dataset disagree on is_fraud")
    return table.drop(columns=["is_fraud_frame"])


def thresholds(table: pd.DataFrame, models=None) -> dict:
    val = table[table["split"] == "validation"]
    return {m: select_threshold(val["is_fraud"], val[m]) for m in (models or MAJOR)}


def analyse(data, report: dict, scores: pd.DataFrame, v1_report: dict | None = None,
            customer_scores: pd.DataFrame | None = None) -> dict:
    table = build_table(data, scores)
    thr = thresholds(table)
    for m in MAJOR:
        table[f"alert_{m}"] = table[m] >= thr[m]
        reported = report["primary"][m]["threshold"] if m in report["primary"] else report["primary"]["baselines"][m]["threshold"]
        if abs(round(thr[m], 6) - reported) > 1e-6:
            raise ValueError(f"threshold for {m} does not reproduce the report ({thr[m]} vs {reported})")
    test = table[table["split"] == "test"]
    out = {
        "thresholds": {m: thr[m] for m in MAJOR},
        "primary": primary_metrics(report),
        "split": split_summary(report),
        "v1_vs_v2": v1_vs_v2(v1_report, report) if v1_report else None,
        "lstm_ablation": lstm_ablation(test),
        "first_fraud": first_fraud(test),
        "warning_period": warning_period(table, data, thr),
        "fraud_types": fraud_types(table),
        "legit_false_positives": legit_false_positives(test),
        "rings": rings(table, data),
        "sanity": sanity(table, data, report),
        "uncertainty": uncertainty(test),
        "secondary_customer_split": secondary_summary(report),
        "lstm_ablation_customer_split": ablation_customer_split(data, customer_scores) if customer_scores is not None else None,
    }
    return out


# ---- sections ---------------------------------------------------------------------------------

def _model_block(report_section, m):
    return report_section[m] if m in report_section else report_section["baselines"][m]


def primary_metrics(report):
    rows = {}
    for m in MAJOR:
        r = _model_block(report["primary"], m)
        rows[m] = {k: r.get(k) for k in ("pr_auc", "auc_roc", "precision", "recall", "f1_score", "threshold",
                                          "false_positive_rate", "alerts_per_1000", "confusion_matrix",
                                          "test_set_size", "fraud_rate_pct")}
        rows[m]["episodes"] = r["episodes"]
        rows[m]["recall_at_fpr"] = r.get("recall_at_fpr")
        rows[m]["validation"] = r.get("validation")
    return rows


def split_summary(report):
    t = report["splits"]["time"]
    return {k: t[k] for k in ("train_before", "validation_before", "episode_rule", "boundary_rule",
                              "episodes_total", "episode_groups_total", "episodes_moved", "splits")} | {
        "evaluated_rows": report["primary"]["evaluated_rows"]}


def secondary_summary(report):
    s = report.get("secondary_customer_grouped")
    if not s:
        return None
    return {m: {k: _model_block(s, m).get(k) for k in ("pr_auc", "auc_roc", "precision", "recall", "f1_score",
                                                     "false_positive_rate", "alerts_per_1000")} |
            {"first_fraud_recall": _model_block(s, m)["episodes"]["first_fraud_recall"],
             "fraud_episodes": _model_block(s, m)["episodes"]["fraud_episodes"]} for m in MAJOR} | {
        "evaluated_rows": s["evaluated_rows"], "split": report["splits"]["customer"]}


def v1_vs_v2(v1, v2):
    def col(r):
        test = r["splits"]["time"]["splits"]["test"]
        p = r["primary"]
        return {
            "test_transactions": test["transactions"], "test_fraud_transactions": test["fraud_transactions"],
            "test_fraud_episodes": test["fraud_episodes"], "test_fraud_rate_pct": test["fraud_rate_pct"],
            "evaluated_test_windows": p["evaluated_rows"]["test"]["windows"],
            "train_before": r["splits"]["time"]["train_before"],
            "validation_before": r["splits"]["time"]["validation_before"],
            "episode_rule": r["splits"]["time"]["episode_rule"],
            **{f"{m}.{k}": _model_block(p, m).get(k) for m in MAJOR for k in
               ("pr_auc", "auc_roc", "precision", "recall", "f1_score", "false_positive_rate", "alerts_per_1000", "threshold")},
            **{f"{m}.first_fraud_recall": _model_block(p, m)["episodes"]["first_fraud_recall"] for m in MAJOR},
        }
    return {"v1": col(v1), "v2": col(v2)}


def lstm_ablation(test, with_lstm: str = "dnn_fraud_classifier", without_lstm: str = "dnn_without_risk_score"):
    a, b = test[f"alert_{with_lstm}"], test[f"alert_{without_lstm}"]
    y = test["is_fraud"] == 1
    per = {}
    for m in (with_lstm, without_lstm):
        s = _prf(y.to_numpy(), test[f"alert_{m}"].to_numpy())
        per[m] = {"pr_auc": average_precision_score(y, test[m]), "auc_roc": roc_auc_score(y, test[m]),
                  **{k: s[k] for k in ("precision", "recall", "f1", "fpr", "alerts_per_1000", "tp", "fp")}}
    diff = {k: per[with_lstm][k] - per[without_lstm][k] for k in per[with_lstm]}
    fr = test[y]
    ep_a = fr.groupby("fraud_episode_id")[f"alert_{with_lstm}"].any()
    ep_b = fr.groupby("fraud_episode_id")[f"alert_{without_lstm}"].any()
    first = fr[fr["fraud_stage"] == "first"]
    corr = float(np.corrcoef(test[with_lstm], test[without_lstm])[0, 1])
    return {
        "per_model": per, "difference_with_minus_without": diff,
        "fraud_caught_only_with_lstm": int((y & a & ~b).sum()),
        "fraud_caught_only_without_lstm": int((y & ~a & b).sum()),
        "fraud_caught_by_both": int((y & a & b).sum()), "fraud_missed_by_both": int((y & ~a & ~b).sum()),
        "false_positives_only_with_lstm": int((~y & a & ~b).sum()),
        "false_positives_only_without_lstm": int((~y & ~a & b).sum()),
        "false_positives_both": int((~y & a & b).sum()),
        "episodes_detected_with": int(ep_a.sum()), "episodes_detected_without": int(ep_b.sum()),
        "episodes_only_with_lstm": int((ep_a & ~ep_b).sum()), "episodes_only_without_lstm": int((~ep_a & ep_b).sum()),
        "first_frauds_only_with_lstm": int((first[f"alert_{with_lstm}"] & ~first[f"alert_{without_lstm}"]).sum()),
        "first_frauds_only_without_lstm": int((~first[f"alert_{with_lstm}"] & first[f"alert_{without_lstm}"]).sum()),
        "score_correlation_test": corr,
    }


def _episode_detail(fr: pd.DataFrame, m: str) -> dict:
    """fr: fraud rows of one or more episodes. Per-episode first-fraud / delay stats for model m."""
    eps = []
    for eid, rows in fr.sort_values("timestamp").groupby("fraud_episode_id"):
        alerts = rows[f"alert_{m}"].to_numpy()
        first_row = rows[rows["fraud_stage"] == "first"]
        det = bool(alerts.any())
        k = int(alerts.argmax()) if det else None
        eps.append({
            "episode": int(eid), "fraud_type": rows["fraud_type"].iloc[0], "fraud_rows": len(rows),
            "first_detected": bool(first_row[f"alert_{m}"].any()) if len(first_row) else None,
            "detected": det,
            "delay_fraud_transactions": k,
            "delay_hours": round((rows["timestamp"].iloc[k] - rows["timestamp"].iloc[0]).total_seconds() / 3600, 2) if det else None,
        })
    return eps


def _summarise_episodes(eps):
    n = len(eps)
    first = sum(e["first_detected"] for e in eps)
    det = [e for e in eps if e["detected"]]
    later = [e for e in det if not e["first_detected"]]
    return {
        "episodes": n, "first_fraud_transactions": n, "first_fraud_detected": first,
        "first_fraud_recall": _r(first / n) if n else None, "first_fraud_recall_wilson95": wilson(first, n),
        "episodes_detected": len(det), "episodes_detected_only_after_onset": len(later),
        "episodes_never_detected": n - len(det),
        "median_delay_fraud_transactions_detected": _r(np.median([e["delay_fraud_transactions"] for e in det]), 2) if det else None,
        "median_delay_hours_detected": _r(np.median([e["delay_hours"] for e in det]), 2) if det else None,
        "median_delay_hours_detected_after_onset": _r(np.median([e["delay_hours"] for e in later]), 2) if later else None,
    }


def first_fraud(test, models=None):
    fr = test[test["is_fraud"] == 1]
    out = {}
    for m in (models or MAJOR):
        eps = _episode_detail(fr, m)
        out[m] = {**_summarise_episodes(eps), "per_episode": eps}
    return out


def warning_period(table, data, thr):
    ep = data.grouping.episodes
    ep = ep[ep["precursor_start"] != ""]
    groups = {"1_before_warning": [], "2_warning_period": [], "3_first_fraud": [], "4_later_fraud": []}
    rows_meta = []
    for _, e in ep.iterrows():
        cust = table[table["customer_id"] == e["customer_id"]]
        start, first = pd.Timestamp(e["precursor_start"]), pd.Timestamp(e["first_fraud_time"])
        ep_rows = cust[cust["fraud_episode_id"] == e["fraud_episode_id"]]
        parts = {
            "1_before_warning": cust[(cust["timestamp"] < start) & (cust["timestamp"] >= start - pd.Timedelta(days=BASELINE_DAYS))
                                     & (cust["is_fraud"] == 0) & (cust["is_precursor"] == 0)],
            "2_warning_period": cust[(cust["is_precursor"] == 1) & (cust["timestamp"] >= start) & (cust["timestamp"] < first)],
            "3_first_fraud": ep_rows[ep_rows["fraud_stage"] == "first"],
            "4_later_fraud": ep_rows[ep_rows["fraud_stage"] == "subsequent"],
        }
        for k, v in parts.items():
            groups[k].append(v.assign(episode=int(e["fraud_episode_id"])))
        rows_meta.append({"episode": int(e["fraud_episode_id"]), "fraud_type": e["fraud_type"],
                          "split_of_first_fraud": parts["3_first_fraud"]["split"].iloc[0] if len(parts["3_first_fraud"]) else None,
                          "warning_rows_scored": len(parts["2_warning_period"])})
    other_forgot = table[(table["is_fraud"] == 0) & (table["is_precursor"] == 0)
                         & table["legit_context"].str.contains("forgot_password")
                         & ~table["customer_id"].isin(ep["customer_id"])]

    def stats(df, m):
        if not len(df):
            return {"rows": 0}
        s = df[m]
        return {"rows": int(len(df)), "episodes": int(df["episode"].nunique()) if "episode" in df else None,
                "median": _r(s.median(), 6), "mean": _r(s.mean(), 6), "p90": _r(s.quantile(0.9), 6),
                "share_at_or_above_threshold": _r((s >= thr[m]).mean())}

    out = {"episodes_with_warning_period": rows_meta, "baseline_window_days": BASELINE_DAYS}
    # LSTM: every split is out-of-sample (out-of-fold on training rows)
    lstm = {k: stats(pd.concat(v), "lstm_risk_predictor") for k, v in groups.items()}
    lstm["5_forgot_password_rows_other_customers"] = stats(other_forgot.assign(episode=-1), "lstm_risk_predictor")
    out["lstm_all_splits_out_of_sample"] = lstm
    # all models: validation + test rows only
    vt = {}
    for m in MAJOR:
        d = {k: stats(pd.concat(v).query("split != 'train'"), m) for k, v in groups.items()}
        d["5_forgot_password_rows_other_customers"] = stats(other_forgot.query("split != 'train'").assign(episode=-1), m)
        vt[m] = d
    out["validation_and_test"] = vt
    return out


def fraud_types(table, models=None):
    out = {"test": {}, "validation_and_test_supplementary": {}}
    for scope, df in (("test", table[table["split"] == "test"]),
                      ("validation_and_test_supplementary", table[table["split"] != "train"])):
        fr = df[df["is_fraud"] == 1]
        for t in sorted(fr["fraud_type"].unique()):
            rows = fr[fr["fraud_type"] == t]
            entry = {"episodes": int(rows["fraud_episode_id"].nunique()), "fraud_transactions": int(len(rows))}
            for m in (models or MAJOR):
                eps = _episode_detail(rows, m)
                summ = _summarise_episodes(eps)
                entry[m] = {"recall": _r(rows[f"alert_{m}"].mean()), "caught": int(rows[f"alert_{m}"].sum()),
                            "first_fraud_detected": summ["first_fraud_detected"],
                            "episodes_detected": summ["episodes_detected"],
                            "median_delay_fraud_transactions": summ["median_delay_fraud_transactions_detected"],
                            "median_delay_hours": summ["median_delay_hours_detected"]}
            out[scope][t] = entry
        present = set(fr["fraud_type"])
        out[scope]["_types_without_episodes"] = sorted(set(table["fraud_type"]) - present - {"none"})
    return out


def legit_false_positives(test, models=None):
    legit = test[test["is_fraud"] == 0].copy()
    legit["contexts"] = legit["legit_context"].str.split("|")
    ex = legit.explode("contexts")
    rows = []
    for ctx, g in [("(all legitimate)", legit), ("(no unusual context)", legit[legit["legit_context"] == "none"])] + \
            [(c, ex[ex["contexts"] == c]) for c in sorted(set(ex["contexts"]) - {"none"})]:
        r = {"context": ctx, "legitimate_rows": int(len(g))}
        for m in (models or MAJOR):
            flagged = int(g[f"alert_{m}"].sum())
            r[f"{m}.flagged"] = flagged
            r[f"{m}.fpr"] = _r(flagged / len(g), 5) if len(g) else None
        rows.append(r)
    return rows


def rings(table, data, models=None):
    models = models or MAJOR
    from ..inference_pipeline import FraudIntelligencePipeline
    ep = data.grouping.episodes
    ring_rows = table[table["fraud_ring_id"] > 0]
    per_ring = []
    for rid, g in ring_rows.groupby("fraud_ring_id"):
        splits = sorted(g["split"].unique())
        entry = {"ring": int(rid), "split": splits, "victims": int(g["customer_id"].nunique()),
                 "fraud_transactions": int(len(g)),
                 "out_of_sample": ("all models" if "train" not in splits else
                                   "LSTM only (out-of-fold); DNN, logistic regression and rule scores are in-sample"),
                 "thresholds_selected_on_these_rows": "validation" in splits}
        for m in models:
            caught = g[f"alert_{m}"]
            entry[m] = {"fraud_caught": int(caught.sum()),
                        "victims_detected": int(g[caught]["customer_id"].nunique())}
        per_ring.append(entry)
    test = table[table["split"] == "test"]
    # legitimate rows on devices that several customers use legitimately (households, borrowed phones)
    dev_c = data.frame.groupby("device_id")["customer_id"].nunique()
    fraud_devices = set(data.frame.loc[data.frame["is_fraud"] == 1, "device_id"])
    shared_legit = set(dev_c[dev_c >= 2].index) - fraud_devices
    lt = test[(test["is_fraud"] == 0) & test["device_id"].isin(shared_legit)]
    shared = {"legitimate_test_rows_on_shared_devices": int(len(lt)),
              **{f"{m}.flagged": int(lt[f"alert_{m}"].sum()) for m in models}}

    # the production ring detector, unchanged, run on the full v2 history
    class _Histories:                        # minimal stand-in carrying what detect_fraud_rings reads
        def __init__(self, frame):
            import threading
            self._lock = threading.RLock()
            self.customer_histories = {c: g[["device_id"]] for c, g in frame.groupby("customer_id")}
    flagged = FraudIntelligencePipeline.detect_fraud_rings(_Histories(data.frame))
    ring_members = ep[ep["fraud_ring_id"] > 0].groupby("fraud_ring_id")["customer_id"].apply(set)
    fr = data.frame[data.frame["is_fraud"] == 1]
    ring_devices = set(fr.loc[data.metadata.loc[fr.index, "fraud_ring_id"] > 0, "device_id"])
    kinds = {"ring": 0, "legitimate_only": 0, "other_fraud": 0}
    for r in flagged:
        d = r["identifier"]
        kinds["ring" if d in ring_devices else ("other_fraud" if d in fraud_devices else "legitimate_only")] += 1
    rings_found = sum(any(set(r["customer_ids"]) & members and r["identifier"] in ring_devices for r in flagged)
                      for members in ring_members)
    return {
        "rings_total": int(ring_members.size),
        "rings_in_test": sorted({int(r["ring"]) for r in per_ring if r["split"] == ["test"]}),
        "per_ring": per_ring,
        "legitimate_shared_devices": shared,
        "production_ring_detector": {
            "rule": "a device used by 2+ customers (app/inference_pipeline.py detect_fraud_rings, unchanged)",
            "flagged_devices": len(flagged), "by_kind": kinds,
            "precision_ring_devices": _r(kinds["ring"] / len(flagged)) if flagged else None,
            "rings_with_a_flagged_ring_device": int(rings_found),
            "scope": "full v2 history (all splits); the detector is not a trained model",
        },
    }


def sanity(table, data, report):
    from ..features.feature_engineering import FEATURE_COLUMNS
    from ..features.ground_truth import GROUND_TRUTH_COLUMNS
    f = data.frame
    rule_all = (f["hour_is_unusual"] == 1) & (f["amount_pct_of_avg"] >= 188)
    test_frame = f[f["transaction_id"].isin(table.loc[table["split"] == "test", "transaction_id"])]
    rule_t = (test_frame["hour_is_unusual"] == 1) & (test_frame["amount_pct_of_avg"] >= 188)
    single = {}
    tf = test_frame
    for c in FEATURE_COLUMNS:
        single[c] = {"pr_auc": _r(average_precision_score(tf["is_fraud"], tf[c])),
                     "auc_roc": _r(roc_auc_score(tf["is_fraud"], tf[c]))}
    return {
        "dataset_sha256": data.sha256, "report_dataset_sha256": report["dataset"]["sha256"],
        "matches_manifest": report["dataset"].get("matches_manifest"),
        "feature_columns": list(FEATURE_COLUMNS), "feature_count": len(FEATURE_COLUMNS),
        "report_feature_version": report.get("feature_version"),
        "model_frame_columns": list(f.columns),
        "ground_truth_in_model_frame": sorted(set(f.columns) & set(GROUND_TRUTH_COLUMNS)),
        "v1_rule_full_dataset": {"precision": _r(f.loc[rule_all, "is_fraud"].mean()),
                                 "recall": _r(rule_all[f["is_fraud"] == 1].mean())},
        "v1_rule_test": {"flagged": int(rule_t.sum()), "precision": _r(test_frame.loc[rule_t, "is_fraud"].mean()),
                         "recall": _r(rule_t[test_frame["is_fraud"] == 1].mean())},
        "single_feature_test": single,
    }


def uncertainty(test, reps: int = BOOTSTRAP_REPS, seed: int = BOOTSTRAP_SEED, models=None,
                diff_pair=("dnn_fraud_classifier", "dnn_without_risk_score")):
    models = models or MAJOR
    t = test.reset_index(drop=True)
    idx_by_c = list(t.groupby("customer_id").indices.values())      # resampling unit: a customer
    y_all = t["is_fraud"].to_numpy()
    rng = np.random.default_rng(seed)
    stats = {m: {"precision": [], "recall": [], "f1": [], "pr_auc": []} for m in models}
    diff = []
    skipped = 0
    for _ in range(reps):
        pick = rng.integers(0, len(idx_by_c), len(idx_by_c))
        idx = np.concatenate([idx_by_c[i] for i in pick])
        y = y_all[idx]
        if y.sum() == 0:
            skipped += 1
            continue
        ap = {}
        for m in models:
            a = t[f"alert_{m}"].to_numpy()[idx]
            s = _prf(y, a)
            stats[m]["precision"].append(s["precision"])
            stats[m]["recall"].append(s["recall"])
            stats[m]["f1"].append(s["f1"])
            ap[m] = average_precision_score(y, t[m].to_numpy()[idx])
            stats[m]["pr_auc"].append(ap[m])
        diff.append(ap[diff_pair[0]] - ap[diff_pair[1]])

    def ci(v):
        v = np.asarray([x for x in v if not math.isnan(x)])
        return [_r(np.percentile(v, 2.5)), _r(np.percentile(v, 97.5))] if len(v) else None

    point = {}
    fr = t[t["is_fraud"] == 1]
    first = fr[fr["fraud_stage"] == "first"]
    for m in models:
        s = _prf(y_all, t[f"alert_{m}"].to_numpy())
        k = int(first[f"alert_{m}"].sum())
        point[m] = {
            "precision": _r(s["precision"]), "precision_ci95": ci(stats[m]["precision"]),
            "recall": _r(s["recall"]), "recall_ci95": ci(stats[m]["recall"]),
            "f1": _r(s["f1"]), "f1_ci95": ci(stats[m]["f1"]),
            "pr_auc": _r(average_precision_score(y_all, t[m])), "pr_auc_ci95": ci(stats[m]["pr_auc"]),
            "first_fraud_recall": f"{k}/{len(first)}", "first_fraud_recall_wilson95": wilson(k, len(first)),
        }
    return {
        "method": f"customer-cluster bootstrap on the test split, {reps} resamples (seed {seed}), thresholds fixed "
                  "at their validation values, 2.5/97.5 percentiles; Wilson 95% intervals for episode-level first-fraud recall",
        "test_customers": len(idx_by_c), "test_fraud_customers": int(t.loc[t["is_fraud"] == 1, "customer_id"].nunique()),
        "resamples_without_fraud_skipped": skipped,
        "models": point,
        "pr_auc_difference_dnn_with_minus_without_lstm": {
            "point": _r(average_precision_score(y_all, t[diff_pair[0]]) - average_precision_score(y_all, t[diff_pair[1]])),
            "ci95": ci(diff),
            "share_of_resamples_above_zero": _r(np.mean(np.asarray(diff) > 0)),
        },
    }


def ablation_customer_split(data, customer_scores, reps: int = BOOTSTRAP_REPS, seed: int = BOOTSTRAP_SEED):
    """Threshold-free check of the LSTM contribution on the secondary (customer-grouped) split."""
    t = customer_scores[customer_scores["split"] == "test"].merge(
        data.frame[["transaction_id", "customer_id"]], on="transaction_id", how="left").reset_index(drop=True)
    y = t["is_fraud"].to_numpy()
    a, b = t["dnn_fraud_classifier"].to_numpy(), t["dnn_without_risk_score"].to_numpy()
    idx_by_c = list(t.groupby("customer_id").indices.values())
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(reps):
        idx = np.concatenate([idx_by_c[i] for i in rng.integers(0, len(idx_by_c), len(idx_by_c))])
        if y[idx].sum():
            diffs.append(average_precision_score(y[idx], a[idx]) - average_precision_score(y[idx], b[idx]))
    point = average_precision_score(y, a) - average_precision_score(y, b)
    return {"pr_auc_with": _r(average_precision_score(y, a)), "pr_auc_without": _r(average_precision_score(y, b)),
            "pr_auc_difference": _r(point),
            "ci95": [_r(np.percentile(diffs, 2.5)), _r(np.percentile(diffs, 97.5))],
            "share_of_resamples_above_zero": _r(np.mean(np.asarray(diffs) > 0)),
            "method": f"customer-cluster bootstrap on the customer-split test customers, {reps} resamples (seed {seed})"}


def main(dataset_version: str = "v2", output_dir=None) -> dict:
    spec, data, report, scores, customer_scores = load_inputs(dataset_version, output_dir)
    v1_path = resolve_dataset("v1").report_path
    v1 = json.loads(v1_path.read_text()) if v1_path.exists() else None
    result = analyse(data, report, scores, v1, customer_scores)
    result["dataset"] = report["dataset"]
    path = spec.output_dir / "analysis.json"
    path.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(f"wrote {path}")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="v2", help="dataset version with metadata (default: v2)")
    p.add_argument("--output-dir", default=None, help="where the evaluation run wrote its report and scores")
    args = p.parse_args()
    main(args.dataset, args.output_dir)
