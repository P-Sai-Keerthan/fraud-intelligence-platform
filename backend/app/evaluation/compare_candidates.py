"""
Candidate comparison on the saved v2 split (Step 4C-2e-c). Evidence only:
it does not choose a model and changes nothing in production.

    cd backend
    python -m app.evaluation.compare_candidates      # -> models/candidates/v2/comparison.json

* Loads the two saved candidates (app/training/candidates.py) exactly as
  they would be used: scaled inputs, +-6 clip on the DNN inputs, and for
  candidate A the risk_score from its final LSTM.
* Scores the validation and test rows of the SAVED v2 time split
  (models/evaluation/v2/split_time.json, checked by load_saved_split).
  Training rows are not scored: the DNNs have seen them.
* Uses each candidate's thresholds from its manifest (chosen on validation
  during training) and checks they reproduce from the validation scores.
  Test labels are only used for measuring.
* Checks the scores are identical to the 4C-2d evaluation's saved scores.
* Reuses app/evaluation/analysis.py for episodes, fraud types,
  false-positive contexts, rings and the customer-cluster bootstrap.

Model scores are NOT calibrated probabilities (class-weighted training).
The JSON output has no timestamps, so reruns are byte-identical.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from ..features.feature_engineering import FEATURE_COLUMNS
from ..features.ground_truth import GROUND_TRUTH_COLUMNS
from ..training.candidates import MANIFEST, load_candidate, load_saved_split
from . import analysis
from .datasets import load_evaluation_data, resolve_dataset
from .metrics import classification_metrics, ranking_metrics, select_threshold
from .run import SCORES_FILES
from .windows import build_windows

A = "candidate_a_dnn_lstm"
B = "candidate_b_dnn_only"
CANDIDATE_DIRS = {A: "dnn_lstm", B: "dnn_only"}
LABELS = {A: "Candidate A (DNN + LSTM)", B: "Candidate B (DNN only)"}
EVAL_COLUMNS = {A: "dnn_fraud_classifier", B: "dnn_without_risk_score"}   # the same models in the 4C-2d scores
ALERT_BANDS = (("Low", 0.0, 0.25), ("Medium", 0.25, 0.50), ("High", 0.50, 0.80), ("Critical", 0.80, 1.01))
SIGNALS = {
    "foreign location": lambda f: f["is_foreign_location"] == 1,
    "new device": lambda f: f["is_new_device"] == 1,
    "unusual hour": lambda f: f["hour_is_unusual"] == 1,
    "unusual category": lambda f: f["category_is_unusual"] == 1,
    "failed logins in previous 24h": lambda f: f["failed_logins_24h"] >= 1,
    "another transaction within the hour (burst)": lambda f: f["txn_velocity_1h"] >= 1,
    "amount >= 300% of customer average": lambda f: f["amount_pct_of_avg"] >= 300,
}


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _r(x, n=6):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


# ---- scoring -------------------------------------------------------------------------------

def score_candidates(spec, candidates_root):
    """Returns (data, scores frame for validation+test rows, candidates, split definition, split sha)."""
    data = load_evaluation_data(spec)
    labels, _, split_def, split_sha = load_saved_split(spec, data)
    df = data.frame
    if set(df.columns) & set(GROUND_TRUTH_COLUMNS):
        raise ValueError("ground-truth columns in the model frame")
    X, y, target = build_windows(df)
    split = np.asarray(labels)[target]
    keep = split != "train"
    F = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)[target]
    cands = {m: load_candidate(CANDIDATE_DIRS[m], spec.version, candidates_root) for m in (A, B)}
    risk = cands[A].risk_score(X[keep])
    scores = pd.DataFrame({
        "transaction_id": df["transaction_id"].to_numpy()[target][keep],
        "split": split[keep],
        "is_fraud": y[keep].astype(int),
        A: cands[A].score(F[keep], risk),
        B: cands[B].score(F[keep]),
        "risk_score_a": risk,
    })
    return data, scores, cands, split_def, split_sha


def check_against_evaluation(scores: pd.DataFrame, spec) -> dict:
    path = spec.output_dir / SCORES_FILES["time"]
    if not path.exists():
        return {"available": False}
    ev = pd.read_csv(path, float_precision="round_trip").set_index("transaction_id").loc[scores["transaction_id"]]
    return {"available": True, **{
        m: {"identical": bool(np.array_equal(scores[m].to_numpy(), ev[EVAL_COLUMNS[m]].to_numpy())),
            "max_abs_difference": float(np.max(np.abs(scores[m].to_numpy() - ev[EVAL_COLUMNS[m]].to_numpy())))}
        for m in (A, B)}}


# ---- sections ------------------------------------------------------------------------------

def overall(test, thr):
    out = {}
    y = test["is_fraud"].to_numpy()
    for m in (A, B):
        c = classification_metrics(y, test[m].to_numpy(), thr[m])
        out[m] = {**ranking_metrics(y, test[m].to_numpy()), **c, "threshold": thr[m],
                  "false_negatives": c["confusion_matrix"]["false_negative"],
                  "false_positives": c["confusion_matrix"]["false_positive"]}
    return out


def operating_points(val, test, cands):
    out = {}
    for m in (A, B):
        pts = {"f1_optimal": cands[m].manifest["thresholds"]["f1_optimal"],
               **{f"fpr_{k}": v for k, v in cands[m].manifest["thresholds"]["fpr_operating_points"].items()}}
        rows = {}
        for name, t in pts.items():
            cv = classification_metrics(val["is_fraud"], val[m], t)
            ct = classification_metrics(test["is_fraud"], test[m], t)
            rows[name] = {"threshold": t, "validation_fpr": cv["false_positive_rate"],
                          "test_precision": ct["precision"], "test_recall": ct["recall"], "test_f1": ct["f1_score"],
                          "test_fpr": ct["false_positive_rate"], "test_alerts_per_1000": ct["alerts_per_1000"],
                          "test_true_positives": ct["confusion_matrix"]["true_positive"],
                          "test_false_positives": ct["confusion_matrix"]["false_positive"]}
        out[m] = rows
    return out


def alert_bands(test):
    """The production /predict bands (25 / 50 / 80 on the score x 100), applied to each candidate."""
    n = len(test)
    out = {}
    for m in (A, B):
        s = np.minimum(test[m].to_numpy(), 0.999)          # production caps the score at 0.999
        bands = {}
        for name, lo, hi in ALERT_BANDS:
            sel = (s >= lo) & (s < hi)
            bands[name] = {"legitimate": int((sel & (test["is_fraud"] == 0)).sum()),
                           "fraud": int((sel & (test["is_fraud"] == 1)).sum()),
                           "per_1000_transactions": _r(sel.sum() / n * 1000, 2)}
        hc = s >= 0.5
        out[m] = {"bands": bands,
                  "high_or_critical_per_1000": _r(hc.sum() / n * 1000, 2),
                  "high_or_critical_legitimate_per_1000": _r((hc & (test["is_fraud"] == 0)).sum() / n * 1000, 2),
                  "high_or_critical_fraud_recall": _r(hc[test["is_fraud"] == 1].mean(), 4)}
    return out


def warning_periods(table, data, thr):
    ep = data.grouping.episodes
    ep = ep[ep["precursor_start"] != ""]
    per_episode = []
    groups = {m: {"before_warning": [], "warning": [], "first_fraud": [], "later_fraud": []} for m in (A, B)}
    for _, e in ep.iterrows():
        cust = table[table["customer_id"] == e["customer_id"]]
        start, first = pd.Timestamp(e["precursor_start"]), pd.Timestamp(e["first_fraud_time"])
        ep_rows = cust[cust["fraud_episode_id"] == e["fraud_episode_id"]]
        firsts = ep_rows[ep_rows["fraud_stage"] == "first"]
        if not len(firsts):                                # first fraud not in validation/test
            continue
        parts = {
            "before_warning": cust[(cust["timestamp"] < start) & (cust["timestamp"] >= start - pd.Timedelta(days=30))
                                   & (cust["is_fraud"] == 0)],
            "warning": cust[(cust["is_precursor"] == 1) & (cust["timestamp"] >= start) & (cust["timestamp"] < first)],
            "first_fraud": firsts,
            "later_fraud": ep_rows[ep_rows["fraud_stage"] == "subsequent"],
        }
        entry = {"episode": int(e["fraud_episode_id"]), "fraud_type": e["fraud_type"],
                 "split": firsts["split"].iloc[0], "warning_rows": int(len(parts["warning"]))}
        for m in (A, B):
            for k, v in parts.items():
                groups[m][k].append(v[m].to_numpy())
            w = parts["warning"]
            entry[m] = {"max_warning_score": _r(w[m].max()) if len(w) else None,
                        "warning_rows_alerted": int((w[m] >= thr[m]).sum()),
                        "alert_during_warning_period": bool((w[m] >= thr[m]).any()),
                        "first_fraud_alerted": bool((firsts[m] >= thr[m]).any())}
        entry["risk_score_a_warning_max"] = _r(parts["warning"]["risk_score_a"].max(), 2) if len(parts["warning"]) else None
        per_episode.append(entry)

    def stats(arrs, m):
        v = np.concatenate(arrs) if arrs else np.array([])
        if not len(v):
            return {"rows": 0}
        return {"rows": int(len(v)), "median": _r(np.median(v)), "mean": _r(v.mean()),
                "share_at_or_above_threshold": _r((v >= thr[m]).mean(), 4)}

    forgot = table[(table["is_fraud"] == 0) & (table["is_precursor"] == 0)
                   & table["legit_context"].str.contains("forgot_password") & ~table["customer_id"].isin(ep["customer_id"])]
    summary = {m: {**{k: stats(v, m) for k, v in groups[m].items()},
                   "forgot_password_rows_other_customers": stats([forgot[m].to_numpy()], m),
                   "episodes_with_an_alert_during_warning_period": sum(e[m]["alert_during_warning_period"] for e in per_episode),
                   "episodes_first_fraud_alerted": sum(e[m]["first_fraud_alerted"] for e in per_episode)}
               for m in (A, B)}
    return {"scope": "episodes whose first fraud is in the validation or test period (out-of-sample for both DNNs); "
                     "an alert on a warning-period row is an alert on a LEGITIMATE transaction, not a fraud detection",
            "episodes": len(per_episode), "per_episode": per_episode, "summary": summary}


def false_positive_signals(test):
    legit = test[test["is_fraud"] == 0]
    rows = []
    for name, fn in [("(all legitimate)", lambda f: pd.Series(True, index=f.index))] + list(SIGNALS.items()):
        flag = fn(legit)
        r = {"signal": name, "legitimate_rows_with_signal": int(flag.sum())}
        for m in (A, B):
            fp = legit[f"alert_{m}"]
            r[f"{m}.false_positives_with_signal"] = int((fp & flag).sum())
            r[f"{m}.share_of_all_false_positives"] = _r((fp & flag).sum() / fp.sum(), 4) if fp.sum() else None
            r[f"{m}.fpr_within_signal"] = _r((fp & flag).sum() / flag.sum(), 5) if flag.sum() else None
        rows.append(r)
    return rows


def false_positive_segments(test):
    legit = test[test["is_fraud"] == 0]
    rows = []
    for seg, g in legit.groupby("customer_segment"):
        r = {"customer_segment": seg, "legitimate_rows": int(len(g))}
        for m in (A, B):
            r[f"{m}.false_positives"] = int(g[f"alert_{m}"].sum())
            r[f"{m}.fpr"] = _r(g[f"alert_{m}"].mean(), 5)
        rows.append(r)
    return rows


def score_differences(test):
    d = test[A] - test[B]
    out = {}
    for name, sel in (("all", slice(None)), ("legitimate", test["is_fraud"] == 0), ("fraud", test["is_fraud"] == 1)):
        v = d[sel].to_numpy()
        out[name] = {"rows": int(len(v)), "mean": _r(v.mean()), "median": _r(np.median(v)),
                     "p05": _r(np.percentile(v, 5)), "p95": _r(np.percentile(v, 95)),
                     "mean_abs": _r(np.abs(v).mean()), "share_a_higher": _r((v > 0).mean(), 4)}
    out["correlation"] = _r(np.corrcoef(test[A], test[B])[0, 1])
    fr = test[test["is_fraud"] == 1].sort_values(["fraud_episode_id", "timestamp"])
    out["fraud_rows"] = [{"episode": int(r.fraud_episode_id), "fraud_type": r.fraud_type, "stage": r.fraud_stage,
                          "a": _r(getattr(r, A)), "b": _r(getattr(r, B)), "risk_score_a": _r(r.risk_score_a, 2)}
                         for r in fr.itertuples()]
    return out


# ---- assemble ------------------------------------------------------------------------------

def compare(spec=None, candidates_root=None) -> dict:
    spec = spec or resolve_dataset("v2")
    root = Path(candidates_root) if candidates_root else config.CANDIDATES_DIR / spec.version
    data, scores, cands, split_def, split_sha = score_candidates(spec, root)
    table = analysis.build_table(data, scores)
    extra = [c for c in FEATURE_COLUMNS if c not in table.columns]
    table = table.merge(data.frame[["transaction_id", *extra]], on="transaction_id", how="left", validate="one_to_one")
    val, test = table[table["split"] == "validation"], table[table["split"] == "test"]
    thr = {m: float(cands[m].manifest["thresholds"]["f1_optimal"]) for m in (A, B)}
    recomputed = {m: select_threshold(val["is_fraud"], val[m]) for m in (A, B)}
    if recomputed != thr:
        raise ValueError(f"manifest thresholds do not reproduce from the validation scores: {thr} vs {recomputed}")
    for m in (A, B):
        table[f"alert_{m}"] = table[m] >= thr[m]
    test = table[table["split"] == "test"]
    val = table[table["split"] == "validation"]

    return {
        "dataset": {"version": spec.version, "file": spec.features_csv.name, "sha256": data.sha256},
        "split": {"file": spec.time_split_path.name, "sha256": split_sha,
                  "train_before": split_def["train_before"], "validation_before": split_def["validation_before"]},
        "features": {"columns": list(FEATURE_COLUMNS),
                     "sha256": hashlib.sha256(json.dumps(list(FEATURE_COLUMNS)).encode()).hexdigest()},
        "candidates": {m: {"label": LABELS[m], "directory": CANDIDATE_DIRS[m],
                           "manifest_sha256": _sha256(root / CANDIDATE_DIRS[m] / MANIFEST),
                           "dnn_weights_sha256": cands[m].manifest["model"]["dnn_weights_sha256"],
                           "lstm_weights_sha256": cands[m].manifest["model"].get("lstm_weights_sha256"),
                           "dnn_input_columns": cands[m].manifest["model"]["dnn_input_columns"],
                           "thresholds": cands[m].manifest["thresholds"]} for m in (A, B)},
        "scores_note": "model scores are not calibrated probabilities (class-weighted training); thresholds come "
                       "from the validation split only",
        "rows": {"validation": int(len(val)), "validation_fraud": int(val["is_fraud"].sum()),
                 "test": int(len(test)), "test_fraud": int(test["is_fraud"].sum()),
                 "test_legitimate": int((test["is_fraud"] == 0).sum()),
                 "test_fraud_episodes": int(test.loc[test["is_fraud"] == 1, "fraud_episode_id"].nunique())},
        "matches_4c2d_evaluation_scores": check_against_evaluation(scores, spec),
        "overall_test": overall(test, thr),
        "operating_points": operating_points(val, test, cands),
        "alert_bands_production_25_50_80": alert_bands(test),
        "episodes_test": analysis.first_fraud(test, models=[A, B]),
        "warning_periods": warning_periods(table, data, thr),
        "false_positives": {"by_legit_context": analysis.legit_false_positives(test, models=[A, B]),
                            "by_signal": false_positive_signals(test),
                            "by_customer_segment": false_positive_segments(test)},
        "fraud_types": analysis.fraud_types(table, models=[A, B]),
        "rings": analysis.rings(table, data, models=[A, B]),
        "lstm_contribution": {"detection_overlap_test": analysis.lstm_ablation(test, A, B),
                              "score_differences_test": score_differences(test),
                              "uncertainty_test": analysis.uncertainty(test, models=[A, B], diff_pair=(A, B))},
        "limitations": [
            "one synthetic dataset (v2 generator 2.0.1); the difficulty comes from generator settings",
            "14 test fraud episodes and 61 test fraud transactions; one episode changes first-fraud recall by ~7 points",
            "no fraud ring and one warning-period episode in the test period",
            "scores are not calibrated probabilities; thresholds come from 119 validation fraud rows",
            "both candidates are the 4C-2d evaluation models; this comparison adds no new training evidence",
            "DNN training inputs were unclipped while scoring clips them to +-6 (as evaluated in 4C-2d)",
        ],
    }


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--candidates-root", default=None, help="default: models/candidates/v2/")
    p.add_argument("--output", default=None, help="default: <candidates-root>/comparison.json")
    args = p.parse_args(argv)
    root = Path(args.candidates_root) if args.candidates_root else config.CANDIDATES_DIR / "v2"
    result = compare(candidates_root=root)
    out = Path(args.output) if args.output else root / "comparison.json"
    out.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(f"wrote {out}")
    return result


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
