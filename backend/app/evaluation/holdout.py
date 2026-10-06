"""
Hold-out evaluation of the saved model sets on fresh-seed v2 datasets (Step 4C-3E).

    cd backend
    python -m app.evaluation.holdout --generate      # data/v2_holdout/seed_<s>/ + the report
    python -m app.evaluation.holdout                 # datasets already generated
    python -m app.evaluation.holdout latency         # scoring time per model set (separate file)
    python -m app.evaluation.holdout new-customer --generate   # the new-customer datasets + their report

The candidates were trained on the seed-42 v2 dataset only. A dataset
generated with another seed has different customers and different fraud, so
every row of it is out of sample for the saved candidates (and for
production, which was trained on v1). Nothing is fitted here: no model, no
scaler, no cut-off. The harness

* refuses the training data (seed 42, or the training file's SHA-256);
* scores every transaction the way the live pipeline does (the LSTM on the
  10 earlier transactions when they exist, the cold-start handling when they
  do not), with fixed-shape batches so a row's score does not depend on the
  rows scored with it;
* applies the cut-offs fixed in 4C-3B (CUTOFFS below). They were measured on
  the seed-42 validation period and cannot be changed by the data scored here;
* reports per seed and pooled, with a group-level bootstrap: whole customer
  components (a ring's victims, a household) are resampled within each seed;
* applies the decision rule that was fixed before any model was scored on
  this data (decision_rule), and never selects, promotes or deploys anything.

Scores are model scores on a 0-1 scale. They are not calibrated probabilities.

It reads the model files and never writes to models/saved/ or
models/candidates/; their checksums are compared before and after every run.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd

from .. import config
from ..features.feature_engineering import FEATURE_COLUMNS
from . import ring_metrics
from .analysis import wilson
from .datasets import load_evaluation_data, resolve_dataset

# ---- fixed inputs (approved before the run; see docs/step4c3e-evidence-closure-plan.md) --------------

TRAINING_SEED = 42
HOLDOUT_SEEDS = (101, 102, 103, 104, 105)
HOLDOUT_CUSTOMERS = 500
MODEL_SETS = ("production", "v2_dnn_lstm", "v2_dnn_only")
CANDIDATES = ("v2_dnn_lstm", "v2_dnn_only")
PAIRS = (("v2_dnn_lstm", "v2_dnn_only"), ("v2_dnn_lstm", "production"), ("v2_dnn_only", "production"))

# Alert if score >= cut-off. Measured on the seed-42 validation period in 4C-3B
# (lowest tie-safe cut-off whose validation false-positive rate is within the
# budget: 0.1% for the Critical tier, 1% for Policy B). Never re-tuned here.
CUTOFFS = MappingProxyType({
    "critical": MappingProxyType({"production": 0.9950721263885498, "v2_dnn_lstm": 0.9680655598640442,
                                  "v2_dnn_only": 0.9119433164596558}),
    "policy_b": MappingProxyType({"production": 0.9680148363113403, "v2_dnn_lstm": 0.76771479845047,
                                  "v2_dnn_only": 0.7744449377059937}),
})

# pre-registered margins for the candidate tie-break (v2_dnn_lstm - v2_dnn_only)
RECALL_MARGIN = 0.05
LEGIT_ALERT_MARGIN_PER_1000 = 1.0
CRITICAL_RECALL_MARGIN = 0.05
# rule against production
ALERT_BUDGET_PER_1000 = 10.0          # Policy B: about 1% of transactions
MIN_RECALL = 0.40

MIN_PRIOR = 10                         # primary population: at least 10 earlier transactions
SECONDARY_START = "2026-06-01"         # secondary population: the calendar period of the original test split
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 42
SCORE_CAP = 0.999                      # the live pipeline never reports a score above this

MIN_EPISODES = 100
MIN_RINGS = 20

HOLDOUT_DATA_DIR = config.DATA_DIR / "v2_holdout"
OUTPUT_DIR = config.EVALUATION_DIR / "v2_holdout"
REPORT_NAME = "holdout_report.json"
LATENCY_NAME = "latency.json"

# New-customer fraud: separate datasets from the generator's off-by-default extension
# (2.1.0), kept apart from the main hold-out so the model comparison above is not
# mixed with data from a changed generator.
NEW_CUSTOMER_SEEDS = (201, 202, 203, 204, 205)
NEW_CUSTOMER_LATE_JOINER_SHARE = 0.2
NEW_CUSTOMER_EPISODES_PER_SEED = 30
NEW_CUSTOMER_DATA_DIR = HOLDOUT_DATA_DIR / "new_customer"
NEW_CUSTOMER_REPORT_NAME = "new_customer_report.json"
NEUTRAL_FLAGS = ("hour_is_unusual", "category_is_unusual")
VARIANT_METRICS = ("recall", "precision", "legit_alerts_per_1000", "false_positive_rate",
                   "first_fraud_recall", "episode_detection_rate")

METRICS = ("recall", "precision", "f1", "legit_alerts_per_1000", "false_positive_rate",
           "critical_recall", "critical_precision", "critical_legit_alerts_per_1000",
           "first_fraud_recall", "episode_detection_rate", "pr_auc", "roc_auc")
PRIMARY_ENDPOINTS = ("recall", "legit_alerts_per_1000", "critical_recall")


class HoldoutError(ValueError):
    """A dataset that must not be used as hold-out, or an unsafe output location."""


# ---- datasets ---------------------------------------------------------------------------------

def seed_dir(seed: int, root=None) -> Path:
    return Path(root or HOLDOUT_DATA_DIR) / f"seed_{int(seed)}"


def training_dataset_sha256() -> set:
    """SHA-256 of the data the candidates were trained on (their manifests, and
    the seed-42 dataset manifest when it is present)."""
    shas = set()
    for name in ("dnn_lstm", "dnn_only"):
        p = config.CANDIDATES_DIR / "v2" / name / "manifest.json"
        if p.exists():
            shas.add(json.loads(p.read_text())["dataset"]["sha256"])
    p = config.DATA_V2_DIR / "manifest.json"
    if p.exists():
        m = json.loads(p.read_text())
        if m.get("seed") == TRAINING_SEED:
            shas.add(m["files"]["transactions_with_features.csv"]["sha256"])
    return shas


def check_holdout(data, expected_seed=None) -> None:
    """Raises HoldoutError unless `data` is a generated dataset that the saved
    candidates have never seen and that matches its own manifest."""
    manifest = data.manifest or {}
    seed = manifest.get("seed")
    if seed is None:
        raise HoldoutError("the dataset has no generator seed in its manifest")
    if seed == TRAINING_SEED:
        raise HoldoutError(f"seed {TRAINING_SEED} is the training dataset; it cannot be used as hold-out")
    if data.sha256 in training_dataset_sha256():
        raise HoldoutError("this file is the dataset the candidates were trained on; it cannot be used as hold-out")
    if expected_seed is not None and seed != expected_seed:
        raise HoldoutError(f"expected a dataset generated with seed {expected_seed}, found seed {seed}")
    recorded = manifest.get("files", {}).get(data.spec.features_csv.name, {}).get("sha256")
    if recorded != data.sha256:
        raise HoldoutError(f"{data.spec.features_csv} does not match the checksum in its manifest")


def generate_dataset(seed: int, root=None, customers: int = HOLDOUT_CUSTOMERS, late_joiner_share: float = 0.0,
                     new_customer_fraud_episodes: int = 0) -> Path:
    """Generates one hold-out dataset with the generator CLI, unless it is already
    there. The two new-customer options are passed only when they are not 0."""
    if int(seed) == TRAINING_SEED:
        raise HoldoutError(f"seed {TRAINING_SEED} is the training dataset; it cannot be used as hold-out")
    out = seed_dir(seed, root)
    if not (out / "manifest.json").exists():
        out.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(config.DATA_V2_DIR / "generate.py"), "--seed", str(int(seed)),
                   "--customers", str(int(customers)), "--out", str(out)]
        if late_joiner_share or new_customer_fraud_episodes:
            command += ["--late-joiner-share", str(late_joiner_share),
                        "--new-customer-fraud-episodes", str(int(new_customer_fraud_episodes))]
        subprocess.run(command, check=True, capture_output=True)
    return out


def load_holdout(seed: int, root=None):
    data = load_evaluation_data(resolve_dataset("v2").with_data_dir(seed_dir(seed, root)))
    check_holdout(data, expected_seed=int(seed))
    return data


def dataset_summary(data) -> dict:
    f, ep = data.frame, data.grouping.episodes
    n_prior = prior_counts(f)
    return {
        "seed": data.manifest["seed"], "generator_version": data.manifest.get("generator_version"),
        "sha256": data.sha256, "matches_manifest": True,
        "customers": int(f["customer_id"].nunique()), "groups": int(data.grouping.component_of.nunique()),
        "transactions": int(len(f)), "fraud_transactions": int(f["is_fraud"].sum()),
        "fraud_episodes": int(len(ep)), "rings": int(ep.loc[ep["fraud_ring_id"] > 0, "fraud_ring_id"].nunique()),
        "ring_victim_episodes": int((ep["fraud_ring_id"] > 0).sum()),
        "fraud_transactions_with_fewer_than_10_earlier": int(((n_prior < MIN_PRIOR) & (f["is_fraud"].to_numpy() == 1)).sum()),
        "first_timestamp": str(f["timestamp"].min()), "last_timestamp": str(f["timestamp"].max()),
    }


# ---- scoring ----------------------------------------------------------------------------------

def prior_counts(frame: pd.DataFrame) -> np.ndarray:
    """Number of earlier transactions of the same customer (frame is in canonical order)."""
    return frame.groupby("customer_id", sort=False).cumcount().to_numpy()


def score_frame(frame: pd.DataFrame, model_set, neutral=()) -> np.ndarray:
    """Score (0-1) of every row of a model frame in canonical order, replicating
    FraudIntelligencePipeline.score_transaction:

    * 10 or more earlier transactions: the LSTM (when the model set has one)
      scores the 10 earlier transactions, risk_score = round(output * 100, 2);
    * fewer than 10: the LSTM is not run and risk_score is the training mean
      (scaled 0); with no earlier transaction the baseline-relative features
      are also set to the training mean;
    * DNN inputs are scaled and clipped to +-CLIP; the score is capped at SCORE_CAP.

    `neutral` (measurement only): extra DNN input columns set to the training
    mean for rows with fewer than 10 earlier transactions."""
    from ..inference_pipeline import BASELINE_RELATIVE_FEATURES, MIN_PRIOR_TRANSACTIONS
    from .stacking import CLIP, predict_fixed_batch
    from .windows import build_windows

    features = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    n_prior = prior_counts(frame)
    cold = n_prior < MIN_PRIOR_TRANSACTIONS
    columns = list(model_set.dnn_input_columns)
    if model_set.uses_lstm:
        risk = np.zeros(len(frame), dtype=np.float64)
        if (~cold).any():
            windows, _, target = build_windows(frame)
            if not np.array_equal(np.sort(target), np.flatnonzero(~cold)):
                raise RuntimeError("LSTM windows do not line up with the rows that have 10 earlier transactions")
            prob = predict_fixed_batch(model_set.lstm_model, (windows - model_set.lstm_mean) / model_set.lstm_std)
            risk[target] = np.round(prob.astype(np.float64) * 100, 2)
        raw = np.column_stack([features, risk])
    else:
        raw = features
    scaled = np.clip((raw.astype(np.float32) - model_set.dnn_mean) / model_set.dnn_std, -CLIP, CLIP).astype(np.float32)
    if model_set.uses_lstm:
        scaled[cold, columns.index("risk_score")] = 0.0
    first = n_prior == 0
    for column in BASELINE_RELATIVE_FEATURES:
        scaled[first, columns.index(column)] = 0.0
    for column in neutral:
        scaled[cold, columns.index(column)] = 0.0
    return np.minimum(predict_fixed_batch(model_set.dnn_model, scaled), np.float32(SCORE_CAP))


def build_table(data, scores: dict, cutoffs=None) -> pd.DataFrame:
    """One row per transaction: identifiers, ground truth (for slicing only),
    the bootstrap group, the scores and the alerts at the fixed cut-offs
    (`cutoffs`: {tier: {model: value}}; default: the 4C-3B table CUTOFFS)."""
    f, meta = data.frame, data.metadata
    seed = int(data.manifest["seed"])
    component = f["customer_id"].map(data.grouping.component_of)
    if component.isna().any():
        raise ValueError("a customer has no group")
    table = pd.DataFrame({
        "seed": seed, "customer_id": f["customer_id"].to_numpy(), "transaction_id": f["transaction_id"].to_numpy(),
        "timestamp": f["timestamp"].to_numpy(), "device_id": f["device_id"].to_numpy(),
        "is_fraud": f["is_fraud"].to_numpy().astype(int), "n_prior": prior_counts(f),
        "group": [f"{seed}:{c}" for c in component],
        "episode": [f"{seed}:{e}" for e in meta["fraud_episode_id"].astype(int)],
    })
    for c in ("fraud_type", "fraud_episode_id", "fraud_stage", "fraud_ring_id", "legit_context"):
        table[c] = meta[c].to_numpy()
    for m, s in scores.items():
        table[m] = np.asarray(s, dtype=np.float32)
    return add_alerts(table, list(scores), cutoffs)


def add_alerts(table: pd.DataFrame, models, cutoffs=None) -> pd.DataFrame:
    cutoffs = CUTOFFS if cutoffs is None else cutoffs
    for m in models:
        table[f"alert_{m}"] = table[m].to_numpy() >= np.float32(cutoffs["policy_b"][m])
        table[f"critical_{m}"] = table[m].to_numpy() >= np.float32(cutoffs["critical"][m])
    return table


def population(table: pd.DataFrame, name: str) -> pd.DataFrame:
    full = table["n_prior"] >= MIN_PRIOR
    if name == "primary":
        return table[full]
    if name == "secondary":
        return table[full & (table["timestamp"] >= pd.Timestamp(SECONDARY_START))]
    if name == "early_history":
        return table[~full]
    raise ValueError(f"unknown population {name!r}")


# ---- weighted, tie-aware ranking metrics --------------------------------------------------------

def rank_prepare(scores: np.ndarray):
    """(order, block_ends): rows by descending score, and the last position of every run of equal scores."""
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="mergesort")
    s = np.asarray(scores)[order]
    ends = np.flatnonzero(np.r_[s[1:] != s[:-1], True])
    return order, ends


def weighted_rank_metrics(pos_w: np.ndarray, neg_w: np.ndarray, ends: np.ndarray):
    """(average precision, ROC AUC) from the weights of positives and negatives
    in descending-score order. Equal scores form one threshold, as in
    scikit-learn's average_precision_score / roc_auc_score with sample_weight."""
    cp, cn = np.cumsum(pos_w)[ends], np.cumsum(neg_w)[ends]
    P, N = cp[-1], cn[-1]
    if P <= 0 or N <= 0:
        return float("nan"), float("nan")
    dp, dn = np.diff(cp, prepend=0.0), np.diff(cn, prepend=0.0)
    total = cp + cn
    precision = np.divide(cp, total, out=np.zeros_like(cp, dtype=np.float64), where=total > 0)
    ap = float(np.sum(dp * precision) / P)
    auc = float(np.sum(dp * (N - cn + 0.5 * dn)) / (P * N))
    return ap, auc


# ---- metrics and the group bootstrap --------------------------------------------------------------

def _ratio(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return np.divide(a, b, out=np.full(np.broadcast(a, b).shape, np.nan), where=b > 0)


def _metrics_from_sums(s: dict, m: str) -> dict:
    P, N = s["fraud"], s["rows"]
    tp, fp, ctp, cfp = s[f"tp_{m}"], s[f"fp_{m}"], s[f"ctp_{m}"], s[f"cfp_{m}"]
    return {
        "recall": _ratio(tp, P), "precision": _ratio(tp, tp + fp), "f1": _ratio(2 * tp, tp + fp + P),
        "legit_alerts_per_1000": 1000 * _ratio(fp, N), "false_positive_rate": _ratio(fp, N - P),
        "critical_recall": _ratio(ctp, P), "critical_precision": _ratio(ctp, ctp + cfp),
        "critical_legit_alerts_per_1000": 1000 * _ratio(cfp, N),
        "first_fraud_recall": _ratio(s[f"first_{m}"], s["first"]),
        "episode_detection_rate": _ratio(s[f"episodes_{m}"], s["episodes"]),
    }


def _r(x, digits: int = 6):
    x = float(x)
    return None if np.isnan(x) else round(x, digits)


def _ci(values):
    v = np.asarray(values, dtype=np.float64)
    v = v[~np.isnan(v)]
    if not len(v):
        return None
    lo, hi = np.percentile(v, [2.5, 97.5])
    return [_r(lo), _r(hi)]


def evaluate_population(table: pd.DataFrame, models=MODEL_SETS, reps: int = BOOTSTRAP_REPS,
                        seed: int = BOOTSTRAP_SEED, return_resamples: bool = False, pairs=None):
    """Point values, 95% bootstrap intervals and paired differences on one set of rows.

    The bootstrap resamples whole groups (customer components) with replacement
    within each dataset seed, so a ring or a household is never split and every
    resample contains every dataset. All model sets are evaluated on the same
    resample, which makes the differences paired.

    return_resamples=True also returns the per-resample values ({model: {metric:
    array}}); two calls on the same rows with the same seed use the same resamples.
    `pairs`: the (a, b) model pairs whose difference a - b is reported (default: PAIRS)."""
    models = list(models)
    t = table.reset_index(drop=True)
    y = t["is_fraud"].to_numpy() == 1
    group_codes, group_names = pd.factorize(t["group"], sort=True)
    n_groups = len(group_names)
    group_seed = pd.Series(t["seed"].to_numpy(), index=group_codes).groupby(level=0).first().to_numpy()
    first = y & (t["fraud_stage"].to_numpy() == "first")

    # per-group sums of everything that is additive over rows
    columns = {"rows": np.ones(len(t)), "fraud": y.astype(float), "first": first.astype(float)}
    for m in models:
        a, c = t[f"alert_{m}"].to_numpy(dtype=bool), t[f"critical_{m}"].to_numpy(dtype=bool)
        columns.update({f"tp_{m}": (a & y).astype(float), f"fp_{m}": (a & ~y).astype(float),
                        f"ctp_{m}": (c & y).astype(float), f"cfp_{m}": (c & ~y).astype(float),
                        f"first_{m}": (a & first).astype(float)})
    names = list(columns)
    per_group = np.column_stack([np.bincount(group_codes, weights=columns[k], minlength=n_groups) for k in names])
    # episodes: an episode is detected when any of its fraud rows (in this population) is alerted
    fraud_rows = t[y]
    ep = fraud_rows.groupby("episode", sort=True).agg(
        group=("group", "first"), **{f"episodes_{m}": (f"alert_{m}", "any") for m in models})
    ep_group = pd.Index(group_names).get_indexer(ep["group"])
    ep_cols = {"episodes": np.ones(len(ep)), **{f"episodes_{m}": ep[f"episodes_{m}"].to_numpy(dtype=float) for m in models}}
    names += list(ep_cols)
    per_group = np.column_stack([per_group] + [np.bincount(ep_group, weights=v, minlength=n_groups) for v in ep_cols.values()])

    rank = {}
    for m in models:
        order, ends = rank_prepare(t[m].to_numpy())
        rank[m] = (group_codes[order], y[order].astype(np.float64), ends)

    def all_metrics(sums: dict, weights=None) -> dict:
        out = {}
        for m in models:
            vals = {k: float(v) for k, v in _metrics_from_sums(sums, m).items()}
            g_sorted, y_sorted, ends = rank[m]
            w = np.ones(len(g_sorted)) if weights is None else weights[g_sorted]
            pos = w * y_sorted
            vals["pr_auc"], vals["roc_auc"] = weighted_rank_metrics(pos, w - pos, ends)
            out[m] = vals
        return out

    total = dict(zip(names, per_group.sum(axis=0)))
    point = all_metrics(total)

    rng = np.random.default_rng(seed)
    draws = np.zeros((reps, n_groups))
    for s in sorted(set(group_seed)):                     # stratified: resample within each dataset
        members = np.flatnonzero(group_seed == s)
        picks = rng.integers(0, len(members), size=(reps, len(members)))
        for r in range(reps):
            draws[r, members] = np.bincount(picks[r], minlength=len(members))
    boot = {m: {k: np.full(reps, np.nan) for k in METRICS} for m in models}
    resampled = draws @ per_group
    for r in range(reps):
        vals = all_metrics(dict(zip(names, resampled[r])), draws[r])
        for m in models:
            for k in METRICS:
                boot[m][k][r] = vals[m][k]

    out_models = {}
    for m in models:
        k_first, n_first = int(total[f"first_{m}"]), int(total["first"])
        out_models[m] = {
            "counts": {"true_positives": int(total[f"tp_{m}"]), "false_positives": int(total[f"fp_{m}"]),
                       "false_negatives": int(total["fraud"] - total[f"tp_{m}"]),
                       "critical_true_positives": int(total[f"ctp_{m}"]), "critical_false_positives": int(total[f"cfp_{m}"]),
                       "first_frauds_detected": k_first, "episodes_detected": int(total[f"episodes_{m}"])},
            "metrics": {k: {"value": _r(point[m][k]), "ci95": _ci(boot[m][k])} for k in METRICS},
            "first_fraud_recall_wilson95": wilson(k_first, n_first),
        }
    paired = {}
    for a, b in (PAIRS if pairs is None else pairs):
        if a not in models or b not in models:
            continue
        entry = {}
        for k in METRICS:
            ci = _ci(boot[a][k] - boot[b][k])
            entry[k] = {"difference": _r(point[a][k] - point[b][k]) if not (np.isnan(point[a][k]) or np.isnan(point[b][k])) else None,
                        "ci95": ci, "excludes_zero": bool(ci is not None and (ci[0] > 0 or ci[1] < 0))}
        paired[f"{a} - {b}"] = entry
    result = {
        "rows": int(len(t)), "fraud_transactions": int(total["fraud"]),
        "legitimate_transactions": int(total["rows"] - total["fraud"]),
        "fraud_episodes": int(total["episodes"]), "first_fraud_transactions": int(total["first"]),
        "customers": int(t["customer_id"].nunique() if t["seed"].nunique() == 1 else t.groupby("seed")["customer_id"].nunique().sum()),
        "groups": int(n_groups), "datasets": int(len(set(group_seed))),
        "models": out_models, "paired_differences": paired,
    }
    return (result, boot) if return_resamples else result


# ---- descriptive breakdowns ------------------------------------------------------------------------

def fraud_type_breakdown(table: pd.DataFrame, models=MODEL_SETS) -> dict:
    fraud = table[table["is_fraud"] == 1]
    out = {}
    for ftype, g in fraud.groupby("fraud_type", sort=True):
        ep = g.groupby("episode")
        entry = {"episodes": int(ep.ngroups), "fraud_transactions": int(len(g))}
        for m in models:
            entry[m] = {"fraud_caught": int(g[f"alert_{m}"].sum()), "recall": _r(g[f"alert_{m}"].mean(), 4),
                        "episodes_detected": int(ep[f"alert_{m}"].any().sum())}
        out[ftype] = entry
    return out


EARLY_BUCKETS = (("0", 0, 0), ("1", 1, 1), ("2", 2, 2), ("3-9", 3, 9))


def early_history(table: pd.DataFrame, models=MODEL_SETS, alert_prefix: str = "alert_") -> dict:
    """Rows with fewer than 10 earlier transactions, by number of earlier transactions."""
    early = table[table["n_prior"] < MIN_PRIOR]
    out = {}
    for label, lo, hi in EARLY_BUCKETS + (("all", 0, MIN_PRIOR - 1),):
        g = early[(early["n_prior"] >= lo) & (early["n_prior"] <= hi)]
        legit, fraud = g[g["is_fraud"] == 0], g[g["is_fraud"] == 1]
        first = fraud[fraud["fraud_stage"] == "first"]
        entry = {"legitimate_transactions": int(len(legit)), "fraud_transactions": int(len(fraud)),
                 "first_fraud_transactions": int(len(first))}
        for m in models:
            col = f"{alert_prefix}{m}"
            entry[m] = {"legitimate_alerts": int(legit[col].sum()),
                        "legit_alerts_per_1000": _r(1000 * legit[col].mean(), 2) if len(legit) else None,
                        "fraud_caught": int(fraud[col].sum()), "first_frauds_caught": int(first[col].sum())}
        out[label] = entry
    return out


# ---- the pre-registered decision rule ----------------------------------------------------------------

def decision_rule(primary: dict) -> dict:
    """Applies the rule fixed before the run to the pooled primary population.
    It reports an outcome; it does not select, promote or deploy anything."""
    pair = primary["paired_differences"]["v2_dnn_lstm - v2_dnn_only"]

    def bound(metric, i):
        ci = pair[metric]["ci95"]
        return None if ci is None else ci[i]

    conditions = {
        "recall_at_least_matches": {
            "test": f"lower bound of the 95% interval of the recall difference > -{RECALL_MARGIN}",
            "ci95": pair["recall"]["ci95"],
            "holds": bound("recall", 0) is not None and bound("recall", 0) > -RECALL_MARGIN},
        "alert_burden_at_least_matches": {
            "test": f"upper bound of the 95% interval of the legitimate-alert difference < +{LEGIT_ALERT_MARGIN_PER_1000} per 1,000",
            "ci95": pair["legit_alerts_per_1000"]["ci95"],
            "holds": bound("legit_alerts_per_1000", 1) is not None and bound("legit_alerts_per_1000", 1) < LEGIT_ALERT_MARGIN_PER_1000},
        "critical_tier_not_worse": {
            "test": f"lower bound of the 95% interval of the Critical-tier recall difference > -{CRITICAL_RECALL_MARGIN}",
            "ci95": pair["critical_recall"]["ci95"],
            "holds": bound("critical_recall", 0) is not None and bound("critical_recall", 0) > -CRITICAL_RECALL_MARGIN},
    }
    all_hold = all(c["holds"] for c in conditions.values())
    separated = [k for k in PRIMARY_ENDPOINTS if pair[k]["excludes_zero"]]

    production_recall = primary["models"]["production"]["metrics"]["recall"]["value"]
    eligibility = {}
    for m in CANDIDATES:
        met = primary["models"][m]["metrics"]
        alerts, recall = met["legit_alerts_per_1000"], met["recall"]
        checks = {
            "within_alert_budget": alerts["ci95"] is not None and alerts["ci95"][1] <= ALERT_BUDGET_PER_1000,
            "recall_at_least_minimum": recall["value"] is not None and recall["value"] >= MIN_RECALL,
            "recall_lower_bound_above_production": recall["ci95"] is not None and recall["ci95"][0] > production_recall,
        }
        eligibility[m] = {"legit_alerts_per_1000": alerts, "recall": recall, "production_recall": production_recall,
                          **checks, "eligible_on_this_evidence": all(checks.values())}

    claims = []
    for name, entry in primary["paired_differences"].items():
        for k in PRIMARY_ENDPOINTS:
            if entry[k]["excludes_zero"]:
                claims.append({"pair": name, "endpoint": k, "difference": entry[k]["difference"], "ci95": entry[k]["ci95"]})
    return {
        "population": "primary (at least 10 earlier transactions), all hold-out datasets pooled",
        "candidate_tie_break": {
            "difference": "v2_dnn_lstm - v2_dnn_only", "conditions": conditions, "all_conditions_hold": all_hold,
            "primary_endpoints_separated": separated,
            "outcome": ("v2_dnn_lstm remains the provisional lead" if all_hold
                        else "v2_dnn_only wins on simplicity (a tie-break condition failed)"),
            "provisional_lead": "v2_dnn_lstm" if all_hold else "v2_dnn_only",
        },
        "against_production": {
            "rule": (f"legitimate alerts per 1,000: upper bound <= {ALERT_BUDGET_PER_1000}; recall >= {MIN_RECALL}; "
                     "lower bound of the recall interval above production's point estimate"),
            "candidates": eligibility,
            "any_candidate_eligible": any(e["eligible_on_this_evidence"] for e in eligibility.values()),
        },
        "primary_endpoint_intervals_excluding_zero": claims,
        "multiplicity_note": ("three primary endpoints and three pairs of model sets give nine primary comparisons; "
                              "about one in twenty such 95% intervals excludes zero by chance"),
        "promotion": "none. This rule reports evidence only; MODEL_SET stays production.",
    }


# ---- model files ---------------------------------------------------------------------------------

def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_file_checksums() -> dict:
    """SHA-256 of every file under models/saved/ and models/candidates/."""
    out = {}
    for label, root in (("production", config.MODELS_SAVED_DIR), ("candidates", config.CANDIDATES_DIR)):
        out[label] = {f.relative_to(root).as_posix(): _sha256(f) for f in sorted(root.rglob("*")) if f.is_file()} \
            if root.exists() else {}
    return out


def assert_safe_output(path) -> Path:
    p = Path(path).resolve()
    for root in (config.MODELS_SAVED_DIR.resolve(), config.CANDIDATES_DIR.resolve()):
        if p == root or root in p.parents:
            raise HoldoutError(f"refusing to write hold-out output into the model directory {root}")
    return p


# ---- the run -------------------------------------------------------------------------------------

def score_dataset(data, model_sets: dict, cutoffs=None) -> pd.DataFrame:
    return build_table(data, {name: score_frame(data.frame, ms) for name, ms in model_sets.items()}, cutoffs)


def gates(datasets: dict, pooled: dict, rings: dict) -> dict:
    prim, sec = pooled["primary"], pooled["secondary"]
    early_fraud = sum(d["fraud_transactions_with_fewer_than_10_earlier"] for d in datasets.values())

    def gate(value, needed):
        return {"value": value, "required": needed, "met": bool(value >= needed)}

    return {
        "fraud_episodes_primary": gate(prim["fraud_episodes"], MIN_EPISODES),
        "fraud_episodes_secondary": gate(sec["fraud_episodes"], MIN_EPISODES),
        "independent_data_seeds": gate(len(datasets), len(HOLDOUT_SEEDS)),
        "rings_primary": gate(rings["rings"], MIN_RINGS),
        "new_customer_fraud": {"value": early_fraud, "required": MIN_EPISODES, "met": False,
                               "note": "fraud with fewer than 10 earlier transactions is not represented in these "
                                       "datasets; it needs the generator extension and is evaluated separately"},
    }


def evaluate(seeds=HOLDOUT_SEEDS, data_root=None, reps: int = BOOTSTRAP_REPS, bootstrap_seed: int = BOOTSTRAP_SEED,
             model_sets: dict | None = None, tables: dict | None = None) -> dict:
    """The full hold-out report (a dict). `model_sets`: name -> LoadedModelSet
    (default: the three saved sets). `tables`: already scored tables by seed."""
    before = model_file_checksums()
    if model_sets is None:
        from ..model_sets import load_model_set
        model_sets = {name: load_model_set(name) for name in MODEL_SETS}
    models = list(model_sets)
    datasets, scored, shared = {}, {}, {}
    for s in seeds:
        data = load_holdout(s, data_root)
        datasets[str(s)] = dataset_summary(data)
        scored[s] = tables[s] if tables is not None and s in tables else score_dataset(data, model_sets)
        shared[str(s)] = ring_metrics.shared_device_rule(data.frame, data.metadata, data.grouping.episodes)
    table = pd.concat([scored[s] for s in seeds], ignore_index=True)
    primary = population(table, "primary")

    pooled = {name: evaluate_population(population(table, name), models, reps, bootstrap_seed)
              for name in ("primary", "secondary")}
    per_seed = {str(s): evaluate_population(population(scored[s], "primary"), models, reps, bootstrap_seed) for s in seeds}
    pairs = [(a, b) for a, b in PAIRS if a in models and b in models]
    rings = {
        "policy_b": ring_metrics.ring_report(primary, models, "alert_", reps, bootstrap_seed, pairs),
        "critical": ring_metrics.ring_report(primary, models, "critical_", reps, bootstrap_seed, pairs),
        "secondary_policy_b": ring_metrics.ring_report(population(table, "secondary"), models, "alert_", reps, bootstrap_seed, pairs),
        "shared_device_rule": {"per_dataset": shared,
                               "note": "the live rule (a device used by 2+ customers); it is the same for every model set"},
    }
    report = {
        "step": "4C-3E hold-out evaluation",
        "fixed_inputs": {
            "holdout_seeds": list(HOLDOUT_SEEDS), "seeds_evaluated": [int(s) for s in seeds], "training_seed": TRAINING_SEED,
            "cutoffs_score_0_1": {tier: dict(v) for tier, v in CUTOFFS.items()},
            "cutoffs_source": "seed-42 validation period (4C-3B); not re-tuned on hold-out data; not probabilities",
            "primary_population": f"transactions with at least {MIN_PRIOR} earlier transactions of the same customer",
            "secondary_population": f"the primary population from {SECONDARY_START}",
            "margins": {"recall": RECALL_MARGIN, "legit_alerts_per_1000": LEGIT_ALERT_MARGIN_PER_1000,
                        "critical_recall": CRITICAL_RECALL_MARGIN},
            "alert_budget_per_1000": ALERT_BUDGET_PER_1000, "minimum_recall": MIN_RECALL,
            "bootstrap": {"resamples": reps, "seed": bootstrap_seed,
                          "unit": "customer component (ring victims and households kept together), resampled within each dataset",
                          "interval": "2.5 / 97.5 percentiles"},
        },
        "model_sets": {name: {"uses_lstm": bool(ms.uses_lstm), "dnn_input_columns": list(ms.dnn_input_columns)}
                       for name, ms in model_sets.items()},
        "model_files_sha256": before,
        "datasets": datasets,
        "pooled": pooled,
        "per_seed_primary": per_seed,
        "fraud_types_primary": fraud_type_breakdown(primary, models),
        "rings": rings,
        "early_history": {"policy_b": early_history(table, models, "alert_"),
                          "critical": early_history(table, models, "critical_")},
    }
    if set(MODEL_SETS) <= set(models):
        report["decision_rule"] = decision_rule(pooled["primary"])
    report["gates"] = gates(datasets, pooled, rings["policy_b"])
    report["state"] = {"model_set_default": _default_model_set(), "promotion": "none", "models_trained": "none",
                       "thresholds_changed": "none"}
    after = model_file_checksums()
    if after != before:
        raise RuntimeError("model files changed during the hold-out evaluation")
    return report


# ---- new-customer fraud (separate datasets, generator extension 2.1.0) ---------------------------------

def _variant_table(early: pd.DataFrame, models, prefix: str) -> pd.DataFrame:
    """The early-history rows with one alert rule (`prefix`) in the alert columns."""
    t = early.copy()
    for m in models:
        t[f"alert_{m}"] = early[f"{prefix}{m}"].to_numpy()
        t[f"critical_{m}"] = early[f"{prefix}{m}"].to_numpy()
    return t


def evaluate_new_customer(seeds=NEW_CUSTOMER_SEEDS, data_root=None, reps: int = BOOTSTRAP_REPS,
                          bootstrap_seed: int = BOOTSTRAP_SEED, model_sets: dict | None = None) -> dict:
    """Transactions with fewer than 10 earlier ones, on datasets that contain
    new-customer fraud. Three alert rules are compared for every model set:

    * current_policy_b            the live cold-start scoring, alert at the Policy B cut-off;
    * critical_only               the same scores, alert only at the Critical cut-off
                                  (the proposed limited-history policy);
    * critical_only_flags_neutral measurement only: Critical cut-off with the unusual-hour
                                  and unusual-category inputs set to the training mean.

    The fraud patterns are the generator's; the result says how each rule behaves
    on those patterns, not on real new-customer fraud."""
    before = model_file_checksums()
    root = data_root or NEW_CUSTOMER_DATA_DIR
    if model_sets is None:
        from ..model_sets import load_model_set
        model_sets = {name: load_model_set(name) for name in MODEL_SETS}
    models = list(model_sets)
    datasets, standard, neutral = {}, [], []
    for s in seeds:
        data = load_holdout(s, root)
        info = dataset_summary(data)
        ep = data.grouping.episodes
        if "prior_transactions_at_first_fraud" in ep.columns:
            prior = ep["prior_transactions_at_first_fraud"].astype(int)
            info["new_customer_fraud_episodes"] = int((prior < MIN_PRIOR).sum())
            info["new_customer_fraud_episodes_by_type"] = {
                k: int(v) for k, v in ep.loc[prior < MIN_PRIOR, "fraud_type"].value_counts().sort_index().items()}
        else:
            info["new_customer_fraud_episodes"] = 0
        info["late_joiners"] = (data.manifest.get("summary") or {}).get("late_joiners", 0)
        datasets[str(s)] = info
        scored = score_dataset(data, model_sets)
        customers = pd.read_csv(data.spec.customers_csv, keep_default_na=False)
        late = set(customers.loc[customers["join_date"] > customers["join_date"].min(), "customer_id"]) \
            if "join_date" in customers.columns else set()
        scored["late_joiner"] = scored["customer_id"].isin(late).to_numpy()
        standard.append(scored)
        neutral.append(build_table(data, {name: score_frame(data.frame, ms, neutral=NEUTRAL_FLAGS)
                                          for name, ms in model_sets.items()}))
    table, table_n = pd.concat(standard, ignore_index=True), pd.concat(neutral, ignore_index=True)
    early, early_n = population(table, "early_history"), population(table_n, "early_history")

    variants = {"current_policy_b": (early, "alert_"), "critical_only": (early, "critical_"),
                "critical_only_flags_neutral": (early_n, "critical_"), "policy_b_flags_neutral": (early_n, "alert_")}
    results, resamples, by_prior = {}, {}, {}
    for name, (rows, prefix) in variants.items():
        res, boot = evaluate_population(_variant_table(rows, models, prefix), models, reps, bootstrap_seed, return_resamples=True)
        results[name] = {
            "models": {m: {"counts": {k: res["models"][m]["counts"][k] for k in
                                      ("true_positives", "false_positives", "false_negatives", "first_frauds_detected", "episodes_detected")},
                           "metrics": {k: res["models"][m]["metrics"][k] for k in VARIANT_METRICS}} for m in models},
            "paired_differences_between_model_sets": {pair: {k: e[k] for k in VARIANT_METRICS}
                                                      for pair, e in res["paired_differences"].items()},
        }
        resamples[name] = (res, boot)
        by_prior[name] = early_history(rows, models, prefix)
    base = resamples["current_policy_b"][0]

    def variant_difference(a, b):
        out = {}
        for m in models:
            entry = {}
            for k in VARIANT_METRICS:
                pa, pb = resamples[a][0]["models"][m]["metrics"][k]["value"], resamples[b][0]["models"][m]["metrics"][k]["value"]
                ci = _ci(resamples[a][1][m][k] - resamples[b][1][m][k])
                entry[k] = {"difference": None if pa is None or pb is None else _r(pa - pb), "ci95": ci,
                            "excludes_zero": bool(ci is not None and (ci[0] > 0 or ci[1] < 0))}
            out[m] = entry
        return out

    n_episodes = sum(d["new_customer_fraud_episodes"] for d in datasets.values())
    primary = population(table, "primary")

    def point(rows):
        res = evaluate_population(rows, models, 0, bootstrap_seed)
        return {"rows": res["rows"], "fraud_transactions": res["fraud_transactions"],
                "models": {m: {k: _r(res["models"][m]["metrics"][k]["value"], 4) for k in ("recall", "legit_alerts_per_1000", "critical_recall")}
                           for m in models}}

    unchanged = primary[~primary["late_joiner"]]
    full_history = {
        "note": ("rows with at least 10 earlier transactions in these datasets, Policy B, point values only. "
                 "'unchanged customers' are the customers the extension does not touch: their transactions are "
                 "exactly those of the default generator with the same seed, so they are five further data seeds. "
                 "Descriptive; not part of the pre-registered primary population"),
        "unchanged_customers_pooled": point(unchanged),
        "unchanged_customers_per_seed": {str(s): point(g) for s, g in unchanged.groupby("seed", sort=True)},
        "late_joiners_pooled": point(primary[primary["late_joiner"]]) if primary["late_joiner"].any() else None,
    }
    report = {
        "step": "4C-3E new-customer fraud evaluation",
        "fixed_inputs": {
            "seeds": [int(s) for s in seeds], "generator_extension": "2.1.0 (off by default)",
            "late_joiner_share": NEW_CUSTOMER_LATE_JOINER_SHARE,
            "new_customer_fraud_episodes_per_seed": NEW_CUSTOMER_EPISODES_PER_SEED,
            "population": f"transactions with fewer than {MIN_PRIOR} earlier transactions of the same customer",
            "cutoffs_score_0_1": {tier: dict(v) for tier, v in CUTOFFS.items()},
            "neutral_flags": list(NEUTRAL_FLAGS),
            "bootstrap": {"resamples": reps, "seed": bootstrap_seed,
                          "unit": "customer component, resampled within each dataset", "interval": "2.5 / 97.5 percentiles"},
        },
        "limit": ("the new-customer fraud patterns are the ones written into the generator for this purpose "
                  "(the existing archetypes placed in a late joiner's first transactions); the result shows how each "
                  "rule behaves on those patterns, not that it works on real new-customer fraud"),
        "model_files_sha256": before,
        "datasets": datasets,
        "population": {k: base[k] for k in ("rows", "fraud_transactions", "legitimate_transactions", "fraud_episodes",
                                            "first_fraud_transactions", "groups", "datasets")},
        "variants": results,
        "variant_differences_per_model_set": {
            "critical_only - current_policy_b": variant_difference("critical_only", "current_policy_b"),
            "critical_only_flags_neutral - critical_only": variant_difference("critical_only_flags_neutral", "critical_only"),
            "policy_b_flags_neutral - current_policy_b": variant_difference("policy_b_flags_neutral", "current_policy_b"),
        },
        "by_earlier_transactions": by_prior,
        "full_history_policy_b": full_history,
        "gates": {"new_customer_fraud_episodes": {"value": n_episodes, "required": MIN_EPISODES, "met": bool(n_episodes >= MIN_EPISODES)}},
        "state": {"model_set_default": _default_model_set(), "promotion": "none", "models_trained": "none",
                  "thresholds_changed": "none", "cold_start_behaviour_changed": "none (analysis only)"},
    }
    if model_file_checksums() != before:
        raise RuntimeError("model files changed during the new-customer evaluation")
    return report


def _default_model_set() -> str:
    from ..model_sets import DEFAULT_MODEL_SET
    return DEFAULT_MODEL_SET


def write_report(report: dict, output_dir=None, name: str = REPORT_NAME) -> Path:
    out = assert_safe_output(output_dir or OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    path.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8", newline="\n")
    return path


# ---- latency (not deterministic; written to its own file) ---------------------------------------------

def _latency_worker(model_set: str, n: int, warmup: int, candidates_root=None) -> dict:
    import time
    from ..inference_pipeline import FraudIntelligencePipeline
    pipeline = FraudIntelligencePipeline(model_set=model_set, candidates_root=candidates_root)
    customers = sorted(pipeline.known_customer_ids())[:50]
    times = []
    for i in range(warmup + n):
        cid = customers[i % len(customers)]
        last = pipeline.customer_histories[cid].iloc[-1]
        txn = {"customer_id": cid, "amount": float(last["amount"]), "merchant_category": last["merchant_category"],
               "device_id": last["device_id"], "location": last["location"], "failed_logins_24h": 0}
        start = time.perf_counter()
        pipeline.score_transaction(txn)
        if i >= warmup:
            times.append((time.perf_counter() - start) * 1000)
    return {"scorings": n, "warm_up_excluded": warmup, "median_ms": round(float(np.median(times)), 2),
            "p95_ms": round(float(np.percentile(times, 95)), 2)}


def measure_latency(n: int = 200, warmup: int = 20, output_dir=None) -> dict:
    """Median and 95th-percentile time of FraudIntelligencePipeline.score_transaction
    (features, models and explanation), each model set in its own process."""
    result = {}
    for name in MODEL_SETS:
        done = subprocess.run([sys.executable, "-m", "app.evaluation.holdout", "latency-worker", "--model-set", name,
                               "--n", str(n), "--warmup", str(warmup)], check=True, capture_output=True, text=True,
                              cwd=str(config.BACKEND_DIR))
        result[name] = json.loads(done.stdout.strip().splitlines()[-1])
    out = assert_safe_output(output_dir or OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"note": "wall-clock time on the machine that ran this; not reproducible bit for bit",
               "method": "live pipeline, known customers, one transaction at a time, warm-up excluded", "model_sets": result}
    (out / LATENCY_NAME).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    return payload


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", nargs="?", default="evaluate", choices=["evaluate", "new-customer", "latency", "latency-worker"])
    p.add_argument("--generate", action="store_true", help="generate missing hold-out datasets first")
    p.add_argument("--data-root", default=None, help=f"where seed_<s>/ directories are (default: {HOLDOUT_DATA_DIR})")
    p.add_argument("--output-dir", default=None, help=f"where the report is written (default: {OUTPUT_DIR})")
    p.add_argument("--reps", type=int, default=BOOTSTRAP_REPS)
    p.add_argument("--model-set", default=None)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--candidates-root", default=None, help="latency-worker only: load the candidate from this directory")
    args = p.parse_args(argv)
    if args.mode == "latency-worker":
        print(json.dumps(_latency_worker(args.model_set, args.n, args.warmup, args.candidates_root)))
        return None
    if args.mode == "latency":
        payload = measure_latency(args.n, args.warmup, args.output_dir)
        print(json.dumps(payload["model_sets"], indent=2))
        return payload
    assert_safe_output(args.output_dir or OUTPUT_DIR)
    if args.mode == "new-customer":
        if args.generate:
            for s in NEW_CUSTOMER_SEEDS:
                generate_dataset(s, args.data_root or NEW_CUSTOMER_DATA_DIR, late_joiner_share=NEW_CUSTOMER_LATE_JOINER_SHARE,
                                 new_customer_fraud_episodes=NEW_CUSTOMER_EPISODES_PER_SEED)
        report = evaluate_new_customer(NEW_CUSTOMER_SEEDS, args.data_root, args.reps)
        path = write_report(report, args.output_dir, NEW_CUSTOMER_REPORT_NAME)
        print(f"wrote {path}")
        return report
    if args.generate:
        for s in HOLDOUT_SEEDS:
            generate_dataset(s, args.data_root)
    report = evaluate(HOLDOUT_SEEDS, args.data_root, args.reps)
    path = write_report(report, args.output_dir)
    print(f"wrote {path}")
    return report


if __name__ == "__main__":
    main()
