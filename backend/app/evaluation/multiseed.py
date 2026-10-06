"""
Evaluation of the multi-seed candidates on the hold-out evidence (Step 4C-3E).

    cd backend
    python -m app.training.multiseed          # trains the seeds (once)
    python -m app.evaluation.multiseed        # this module -> models/evaluation/v2_holdout/multiseed_report.json
    python -m app.evaluation.multiseed latency

Question: is the v2_dnn_lstm advantage over v2_dnn_only a property of the
architecture, or of the single seed-42 training run?

Every trained model (5 training seeds x 2 architectures) is scored on the
same hold-out datasets as the seed-42 candidates (data seeds 101-105, and
the new-customer datasets 201-205), on the same populations, with the same
harness (app.evaluation.holdout).

Cut-offs. A cut-off is a property of one trained model: the seed-42 numbers
cannot be applied to another model's scores. Each trained model therefore
gets its cut-offs by the SAME RULE that produced the fixed seed-42 table in
4C-3B: on the validation period of the seed-42 training dataset, the lowest
tie-safe score whose false-positive rate is within 1% (Policy B) or 0.1%
(Critical). Applied to the saved seed-42 models this rule reproduces
holdout.CUTOFFS exactly (tested). Nothing is tuned on hold-out data.

Two sources of variation are kept apart:

* data:           which customers are in the hold-out (group bootstrap within
                  each dataset, as in the hold-out report);
* training seed:  which training run produced the model (spread across the
                  5 seeds of an architecture).

The architecture comparison is the difference between the two
architectures' means over training seeds. Its decision interval combines
both sources: every bootstrap resample redraws the customer groups AND
redraws the 5 training seeds of each architecture with replacement.

It reads model files and never writes to models/saved/, models/candidates/
or models/candidates_multiseed/. It does not select, promote or deploy.
"""

import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from ..training import multiseed as training
from . import holdout as h
from . import ring_metrics

ARCHITECTURES = ("v2_dnn_lstm", "v2_dnn_only")
CANDIDATE_DIR = {"v2_dnn_lstm": "dnn_lstm", "v2_dnn_only": "dnn_only"}
REFERENCE = ("production", "v2_dnn_lstm", "v2_dnn_only")          # the saved seed-42 candidates and production
POLICY_FPR = {"critical": 0.001, "policy_b": 0.01}                 # the 4C-3B budgets behind holdout.CUTOFFS
REPORT_NAME = "multiseed_report.json"
LATENCY_NAME = "multiseed_latency.json"
SUMMARY_METRICS = ("recall", "legit_alerts_per_1000", "critical_recall", "first_fraud_recall",
                   "episode_detection_rate", "precision", "critical_legit_alerts_per_1000", "pr_auc", "roc_auc")
RING_MEASURES = ("victim_episode_detection_rate", "ring_fraud_transaction_recall", "rings_with_at_least_one_victim_alerted")


def key(architecture: str, seed: int) -> str:
    return f"{architecture}@{int(seed)}"


# ---- cut-offs (the 4C-3B rule) ----------------------------------------------------------------------

def tie_safe_cutoff(y, scores, target_fpr: float):
    """Lowest cut-off v (alert if score >= v) whose false-positive rate on these
    rows is at most target_fpr. Rows with equal scores are alerted together, so
    ties never push the rate over the target. None if no cut-off qualifies."""
    y, scores = np.asarray(y), np.asarray(scores)
    negatives = np.sort(scores[y == 0])
    allowed = int(np.floor(target_fpr * len(negatives)))
    values = np.unique(scores)
    flagged = len(negatives) - np.searchsorted(negatives, values, "left")      # negatives with score >= value
    ok = np.flatnonzero(flagged <= allowed)
    return float(values[ok[0]]) if len(ok) else None


def validation_mask(data, labels) -> np.ndarray:
    """Validation rows of the training dataset that have at least 10 earlier transactions."""
    return (np.asarray(labels) == "validation") & (h.prior_counts(data.frame) >= h.MIN_PRIOR)


def policy_cutoffs(model_set, data, mask) -> dict:
    scores = h.score_frame(data.frame, model_set)[mask]
    y = data.frame["is_fraud"].to_numpy()[mask]
    out = {"validation_rows": int(mask.sum()), "validation_fraud": int(y.sum())}
    for tier, target in POLICY_FPR.items():
        cut = tie_safe_cutoff(y, scores, target)
        if cut is None:
            raise ValueError(f"no cut-off keeps the validation false-positive rate within {target}")
        alert = scores >= np.float32(cut)
        out[tier] = cut
        out[f"{tier}_validation"] = {"target_false_positive_rate": target,
                                     "false_positive_rate": h._r(float((alert & (y == 0)).sum() / (y == 0).sum())),
                                     "legit_alerts_per_1000": h._r(1000 * float((alert & (y == 0)).sum()) / len(y), 2),
                                     "recall": h._r(float((alert & (y == 1)).sum() / max(y.sum(), 1)), 4)}
    return out


# ---- models -------------------------------------------------------------------------------------

def load_trained(seeds, root=None) -> dict:
    from ..model_sets import load_model_set
    return {key(a, s): load_model_set(a, candidates_root=training.seed_root(s, root))
            for s in seeds for a in ARCHITECTURES}


def multiseed_file_checksums(seeds, root=None) -> dict:
    out = {}
    for s in seeds:
        d = training.seed_root(s, root)
        out[str(s)] = {f.relative_to(d).as_posix(): h._sha256(f) for f in sorted(d.rglob("*")) if f.is_file()}
    return out


# ---- statistics across training seeds ------------------------------------------------------------

def _t(df: float) -> float:
    from scipy import stats
    return float(stats.t.ppf(0.975, df))


def across_seeds(values) -> dict:
    """Spread of one metric across the training seeds of an architecture (data fixed)."""
    v = np.asarray([x for x in values if x is not None], dtype=np.float64)
    n = len(v)
    if n == 0:
        return {"n": 0}
    sd = float(v.std(ddof=1)) if n > 1 else None
    half = _t(n - 1) * sd / np.sqrt(n) if n > 1 else None
    return {"n": n, "mean": h._r(v.mean()), "median": h._r(np.median(v)), "sd": None if sd is None else h._r(sd),
            "min": h._r(v.min()), "max": h._r(v.max()),
            "mean_ci95_training_seeds": None if half is None else [h._r(v.mean() - half), h._r(v.mean() + half)]}


def _welch(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if len(a) < 2 or len(b) < 2:
        return None
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    d = a.mean() - b.mean()
    if se == 0:
        return [h._r(d), h._r(d)]
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    half = _t(df) * se
    return [h._r(d - half), h._r(d + half)]


def architecture_summary(result: dict, boot: dict, seeds, metrics, bootstrap_seed: int = h.BOOTSTRAP_SEED) -> dict:
    """Per architecture: the metric across training seeds. Between architectures:
    the difference of the means over training seeds (v2_dnn_lstm - v2_dnn_only) with
    three intervals, each labelled with the variation it includes."""
    seeds = list(seeds)
    n = len(seeds)
    reps = len(next(iter(next(iter(boot.values())).values()))) if boot else 0
    rng = np.random.default_rng(bootstrap_seed + 1)
    draws = {a: rng.integers(0, n, size=(reps, n)) for a in ARCHITECTURES}      # training seeds redrawn per resample
    out = {"architectures": {a: {} for a in ARCHITECTURES}, "comparison": {}}
    for m in metrics:
        point = {a: [result["models"][key(a, s)]["metrics"][m]["value"] for s in seeds] for a in ARCHITECTURES}
        stacked = {a: np.vstack([boot[key(a, s)][m] for s in seeds]) for a in ARCHITECTURES}      # [seed, resample]
        for a in ARCHITECTURES:
            stats = across_seeds(point[a])
            stats["per_training_seed"] = {str(s): v for s, v in zip(seeds, point[a])}
            stats["mean_ci95_data"] = h._ci(stacked[a].mean(axis=0))
            out["architectures"][a][m] = stats
        la, on = "v2_dnn_lstm", "v2_dnn_only"
        if any(v is None for v in point[la] + point[on]):
            continue
        d = float(np.mean(point[la]) - np.mean(point[on]))
        cols = np.arange(reps)
        combined = (stacked[la][draws[la].T, cols].mean(axis=0) - stacked[on][draws[on].T, cols].mean(axis=0)) if reps else []
        ci_data, ci_combined = h._ci(stacked[la].mean(axis=0) - stacked[on].mean(axis=0)), h._ci(combined)
        out["comparison"][m] = {
            "difference_of_means": h._r(d),
            "ci95_data_only": ci_data,
            "ci95_training_seeds_only": _welch(point[la], point[on]),
            "ci95_data_and_training_seeds": ci_combined,
            "excludes_zero": bool(ci_combined is not None and (ci_combined[0] > 0 or ci_combined[1] < 0)),
            "same_seed_differences": {str(s): h._r(a - b) for s, a, b in zip(seeds, point[la], point[on])},
            "seed_pairs_with_v2_dnn_lstm_higher": f"{sum(a > b for a in point[la] for b in point[on])}/{n * n}",
        }
    return out


def decision(comparison: dict) -> dict:
    """The existing tie-break rule, on the multi-seed evidence: the interval is the
    one that includes both the data and the training-seed variation."""
    def bounds(metric):
        return comparison[metric]["ci95_data_and_training_seeds"]

    def also(metric, test):
        ci = comparison[metric]["ci95_training_seeds_only"]
        return bool(ci is not None and test(ci))

    tests = {
        "recall_at_least_matches": ("recall", f"lower bound > -{h.RECALL_MARGIN}", lambda ci: ci[0] > -h.RECALL_MARGIN),
        "alert_burden_at_least_matches": ("legit_alerts_per_1000", f"upper bound < +{h.LEGIT_ALERT_MARGIN_PER_1000} per 1,000",
                                          lambda ci: ci[1] < h.LEGIT_ALERT_MARGIN_PER_1000),
        "critical_tier_not_worse": ("critical_recall", f"lower bound > -{h.CRITICAL_RECALL_MARGIN}",
                                    lambda ci: ci[0] > -h.CRITICAL_RECALL_MARGIN),
    }
    conditions = {}
    for name, (metric, text, test) in tests.items():
        ci = bounds(metric)
        conditions[name] = {"metric": metric, "test": text, "difference_of_means": comparison[metric]["difference_of_means"],
                            "ci95_data_and_training_seeds": ci, "holds": bool(ci is not None and test(ci)),
                            "ci95_training_seeds_only": comparison[metric]["ci95_training_seeds_only"],
                            "holds_on_training_seed_interval_alone": also(metric, test)}
    ok = all(c["holds"] for c in conditions.values())
    return {
        "difference": "mean over training seeds of v2_dnn_lstm - mean over training seeds of v2_dnn_only",
        "population": "primary (at least 10 earlier transactions), hold-out data seeds pooled",
        "conditions": conditions, "all_conditions_hold": ok,
        "all_hold_on_training_seed_interval_alone": all(c["holds_on_training_seed_interval_alone"] for c in conditions.values()),
        "primary_endpoints_separated": [m for m in h.PRIMARY_ENDPOINTS if comparison[m]["excludes_zero"]],
        "outcome": ("v2_dnn_lstm remains the provisional lead" if ok
                    else "v2_dnn_only is selected on simplicity (a tie-break condition failed)"),
        "provisional_lead": "v2_dnn_lstm" if ok else "v2_dnn_only",
        "promotion": "none. MODEL_SET stays production; this result changes no setting.",
    }


# ---- pieces of the report ---------------------------------------------------------------------------

def _model_block(res: dict, m: str, metrics=h.METRICS) -> dict:
    counts = res["models"][m]["counts"]
    if "critical_recall" not in metrics:           # one alert rule only: no separate Critical tier to count
        counts = {k: v for k, v in counts.items() if not k.startswith("critical_")}
    return {"counts": counts, "metrics": {k: res["models"][m]["metrics"][k] for k in metrics}}


def _points(rows: pd.DataFrame, models, metrics=("recall", "legit_alerts_per_1000", "critical_recall")) -> dict:
    res = h.evaluate_population(rows, models, 0)
    return {m: {k: h._r(res["models"][m]["metrics"][k]["value"], 4) for k in metrics} for m in models}


def _population_report(rows, models, seeds, pairs, reps, bootstrap_seed, metrics=h.METRICS) -> dict:
    res, boot = h.evaluate_population(rows, models, reps, bootstrap_seed, return_resamples=True, pairs=pairs)
    summary = architecture_summary(res, boot, seeds, [m for m in SUMMARY_METRICS if m in metrics], bootstrap_seed)
    return {
        "population": {k: res[k] for k in ("rows", "fraud_transactions", "legitimate_transactions", "fraud_episodes",
                                           "first_fraud_transactions", "groups", "datasets")},
        "per_model": {m: _model_block(res, m, metrics) for m in models},
        "paired_differences": {p: {k: e[k] for k in metrics} for p, e in res["paired_differences"].items()},
        **summary,
    }


def _ring_block(rows, models, seeds, prefix) -> dict:
    rings = ring_metrics.per_ring(rows, models, prefix)
    summary = ring_metrics.summarise(rings, rows, models, prefix)
    per_model = {}
    for m in models:
        s = summary[m]
        per_model[m] = {
            "victim_episodes_detected": s["victim_level"]["victim_episodes_detected"],
            "victim_first_frauds_detected": s["victim_level"]["victim_first_frauds_detected"],
            "ring_fraud_caught": s["transaction_level"]["ring_fraud_caught"],
            "rings_with_at_least_one_victim_alerted": s["ring_level"]["rings_with_at_least_one_victim_alerted"],
            "rings_with_at_least_two_victims_alerted": s["ring_level"]["rings_with_at_least_two_victims_alerted"],
        }
    arch = {a: {name: across_seeds([per_model[key(a, s)][field]["rate"] for s in seeds])
                for name, field in (("victim_episode_detection_rate", "victim_episodes_detected"),
                                    ("ring_fraud_transaction_recall", "ring_fraud_caught"),
                                    ("rings_with_at_least_one_victim_alerted", "rings_with_at_least_one_victim_alerted"))}
            for a in ARCHITECTURES}
    return {"rings": len(rings), "victim_episodes": int(sum(r["victims"] for r in rings)),
            "ring_fraud_transactions": int(sum(r["fraud_transactions"] for r in rings)),
            "per_model": per_model, "architectures": arch}


def _first_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Each customer's first MIN_PRIOR transactions (the rows with fewer than 10 earlier ones)."""
    return frame[h.prior_counts(frame) < h.MIN_PRIOR].reset_index(drop=True)


def alert_budget(report: dict, seeds) -> dict:
    """Is the Policy B budget (legitimate alerts per 1,000) stable across training seeds?"""
    prim = report["holdout"]["primary"]
    per_data = report["holdout"]["per_data_seed"]
    unchanged = report["new_customer"]["unchanged_customers_full_history"]
    out = {"budget_per_1000": h.ALERT_BUDGET_PER_1000, "architectures": {}}
    for a in ARCHITECTURES:
        keys = [key(a, s) for s in seeds]
        pooled = {k: prim["per_model"][k]["metrics"]["legit_alerts_per_1000"] for k in keys}
        cells = {k: [per_data[d][k]["legit_alerts_per_1000"] for d in per_data] +
                    [unchanged["per_data_seed"][d][k]["legit_alerts_per_1000"] for d in unchanged["per_data_seed"]] for k in keys}
        by_data = [float(np.mean([per_data[d][k]["legit_alerts_per_1000"] for k in keys])) for d in per_data] + \
                  [float(np.mean([unchanged["per_data_seed"][d][k]["legit_alerts_per_1000"] for k in keys]))
                   for d in unchanged["per_data_seed"]]
        all_cells = [v for k in keys for v in cells[k]]
        out["architectures"][a] = {
            "holdout_pooled_per_training_seed": {k.split("@")[1]: pooled[k] for k in keys},
            "holdout_pooled_across_training_seeds": across_seeds([pooled[k]["value"] for k in keys]),
            "training_seeds_over_budget_point": sum(pooled[k]["value"] > h.ALERT_BUDGET_PER_1000 for k in keys),
            "training_seeds_with_upper_bound_over_budget": sum(pooled[k]["ci95"][1] > h.ALERT_BUDGET_PER_1000
                                                               for k in keys if pooled[k]["ci95"]),
            "unchanged_customers_201_205_per_training_seed": {
                k.split("@")[1]: unchanged["pooled"][k]["legit_alerts_per_1000"] for k in keys},
            "unchanged_customers_201_205_across_training_seeds": across_seeds(
                [unchanged["pooled"][k]["legit_alerts_per_1000"] for k in keys]),
            "cells_training_seed_x_data_seed": {"n": len(all_cells), "min": h._r(min(all_cells), 2), "max": h._r(max(all_cells), 2),
                                                "over_budget": int(sum(v > h.ALERT_BUDGET_PER_1000 for v in all_cells))},
            "sd_between_training_seeds": h._r(float(np.std([pooled[k]["value"] for k in keys], ddof=1)), 3) if len(keys) > 1 else None,
            "sd_between_data_seeds": h._r(float(np.std(by_data, ddof=1)), 3) if len(by_data) > 1 else None,
            "validation_legit_alerts_per_1000": {k.split("@")[1]: report["cutoffs"][k]["policy_b_validation"]["legit_alerts_per_1000"]
                                                 for k in keys},
        }
    return out


def seed_42_position(report: dict, seeds, metrics=("recall", "legit_alerts_per_1000", "critical_recall", "first_fraud_recall",
                                                     "episode_detection_rate", "roc_auc")) -> dict:
    """Where the saved seed-42 candidate sits relative to the 5 new training runs of its architecture."""
    prim = report["holdout"]["primary"]
    out = {}
    for a in ARCHITECTURES:
        out[a] = {}
        for m in metrics:
            ref = prim["per_model"][a]["metrics"][m]["value"]
            stats = prim["architectures"][a][m]
            out[a][m] = {"seed_42": ref, "multi_seed_mean": stats["mean"], "multi_seed_min": stats["min"],
                         "multi_seed_max": stats["max"], "seed_42_inside_multi_seed_range": bool(stats["min"] <= ref <= stats["max"])}
    out["difference_v2_dnn_lstm_minus_v2_dnn_only"] = {
        m: {"seed_42": prim["paired_differences"]["v2_dnn_lstm - v2_dnn_only"][m]["difference"],
            "multi_seed_difference_of_means": prim["comparison"][m]["difference_of_means"]} for m in metrics}
    return out


# ---- the run ------------------------------------------------------------------------------------

def _scored(cache_dir, name, build):
    if cache_dir is None:
        return build()
    path = Path(cache_dir) / f"{name}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    value = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(value))
    return value


def evaluate(training_seeds=training.TRAINING_SEEDS, root=None, holdout_seeds=h.HOLDOUT_SEEDS, holdout_root=None,
             new_customer_seeds=h.NEW_CUSTOMER_SEEDS, new_customer_root=None, reps: int = h.BOOTSTRAP_REPS,
             bootstrap_seed: int = h.BOOTSTRAP_SEED, spec=None, cache_dir=None, log=lambda *_: None) -> dict:
    """The multi-seed report (a dict). `spec`: the training dataset whose validation
    period gives the cut-offs (default: the seed-42 v2 dataset). `cache_dir`: keep the
    scored tables there, so a rerun does not score again."""
    from ..model_sets import load_model_set
    from ..training import candidates as cand
    from .datasets import load_evaluation_data, resolve_dataset
    seeds = [int(s) for s in training_seeds]
    before = h.model_file_checksums()
    files_before = multiseed_file_checksums(seeds, root)

    spec = spec or resolve_dataset("v2")
    train_data = load_evaluation_data(spec)
    labels, _, _, _ = cand.load_saved_split(spec, train_data)
    mask = validation_mask(train_data, labels)

    models = {name: load_model_set(name) for name in REFERENCE}
    models.update(load_trained(seeds, root))
    names = list(models)
    trained = [k for k in names if "@" in k]
    for k in trained:
        if models[k].manifest["dataset"]["sha256"] != train_data.sha256:
            raise ValueError(f"{k} was not trained on the dataset used for the cut-offs")
        if models[k].manifest["seed"] != int(k.split("@")[1]):
            raise ValueError(f"{k}: the manifest records another training seed")
    log("cut-offs")
    cut_info = _scored(cache_dir, "cutoffs", lambda: {k: policy_cutoffs(models[k], train_data, mask) for k in names})
    for name in REFERENCE:                         # the rule must reproduce the fixed seed-42 table
        for tier in POLICY_FPR:
            if cut_info[name][tier] != h.CUTOFFS[tier][name]:
                if spec.data_dir == config.DATA_V2_DIR:
                    raise RuntimeError(f"the cut-off rule does not reproduce the fixed {tier} cut-off of {name}")
    cutoffs = {tier: {k: (h.CUTOFFS[tier][k] if k in REFERENCE else cut_info[k][tier]) for k in names} for tier in POLICY_FPR}
    pairs = [(key("v2_dnn_lstm", s), key("v2_dnn_only", s)) for s in seeds] + list(h.PAIRS) + \
            [(k, "production") for k in trained]

    # ---- hold-out data seeds
    tables, datasets = [], {}
    for s in holdout_seeds:
        log(f"hold-out seed {s}")
        data = h.load_holdout(s, holdout_root)
        datasets[str(s)] = {"sha256": data.sha256, "transactions": int(len(data.frame))}
        tables.append(_scored(cache_dir, f"holdout_{s}", lambda: h.score_dataset(data, models, cutoffs)))
    table = pd.concat(tables, ignore_index=True)
    primary, secondary = h.population(table, "primary"), h.population(table, "secondary")
    log("bootstrap, primary")
    holdout = {"primary": _population_report(primary, names, seeds, pairs, reps, bootstrap_seed)}
    log("bootstrap, secondary")
    holdout["secondary"] = _population_report(secondary, names, seeds, pairs, reps, bootstrap_seed)
    holdout["per_data_seed"] = {str(s): _points(g, names) for s, g in primary.groupby("seed", sort=True)}
    holdout["rings"] = {"policy_b": _ring_block(primary, names, seeds, "alert_"),
                        "critical": _ring_block(primary, names, seeds, "critical_")}
    del table, tables

    # ---- new-customer datasets
    standard, neutral, nc_datasets = [], [], {}
    root_nc = new_customer_root or h.NEW_CUSTOMER_DATA_DIR
    for s in new_customer_seeds:
        log(f"new-customer seed {s}")
        data = h.load_holdout(s, root_nc)
        nc_datasets[str(s)] = {"sha256": data.sha256, "transactions": int(len(data.frame))}

        def build():
            scored = h.score_dataset(data, models, cutoffs)
            customers = pd.read_csv(data.spec.customers_csv, keep_default_na=False)
            late = set(customers.loc[customers["join_date"] > customers["join_date"].min(), "customer_id"]) \
                if "join_date" in customers.columns else set()
            scored["late_joiner"] = scored["customer_id"].isin(late).to_numpy()
            early = h.population(scored, "early_history").reset_index(drop=True)
            first = _first_rows(data.frame)                 # only these rows are affected by the neutral flags
            if not np.array_equal(first["transaction_id"].to_numpy(), early["transaction_id"].to_numpy()):
                raise RuntimeError("early-history rows are not aligned")
            flags = early.copy()
            for k in names:
                flags[k] = h.score_frame(first, models[k], neutral=h.NEUTRAL_FLAGS)
            return scored, h.add_alerts(flags, names, cutoffs)
        scored, flags = _scored(cache_dir, f"new_customer_{s}", build)
        standard.append(scored)
        neutral.append(flags)
    nc = pd.concat(standard, ignore_index=True)
    early, early_n = h.population(nc, "early_history"), pd.concat(neutral, ignore_index=True)
    variants = {"current_policy_b": (early, "alert_"), "critical_only": (early, "critical_"),
                "policy_b_flags_neutral": (early_n, "alert_"), "critical_only_flags_neutral": (early_n, "critical_")}
    nc_report = {"variants": {}}
    for name, (rows, prefix) in variants.items():
        log(f"new-customer variant {name}")
        rep = _population_report(h._variant_table(rows, names, prefix), names, seeds, pairs, reps, bootstrap_seed,
                                 metrics=h.VARIANT_METRICS)
        nc_report["population"] = rep.pop("population")
        nc_report["variants"][name] = rep
    full = h.population(nc, "primary")
    unchanged = full[~full["late_joiner"]]
    nc_report["unchanged_customers_full_history"] = {
        "note": "customers the generator extension does not touch, rows with at least 10 earlier transactions, Policy B; "
                "five further data seeds, descriptive",
        "rows": int(len(unchanged)), "pooled": _points(unchanged, names),
        "per_data_seed": {str(s): _points(g, names) for s, g in unchanged.groupby("seed", sort=True)},
    }

    runs = {}
    for s in seeds:
        record = json.loads((training.seed_root(s, root) / training.RUN_FILE).read_text())
        record.pop("protected_files_sha256", None)
        runs[str(s)] = record
    report = {
        "step": "4C-3E multi-seed candidate training: hold-out evaluation",
        "fixed_inputs": {
            "training_seeds": seeds, "reference_seed": training.REFERENCE_SEED,
            "architectures": list(ARCHITECTURES),
            "what_varies": "the training seed only (weight initialisation, shuffling, dropout, out-of-fold LSTM folds)",
            "holdout_data_seeds": [int(s) for s in holdout_seeds], "new_customer_data_seeds": [int(s) for s in new_customer_seeds],
            "primary_population": f"transactions with at least {h.MIN_PRIOR} earlier transactions of the same customer",
            "secondary_population": f"the primary population from {h.SECONDARY_START}",
            "cutoff_rule": ("per trained model, on the validation period of the training dataset: lowest tie-safe score with "
                            "false-positive rate <= 1% (Policy B) and <= 0.1% (Critical); the rule behind the fixed seed-42 "
                            "cut-offs (4C-3B), which it reproduces exactly; never tuned on hold-out data"),
            "seed_42_cutoffs": {tier: dict(v) for tier, v in h.CUTOFFS.items()},
            "margins": {"recall": h.RECALL_MARGIN, "legit_alerts_per_1000": h.LEGIT_ALERT_MARGIN_PER_1000,
                        "critical_recall": h.CRITICAL_RECALL_MARGIN},
            "alert_budget_per_1000": h.ALERT_BUDGET_PER_1000,
            "bootstrap": {"resamples": reps, "seed": bootstrap_seed,
                          "data": "customer components resampled within each dataset",
                          "training_seeds": "the 5 seeds of each architecture redrawn with replacement in every resample",
                          "decision_interval": "ci95_data_and_training_seeds"},
        },
        "training_dataset": {"version": spec.version, "sha256": train_data.sha256,
                             "generator_seed": (train_data.manifest or {}).get("seed")},
        "training_runs": runs,
        "multiseed_files_sha256": files_before,
        "protected_model_files_sha256": before,
        "cutoffs": cut_info,
        "datasets": {"holdout": datasets, "new_customer": nc_datasets},
        "holdout": holdout,
        "new_customer": nc_report,
    }
    report["decision"] = decision(holdout["primary"]["comparison"])
    report["alert_budget"] = alert_budget(report, seeds)
    report["seed_42_and_multi_seed"] = seed_42_position(report, seeds)
    report["state"] = {"model_set_default": h._default_model_set(), "promotion": "none",
                       "production_models_trained": "none", "seed_42_candidates_retrained": "none",
                       "thresholds_changed": "none"}
    if h.model_file_checksums() != before or multiseed_file_checksums(seeds, root) != files_before:
        raise RuntimeError("model files changed during the multi-seed evaluation")
    return report


def measure_latency(seed: int = training.TRAINING_SEEDS[0], root=None, n: int = 200, warmup: int = 20, output_dir=None) -> dict:
    """Scoring time of one multi-seed model per architecture (latency depends on the
    architecture, not on the trained weights), each in its own process."""
    result = {}
    for a in ARCHITECTURES:
        done = subprocess.run([sys.executable, "-m", "app.evaluation.holdout", "latency-worker", "--model-set", a,
                               "--n", str(n), "--warmup", str(warmup), "--candidates-root", str(training.seed_root(seed, root))],
                              check=True, capture_output=True, text=True, cwd=str(config.BACKEND_DIR))
        result[key(a, seed)] = json.loads(done.stdout.strip().splitlines()[-1])
    out = h.assert_safe_output(output_dir or h.OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"note": "wall-clock time on the machine that ran this; not reproducible bit for bit",
               "method": "live pipeline, known customers, one transaction at a time, warm-up excluded", "models": result}
    (out / LATENCY_NAME).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    return payload


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", nargs="?", default="evaluate", choices=["evaluate", "latency"])
    p.add_argument("--output-dir", default=None, help=f"default: {h.OUTPUT_DIR}")
    p.add_argument("--reps", type=int, default=h.BOOTSTRAP_REPS)
    p.add_argument("--cache-dir", default=None, help="keep scored tables here (optional)")
    args = p.parse_args(argv)
    if args.mode == "latency":
        payload = measure_latency(output_dir=args.output_dir)
        print(json.dumps(payload["models"], indent=2))
        return payload
    missing = [s for s in training.TRAINING_SEEDS if s not in training.trained_seeds()]
    if missing:
        raise SystemExit(f"training seeds {missing} are not trained yet; run python -m app.training.multiseed")
    report = evaluate(reps=args.reps, cache_dir=args.cache_dir, log=lambda *a: print(*a, flush=True))
    path = h.write_report(report, args.output_dir, REPORT_NAME)
    print(f"wrote {path}")
    return report


if __name__ == "__main__":
    main()
