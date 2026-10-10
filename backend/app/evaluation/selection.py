"""
Pre-registered model-selection protocol (Step 4C-3E.6).

The rules are in docs/step4c3e-model-selection-protocol.md. This module is
their executable form, so the selection is mechanical and repeatable:

    cd backend
    python -m app.evaluation.selection --generate     # NOT run in the protocol step; needs approval

Three separate things:

A. Architecture.  The existing tie-break rule on the multi-seed evidence
   (app.evaluation.multiseed.decision). Nothing new is decided here.
B. Artifact.      For each architecture, which of the five trained seeds
   (11-15) represents it. Decided by select_seed() from DEVELOPMENT datasets
   only. The rule picks the most typical run, never the best-scoring one.
C. Final test.    A fresh hold-out that does not exist yet and is scored once,
   after the artifact selection has been written down. Not implemented here.

What the artifact rule may read: the five trained models of an architecture,
their validation cut-offs (the 4C-3B rule) and the development datasets
(DEVELOPMENT_SEEDS). What it may never read: the existing hold-out datasets
(101-105, 201-205), whose results are already known, and the final hold-out
(FINAL_HOLDOUT_SEEDS, FINAL_NEW_CUSTOMER_SEEDS). assert_development_seed()
enforces this, and write_record() refuses to run once a final hold-out exists.

Nothing here trains, promotes or deploys, and nothing writes to models/saved/,
models/candidates/ or models/candidates_multiseed/. MODEL_SET is not involved.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from ..training import multiseed as training
from . import holdout as h
from . import multiseed as ev

PROTOCOL_VERSION = "4C-3E.6 v1"

# ---- fixed by the protocol -------------------------------------------------------------------------
ELIGIBLE_ARCHITECTURES = ev.ARCHITECTURES                 # ("v2_dnn_lstm", "v2_dnn_only")
ELIGIBLE_SEEDS = training.TRAINING_SEEDS                   # (11, 12, 13, 14, 15)

DEVELOPMENT_SEEDS = (301, 302, 303, 304, 305)              # artifact selection; default generator, 500 customers
FINAL_HOLDOUT_SEEDS = (401, 402, 403, 404, 405)            # the fresh, untouched final hold-out
FINAL_NEW_CUSTOMER_SEEDS = (411, 412, 413, 414, 415)       # final hold-out with new-customer fraud
# Seeds whose data has already been used and can never be selection or final data.
SPENT_SEEDS = (h.TRAINING_SEED,) + tuple(h.HOLDOUT_SEEDS) + tuple(h.NEW_CUSTOMER_SEEDS)

DEVELOPMENT_DATA_DIR = h.HOLDOUT_DATA_DIR / "development"
FINAL_DATA_DIR = h.HOLDOUT_DATA_DIR / "final"
RECORD_NAME = "selection_record.json"

# Screens: every number is one the project already approved (4C-3B / 4C-3E).
MIN_RECALL = h.MIN_RECALL                                              # 0.40 at Policy B
ALERT_SCREEN_PER_1000 = h.ALERT_BUDGET_PER_1000 + h.LEGIT_ALERT_MARGIN_PER_1000   # 10 target + 1.0 margin = 11.0
CRITICAL_RECALL_MARGIN = h.CRITICAL_RECALL_MARGIN                      # 0.05 below the architecture median

# What "typical" is measured on. First-fraud and episode detection are secondary
# evidence: they shape which run is typical, they are never a screen or a target.
SELECTION_METRICS = ("recall", "legit_alerts_per_1000", "critical_recall", "first_fraud_recall", "episode_detection_rate")
DIGITS = 6


class SelectionError(ValueError):
    """Data that the selection may not use, or a selection that may not be made (again)."""


def majority(n: int) -> int:
    """More than half of the trained seeds."""
    return n // 2 + 1


# ---- which data may be used --------------------------------------------------------------------------

def assert_development_seed(seed: int, allowed=DEVELOPMENT_SEEDS) -> int:
    seed = int(seed)
    if seed in SPENT_SEEDS:
        raise SelectionError(f"data seed {seed} has already been used (training or existing hold-out); "
                             "its results are known and it may not be used to select an artifact")
    if seed in FINAL_HOLDOUT_SEEDS or seed in FINAL_NEW_CUSTOMER_SEEDS:
        raise SelectionError(f"data seed {seed} belongs to the final hold-out; it must stay untouched until the "
                             "selection is recorded")
    if seed in ELIGIBLE_SEEDS:
        raise SelectionError(f"{seed} is a training seed, not a development data seed")
    if seed not in tuple(allowed):
        raise SelectionError(f"data seed {seed} is not a development seed {tuple(allowed)}")
    return seed


def final_holdout_exists(final_root=None) -> bool:
    root = Path(final_root or FINAL_DATA_DIR)
    return root.exists() and any(root.iterdir())


# ---- B. the artifact rule (pure, deterministic) -----------------------------------------------------------

def select_seed(metrics: dict) -> dict:
    """metrics: {training seed: {metric: value}} for ONE architecture, measured on the
    pooled development datasets (primary population, each model at its own validation
    cut-offs). Returns the screens, the distances and the selected seed (or None).

    1. Screens, per seed:
         recall                 >= MIN_RECALL
         legit_alerts_per_1000  <= ALERT_SCREEN_PER_1000
         critical_recall        >= median over the seeds - CRITICAL_RECALL_MARGIN
    2. The architecture passes only if a majority of its seeds pass every screen.
    3. Among the seeds that pass, select the most typical one: the smallest sum, over
       SELECTION_METRICS, of |value - median over all seeds| / (standard deviation over
       all seeds). Ties: closest to the median recall, then the lowest seed number.

    The result does not depend on the order of the input and uses no randomness."""
    metrics = {int(s): v for s, v in metrics.items()}
    seeds = sorted(metrics)
    if len(seeds) < 3:
        raise SelectionError("the rule needs at least three trained seeds")
    for s in seeds:
        if any(metrics[s].get(m) is None or np.isnan(float(metrics[s][m])) for m in SELECTION_METRICS):
            raise SelectionError("a selection metric is missing")
    # inputs are taken at 6 decimals (as the reports store them), so the result is the same on every machine
    values = {m: np.round(np.array([float(metrics[s][m]) for s in seeds]), DIGITS) for m in SELECTION_METRICS}
    centre = {m: round(float(np.median(v)), DIGITS) for m, v in values.items()}
    scale = {m: float(np.std(v, ddof=1)) for m, v in values.items()}
    critical_floor = round(centre["critical_recall"] - CRITICAL_RECALL_MARGIN, DIGITS)

    per_seed = {}
    for i, s in enumerate(seeds):
        screens = {
            "recall_at_least_minimum": bool(values["recall"][i] >= MIN_RECALL),
            "alert_burden_within_screen": bool(values["legit_alerts_per_1000"][i] <= ALERT_SCREEN_PER_1000),
            "critical_recall_not_an_outlier": bool(values["critical_recall"][i] >= critical_floor),
        }
        parts = {m: (abs(values[m][i] - centre[m]) / scale[m] if scale[m] > 0 else 0.0) for m in SELECTION_METRICS}
        per_seed[s] = {
            "metrics": {m: round(float(values[m][i]), DIGITS) for m in SELECTION_METRICS},
            "screens": screens, "eligible": all(screens.values()),
            "distance_parts": {m: round(v, DIGITS) for m, v in parts.items()},
            "distance": round(sum(parts.values()), DIGITS),
            "recall_distance_from_median": round(abs(values["recall"][i] - centre["recall"]), DIGITS),
        }
    eligible = [s for s in seeds if per_seed[s]["eligible"]]
    needed = majority(len(seeds))
    passes = len(eligible) >= needed
    selected = None
    if passes:
        selected = min(eligible, key=lambda s: (per_seed[s]["distance"], per_seed[s]["recall_distance_from_median"], s))
    best_recall = max(seeds, key=lambda s: (per_seed[s]["metrics"]["recall"], -s))
    return {
        "seeds": seeds,
        "median": {m: round(v, DIGITS) for m, v in centre.items()},
        "standard_deviation": {m: round(v, DIGITS) for m, v in scale.items()},
        "per_seed": {str(s): per_seed[s] for s in seeds},
        "eligible_seeds": eligible, "eligible_needed": needed, "architecture_passes": passes,
        "selected_seed": selected,
        "highest_recall_seed": best_recall,
        "selected_is_highest_recall_seed": selected is not None and selected == best_recall,
        "note": "the selected seed is the most typical eligible run; recall is never maximised",
    }


def stage_outcome(selection: dict, lead: str) -> dict:
    """What goes to the final hold-out. `lead` is the architecture decided in stage A."""
    if lead not in ELIGIBLE_ARCHITECTURES:
        raise SelectionError(f"unknown lead architecture {lead!r}")
    challenger = next(a for a in ELIGIBLE_ARCHITECTURES if a != lead)
    passed = [a for a in (lead, challenger) if selection[a]["architecture_passes"]]
    if not passed:
        return {"artifacts": [], "primary_candidate": None, "challenger": None,
                "outcome": "neither architecture passes the development screens: no artifact is selected, "
                           "no final hold-out is generated, production stays"}
    primary = passed[0]
    other = passed[1] if len(passed) > 1 else None
    text = (f"{primary} seed {selection[primary]['selected_seed']} is the primary candidate"
            + (f"; {other} seed {selection[other]['selected_seed']} is the required challenger" if other else
               f"; {challenger if primary == lead else lead} did not pass the development screens and is not carried forward"))
    return {"artifacts": [ev.key(a, selection[a]["selected_seed"]) for a in passed],
            "primary_candidate": ev.key(primary, selection[primary]["selected_seed"]),
            "challenger": ev.key(other, selection[other]["selected_seed"]) if other else None, "outcome": text}


# ---- A. the architecture (the existing rule, re-read from the multi-seed evidence) --------------------------

def architecture_lead(multiseed_report: dict) -> dict:
    """Applies the existing tie-break rule to the multi-seed comparison. Only the
    architecture-level comparison is read; no per-seed hold-out result is."""
    d = ev.decision(multiseed_report["holdout"]["primary"]["comparison"])
    return {"rule": "existing pre-registered tie-break (4C-3E)", "lead": d["provisional_lead"],
            "all_conditions_hold": d["all_conditions_hold"],
            "conditions": {k: {"ci95": c["ci95_data_and_training_seeds"], "holds": c["holds"]} for k, c in d["conditions"].items()}}


# ---- running B on development data (not executed in the protocol step) --------------------------------------

def development_metrics(training_seeds=ELIGIBLE_SEEDS, root=None, development_seeds=DEVELOPMENT_SEEDS,
                        development_root=None, spec=None) -> dict:
    """Scores the trained models on the development datasets (primary population, own
    validation cut-offs) and returns the selection metrics per architecture and seed."""
    import pandas as pd
    from ..training import candidates as cand
    from .datasets import load_evaluation_data, resolve_dataset
    seeds = [int(s) for s in training_seeds]
    for s in development_seeds:
        assert_development_seed(s, development_seeds)
    spec = spec or resolve_dataset("v2")
    train_data = load_evaluation_data(spec)
    labels, _, _, _ = cand.load_saved_split(spec, train_data)
    mask = ev.validation_mask(train_data, labels)
    models = ev.load_trained(seeds, root)
    names = list(models)
    cut_info = {k: ev.policy_cutoffs(models[k], train_data, mask) for k in names}
    cutoffs = {tier: {k: cut_info[k][tier] for k in names} for tier in ev.POLICY_FPR}
    tables, datasets = [], {}
    for s in development_seeds:
        data = h.load_holdout(s, development_root or DEVELOPMENT_DATA_DIR)
        datasets[str(s)] = {"sha256": data.sha256, "transactions": int(len(data.frame)),
                            "generator_version": data.manifest.get("generator_version")}
        tables.append(h.score_dataset(data, models, cutoffs))
    primary = h.population(pd.concat(tables, ignore_index=True), "primary")
    res = h.evaluate_population(primary, names, 0)
    metrics = {a: {s: {m: res["models"][ev.key(a, s)]["metrics"][m]["value"] for m in SELECTION_METRICS} for s in seeds}
               for a in ELIGIBLE_ARCHITECTURES}
    return {"metrics": metrics, "cutoffs": cut_info, "datasets": datasets,
            "population": {k: res[k] for k in ("rows", "fraud_transactions", "fraud_episodes", "datasets")}}


def build_record(development: dict, lead: dict, training_seeds=ELIGIBLE_SEEDS, root=None) -> dict:
    selection = {a: select_seed(development["metrics"][a]) for a in ELIGIBLE_ARCHITECTURES}
    outcome = stage_outcome(selection, lead["lead"])
    artifacts = {}
    for name in outcome["artifacts"]:
        arch, seed = name.split("@")
        directory = training.seed_root(int(seed), root) / ev.CANDIDATE_DIR[arch]
        manifest = json.loads((directory / "manifest.json").read_text())
        artifacts[name] = {
            "architecture": arch, "training_seed": int(seed),
            "weights_sha256": {k: v for k, v in manifest["model"].items() if k.endswith("weights_sha256")},
            "files_sha256": manifest["files"],
            "cutoffs": {tier: development["cutoffs"][name][tier] for tier in ev.POLICY_FPR},
        }
    return {
        "protocol": PROTOCOL_VERSION,
        "fixed_inputs": {
            "eligible_architectures": list(ELIGIBLE_ARCHITECTURES), "eligible_training_seeds": [int(s) for s in training_seeds],
            "development_seeds": list(development["datasets"]), "selection_metrics": list(SELECTION_METRICS),
            "screens": {"minimum_recall": MIN_RECALL, "alert_screen_per_1000": ALERT_SCREEN_PER_1000,
                        "critical_recall_margin_below_median": CRITICAL_RECALL_MARGIN,
                        "eligible_seeds_needed": majority(len(training_seeds))},
            "final_holdout_seeds": list(FINAL_HOLDOUT_SEEDS), "final_new_customer_seeds": list(FINAL_NEW_CUSTOMER_SEEDS),
            "never_used_for_selection": list(SPENT_SEEDS),
        },
        "stage_a_architecture": lead,
        "development": {"datasets": development["datasets"], "population": development["population"]},
        "stage_b_selection": selection,
        "stage_b_outcome": outcome,
        "selected_artifacts": artifacts,
        "stage_c_final_holdout": "not generated, not scored",
        "state": {"model_set_default": h._default_model_set(), "promotion": "none", "models_trained": "none"},
    }


def write_record(record: dict, output_dir=None, final_root=None) -> Path:
    """Writes the selection record once. Refuses if a record or a final hold-out already exists."""
    out = h.assert_safe_output(output_dir or h.OUTPUT_DIR)
    if final_holdout_exists(final_root):
        raise SelectionError("a final hold-out already exists; the selection must be recorded before it is generated")
    path = out / RECORD_NAME
    if path.exists():
        raise SelectionError(f"{path} already exists; the selection is made once and is never repeated")
    out.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--generate", action="store_true", help="generate missing development datasets first")
    p.add_argument("--output-dir", default=None, help=f"default: {h.OUTPUT_DIR}")
    args = p.parse_args(argv)
    out = h.assert_safe_output(args.output_dir or h.OUTPUT_DIR)
    if (out / RECORD_NAME).exists():
        raise SystemExit(f"{out / RECORD_NAME} already exists; the selection is made once")
    if final_holdout_exists():
        raise SystemExit("a final hold-out already exists; the selection must be recorded before it is generated")
    before = h.model_file_checksums()
    files = ev.multiseed_file_checksums(ELIGIBLE_SEEDS)
    if args.generate:
        for s in DEVELOPMENT_SEEDS:
            h.generate_dataset(assert_development_seed(s), DEVELOPMENT_DATA_DIR)
    lead = architecture_lead(json.loads((h.OUTPUT_DIR / ev.REPORT_NAME).read_text()))
    record = build_record(development_metrics(), lead)
    if h.model_file_checksums() != before or ev.multiseed_file_checksums(ELIGIBLE_SEEDS) != files:
        raise RuntimeError("model files changed during the selection")
    path = write_record(record, args.output_dir)
    print(json.dumps(record["stage_b_outcome"], indent=2))
    print(f"wrote {path}")
    return record


if __name__ == "__main__":
    main()
