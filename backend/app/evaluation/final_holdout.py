"""
Stage C of the pre-registered model-selection protocol (Step 4C-3E.6): the
final evaluation on a fresh hold-out.

    cd backend
    python -m app.evaluation.final_holdout            # generate 401-405 / 411-415, score once, write the report
    python -m app.evaluation.final_holdout latency

Rules: docs/step4c3e-model-selection-protocol.md, section 7. Nothing in this
module is a choice made after seeing final data; it was written and tested
(on samples and on the development data) before the final hold-out existed.

What it does, in this order:

1. Requires the Stage B selection record and checks that the selected
   artifacts on disk are the recorded ones (weight and file checksums).
2. Generates the final hold-out: seeds 401-405 with the default generator and
   seeds 411-415 with the new-customer extension (20% late joiners, 30
   episodes), and keeps every dataset's manifest as generation evidence.
3. Scores exactly three models: `production` and the selected artifacts, at
   FROZEN cut-offs (the selection record's for the artifacts, the fixed 4C-3B
   table for production). No cut-off is computed here.
4. Applies the approved acceptance gates and the documented tie-break.
5. Writes the report once. A second run is refused.

It promotes nothing and selects nothing new: the output is "eligible for a
controlled-promotion decision" or "not eligible". MODEL_SET is not involved.
It never writes to models/saved/, models/candidates/ or
models/candidates_multiseed/, and compares their checksums before and after.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from .. import config
from ..training import multiseed as training
from . import holdout as h
from . import multiseed as ev
from . import ring_metrics
from . import selection as sel

REPORT_NAME = "final_holdout_report.json"
LATENCY_NAME = "final_holdout_latency.json"
MANIFEST_DIR = "final_holdout_manifests"
FINAL_NEW_CUSTOMER_DIR = sel.FINAL_DATA_DIR / "new_customer"
PRODUCTION = "production"


class StageCError(ValueError):
    """Stage C may not run: no selection record, changed artifacts, or already run."""


# ---- the selection record and the frozen inputs -----------------------------------------------------

def load_record(output_dir=None) -> tuple:
    path = Path(output_dir or h.OUTPUT_DIR) / sel.RECORD_NAME
    if not path.exists():
        raise StageCError(f"{path} not found: the artifact selection (Stage B) must be recorded before Stage C")
    record = json.loads(path.read_text())
    if record.get("protocol") != sel.PROTOCOL_VERSION:
        raise StageCError(f"the selection record is for protocol {record.get('protocol')!r}, not {sel.PROTOCOL_VERSION!r}")
    if not record["stage_b_outcome"]["artifacts"]:
        raise StageCError("Stage B selected no artifact; there is nothing to evaluate")
    return record, h._sha256(path)


def load_selected(record: dict, root=None) -> dict:
    """{name: LoadedModelSet} for production and the selected artifacts, after checking that
    every artifact on disk is the recorded one."""
    from ..model_sets import load_model_set
    models = {PRODUCTION: load_model_set(PRODUCTION)}
    for name in record["stage_b_outcome"]["artifacts"]:
        arch, seed = name.split("@")
        art = record["selected_artifacts"][name]
        directory = training.seed_root(int(seed), root) / ev.CANDIDATE_DIR[arch]
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["files"] != art["files_sha256"]:
            raise StageCError(f"{name}: the files on disk are not the ones in the selection record")
        for k, sha in art["weights_sha256"].items():
            if manifest["model"].get(k) != sha:
                raise StageCError(f"{name}: {k} differs from the selection record")
        for fname, sha in art["files_sha256"].items():
            if not fname.endswith(".keras") and h._sha256(directory / fname) != sha:
                raise StageCError(f"{name}: {fname} does not match the selection record")
        models[name] = load_model_set(arch, candidates_root=training.seed_root(int(seed), root))   # checks the weights
    return models


def frozen_cutoffs(record: dict) -> dict:
    """{tier: {model: cut-off}}: the selection record's values for the artifacts and the
    fixed 4C-3B table for production. Nothing is recomputed."""
    names = record["stage_b_outcome"]["artifacts"]
    return {tier: {PRODUCTION: h.CUTOFFS[tier][PRODUCTION],
                   **{n: record["selected_artifacts"][n]["cutoffs"][tier] for n in names}} for tier in ("policy_b", "critical")}


def comparison_pairs(names) -> list:
    """(v2_dnn_lstm artifact, v2_dnn_only artifact) first when both are present, then each artifact against production."""
    arts = [n for n in names if n != PRODUCTION]
    lstm = [n for n in arts if n.startswith("v2_dnn_lstm@")]
    only = [n for n in arts if n.startswith("v2_dnn_only@")]
    pairs = [(lstm[0], only[0])] if lstm and only else []
    return pairs + [(n, PRODUCTION) for n in arts]


# ---- generation -------------------------------------------------------------------------------------

def generate(final_seeds=sel.FINAL_HOLDOUT_SEEDS, new_customer_seeds=sel.FINAL_NEW_CUSTOMER_SEEDS, final_root=None,
             new_customer_root=None, customers: int = h.HOLDOUT_CUSTOMERS,
             new_customer_episodes: int = h.NEW_CUSTOMER_EPISODES_PER_SEED) -> None:
    for s in final_seeds:
        h.generate_dataset(s, final_root or sel.FINAL_DATA_DIR, customers=customers)
    for s in new_customer_seeds:
        h.generate_dataset(s, new_customer_root or FINAL_NEW_CUSTOMER_DIR, customers=customers,
                           late_joiner_share=h.NEW_CUSTOMER_LATE_JOINER_SHARE, new_customer_fraud_episodes=new_customer_episodes)


def _evidence(data, root, seed) -> dict:
    manifest = data.manifest
    return {**h.dataset_summary(data), "command": manifest.get("command"),
            "files_sha256": {k: v["sha256"] for k, v in manifest["files"].items()},
            "manifest_sha256": h._sha256(h.seed_dir(seed, root) / "manifest.json")}


# ---- the decision ---------------------------------------------------------------------------------

def stage_c_decision(primary: dict, artifacts) -> dict:
    """The approved gates and the documented tie-break, on the pooled primary population.
    `primary` is an evaluate_population result; `artifacts` the selected artifact names."""
    production_recall = primary["models"][PRODUCTION]["metrics"]["recall"]["value"]
    gates = {}
    for n in artifacts:
        met = primary["models"][n]["metrics"]
        alerts, recall = met["legit_alerts_per_1000"], met["recall"]
        checks = {
            "alert_budget_upper_bound_at_most_10": alerts["ci95"] is not None and alerts["ci95"][1] <= h.ALERT_BUDGET_PER_1000,
            "recall_at_least_0_40": recall["value"] is not None and recall["value"] >= h.MIN_RECALL,
            "recall_lower_bound_above_production": recall["ci95"] is not None and recall["ci95"][0] > production_recall,
        }
        gates[n] = {"legit_alerts_per_1000": alerts, "recall": recall, "production_recall": production_recall,
                    **checks, "eligible": all(checks.values())}
    eligible = [n for n in artifacts if gates[n]["eligible"]]
    lstm = next((n for n in artifacts if n.startswith("v2_dnn_lstm@")), None)
    only = next((n for n in artifacts if n.startswith("v2_dnn_only@")), None)

    tie_break = None
    pair = primary["paired_differences"].get(f"{lstm} - {only}") if lstm and only else None
    if pair is not None:
        def cond(metric, text, test):
            ci = pair[metric]["ci95"]
            return {"metric": metric, "test": text, "difference": pair[metric]["difference"], "ci95": ci,
                    "holds": bool(ci is not None and test(ci))}
        conditions = {
            "recall_at_least_matches": cond("recall", f"lower bound > -{h.RECALL_MARGIN}", lambda ci: ci[0] > -h.RECALL_MARGIN),
            "alert_burden_at_least_matches": cond("legit_alerts_per_1000", f"upper bound < +{h.LEGIT_ALERT_MARGIN_PER_1000} per 1,000",
                                                  lambda ci: ci[1] < h.LEGIT_ALERT_MARGIN_PER_1000),
            "critical_tier_not_worse": cond("critical_recall", f"lower bound > -{h.CRITICAL_RECALL_MARGIN}",
                                            lambda ci: ci[0] > -h.CRITICAL_RECALL_MARGIN),
        }
        holds = all(c["holds"] for c in conditions.values())
        tie_break = {"difference": f"{lstm} - {only}", "conditions": conditions, "all_conditions_hold": holds,
                     "applies": lstm in eligible and only in eligible,
                     "favours": lstm if holds else only,
                     "note": "applied only when both artifacts are eligible; otherwise reported for information"}

    if lstm in eligible and only in eligible:
        chosen = tie_break["favours"]
        reason = ("both artifacts pass the gates and all three tie-break conditions hold" if chosen == lstm
                  else "both artifacts pass the gates and a tie-break condition fails: the simpler model")
    elif eligible:
        chosen = eligible[0]
        reason = "it is the only artifact that passes the gates"
    else:
        chosen, reason = None, "no artifact passes the gates"

    first_fraud = None
    if pair is not None and pair["first_fraud_recall"]["ci95"] is not None:
        ff = pair["first_fraud_recall"]
        first_fraud = {"difference": ff["difference"], "ci95": ff["ci95"], "excludes_zero": ff["excludes_zero"],
                       "statement": (f"first-fraud detection, {lstm} - {only}: {ff['difference']:+.3f} "
                                     f"[{ff['ci95'][0]:+.3f}, {ff['ci95'][1]:+.3f}]; "
                                     + ("the interval excludes zero" if ff["excludes_zero"] else "the interval includes zero"))}
    return {
        "population": "primary (at least 10 earlier transactions), final hold-out datasets pooled",
        "gates": gates, "eligible_artifacts": eligible, "tie_break": tie_break,
        "eligible_for_a_controlled_promotion_decision": chosen, "reason": reason,
        "first_fraud_detection": first_fraud,
        "outcome": (f"{chosen} is eligible for a controlled-promotion decision" if chosen
                    else "no artifact is eligible; production stays"),
        "promotion": "none. This is evidence for a decision the owner takes; MODEL_SET stays production.",
    }


# ---- the evaluation ---------------------------------------------------------------------------------

def _per_dataset(rows: pd.DataFrame, names) -> dict:
    out = {}
    for s, g in rows.groupby("seed", sort=True):
        res = h.evaluate_population(g, names, 0)
        out[str(s)] = {"rows": res["rows"], "fraud_transactions": res["fraud_transactions"], "fraud_episodes": res["fraud_episodes"],
                       "models": {m: {k: (None if res["models"][m]["metrics"][k]["value"] is None
                                          else h._r(res["models"][m]["metrics"][k]["value"], 4))
                                      for k in ("recall", "legit_alerts_per_1000", "critical_recall", "first_fraud_recall",
                                                "episode_detection_rate")} for m in names}}
    return out


def evaluate(record: dict, root=None, final_seeds=sel.FINAL_HOLDOUT_SEEDS, final_root=None,
             new_customer_seeds=sel.FINAL_NEW_CUSTOMER_SEEDS, new_customer_root=None, reps: int = h.BOOTSTRAP_REPS,
             bootstrap_seed: int = h.BOOTSTRAP_SEED, record_sha256: str | None = None, log=lambda *_: None) -> dict:
    """The Stage C report (a dict). Scores production and the selected artifacts on the
    final hold-out at the frozen cut-offs and applies the decision rule."""
    before = h.model_file_checksums()
    artifacts = list(record["stage_b_outcome"]["artifacts"])
    files_before = ev.multiseed_file_checksums(sorted({int(n.split("@")[1]) for n in artifacts}), root)
    models = load_selected(record, root)
    names = list(models)
    cutoffs = frozen_cutoffs(record)
    pairs = comparison_pairs(names)
    final_root, nc_root = final_root or sel.FINAL_DATA_DIR, new_customer_root or FINAL_NEW_CUSTOMER_DIR

    datasets, tables, shared = {}, [], {}
    for s in final_seeds:
        log(f"final hold-out seed {s}")
        data = h.load_holdout(s, final_root)
        datasets[str(s)] = _evidence(data, final_root, s)
        tables.append(h.score_dataset(data, models, cutoffs))
        shared[str(s)] = ring_metrics.shared_device_rule(data.frame, data.metadata, data.grouping.episodes)
    table = pd.concat(tables, ignore_index=True)
    primary, secondary = h.population(table, "primary"), h.population(table, "secondary")
    log("bootstrap, primary")
    pooled = {"primary": h.evaluate_population(primary, names, reps, bootstrap_seed, pairs=pairs)}
    log("bootstrap, secondary")
    pooled["secondary"] = h.evaluate_population(secondary, names, reps, bootstrap_seed, pairs=pairs)
    per_dataset = _per_dataset(primary, names)
    over = {m: sum((d["models"][m]["legit_alerts_per_1000"] or 0.0) > h.ALERT_BUDGET_PER_1000 for d in per_dataset.values())
            for m in names}
    rings = {"policy_b": ring_metrics.ring_report(primary, names, "alert_", reps, bootstrap_seed, pairs),
             "critical": ring_metrics.ring_report(primary, names, "critical_", reps, bootstrap_seed, pairs),
             "shared_device_rule": shared}
    fraud_types = h.fraud_type_breakdown(primary, names)
    del table, tables

    nc_datasets, nc_tables = {}, []
    for s in new_customer_seeds:
        log(f"final new-customer seed {s}")
        data = h.load_holdout(s, nc_root)
        info = _evidence(data, nc_root, s)
        prior = data.grouping.episodes["prior_transactions_at_first_fraud"].astype(int)
        info["new_customer_fraud_episodes"] = int((prior < h.MIN_PRIOR).sum())
        nc_datasets[str(s)] = info
        scored = h.score_dataset(data, models, cutoffs)
        customers = pd.read_csv(data.spec.customers_csv, keep_default_na=False)
        scored["late_joiner"] = scored["customer_id"].isin(
            set(customers.loc[customers["join_date"] > customers["join_date"].min(), "customer_id"])).to_numpy()
        nc_tables.append(scored)
    nc = pd.concat(nc_tables, ignore_index=True)
    early = h.population(nc, "early_history")
    log("bootstrap, early transactions")
    early_res = h.evaluate_population(early, names, reps, bootstrap_seed, pairs=pairs)
    nc_full = h.population(nc, "primary")
    unchanged = nc_full[~nc_full["late_joiner"]]
    new_customer = {
        "note": ("transactions with fewer than 10 earlier ones, scored with the current cold-start behaviour; "
                 "no new-customer policy was approved before the data was generated, so no other rule is evaluated. "
                 "The fraud patterns are the generator's. Not part of the acceptance gates"),
        "new_customer_fraud_episodes": int(sum(d["new_customer_fraud_episodes"] for d in nc_datasets.values())),
        "current_behaviour": early_res,
        "by_earlier_transactions": {"policy_b": h.early_history(nc, names, "alert_"), "critical": h.early_history(nc, names, "critical_")},
        "full_history_rows_description_only": {
            "note": "rows with at least 10 earlier transactions in seeds 411-415; not part of the acceptance gates",
            "unchanged_customers_per_dataset": _per_dataset(unchanged, names),
            "late_joiners_pooled": (_per_dataset(nc_full[nc_full["late_joiner"]].assign(seed=0), names)["0"]
                                    if nc_full["late_joiner"].any() else None),
        },
    }

    prim = pooled["primary"]
    requirements = {
        "fraud_episodes_primary": {"value": prim["fraud_episodes"], "required": h.MIN_EPISODES, "met": prim["fraud_episodes"] >= h.MIN_EPISODES},
        "fraud_episodes_secondary": {"value": pooled["secondary"]["fraud_episodes"], "required": h.MIN_EPISODES,
                                     "met": pooled["secondary"]["fraud_episodes"] >= h.MIN_EPISODES},
        "rings_primary": {"value": rings["policy_b"]["rings"], "required": h.MIN_RINGS, "met": rings["policy_b"]["rings"] >= h.MIN_RINGS},
    }
    report = {
        "step": "4C-3E.6 Stage C: final hold-out evaluation",
        "protocol": record["protocol"],
        "selection_record_sha256": record_sha256,
        "fixed_inputs": {
            "final_holdout_seeds": [int(s) for s in final_seeds], "final_new_customer_seeds": [int(s) for s in new_customer_seeds],
            "models_scored": names, "primary_candidate": record["stage_b_outcome"]["primary_candidate"],
            "challenger": record["stage_b_outcome"]["challenger"],
            "frozen_cutoffs_score_0_1": cutoffs,
            "cutoffs_source": "selection record (artifacts) and the fixed 4C-3B table (production); not recomputed",
            "gates": {"legit_alerts_per_1000_upper_bound_at_most": h.ALERT_BUDGET_PER_1000, "recall_at_least": h.MIN_RECALL,
                      "recall_lower_bound": "above production's point estimate"},
            "tie_break_margins": {"recall": h.RECALL_MARGIN, "legit_alerts_per_1000": h.LEGIT_ALERT_MARGIN_PER_1000,
                                  "critical_recall": h.CRITICAL_RECALL_MARGIN},
            "primary_population": f"transactions with at least {h.MIN_PRIOR} earlier transactions of the same customer",
            "secondary_population": f"the primary population from {h.SECONDARY_START}",
            "bootstrap": {"resamples": reps, "seed": bootstrap_seed, "unit": "customer component, resampled within each dataset",
                          "interval": "2.5 / 97.5 percentiles"},
        },
        "selected_artifacts": record["selected_artifacts"],
        "protected_model_files_sha256": before,
        "datasets": {"final_holdout": datasets, "final_new_customer": nc_datasets},
        "size_requirements": requirements,
        "pooled": pooled,
        "per_dataset_primary": per_dataset,
        "datasets_over_alert_budget": {m: f"{over[m]} of {len(per_dataset)}" for m in names},
        "fraud_types_primary": fraud_types,
        "rings": rings,
        "new_customer": new_customer,
        "decision": stage_c_decision(prim, artifacts),
        "training_seed_variability_note": (
            "one trained model per architecture is tested; between training runs of one architecture the multi-seed "
            "report measured a standard deviation of about 0.05-0.06 in Policy B recall. A difference between the two "
            "artifacts is a difference between two trained models, not between architectures"),
        "state": {"model_set_default": h._default_model_set(), "promotion": "none", "models_trained": "none",
                  "cutoffs_recalibrated": "none", "seed_reselected": "none"},
    }
    if h.model_file_checksums() != before or ev.multiseed_file_checksums(sorted(int(s) for s in files_before), root) != files_before:
        raise RuntimeError("model files changed during the final evaluation")
    return report


def write_report(report: dict, output_dir=None) -> Path:
    """Writes the Stage C report once; a second report is refused."""
    out = h.assert_safe_output(output_dir or h.OUTPUT_DIR)
    path = out / REPORT_NAME
    if path.exists():
        raise StageCError(f"{path} already exists: the final hold-out is scored and reported once")
    out.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    return path


def save_manifests(final_seeds, final_root, new_customer_seeds, new_customer_root, output_dir=None) -> Path:
    """Copies every final dataset's manifest.json next to the report (generation evidence)."""
    out = h.assert_safe_output(output_dir or h.OUTPUT_DIR) / MANIFEST_DIR
    out.mkdir(parents=True, exist_ok=True)
    for seeds, root in ((final_seeds, final_root or sel.FINAL_DATA_DIR), (new_customer_seeds, new_customer_root or FINAL_NEW_CUSTOMER_DIR)):
        for s in seeds:
            shutil.copyfile(h.seed_dir(s, root) / "manifest.json", out / f"seed_{int(s)}_manifest.json")
    return out


def run(output_dir=None, root=None, final_seeds=sel.FINAL_HOLDOUT_SEEDS, new_customer_seeds=sel.FINAL_NEW_CUSTOMER_SEEDS,
        final_root=None, new_customer_root=None, reps: int = h.BOOTSTRAP_REPS, log=lambda *_: None, **generate_kwargs) -> dict:
    out = h.assert_safe_output(output_dir or h.OUTPUT_DIR)
    if (out / REPORT_NAME).exists():
        raise StageCError(f"{out / REPORT_NAME} already exists: the final hold-out is scored and reported once")
    record, record_sha = load_record(out)
    load_selected(record, root)                       # the artifacts must be the recorded ones before any data exists
    log("generating the final hold-out")
    generate(final_seeds, new_customer_seeds, final_root, new_customer_root, **generate_kwargs)
    report = evaluate(record, root, final_seeds, final_root, new_customer_seeds, new_customer_root, reps,
                      record_sha256=record_sha, log=log)
    if h._sha256(out / sel.RECORD_NAME) != record_sha:
        raise RuntimeError("the selection record changed during Stage C")
    save_manifests(final_seeds, final_root, new_customer_seeds, new_customer_root, out)
    write_report(report, out)
    return report


def measure_latency(output_dir=None, root=None, n: int = 200, warmup: int = 20) -> dict:
    """Scoring time of production and the selected artifacts, each in its own process."""
    out = h.assert_safe_output(output_dir or h.OUTPUT_DIR)
    record, _ = load_record(out)
    result = {}
    for name in [PRODUCTION] + list(record["stage_b_outcome"]["artifacts"]):
        command = [sys.executable, "-m", "app.evaluation.holdout", "latency-worker", "--n", str(n), "--warmup", str(warmup)]
        if name == PRODUCTION:
            command += ["--model-set", PRODUCTION]
        else:
            arch, seed = name.split("@")
            command += ["--model-set", arch, "--candidates-root", str(training.seed_root(int(seed), root))]
        done = subprocess.run(command, check=True, capture_output=True, text=True, cwd=str(config.BACKEND_DIR))
        result[name] = json.loads(done.stdout.strip().splitlines()[-1])
    payload = {"note": "wall-clock time on the machine that ran this; not reproducible bit for bit",
               "method": "live pipeline, known customers, one transaction at a time, warm-up excluded", "models": result}
    (out / LATENCY_NAME).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    return payload


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", nargs="?", default="evaluate", choices=["evaluate", "latency"])
    args = p.parse_args(argv)
    if args.mode == "latency":
        payload = measure_latency()
        print(json.dumps(payload["models"], indent=2))
        return payload
    try:
        report = run(log=lambda *a: print(*a, flush=True))
    except StageCError as e:
        raise SystemExit(str(e))
    print(json.dumps({k: report["decision"][k] for k in ("eligible_artifacts", "outcome")}, indent=2))
    print(f"wrote {h.OUTPUT_DIR / REPORT_NAME}")
    return report


if __name__ == "__main__":
    main()
