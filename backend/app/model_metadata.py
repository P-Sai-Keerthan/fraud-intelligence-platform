"""
Model metadata for observability (Step 4C-2f-2).

model_metadata(loaded_model_set) describes the loaded model set: its
version (content checksums), training dataset, feature version, whether an
LSTM is used, what the scores mean, and the alert/threshold configuration.
It is served by GET /model-info, embedded in /metrics, and used for
provenance (the model_set / model_version stored with every scored
transaction) and for the PDF wording.

It only reads files that the loader has already validated, and it never
changes scoring. Scores are NOT calibrated probabilities (class-weighted
training); the metadata says so.

Feature versions (docs/step4c2f-1-v1-feature-version-audit.md):
* LEGACY_V1  -- the feature code that produced data/transactions_with_features.csv
  (older than the initial commit: no std floor, no clipping). The production
  models' scalers and weights were fitted on these values.
* CURRENT    -- app/features/feature_engineering.build_point_features (std floor
  max(std, 0.1*avg, 1); amount_zscore clipped to +-10; amount_pct_of_avg clipped
  to [0, 1000]). Live inference uses it for every model set, and the v2 data
  the candidates were trained on was built with it.
"""

import hashlib
import inspect
import json
from functools import lru_cache
from pathlib import Path

from . import config
from .features.feature_engineering import FEATURE_COLUMNS, build_point_features
from .model_sets import (DEFAULT_MODEL_SET, LEGACY_PRODUCTION, MODEL_SETS, RF_FROZEN_CUTOFFS, SEED14_FROZEN_CUTOFFS,
                         architecture_name)
from .models.downstream_classifier import FAMILY_LABELS
from .models.dnn_model import alert_level_from_probability  # noqa: F401  (the bands documented below)

ALERT_BANDS = {
    "Low Risk": "fraud_probability < 25",
    "Medium Risk": "25 <= fraud_probability < 50",
    "High Risk": "50 <= fraud_probability < 80",
    "Critical Risk": "fraud_probability >= 80",
}
ALERT_BANDS_STATUS = ("legacy fixed bands, applied by /predict to every model set; not derived from any "
                      "evaluation of the loaded model; candidate-specific thresholds have not been approved")

FEATURE_VERSION_LEGACY_V1 = {
    "name": "legacy-v1",
    "description": "feature code that produced data/transactions_with_features.csv (older than the initial "
                   "commit): amount_zscore without the std floor and without clipping; amount_pct_of_avg not "
                   "clipped. Differs from the current code on amount_zscore (43,442 of 93,913 v1 rows) and "
                   "amount_pct_of_avg (4 rows).",
    "reference": "docs/step4c2f-1-v1-feature-version-audit.md",
}


@lru_cache(maxsize=None)
def current_feature_implementation() -> dict:
    source = inspect.getsource(build_point_features)
    return {
        "name": "current",
        "function": "app.features.feature_engineering.build_point_features",
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "description": "std floor max(std, 0.1*avg, 1); amount_zscore clipped to +-10; amount_pct_of_avg "
                       "clipped to [0, 1000]",
    }


def feature_list_sha256(columns=FEATURE_COLUMNS) -> str:
    return hashlib.sha256(json.dumps(list(columns)).encode()).hexdigest()


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rel(path) -> str:
    p = Path(path).resolve()
    try:
        return p.relative_to(config.BACKEND_DIR.parent.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


@lru_cache(maxsize=None)
def _production_dataset_sha256() -> str | None:
    return _sha256(config.FEATURES_CSV) if config.FEATURES_CSV.exists() else None


RISK_SCORE_SEMANTICS = {
    "production": "LSTM risk score: the production LSTM's output x 100 for the customer's 10 previous "
                  "transactions (before this one); an input to the DNN. With fewer than 10 earlier "
                  "transactions the LSTM is not run and the training-average value is reported (4C-2f-1).",
    "v2_dnn_lstm": "LSTM behavioral-risk component: the v2 LSTM's output x 100 for the customer's 10 previous "
                   "transactions; an input to the DNN. With fewer than 10 earlier transactions the LSTM is not "
                   "run and the training-average value is reported (4C-2f-1).",
    "v2_dnn_only": "No LSTM in this model set: risk_score repeats the DNN fraud score (equal to "
                   "fraud_probability) and is kept only for API compatibility.",
    "v2_lstm_rf_seed14": "Temporal risk signal: the seed-14 v2 LSTM's output x 100 for the customer's 10 previous "
                         "transactions (the same LSTM as v2_dnn_lstm_seed14, unchanged); an input to the random "
                         "forest. With fewer than 10 earlier transactions the LSTM is not run and the "
                         "training-average value is reported (4C-2f-1).",
}
RF_FRAUD_SCORE_SEMANTICS = ("Random forest output (predict_proba of the fraud class) x 100, capped at 99.9. A model "
                            "score, not a calibrated probability: on the synthetic development data its overall "
                            "calibration error is small because almost every transaction scores near 0, but in the "
                            "middle and upper range it under-states how often the transaction is fraud "
                            "(docs/model_selection_report.md, section 8).")
FRAUD_PROBABILITY_SEMANTICS = ("DNN output x 100, capped at 99.9. A fraud SCORE from class-weighted training, "
                               "not a calibrated probability.")


STATUS_PRODUCTION = "PRODUCTION"
STATUS_EVALUATION = "EVALUATION / NOT DEPLOYED"
STATUS_LEGACY = "PREVIOUS DEFAULT (v1, LSTM -> DNN)"

DOWNSTREAM_DIR = config.BACKEND_DIR / "models" / "evaluation" / "downstream"
DOWNSTREAM_DEVELOPMENT_REPORT = DOWNSTREAM_DIR / "development_report.json"
DOWNSTREAM_FINAL_REPORT = DOWNSTREAM_DIR / "final_holdout_report.json"
DOWNSTREAM_RECORD = DOWNSTREAM_DIR / "selection_record.json"

V2_HOLDOUT_DIR = config.BACKEND_DIR / "models" / "evaluation" / "v2_holdout"
SELECTION_RECORD_PATH = V2_HOLDOUT_DIR / "selection_record.json"
FINAL_HOLDOUT_REPORT_PATH = V2_HOLDOUT_DIR / "final_holdout_report.json"

NEW_CUSTOMER_LIMITATION = (
    "Customers with fewer than 10 earlier transactions: on the Stage C new-customer population this artifact "
    "raised about 25.6 legitimate alerts per 1,000 transactions at the frozen Policy B cut-off under the "
    "current cold-start behaviour, against 9.1 for customers with full history. Policy B performance is "
    "therefore NOT claimed for cold-start customers, and no new-customer policy has been approved.")
PROMOTION_BLOCKERS = (
    "no approved new-customer (cold-start) policy",
    "the frozen Policy B / Critical cut-offs are not applied by /predict, which keeps the legacy 25/50/80 bands; "
    "the Stage C figures describe the frozen cut-offs, not the live bands",
    "all evidence is on synthetic data (generator v2)",
    "promotion is the owner's decision; none has been made and MODEL_SET stays production",
)


def model_version(ms) -> str:
    """A short, content-derived identifier of the loaded weights."""
    if ms.manifest is not None and ms.manifest.get("model_set_kind") == "lstm_classifier":
        return f"{ms.name}-{ms.manifest['files']['classifier.joblib'][:12]}"
    if ms.manifest is None:
        digest = hashlib.sha256("".join(_sha256(ms.files[k]) for k in sorted(ms.files)).encode()).hexdigest()
        return f"{ms.name}-{digest[:12]}"
    return f"{ms.name}-{ms.manifest['model']['dnn_weights_sha256'][:12]}"


def _status(ms) -> str:
    if ms.name == DEFAULT_MODEL_SET:
        return STATUS_PRODUCTION
    if ms.name == LEGACY_PRODUCTION:
        return STATUS_LEGACY
    return STATUS_EVALUATION


def _read_json(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


def lstm_classifier_metadata(ms) -> dict:
    """/model-info for an LSTM -> scikit-learn classifier model set (Step 4D)."""
    man = ms.manifest
    spec = MODEL_SETS[ms.name]
    live_features = current_feature_implementation()
    label = FAMILY_LABELS.get(spec.family, spec.family)
    final = _read_json(DOWNSTREAM_FINAL_REPORT) or {}
    dev = _read_json(DOWNSTREAM_DEVELOPMENT_REPORT) or {}
    calib = (dev.get("calibration_by_family") or {}).get(spec.family)
    return {
        "model_set": ms.name,
        "model_version": model_version(ms),
        "status": _status(ms),
        "architecture": f"LSTM + {label}",
        "downstream_classifier": {
            "family": spec.family, "label": label, "class": man.get("classifier_class"), "params": man.get("params"),
            "library": f"scikit-learn {man.get('versions', {}).get('scikit_learn', '')}".strip(),
            "selected_by": "pre-registered protocol 4D v1 (docs/step4d-downstream-selection-protocol.md): highest mean "
                           "PR-AUC on development data among the candidates that clearly beat the DNN; confirmed once "
                           "on a fresh hold-out (docs/model_selection_report.md)",
        },
        "training_seed": man.get("seed"),
        "directory": _rel(ms.directory),
        "uses_lstm": True,
        "lstm": {"window": f"{config.SEQUENCE_LENGTH} previous transactions x {len(FEATURE_COLUMNS)} behavioral features",
                 "source": man["lstm"]["source"], "lstm_weights_sha256": man["lstm"]["lstm_weights_sha256"],
                 "output": "risk_score = LSTM output x 100 (an input to the classifier)"},
        "dnn_input_columns": list(ms.dnn_input_columns),
        "classifier_input_columns": list(ms.dnn_input_columns),
        "score_semantics": {
            "fraud_probability": RF_FRAUD_SCORE_SEMANTICS,
            "risk_score": RISK_SCORE_SEMANTICS[ms.name],
            "calibrated_probabilities": False,
        },
        "calibration": {
            "status": "not demonstrated: the score is shown as a Fraud Score (model score)",
            "development_data_mean_over_training_seeds": calib,
            "source": _rel(DOWNSTREAM_DEVELOPMENT_REPORT),
        },
        "explanations": {"method": "shap.TreeExplainer (exact interventional Tree SHAP on the classifier's score), "
                                   "background: 200 training rows", "explains": "this model set's random forest"},
        "thresholds": {
            "alert_bands": ALERT_BANDS,
            "alert_bands_status": ALERT_BANDS_STATUS,
            "frozen_cutoffs": {
                "policy_b": RF_FROZEN_CUTOFFS["policy_b"], "critical": RF_FROZEN_CUTOFFS["critical"],
                "scale": "classifier output, 0-1 (fraud score / 100)",
                "source": "validation period of the training dataset, 4C-3B rule (false-positive rate within 1% / "
                          "0.1%), recorded in selection_record.json",
                "status": "used for every reported metric; NOT applied by /predict, which keeps the fixed bands",
            },
        },
        "cold_start": {
            "min_prior_transactions": config.SEQUENCE_LENGTH,
            "policy": "fewer than 10 earlier transactions: LSTM not run, risk_score input = training mean; "
                      "first transaction: baseline-relative features treated as missing (4C-2f-1)",
        },
        "model": {"artifact": spec.artifact, "manifest_sha256": _sha256(ms.directory / "manifest.json"),
                  "files_sha256": dict(spec.pinned["files"]),
                  "files_verified": "every file matched its pinned SHA-256 before it was loaded",
                  "seed": man.get("seed")},
        "dataset": {"version": man["dataset"]["version"], "file": man["dataset"].get("file"),
                    "sha256": man["dataset"]["sha256"], "generator_version": man["dataset"].get("generator_version"),
                    "split_sha256": (man.get("split") or {}).get("sha256"), "synthetic": True},
        "features": {
            "columns": list(man["features"]["columns"]), "list_sha256": man["features"]["sha256"],
            "training_feature_version": {"name": "current", "description": live_features["description"],
                                         "source": "v2 generator, built with build_point_features"},
            "live_feature_version": live_features,
            "training_and_live_features_consistent": True,
        },
        "evaluation": {
            "status": final.get("outcome", "final hold-out report not found"),
            "confirmed_on_fresh_holdout": final.get("confirmed"),
            "source": _rel(DOWNSTREAM_FINAL_REPORT),
            "development_source": _rel(DOWNSTREAM_DEVELOPMENT_REPORT),
            "dataset_version": "v2 (synthetic)",
            "evaluates": "exactly these files: development datasets 301-305 (selection) and the fresh hold-out "
                         "501-505 / 511-515 (confirmation, scored once), customers with at least 10 earlier "
                         "transactions, at the frozen cut-offs",
        },
        "limitations": [
            "all data is synthetic (generator v2); no result describes real banking traffic",
            "the application's customer histories are the v1 seed data, while this model was trained on v2 data",
            "the score is not a calibrated probability",
            "the fixed 25/50/80 alert bands were not derived for this model; see the model selection report",
        ],
    }


def model_metadata(ms) -> dict:
    if ms.manifest is not None and ms.manifest.get("model_set_kind") == "lstm_classifier":
        return lstm_classifier_metadata(ms)
    live_features = current_feature_implementation()
    meta = {
        "model_set": ms.name,
        "model_version": model_version(ms),
        "status": _status(ms),
        "architecture": "DNN + LSTM" if ms.uses_lstm else "DNN only",
        "downstream_classifier": {"family": "dnn", "label": "DNN", "class": "Keras Sequential (dense)"},
        "training_seed": None if ms.manifest is None else ms.manifest.get("seed"),
        "directory": _rel(ms.directory),
        "uses_lstm": bool(ms.uses_lstm),
        "dnn_input_columns": list(ms.dnn_input_columns),
        "score_semantics": {
            "fraud_probability": FRAUD_PROBABILITY_SEMANTICS,
            "risk_score": RISK_SCORE_SEMANTICS[architecture_name(ms.name)],
            "calibrated_probabilities": False,
        },
        "thresholds": {
            "alert_bands": ALERT_BANDS,
            "alert_bands_status": ALERT_BANDS_STATUS,
        },
        "cold_start": {
            "min_prior_transactions": config.SEQUENCE_LENGTH,
            "policy": "fewer than 10 earlier transactions: LSTM not run, risk_score input = training mean; "
                      "first transaction: baseline-relative features treated as missing (4C-2f-1)",
        },
    }
    if ms.manifest is None:                                    # production
        meta["model"] = {
            "files_sha256": {Path(p).name: _sha256(p) for p in sorted(map(str, ms.files.values()))},
            "training": "legacy production recipe: random 80/20 split, in-sample LSTM -> DNN stacking "
                        "(docs/step4c2e-retraining-design.md)",
        }
        meta["dataset"] = {"version": "v1", "file": _rel(config.FEATURES_CSV), "sha256": _production_dataset_sha256()}
        meta["features"] = {
            "columns": list(FEATURE_COLUMNS), "list_sha256": feature_list_sha256(),
            "training_feature_version": FEATURE_VERSION_LEGACY_V1,
            "live_feature_version": live_features,
            "training_and_live_features_consistent": False,
            "note": "production was trained on legacy-v1 feature values; live inference computes current "
                    "values (19 of 88,913 v1 windows change alert level)",
        }
        meta["evaluation"] = {
            "source": _rel(config.EVAL_REPORT_PATH),
            "dataset_version": "v1",
            "evaluates": "evaluation copies of the production architecture retrained on the v1 time split "
                         "(legacy-v1 feature values); not the deployed weights in models/saved/",
        }
    else:                                                      # candidate
        man = ms.manifest
        meta["model"] = {
            "candidate": man["candidate"],
            "manifest_sha256": _sha256(ms.directory / "manifest.json"),
            "dnn_weights_sha256": man["model"]["dnn_weights_sha256"],
            "lstm_weights_sha256": man["model"].get("lstm_weights_sha256"),
            "seed": man.get("seed"),
            "recipe": man.get("recipe"),
        }
        meta["dataset"] = {"version": man["dataset"]["version"], "file": man["dataset"].get("file"),
                           "sha256": man["dataset"]["sha256"], "generator_version": man["dataset"].get("generator_version"),
                           "split_sha256": (man.get("split") or {}).get("sha256")}
        meta["features"] = {
            "columns": list(man["features"]["columns"]), "list_sha256": man["features"]["sha256"],
            "training_feature_version": {"name": "current", "description": live_features["description"],
                                         "source": "v2 generator, built with build_point_features"},
            "live_feature_version": live_features,
            "training_and_live_features_consistent": True,
        }
        meta["thresholds"]["candidate_thresholds"] = {
            "f1_optimal": man["thresholds"]["f1_optimal"],
            "fpr_operating_points": man["thresholds"]["fpr_operating_points"],
            "source": man["thresholds"].get("source"),
            "status": "recorded in the manifest (validation split); NOT approved and NOT applied by /predict",
        }
        meta["evaluation"] = {
            "source": _rel(ms.directory.parent / "comparison.json"),
            "dataset_version": man["dataset"]["version"],
            "evaluates": "this candidate's saved weights on the v2 time-split test period (synthetic data)",
        }
        spec = MODEL_SETS[ms.name]
        if spec.pinned is not None:                            # the selected seed-14 artifact (4C-3F)
            meta["model"]["files_sha256"] = dict(spec.pinned["files"])
            meta["model"]["files_verified"] = "every file matched its pinned SHA-256 when the model set was loaded"
            meta["selection"] = {
                "artifact": spec.artifact,
                "architecture_model_set": spec.architecture_of,
                "training_seed": spec.training_seed,
                "protocol": "4C-3E.6 (pre-registered): Stage B artifact selection, Stage C final hold-out",
                "selection_record": _rel(SELECTION_RECORD_PATH),
                "selection_record_sha256": _sha256(SELECTION_RECORD_PATH) if SELECTION_RECORD_PATH.exists() else None,
                "outcome": "passed the Stage C gates; eligible for a controlled-promotion decision",
                "presentation": "Validated candidate - not yet deployed",
                "new_customer_limitation": NEW_CUSTOMER_LIMITATION,
                "promotion_blockers": list(PROMOTION_BLOCKERS),
            }
            meta["thresholds"]["frozen_cutoffs"] = {
                "policy_b": SEED14_FROZEN_CUTOFFS["policy_b"],
                "critical": SEED14_FROZEN_CUTOFFS["critical"],
                "scale": "DNN output, 0-1 (fraud score / 100)",
                "source": "frozen at Stage B (selection_record.json) and used unchanged in Stage C",
                "status": "recorded for provenance; NOT applied by /predict, which keeps the legacy fixed bands",
            }
            meta["evaluation"] = {
                "source": _rel(FINAL_HOLDOUT_REPORT_PATH),
                "dataset_version": man["dataset"]["version"],
                "evaluates": "exactly these weights on the Stage C final hold-out (synthetic v2 data, seeds "
                             "401-405, customers with at least 10 earlier transactions) at the frozen cut-offs",
            }
    return meta


CANDIDATE_KEYS = {"v2_dnn_lstm": "candidate_a_dnn_lstm", "v2_dnn_only": "candidate_b_dnn_only"}


def candidate_evaluation(ms) -> dict:
    """The loaded candidate's slice of comparison.json, or why it is unavailable.
    Only returned when comparison.json describes exactly the loaded weights."""
    path = ms.directory.parent / "comparison.json"
    key = CANDIDATE_KEYS.get(ms.name)
    if MODEL_SETS[ms.name].pinned is not None:
        return {"available": False, "reason": "comparison.json describes the seed-42 candidates; this artifact is "
                                              "evaluated by the Stage C final hold-out (final_holdout_evaluation)"}
    if key is None:
        return {"available": False, "reason": "not a candidate model set"}
    if not path.exists():
        return {"available": False, "reason": f"{_rel(path)} not found"}
    comp = json.loads(path.read_text())
    entry = comp["candidates"][key]
    man = ms.manifest
    if (entry.get("dnn_weights_sha256") != man["model"]["dnn_weights_sha256"]
            or entry.get("lstm_weights_sha256") != man["model"].get("lstm_weights_sha256")):
        return {"available": False, "reason": "comparison.json was produced for different weights"}
    return {
        "available": True,
        "source": _rel(path),
        "source_sha256": _sha256(path),
        "dataset": comp["dataset"],
        "split": comp["split"],
        "scores_note": comp["scores_note"],
        "rows": comp["rows"],
        "overall_test": comp["overall_test"][key],
        "operating_points": comp["operating_points"][key],
        "alert_bands_production_25_50_80": comp["alert_bands_production_25_50_80"][key],
        "episodes_test": {k: v for k, v in comp["episodes_test"][key].items() if k != "per_episode"},
        "limitations": comp["limitations"],
    }


_HOLDOUT_METRICS = ("recall", "precision", "legit_alerts_per_1000", "critical_recall", "critical_legit_alerts_per_1000",
                    "first_fraud_recall", "episode_detection_rate", "pr_auc", "roc_auc")


def final_holdout_evaluation(ms) -> dict:
    """The loaded artifact's slice of final_holdout_report.json (Stage C), or why
    it is unavailable. Only returned when the report scored exactly the loaded
    weights. Values are copied from the report; nothing is recomputed."""
    spec = MODEL_SETS[ms.name]
    if spec.pinned is None:
        return {"available": False, "reason": "the final hold-out evaluated one selected artifact; "
                                              "this model set is not it"}
    path = FINAL_HOLDOUT_REPORT_PATH
    if not path.exists():
        return {"available": False, "reason": f"{_rel(path)} not found"}
    report = json.loads(path.read_text())
    art = spec.artifact
    recorded = (report.get("selected_artifacts") or {}).get(art) or {}
    man = ms.manifest["model"]
    if (recorded.get("weights_sha256") or {}) != {"dnn_weights_sha256": man["dnn_weights_sha256"],
                                                  "lstm_weights_sha256": man.get("lstm_weights_sha256")}:
        return {"available": False, "reason": "final_holdout_report.json was produced for different weights"}

    def pick(models):
        m = models[art]
        return {"counts": m["counts"], "metrics": {k: m["metrics"][k] for k in _HOLDOUT_METRICS if k in m["metrics"]}}

    primary, new = report["pooled"]["primary"], report["new_customer"]["current_behaviour"]
    population = lambda p: {k: p[k] for k in ("rows", "fraud_transactions", "legitimate_transactions",
                                              "fraud_episodes", "customers", "datasets")}
    decision = report["decision"]
    return {
        "available": True,
        "artifact": art,
        "source": _rel(path),
        "source_sha256": _sha256(path),
        "step": report["step"],
        "frozen_cutoffs_score_0_1": {k: v[art] for k, v in report["fixed_inputs"]["frozen_cutoffs_score_0_1"].items()},
        "primary": {"population": decision["population"], **population(primary), **pick(primary["models"]),
                    "production_recall": decision["gates"][art]["production_recall"]},
        "gates": {"definition": report["fixed_inputs"]["gates"], "result": decision["gates"][art]},
        "new_customer": {"note": report["new_customer"]["note"], **population(new), **pick(new["models"]),
                         "limitation": NEW_CUSTOMER_LIMITATION},
        "outcome": decision["outcome"],
        "promotion": decision["promotion"],
        "scores_note": "Scores are not calibrated probabilities. Alerts are counted at the frozen Policy B "
                       "cut-off, which /predict does not apply.",
    }


# ---- Step 4D: evaluation of the LSTM -> classifier model set --------------------------------------------

_COMPARISON_METRICS = ("pr_auc", "roc_auc", "precision", "recall", "f1", "legit_alerts_per_1000", "critical_recall",
                       "first_fraud_recall", "episode_detection_rate")
_FINAL_NAMES = {"selected": "Selected model", "dnn_seed14": "DNN (same LSTM, seed 14)", "production": "Previous default (v1)"}


def downstream_evaluation(ms) -> dict:
    """The model-selection evidence for an LSTM -> classifier model set, copied from
    models/evaluation/downstream/*.json (nothing is recomputed). Only returned when
    the reports describe exactly the loaded classifier file."""
    if ms.manifest is None or ms.manifest.get("model_set_kind") != "lstm_classifier":
        return {"available": False, "reason": "not an LSTM -> classifier model set"}
    dev, final, record = (_read_json(p) for p in (DOWNSTREAM_DEVELOPMENT_REPORT, DOWNSTREAM_FINAL_REPORT, DOWNSTREAM_RECORD))
    if dev is None or final is None or record is None:
        return {"available": False, "reason": "model-selection reports not found under models/evaluation/downstream/"}
    loaded = ms.manifest["files"]["classifier.joblib"]
    if (final.get("artifact_manifest") or {}).get("files", {}).get("classifier.joblib") != loaded:
        return {"available": False, "reason": "final_holdout_report.json was produced for a different classifier file"}
    spec = MODEL_SETS[ms.name]
    summary = dev["summary"]
    families = list(summary["families"])
    table = []
    for f in families:
        row = {"family": f, "label": FAMILY_LABELS.get(f, f), "selected": f == spec.family,
               "params": (dev.get("chosen_params") or {}).get(f, {}),
               "eligible": (dev["eligibility"].get(f) or {}).get("eligible") if f != "dnn" else None}
        for m in _COMPARISON_METRICS:
            st = summary["families"][f][m]
            row[m] = {k: st.get(k) for k in ("mean", "sd", "min", "max")}
        row["accuracy"] = {k: dev["accuracy_at_policy_b"][f].get(k) for k in ("mean", "sd", "min", "max")}
        row["brier_score"] = dev["calibration_by_family"][f]["brier_score_mean"]
        row["ece"] = dev["calibration_by_family"][f]["ece_equal_width_mean"]
        cost = dev["inference_cost_seed_14"][f]
        row["single_row_ms"] = cost["single_row_ms_median"]
        row["size_bytes"] = cost["size_bytes"]
        if f in summary["versus_incumbent"]:
            row["versus_dnn"] = {m: {"difference": summary["versus_incumbent"][f][m]["difference_of_means"],
                                     "ci95": summary["versus_incumbent"][f][m]["ci95_data_and_training_seeds"]}
                                 for m in ("pr_auc", "recall", "legit_alerts_per_1000", "first_fraud_recall")}
        table.append(row)

    def holdout_block(block):
        out = {k: block[k] for k in ("rows", "fraud_transactions", "legitimate_transactions", "fraud_episodes",
                                     "first_fraud_transactions", "datasets")}
        out["models"] = {}
        for name, label in _FINAL_NAMES.items():
            m = block["models"][name]
            out["models"][name] = {"label": label,
                                   "metrics": {k: m["metrics"][k] for k in _COMPARISON_METRICS},
                                   "confusion_matrix": m["confusion_matrix_policy_b"]}
            if "calibration" in block:
                c = block["calibration"][name]
                out["models"][name]["calibration"] = {k: c[k] for k in ("brier_score", "log_loss", "ece_equal_width_10_bins")}
        out["paired_differences"] = {k: {m: v[m] for m in ("pr_auc", "recall", "legit_alerts_per_1000", "first_fraud_recall")}
                                     for k, v in block["paired_differences"].items()}
        return out

    sel_cal = final["final"]["primary"]["calibration"]["selected"]
    return {
        "available": True,
        "protocol": dev["protocol"],
        "sources": {"development": _rel(DOWNSTREAM_DEVELOPMENT_REPORT), "final_holdout": _rel(DOWNSTREAM_FINAL_REPORT),
                    "selection_record": _rel(DOWNSTREAM_RECORD),
                    "report": "docs/model_selection_report.md"},
        "development": {
            "population": dev["population"], "training_seeds": dev["training_seeds"],
            "datasets": list(dev["development_datasets"]),
            "note": "mean over training seeds 11-15 (sd, min, max across seeds); each model at its own frozen "
                    "validation cut-offs; accuracy is reported but was not used to choose",
            "comparison": table,
            "decision": record["decision"],
        },
        "final_holdout": {
            "confirmed": final["confirmed"], "outcome": final["outcome"], "gates": final["gates"],
            "cutoffs": final["cutoffs"],
            "primary": holdout_block(final["final"]["primary"]),
            "new_customer_full_history": holdout_block(final["new_customer"]["primary"]),
            "new_customer_early_history": holdout_block(final["new_customer"]["early_history"]),
            "selected_reliability": sel_cal["reliability_equal_width"],
        },
        "scores_note": "Scores are model scores, not calibrated probabilities. All data is synthetic (generator v2).",
    }
