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
}
FRAUD_PROBABILITY_SEMANTICS = ("DNN output x 100, capped at 99.9. A fraud SCORE from class-weighted training, "
                               "not a calibrated probability.")


def model_version(ms) -> str:
    """A short, content-derived identifier of the loaded weights."""
    if ms.manifest is None:
        digest = hashlib.sha256("".join(_sha256(ms.files[k]) for k in sorted(ms.files)).encode()).hexdigest()
        return f"{ms.name}-{digest[:12]}"
    return f"{ms.name}-{ms.manifest['model']['dnn_weights_sha256'][:12]}"


def model_metadata(ms) -> dict:
    live_features = current_feature_implementation()
    meta = {
        "model_set": ms.name,
        "model_version": model_version(ms),
        "directory": _rel(ms.directory),
        "uses_lstm": bool(ms.uses_lstm),
        "dnn_input_columns": list(ms.dnn_input_columns),
        "score_semantics": {
            "fraud_probability": FRAUD_PROBABILITY_SEMANTICS,
            "risk_score": RISK_SCORE_SEMANTICS[ms.name],
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
    return meta


CANDIDATE_KEYS = {"v2_dnn_lstm": "candidate_a_dnn_lstm", "v2_dnn_only": "candidate_b_dnn_only"}


def candidate_evaluation(ms) -> dict:
    """The loaded candidate's slice of comparison.json, or why it is unavailable.
    Only returned when comparison.json describes exactly the loaded weights."""
    path = ms.directory.parent / "comparison.json"
    key = CANDIDATE_KEYS.get(ms.name)
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
