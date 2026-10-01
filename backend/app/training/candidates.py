"""
Train v2 candidate models with the corrected evaluation recipe.

    cd backend
    python -m app.training.candidates                        # v2, both candidates -> models/candidates/v2/
    python -m app.training.candidates --output-root /tmp/c   # same training, written elsewhere (e.g. a rerun)

Candidates
----------
* dnn_lstm: LSTM (out-of-fold risk scores for the DNN's training rows) + DNN
            on the 9 features + risk_score. Identical to the evaluation's
            "dnn_fraud_classifier" (and its "lstm_risk_predictor").
* dnn_only: DNN on the 9 features. Identical to the evaluation's
            "dnn_without_risk_score" baseline.

Recipe (no second implementation): the models are fitted by
app.evaluation.stacking (fit_lstm, fit_dnn, stacked_risk_scores) with the
same arguments app.evaluation.run.evaluate_split uses, seed 42 and
TensorFlow op determinism. Data: the saved v2 time split
(models/evaluation/v2/split_time.json), recomputed and checked against the
saved file (definition and per-split transaction-id hashes); training rows
fit the models, validation rows choose the thresholds, test rows are not
used at all. No refit on train + validation.

Clipping, exactly as evaluated in 4C-2d: DNN inputs are scaled with the
training mean/std and trained UNCLIPPED; when scoring, the scaled DNN
inputs are clipped to +-6 (DNNScorer.predict; production inference does
the same). LSTM inputs are scaled, not clipped.

Safety: the output root may not be models/saved/ or anything inside it,
and the SHA-256 of every production file is compared before and after;
any change raises RuntimeError.
"""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

from .. import config
from ..evaluation.datasets import load_evaluation_data, resolve_dataset
from ..evaluation.metrics import select_threshold, threshold_for_fpr
from ..evaluation.split import build_time_split
from ..evaluation.stacking import (CLIP, EVAL_SEED, N_FOLDS, _class_weight, fit_dnn, fit_lstm, predict_fixed_batch,
                                   stacked_risk_scores)
from ..evaluation.windows import assert_past_only, build_windows
from ..features.feature_engineering import FEATURE_COLUMNS, SEQUENCE_LENGTH
from ..features.ground_truth import assert_no_ground_truth

CANDIDATES = ("dnn_lstm", "dnn_only")
REQUIRED_DATASET = "v2"
SHAP_BACKGROUND_ROWS = 200
FPR_OPERATING_POINTS = (0.001, 0.01, 0.05)
MANIFEST = "manifest.json"
FILES = {
    "lstm_model": "lstm_risk_model.keras", "lstm_mean": "lstm_feature_mean.npy", "lstm_std": "lstm_feature_std.npy",
    "dnn_model": "dnn_fraud_model.keras", "dnn_mean": "dnn_feature_mean.npy", "dnn_std": "dnn_feature_std.npy",
    "shap_background": "shap_background.npy",
}


# ---- safety ------------------------------------------------------------------------------

def assert_safe_output(path) -> Path:
    """Refuses models/saved/ (production) and anything inside it."""
    p = Path(path).resolve()
    saved = config.MODELS_SAVED_DIR.resolve()
    if p == saved or saved in p.parents:
        raise ValueError(f"refusing to write candidates into the production model directory {saved}")
    return p


def production_checksums() -> dict:
    """SHA-256 of every file in models/saved/ (the production models)."""
    root = config.MODELS_SAVED_DIR
    return {f.relative_to(root).as_posix(): _sha256(f) for f in sorted(root.rglob("*")) if f.is_file()}


def verify_production_unchanged(before: dict, after: dict) -> None:
    if before != after:
        changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
        raise RuntimeError(f"production model files changed during candidate training: {changed}")


def candidate_dir(name: str, dataset_version: str = REQUIRED_DATASET, output_root=None) -> Path:
    if name not in CANDIDATES:
        raise ValueError(f"unknown candidate {name!r}; expected one of {CANDIDATES}")
    root = Path(output_root) if output_root is not None else config.CANDIDATES_DIR / dataset_version
    return assert_safe_output(root) / name


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weights_sha256(model) -> str:
    """Checksum of the numerical weights (deterministic, unlike the .keras
    archive, which stores the save time in its metadata)."""
    h = hashlib.sha256()
    for w in model.get_weights():
        h.update(np.ascontiguousarray(w).tobytes())
    return h.hexdigest()


# ---- data --------------------------------------------------------------------------------

def load_saved_split(spec, data):
    """Recomputes the time split and requires it to equal the saved definition."""
    path = spec.time_split_path
    if not path.exists():
        raise FileNotFoundError(f"saved split {path} not found; run the {spec.version} evaluation first "
                                f"(python -m app.evaluation.run --dataset {spec.version})")
    saved = json.loads(path.read_text())
    labels, episode_id, definition = build_time_split(data.frame, data.grouping)
    if json.loads(json.dumps(definition)) != saved:
        raise ValueError(f"the recomputed split does not match the saved split {path.name}; refusing to train")
    return labels, episode_id, saved, _sha256(path)


# ---- training ------------------------------------------------------------------------------

def _thresholds(y_val, p_val) -> dict:
    return {
        "f1_optimal": float(select_threshold(y_val, p_val)),
        "source": "validation split only (F1-maximizing; metrics.select_threshold)",
        "fpr_operating_points": {f"{t:g}": float(threshold_for_fpr(y_val, p_val, t)) for t in FPR_OPERATING_POINTS},
        "fpr_operating_points_source": "validation split only (lowest threshold with validation FPR <= target)",
    }


def _layers(model) -> list:
    out = []
    for layer in model.layers:
        cfg = layer.get_config()
        entry = {"type": layer.__class__.__name__}
        for key in ("units", "rate", "activation", "mask_value", "return_sequences"):
            if key in cfg:
                entry[key] = cfg[key]
        out.append(entry)
    return out


def train_candidates(dataset_version: str = REQUIRED_DATASET, names=CANDIDATES, output_root=None, spec=None,
                     fit_lstm_fn=fit_lstm, fit_dnn_fn=fit_dnn, n_folds: int = N_FOLDS, seed: int = EVAL_SEED,
                     log=print) -> dict:
    """Trains and saves the candidates; returns {"manifests": ..., "predictions": ...}.
    `spec` (a DatasetSpec) lets tests point at a small generated dataset."""
    spec = spec or resolve_dataset(dataset_version)
    if spec.version != REQUIRED_DATASET:
        raise ValueError(f"candidates are trained on dataset {REQUIRED_DATASET} only, got {spec.version!r}")
    dirs = {n: candidate_dir(n, spec.version, output_root) for n in names}      # safety check before any work
    data = load_evaluation_data(spec)
    labels, _, split_def, split_sha = load_saved_split(spec, data)
    df = data.frame
    assert_no_ground_truth([c for c in df.columns if c != "is_fraud"], "columns of the training frame")

    X, y, target = build_windows(df)
    assert_past_only(df, target)
    split = np.asarray(labels)[target]
    tr, va = split == "train", split == "validation"
    ts = df["timestamp"].to_numpy()[target]
    groups = df["customer_id"].to_numpy()[target]
    F = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)[target]
    log(f"windows: {len(y)} (train {int(tr.sum())}, validation {int(va.sum())}; test rows are not used)")

    common = {
        "dataset": {"version": spec.version, "file": spec.features_csv.name, "sha256": data.sha256,
                    "generator_version": (data.manifest or {}).get("generator_version")},
        "split": {"file": spec.time_split_path.name, "sha256": split_sha, "method": split_def["method"],
                  "train_before": split_def["train_before"], "validation_before": split_def["validation_before"],
                  "rows_used": {"train": int(tr.sum()), "validation": int(va.sum())},
                  "fraud_used": {"train": int(y[tr].sum()), "validation": int(y[va].sum())},
                  "test_rows_used": 0},
        "features": {"columns": list(FEATURE_COLUMNS), "count": len(FEATURE_COLUMNS),
                     "sha256": hashlib.sha256(json.dumps(list(FEATURE_COLUMNS)).encode()).hexdigest()},
        "seed": seed,
        "sequence_length": SEQUENCE_LENGTH,
        "clipping": {"dnn_training": "none (scaled only)",
                     "dnn_scoring": f"scaled inputs clipped to +-{CLIP:g} (as in 4C-2d and production inference)",
                     "lstm": "scaled only, never clipped"},
        "recipe": "app/evaluation/stacking.py fit_lstm / fit_dnn / stacked_risk_scores, called as in "
                  "app/evaluation/run.py evaluate_split (4C-2d)",
        "versions": _versions(),
    }
    out = {"manifests": {}, "predictions": {}}

    lstm_p = final_lstm = prov = None
    if "dnn_lstm" in names:
        log(f"dnn_lstm: LSTM, {n_folds} out-of-fold models + final model on the training rows")
        lstm_p, final_lstm, prov = stacked_risk_scores(X, y, ts, groups, tr, fit_fn=fit_lstm_fn, n_folds=n_folds, seed=seed)
        risk = np.round(lstm_p * 100, 2)
        log("dnn_lstm: DNN on the 9 features + out-of-fold risk_score")
        XD = np.column_stack([F, risk]).astype(np.float32)
        dnn = fit_dnn_fn(XD[tr], y[tr], ts[tr], seed=seed)
        p = dnn.predict(XD)
        out["predictions"]["dnn_lstm"] = {"dnn": p, "lstm": lstm_p, "split": split}
        out["manifests"]["dnn_lstm"] = _save(
            "dnn_lstm", dirs["dnn_lstm"], common, dnn, XD[tr], y, tr, va, p, seed,
            input_columns=list(FEATURE_COLUMNS) + ["risk_score"], lstm=final_lstm, lstm_p=lstm_p, prov=prov,
            n_folds=n_folds, log=log)
    if "dnn_only" in names:
        log("dnn_only: DNN on the 9 features")
        dnn = fit_dnn_fn(F[tr], y[tr], ts[tr], seed=seed)
        p = dnn.predict(F)
        out["predictions"]["dnn_only"] = {"dnn": p, "split": split}
        out["manifests"]["dnn_only"] = _save(
            "dnn_only", dirs["dnn_only"], common, dnn, F[tr], y, tr, va, p, seed,
            input_columns=list(FEATURE_COLUMNS), log=log)
    out["transaction_ids"] = df["transaction_id"].to_numpy()[target]
    out["y"] = y
    return out


def _save(name, directory, common, dnn, X_train, y, tr, va, p, seed, input_columns, lstm=None, lstm_p=None,
          prov=None, n_folds=None, log=print) -> dict:
    assert_no_ground_truth(input_columns, f"{name} input columns")
    directory.mkdir(parents=True, exist_ok=True)
    files = {}

    def save_npy(key, arr):
        np.save(directory / FILES[key], np.asarray(arr))
        files[FILES[key]] = _sha256(directory / FILES[key])

    # SHAP background: 200 training rows, scaled and clipped exactly like scoring inputs, fixed seed
    Xn = np.clip((X_train - dnn.mean) / dnn.std, -CLIP, CLIP).astype(np.float32)
    pick = np.sort(np.random.default_rng(seed).choice(len(Xn), size=min(SHAP_BACKGROUND_ROWS, len(Xn)), replace=False))
    dnn.model.save(directory / FILES["dnn_model"])
    files[FILES["dnn_model"]] = _sha256(directory / FILES["dnn_model"])
    save_npy("dnn_mean", dnn.mean)
    save_npy("dnn_std", dnn.std)
    save_npy("shap_background", Xn[pick])
    manifest = {
        "candidate": name,
        **common,
        "model": {
            "type": "LSTM risk score -> DNN" if lstm is not None else "DNN only",
            "architecture": "production architectures: app/models/lstm_model.build_lstm_model, "
                            "app/models/dnn_model.build_dnn_model (unchanged)",
            "dnn_input_columns": input_columns,
            "dnn_input_shape": [None, len(input_columns)],
            "dnn_layers": _layers(dnn.model),
            "dnn_parameters": int(dnn.model.count_params()),
            "dnn_weights_sha256": weights_sha256(dnn.model),
            "dnn_training": {**getattr(dnn, "info", {}), "class_weight": _class_weight(y[tr]),
                             "epochs_max": 40, "batch_size": 128,
                             "early_stopping": "val_auc (val_loss if no fraud), patience 5, best weights restored, "
                                               "on the chronologically last 15% of the training rows"},
        },
        "scalers": {"dnn_mean": FILES["dnn_mean"], "dnn_std": FILES["dnn_std"]},
        "thresholds": _thresholds(y[va], p[va]),
        "threshold_score_scale": "DNN output (0-1); /predict shows it x100 as fraud_probability",
        "shap_background": {"file": FILES["shap_background"], "rows": int(len(pick)), "seed": seed,
                            "source": "training rows, scaled and clipped to +-6 like scoring inputs"},
        "files": files,
        "file_checksum_note": ".keras archives embed their save time (metadata.json) and Keras's auto-generated "
                              "layer names (config.json), so their file SHA-256 differs between otherwise identical "
                              "runs; dnn_weights_sha256 / lstm_weights_sha256 are the reproducible identity of the "
                              "trained weights. The .npy files are byte-identical between runs.",
    }
    if lstm is not None:
        lstm.model.save(directory / FILES["lstm_model"])
        files[FILES["lstm_model"]] = _sha256(directory / FILES["lstm_model"])
        save_npy("lstm_mean", lstm.mean)
        save_npy("lstm_std", lstm.std)
        manifest["scalers"].update({"lstm_mean": FILES["lstm_mean"], "lstm_std": FILES["lstm_std"]})
        manifest["model"].update({
            "lstm_input_shape": [None, SEQUENCE_LENGTH, len(FEATURE_COLUMNS)],
            "lstm_layers": _layers(lstm.model),
            "lstm_parameters": int(lstm.model.count_params()),
            "lstm_weights_sha256": weights_sha256(lstm.model),
            "lstm_training": {**getattr(lstm, "info", {}), "class_weight": _class_weight(y[tr]),
                              "epochs_max": 30, "batch_size": 64},
            "risk_score": "LSTM probability x 100, rounded to 2 decimals; DNN training rows get out-of-fold "
                          f"scores from {n_folds} customer-grouped folds",
            "out_of_fold_models": prov["fold_info"] if prov else None,
        })
        manifest["lstm_thresholds"] = _thresholds(y[va], lstm_p[va])
    manifest["scalers"]["sha256"] = {manifest["scalers"][k]: files[manifest["scalers"][k]]
                                     for k in manifest["scalers"] if k != "sha256"}
    (directory / MANIFEST).write_text(json.dumps(manifest, indent=2, default=_json_default) + "\n")
    log(f"{name}: saved to {directory.name}/ (threshold {manifest['thresholds']['f1_optimal']:.6f})")
    return manifest


def _json_default(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    raise TypeError(type(x))


def _versions() -> dict:
    import sklearn
    import tensorflow as tf
    import pandas as pd
    return {"python": platform.python_version(), "tensorflow": tf.__version__, "scikit_learn": sklearn.__version__,
            "numpy": np.__version__, "pandas": pd.__version__}


# ---- loading ---------------------------------------------------------------------------------

class Candidate:
    """A saved candidate: manifest + models + scalers. score() applies the
    same scaling / clipping as training-time scoring."""

    def __init__(self, directory):
        from tensorflow import keras
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / MANIFEST).read_text())
        for fname, sha in self.manifest["files"].items():
            if not (self.directory / fname).exists():
                raise FileNotFoundError(f"{self.directory.name}: missing {fname}")
            if not fname.endswith(".keras") and _sha256(self.directory / fname) != sha:
                raise ValueError(f"{self.directory.name}: {fname} does not match its manifest checksum")
        if self.manifest["features"]["columns"] != list(FEATURE_COLUMNS):
            raise ValueError("candidate was trained on a different feature list")
        self.dnn = keras.models.load_model(self.directory / FILES["dnn_model"])
        self.dnn_mean = np.load(self.directory / FILES["dnn_mean"])
        self.dnn_std = np.load(self.directory / FILES["dnn_std"])
        self.threshold = float(self.manifest["thresholds"]["f1_optimal"])
        self.input_columns = self.manifest["model"]["dnn_input_columns"]
        if self.dnn.input_shape[-1] != len(self.input_columns):
            raise ValueError("DNN input width does not match the manifest")
        if weights_sha256(self.dnn) != self.manifest["model"]["dnn_weights_sha256"]:
            raise ValueError("DNN weights do not match the manifest")
        self.lstm = None
        if "risk_score" in self.input_columns:
            self.lstm = keras.models.load_model(self.directory / FILES["lstm_model"])
            self.lstm_mean = np.load(self.directory / FILES["lstm_mean"])
            self.lstm_std = np.load(self.directory / FILES["lstm_std"])
            if weights_sha256(self.lstm) != self.manifest["model"]["lstm_weights_sha256"]:
                raise ValueError("LSTM weights do not match the manifest")

    def risk_probability(self, windows: np.ndarray) -> np.ndarray:
        return self.lstm.predict((windows - self.lstm_mean) / self.lstm_std, batch_size=2048, verbose=0).ravel()

    def risk_score(self, windows: np.ndarray) -> np.ndarray:
        """risk_score as the DNN was trained on it: probability x 100 in float64, rounded to 2
        decimals (production: round(float(prob) * 100, 2))."""
        return np.round(self.risk_probability(windows).astype(np.float64) * 100, 2)

    def score(self, features: np.ndarray, risk_score: np.ndarray | None = None) -> np.ndarray:
        """features: [n, 9] production features; risk_score (x100 scale) for dnn_lstm."""
        X = features if risk_score is None else np.column_stack([features, risk_score])
        Xn = np.clip((X.astype(np.float32) - self.dnn_mean) / self.dnn_std, -CLIP, CLIP)
        # same fixed-shape batching as training-time scoring (DNNScorer), so a
        # loaded candidate reproduces the in-memory model's scores bit for bit
        # for any subset of rows
        return predict_fixed_batch(self.dnn, Xn)


def load_candidate(name: str, dataset_version: str = REQUIRED_DATASET, output_root=None) -> Candidate:
    return Candidate(candidate_dir(name, dataset_version, output_root))


# ---- reproduction check against the 4C-2d evaluation --------------------------------------

def compare_with_evaluation(result: dict, spec) -> dict:
    """Compares the candidates with the saved v2 evaluation run (models/evaluation/v2)."""
    import pandas as pd
    from tensorflow import keras
    from ..evaluation.run import SCORES_FILES
    out = {}
    ev = spec.output_dir
    report = json.loads(spec.report_path.read_text())
    scores = pd.read_csv(ev / SCORES_FILES["time"], float_precision="round_trip")
    if not (scores["transaction_id"].to_numpy() == result["transaction_ids"]).all():
        raise ValueError("evaluation scores are not in the same row order as the candidate windows")
    pairs = {"dnn_lstm": [("dnn", "dnn_fraud_classifier"), ("lstm", "lstm_risk_predictor")],
             "dnn_only": [("dnn", "dnn_without_risk_score")]}
    for name, m in result["manifests"].items():
        entry = {}
        for key, col in pairs[name]:
            ours = result["predictions"][name][key]
            theirs = scores[col].to_numpy()
            entry[f"{key}_predictions_identical"] = bool(np.array_equal(ours, theirs))
            entry[f"{key}_max_abs_prediction_difference"] = float(np.max(np.abs(ours - theirs)))
        block = (report["primary"]["dnn_fraud_classifier"] if name == "dnn_lstm"
                 else report["primary"]["baselines"]["dnn_without_risk_score"])
        entry["threshold_candidate"] = round(m["thresholds"]["f1_optimal"], 6)
        entry["threshold_evaluation_report"] = block["threshold"]
        entry["threshold_identical"] = entry["threshold_candidate"] == block["threshold"]
        if name == "dnn_lstm":
            d = spec.models_dir
            dnn_eval = keras.models.load_model(d / FILES["dnn_model"])
            lstm_eval = keras.models.load_model(d / FILES["lstm_model"])
            entry["dnn_weights_identical"] = weights_sha256(dnn_eval) == m["model"]["dnn_weights_sha256"]
            entry["lstm_weights_identical"] = weights_sha256(lstm_eval) == m["model"]["lstm_weights_sha256"]
            for k in ("dnn_mean", "dnn_std", "lstm_mean", "lstm_std"):
                entry[f"{k}_identical"] = _sha256(d / FILES[k]) == m["files"][FILES[k]]
        else:
            entry["note"] = "the evaluation did not save its DNN-only model; predictions and threshold are compared"
        out[name] = entry
    return out


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=REQUIRED_DATASET)
    p.add_argument("--output-root", default=None, help="default: models/candidates/<dataset>/")
    p.add_argument("--candidates", nargs="+", default=list(CANDIDATES), choices=CANDIDATES)
    args = p.parse_args(argv)

    before = production_checksums()
    started = time.time()
    spec = resolve_dataset(args.dataset)
    result = train_candidates(args.dataset, names=tuple(args.candidates), output_root=args.output_root, spec=spec)
    duration = round(time.time() - started, 1)
    after = production_checksums()
    verify_production_unchanged(before, after)             # raises if anything in models/saved changed
    reproduction = compare_with_evaluation(result, spec) if spec.report_path.exists() else None
    root = Path(args.output_root) if args.output_root else config.CANDIDATES_DIR / spec.version
    run_report = {
        "command": "python -m app.training.candidates " + " ".join(argv if argv is not None else []),
        "candidates": sorted(result["manifests"]),
        "training_seconds": duration,
        "production_models_unchanged": True,
        "production_checksums": before,
        "reproduction_of_4c2d_evaluation": reproduction,
    }
    (root / "training_run.json").write_text(json.dumps(run_report, indent=2) + "\n")
    print(json.dumps({"training_seconds": duration, "reproduction": reproduction}, indent=2))
    return run_report


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
