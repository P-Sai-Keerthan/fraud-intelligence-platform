"""
Model sets: which trained models the inference pipeline loads.

Selected with the MODEL_SET environment variable, read once when the
pipeline is created at application startup:

    MODEL_SET unset        -> production (the default)
    MODEL_SET=production   -> models/saved/                    LSTM risk score -> DNN (unchanged)
    MODEL_SET=v2_dnn_lstm  -> models/candidates/v2/dnn_lstm/   LSTM risk score -> DNN (4C-2e-b candidate A)
    MODEL_SET=v2_dnn_only  -> models/candidates/v2/dnn_only/   DNN on the 9 features (4C-2e-b candidate B)

Any other value (including an empty string) raises ModelSetError, so a
mistyped or explicitly requested candidate never silently falls back to
production. Loading only reads files; nothing here writes anywhere, and a
candidate directory inside models/saved/ is refused.

Production loads exactly the files it always has (config.*_PATH) and scores
exactly as before. The checks added for it are read-only shape checks.

Candidates are validated against their manifest.json before use: every
listed file exists, the .npy files match their SHA-256, the trained weights
match dnn_weights_sha256 / lstm_weights_sha256, the feature list (names,
order, count and hash) equals the application's FEATURE_COLUMNS, the DNN
input columns are exactly what the model set expects, model input shapes,
scaler shapes and the SHAP background width agree, the sequence length is
the application's, and the dataset / clipping metadata is what the
candidates were trained with.
"""

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from . import config
from .features.feature_engineering import FEATURE_COLUMNS
from .features.ground_truth import assert_no_ground_truth

MODEL_SET_ENV = "MODEL_SET"
DEFAULT_MODEL_SET = "production"

RISK_SCORE = "risk_score"


@dataclass(frozen=True)
class ModelSetSpec:
    name: str
    uses_lstm: bool
    candidate: Optional[str] = None           # candidate directory name under models/candidates/<dataset>/
    dataset_version: Optional[str] = None
    description: str = ""

    @property
    def dnn_input_columns(self) -> list:
        return list(FEATURE_COLUMNS) + ([RISK_SCORE] if self.uses_lstm else [])


MODEL_SETS = {
    "production": ModelSetSpec("production", uses_lstm=True,
                               description="existing production models in models/saved/"),
    "v2_dnn_lstm": ModelSetSpec("v2_dnn_lstm", uses_lstm=True, candidate="dnn_lstm", dataset_version="v2",
                                description="v2 candidate A: LSTM risk score -> DNN"),
    "v2_dnn_only": ModelSetSpec("v2_dnn_only", uses_lstm=False, candidate="dnn_only", dataset_version="v2",
                                description="v2 candidate B: DNN on the 9 behavioral features, no LSTM"),
}


class ModelSetError(ValueError):
    """An unknown model set, or model-set files that fail validation."""


def resolve_model_set_name(value: Optional[str] = None) -> str:
    """The model set to load: `value` if given, else $MODEL_SET, else production.
    Unknown values raise ModelSetError; there is no fallback."""
    if value is None:
        value = os.environ.get(MODEL_SET_ENV)
        if value is None:
            return DEFAULT_MODEL_SET
        source = f"environment variable {MODEL_SET_ENV}"
    else:
        source = "model_set argument"
    if value not in MODEL_SETS:
        raise ModelSetError(f"invalid {source} value {value!r}; expected one of {sorted(MODEL_SETS)} "
                            f"(unset means {DEFAULT_MODEL_SET!r})")
    return value


def model_set_directory(name: str, candidates_root=None) -> Path:
    """Where a model set's files live. candidates_root replaces
    models/candidates/<dataset>/ (tests use it to point at copies)."""
    spec = MODEL_SETS[resolve_model_set_name(name)]
    if spec.candidate is None:
        return config.MODELS_SAVED_DIR
    root = Path(candidates_root) if candidates_root is not None else config.CANDIDATES_DIR / spec.dataset_version
    directory = (root / spec.candidate).resolve()
    saved = config.MODELS_SAVED_DIR.resolve()
    if directory == saved or saved in directory.parents:
        raise ModelSetError(f"{name}: candidate directory {directory} is inside the production model directory")
    return directory


@dataclass
class LoadedModelSet:
    name: str
    directory: Path
    uses_lstm: bool
    dnn_model: object
    dnn_mean: np.ndarray
    dnn_std: np.ndarray
    dnn_input_columns: list
    shap_background: np.ndarray
    sequence_length: int
    lstm_model: object = None
    lstm_mean: Optional[np.ndarray] = None
    lstm_std: Optional[np.ndarray] = None
    manifest: Optional[dict] = None
    files: dict = field(default_factory=dict)   # role -> path actually loaded

    def describe(self) -> str:
        parts = [f"model set {self.name!r} from {self.directory}",
                 "LSTM risk score -> DNN" if self.uses_lstm else "DNN only (no LSTM)",
                 f"DNN inputs {self.dnn_input_columns}"]
        return "; ".join(parts)


def load_model_set(name: Optional[str] = None, candidates_root=None) -> LoadedModelSet:
    """Resolves (argument, then $MODEL_SET, then production) and loads a model set."""
    name = resolve_model_set_name(name)
    spec = MODEL_SETS[name]
    if spec.candidate is None:
        return _load_production(spec)
    return _load_candidate(spec, model_set_directory(name, candidates_root))


# ---- production ----------------------------------------------------------------------------

def _load_production(spec: ModelSetSpec) -> LoadedModelSet:
    """The same files, loaded the same way, as the pipeline always has."""
    from tensorflow import keras
    files = {
        "lstm_model": config.LSTM_MODEL_PATH, "dnn_model": config.DNN_MODEL_PATH,
        "lstm_mean": config.LSTM_FEATURE_MEAN_PATH, "lstm_std": config.LSTM_FEATURE_STD_PATH,
        "dnn_mean": config.DNN_FEATURE_MEAN_PATH, "dnn_std": config.DNN_FEATURE_STD_PATH,
        "shap_background": config.SHAP_BACKGROUND_PATH,
    }
    ms = LoadedModelSet(
        name=spec.name, directory=config.MODELS_SAVED_DIR, uses_lstm=True,
        lstm_model=keras.models.load_model(files["lstm_model"]),
        dnn_model=keras.models.load_model(files["dnn_model"]),
        lstm_mean=np.load(files["lstm_mean"]), lstm_std=np.load(files["lstm_std"]),
        dnn_mean=np.load(files["dnn_mean"]), dnn_std=np.load(files["dnn_std"]),
        shap_background=np.load(files["shap_background"]),
        dnn_input_columns=spec.dnn_input_columns, sequence_length=config.SEQUENCE_LENGTH,
        files={k: Path(v) for k, v in files.items()},
    )
    validate_shapes(ms)
    return ms


# ---- candidates ----------------------------------------------------------------------------

def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def feature_list_sha256(columns) -> str:
    """The hash stored in candidate manifests (features.sha256)."""
    return hashlib.sha256(json.dumps(list(columns)).encode()).hexdigest()


def _fail(spec, message):
    raise ModelSetError(f"{spec.name}: {message}")


def _load_candidate(spec: ModelSetSpec, directory: Path) -> LoadedModelSet:
    from tensorflow import keras
    from .training.candidates import FILES, MANIFEST, weights_sha256

    manifest_path = directory / MANIFEST
    if not manifest_path.exists():
        _fail(spec, f"manifest {manifest_path} not found")
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(spec, manifest)

    required = ["dnn_model", "dnn_mean", "dnn_std", "shap_background"]
    if spec.uses_lstm:
        required += ["lstm_model", "lstm_mean", "lstm_std"]
    listed = manifest.get("files", {})
    for role in required:
        if FILES[role] not in listed:
            _fail(spec, f"manifest does not list the required file {FILES[role]}")
    for fname, sha in listed.items():
        path = directory / fname
        if not path.is_file():
            _fail(spec, f"missing artifact {path}")
        if not fname.endswith(".keras") and _sha256(path) != sha:
            _fail(spec, f"{fname} does not match its manifest checksum")

    files = {role: directory / FILES[role] for role in required}
    dnn = keras.models.load_model(files["dnn_model"])
    if weights_sha256(dnn) != manifest["model"]["dnn_weights_sha256"]:
        _fail(spec, "DNN weights do not match the manifest (dnn_weights_sha256)")
    ms = LoadedModelSet(
        name=spec.name, directory=directory, uses_lstm=spec.uses_lstm, dnn_model=dnn,
        dnn_mean=np.load(files["dnn_mean"]), dnn_std=np.load(files["dnn_std"]),
        shap_background=np.load(files["shap_background"]),
        dnn_input_columns=list(manifest["model"]["dnn_input_columns"]),
        sequence_length=int(manifest["sequence_length"]), manifest=manifest, files=files,
    )
    if spec.uses_lstm:
        ms.lstm_model = keras.models.load_model(files["lstm_model"])
        if weights_sha256(ms.lstm_model) != manifest["model"]["lstm_weights_sha256"]:
            _fail(spec, "LSTM weights do not match the manifest (lstm_weights_sha256)")
        ms.lstm_mean = np.load(files["lstm_mean"])
        ms.lstm_std = np.load(files["lstm_std"])
    validate_shapes(ms)
    return ms


def validate_manifest(spec: ModelSetSpec, manifest: dict) -> None:
    """Metadata checks that need no model files."""
    if manifest.get("candidate") != spec.candidate:
        _fail(spec, f"manifest is for candidate {manifest.get('candidate')!r}, expected {spec.candidate!r}")
    dataset = manifest.get("dataset") or {}
    if dataset.get("version") != spec.dataset_version:
        _fail(spec, f"manifest dataset version {dataset.get('version')!r}, expected {spec.dataset_version!r}")
    if not isinstance(dataset.get("sha256"), str) or len(dataset["sha256"]) != 64:
        _fail(spec, "manifest has no dataset sha256")

    feats = manifest.get("features") or {}
    expected = list(FEATURE_COLUMNS)
    if feats.get("columns") != expected:
        if sorted(feats.get("columns") or []) == sorted(expected):
            _fail(spec, f"feature order mismatch: manifest {feats.get('columns')} vs application {expected}")
        _fail(spec, f"feature list mismatch: manifest {feats.get('columns')} vs application {expected}")
    if feats.get("count") != len(expected):
        _fail(spec, f"feature count {feats.get('count')} != {len(expected)}")
    if feats.get("sha256") != feature_list_sha256(expected):
        _fail(spec, "feature list hash does not match the application's FEATURE_COLUMNS")

    model = manifest.get("model") or {}
    inputs = model.get("dnn_input_columns")
    if inputs != spec.dnn_input_columns:
        _fail(spec, f"DNN input columns {inputs} differ from the expected {spec.dnn_input_columns}")
    assert_no_ground_truth(inputs, f"{spec.name} DNN input columns")
    if model.get("dnn_input_shape") != [None, len(inputs)]:
        _fail(spec, f"manifest dnn_input_shape {model.get('dnn_input_shape')} != [None, {len(inputs)}]")
    if manifest.get("sequence_length") != config.SEQUENCE_LENGTH:
        _fail(spec, f"sequence length {manifest.get('sequence_length')} != {config.SEQUENCE_LENGTH}")
    if spec.uses_lstm:
        if model.get("lstm_input_shape") != [None, config.SEQUENCE_LENGTH, len(expected)]:
            _fail(spec, f"manifest lstm_input_shape {model.get('lstm_input_shape')} "
                        f"!= [None, {config.SEQUENCE_LENGTH}, {len(expected)}]")
        if not model.get("lstm_weights_sha256"):
            _fail(spec, "manifest has no lstm_weights_sha256")
    elif "lstm_input_shape" in model:
        _fail(spec, "manifest describes an LSTM, but this model set has none")
    if not model.get("dnn_weights_sha256"):
        _fail(spec, "manifest has no dnn_weights_sha256")

    clip = manifest.get("clipping") or {}
    if not (str(clip.get("dnn_training", "")).startswith("none") and "+-6" in str(clip.get("dnn_scoring", ""))
            and "never clipped" in str(clip.get("lstm", ""))):
        _fail(spec, f"unexpected clipping metadata {clip}; the pipeline clips scaled DNN inputs to +-6 "
                    "when scoring and never clips LSTM inputs")


def validate_shapes(ms: LoadedModelSet) -> None:
    """Model inputs, scalers, SHAP background and feature names must agree."""
    spec = MODEL_SETS[ms.name]
    n_in = len(ms.dnn_input_columns)
    if ms.dnn_input_columns != spec.dnn_input_columns:
        _fail(spec, f"DNN input columns {ms.dnn_input_columns} != {spec.dnn_input_columns}")
    if tuple(ms.dnn_model.input_shape[1:]) != (n_in,):
        _fail(spec, f"DNN input shape {ms.dnn_model.input_shape} does not take {n_in} inputs")
    for label, arr in (("dnn mean", ms.dnn_mean), ("dnn std", ms.dnn_std)):
        if arr.shape != (n_in,):
            _fail(spec, f"{label} has shape {arr.shape}, expected ({n_in},)")
    if not (ms.dnn_std > 0).all():
        _fail(spec, "dnn std has non-positive entries")
    if ms.shap_background.ndim != 2 or ms.shap_background.shape[1] != n_in:
        _fail(spec, f"SHAP background has shape {ms.shap_background.shape}, expected (n, {n_in})")
    if ms.sequence_length != config.SEQUENCE_LENGTH:
        _fail(spec, f"sequence length {ms.sequence_length} != {config.SEQUENCE_LENGTH}")
    if ms.uses_lstm:
        if ms.lstm_model is None:
            _fail(spec, "LSTM model missing")
        want = (config.SEQUENCE_LENGTH, len(FEATURE_COLUMNS))
        if tuple(ms.lstm_model.input_shape[1:]) != want:
            _fail(spec, f"LSTM input shape {ms.lstm_model.input_shape} does not take {want} windows")
        for label, arr in (("lstm mean", ms.lstm_mean), ("lstm std", ms.lstm_std)):
            if arr is None or arr.shape != (len(FEATURE_COLUMNS),):
                _fail(spec, f"{label} has shape {getattr(arr, 'shape', None)}, expected ({len(FEATURE_COLUMNS)},)")
    elif ms.lstm_model is not None:
        _fail(spec, "a DNN-only model set must not load an LSTM")
