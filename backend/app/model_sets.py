"""
Model sets: which trained models the inference pipeline loads.

Selected with the MODEL_SET environment variable, read once when the
pipeline is created at application startup:

    MODEL_SET unset        -> production (the default)
    MODEL_SET=production   -> models/saved/                    LSTM risk score -> DNN (unchanged)
    MODEL_SET=v2_dnn_lstm  -> models/candidates/v2/dnn_lstm/   LSTM risk score -> DNN (4C-2e-b candidate A)
    MODEL_SET=v2_dnn_only  -> models/candidates/v2/dnn_only/   DNN on the 9 features (4C-2e-b candidate B)
    MODEL_SET=v2_dnn_lstm_seed14
                           -> models/candidates_multiseed/v2/seed_14/dnn_lstm/
                              the exact artifact selected in 4C-3E.6 (Stage B) and tested on the final
                              hold-out (Stage C). Evaluation only, NOT deployed (4C-3F): it loads only
                              when asked for by name, and only if every file and both weight hashes
                              equal the values recorded in selection_record.json.

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
from types import MappingProxyType
from typing import Mapping, Optional

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
    # --- an exact, pinned artifact (4C-3F); all None for the other model sets ---
    root: Optional[tuple] = None              # directory holding <candidate>/, as parts under backend/models/
    training_seed: Optional[int] = None       # the manifest's seed must equal this
    pinned: Optional[Mapping] = field(default=None, compare=False)   # {"weights": {...}, "files": {...}} SHA-256
    architecture_of: Optional[str] = None     # the model set whose score wording applies (same architecture)
    artifact: Optional[str] = None            # its name in selection_record.json / final_holdout_report.json

    @property
    def dnn_input_columns(self) -> list:
        return list(FEATURE_COLUMNS) + ([RISK_SCORE] if self.uses_lstm else [])


# The artifact selected by the pre-registered protocol (docs/step4c3e-model-selection-protocol.md,
# models/evaluation/v2_holdout/selection_record.json -> selected_artifacts["v2_dnn_lstm@14"]).
# These values are copied from that record; tests/test_seed14_model_set.py checks they still equal it.
SEED14_MODEL_SET = "v2_dnn_lstm_seed14"
SEED14_ARTIFACT = "v2_dnn_lstm@14"
SEED14_PINNED = MappingProxyType({
    "weights": MappingProxyType({
        "dnn_weights_sha256": "32fc53979e06db730742423133e54bee2305f89a439cbd634ae03b76f8c94f0f",
        "lstm_weights_sha256": "e4851c4c20e864b14f1f54a7051b0ad5dad5d0e4920270737ded6bad6db1a077",
    }),
    "files": MappingProxyType({
        "dnn_fraud_model.keras": "c033fa8d4ed9888b135a308959ab7596c5db8824665121876cde0f2a28655351",
        "dnn_feature_mean.npy": "2071d1baa335af2c06d26ec63cc6623e0e46e6626ee349f79face3bc0f01c681",
        "dnn_feature_std.npy": "b472068a64760b13ab4f959b81309cf5df7abd1c56afbb8372d878aa515eadd8",
        "shap_background.npy": "ef63280f701e04b5a25dfc47e41457782c1e3152474104560d6c71b3d9761a59",
        "lstm_risk_model.keras": "5a327ccba85f86e90a927cf5c0a4cd2f1cc6d040f8ded2e47734705b0c1a7600",
        "lstm_feature_mean.npy": "fbfc1a0afd79ad236b27932198a6a6a6c1d6ac4b2e61b335db1269b9be91e90c",
        "lstm_feature_std.npy": "e3e7765d84d6b6d2f3ebc665c0798ef36d015cbf604acc53f00d8cd8aaff5a35",
    }),
})
# Cut-offs frozen at Stage B (DNN output, 0-1) and used unchanged in Stage C. They are recorded
# here for provenance only: /predict does NOT apply them (it keeps the fixed 25/50/80 bands).
SEED14_FROZEN_CUTOFFS = MappingProxyType({"policy_b": 0.7928694486618042, "critical": 0.9430845379829407})

MODEL_SETS = {
    "production": ModelSetSpec("production", uses_lstm=True,
                               description="existing production models in models/saved/"),
    "v2_dnn_lstm": ModelSetSpec("v2_dnn_lstm", uses_lstm=True, candidate="dnn_lstm", dataset_version="v2",
                                description="v2 candidate A: LSTM risk score -> DNN"),
    "v2_dnn_only": ModelSetSpec("v2_dnn_only", uses_lstm=False, candidate="dnn_only", dataset_version="v2",
                                description="v2 candidate B: DNN on the 9 behavioral features, no LSTM"),
    SEED14_MODEL_SET: ModelSetSpec(SEED14_MODEL_SET, uses_lstm=True, candidate="dnn_lstm", dataset_version="v2",
                                   root=("candidates_multiseed", "v2", "seed_14"), training_seed=14,
                                   pinned=SEED14_PINNED, architecture_of="v2_dnn_lstm", artifact=SEED14_ARTIFACT,
                                   description="the selected v2_dnn_lstm artifact, training seed 14 "
                                               "(4C-3E.6); evaluation only, not deployed"),
}


def architecture_name(name: str) -> str:
    """The model set whose architecture (and score wording) `name` shares; itself unless pinned."""
    spec = MODEL_SETS.get(name)
    return spec.architecture_of if spec is not None and spec.architecture_of else name


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
    if candidates_root is not None:
        root = Path(candidates_root)
    elif spec.root is not None:
        root = config.BACKEND_DIR.joinpath("models", *spec.root)
    else:
        root = config.CANDIDATES_DIR / spec.dataset_version
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
    validate_pinned(spec, manifest, directory)

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


def validate_pinned(spec: ModelSetSpec, manifest: dict, directory: Path) -> None:
    """For a pinned model set: the directory must hold exactly the selected artifact.
    Training seed, both weight hashes and the SHA-256 of every file (the .keras
    archives included) must equal the pinned values; nothing else is accepted."""
    if spec.training_seed is not None and manifest.get("seed") != spec.training_seed:
        _fail(spec, f"manifest training seed {manifest.get('seed')!r}, expected {spec.training_seed}")
    if spec.pinned is None:
        return
    model = manifest.get("model") or {}
    for key, sha in spec.pinned["weights"].items():
        if model.get(key) != sha:
            _fail(spec, f"manifest {key} is not the selected artifact's ({spec.artifact})")
    listed = manifest.get("files") or {}
    if set(listed) != set(spec.pinned["files"]):
        _fail(spec, f"manifest lists files {sorted(listed)}, the selected artifact has {sorted(spec.pinned['files'])}")
    for fname, sha in spec.pinned["files"].items():
        path = directory / fname
        if not path.is_file():
            _fail(spec, f"missing artifact {path}")
        if _sha256(path) != sha:
            _fail(spec, f"{fname} is not the selected artifact's file ({spec.artifact}): SHA-256 differs")


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
