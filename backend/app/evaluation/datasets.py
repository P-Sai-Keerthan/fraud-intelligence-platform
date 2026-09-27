"""
Dataset versions for the evaluation.

    resolve_dataset()        -> v1 (the default: the production dataset)
    resolve_dataset("v2")    -> the generated dataset in data/v2
    resolve_dataset("v3")    -> ValueError

Every path the evaluation reads or writes comes from the DatasetSpec, so
the rest of the evaluation code has no version checks. There is no fallback:
asking for v2 when data/v2 has not been generated raises FileNotFoundError.

load_evaluation_data(spec) separates two things:

* frame    -- the MODEL frame: identifiers, the is_fraud target and exactly
              the production FEATURE_COLUMNS. Nothing else, so v2 metadata
              cannot reach a feature matrix.
* metadata -- v2's ground-truth / analysis columns (fraud type, episode,
              stage, ring, warning period, context, segment, merchant,
              network), row-aligned with the frame, for analysis only.
              None for v1.

For v2 it also builds the grouping (explicit episode ids, ring units, and
customer components from rings + households) with data/v2/synth_v2/groups.py.
v1 has no grouping; its evaluation keeps the 14-day episode heuristic.

Production inference and GET /metrics are not affected: /predict loads
config.FEATURES_CSV and /metrics reads config.EVAL_REPORT_PATH, which are
the v1 paths, and v2 output goes to config.EVALUATION_V2_DIR.
"""

import hashlib
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from ..features.feature_engineering import FEATURE_COLUMNS
from ..features.ground_truth import GROUND_TRUTH_COLUMNS, assert_no_ground_truth

DEFAULT_DATASET_VERSION = "v1"

# identifiers + target + production features: the only columns a model frame may have
ID_COLUMNS = ["customer_id", "transaction_id", "timestamp", "amount", "merchant_category", "device_id", "location"]
TARGET_COLUMN = "is_fraud"
MODEL_FRAME_COLUMNS = ID_COLUMNS + [TARGET_COLUMN] + list(FEATURE_COLUMNS)

assert_no_ground_truth(FEATURE_COLUMNS, "evaluation features")


@dataclass(frozen=True)
class DatasetSpec:
    version: str
    description: str
    data_dir: Path
    features_csv: Path
    episodes_csv: Path | None          # v2 only
    customers_csv: Path | None         # v2 only
    manifest_path: Path | None         # v2 only
    output_dir: Path                   # where this version's evaluation output goes
    report_path: Path
    time_split_path: Path
    customer_split_path: Path
    has_metadata: bool
    legacy_comparison: bool            # compare with the production models' old random-split metrics
    regenerate_hint: str = ""

    @property
    def models_dir(self) -> Path:
        return self.output_dir / "time_split"

    def with_data_dir(self, data_dir) -> "DatasetSpec":
        """Same version, data read from another directory with the same file names
        (e.g. a small generated sample in tests)."""
        d = Path(data_dir)
        moved = {k: d / getattr(self, k).name for k in ("features_csv", "episodes_csv", "customers_csv", "manifest_path")
                 if getattr(self, k) is not None}
        return DatasetSpec(**{**self.__dict__, "data_dir": d, **moved})

    def with_output_dir(self, output_dir) -> "DatasetSpec":
        """Same dataset, evaluation output written somewhere else (e.g. a regression run)."""
        out = Path(output_dir).resolve()
        return DatasetSpec(**{**self.__dict__, "output_dir": out, "report_path": out / self.report_path.name,
                              "time_split_path": out / self.time_split_path.name,
                              "customer_split_path": out / self.customer_split_path.name})


DATASETS = {
    "v1": DatasetSpec(
        version="v1",
        description="original synthetic dataset (production data used by /predict)",
        data_dir=config.DATA_DIR,
        features_csv=config.FEATURES_CSV,
        episodes_csv=None, customers_csv=None, manifest_path=None,
        output_dir=config.EVALUATION_DIR,
        report_path=config.EVAL_REPORT_PATH,
        time_split_path=config.EVAL_TIME_SPLIT_PATH,
        customer_split_path=config.EVAL_CUSTOMER_SPLIT_PATH,
        has_metadata=False,
        legacy_comparison=True,
    ),
    "v2": DatasetSpec(
        version="v2",
        description="generated v2 synthetic dataset (data/v2), evaluation only",
        data_dir=config.DATA_V2_DIR,
        features_csv=config.DATA_V2_DIR / "transactions_with_features.csv",
        episodes_csv=config.DATA_V2_DIR / "episodes.csv",
        customers_csv=config.DATA_V2_DIR / "customers.csv",
        manifest_path=config.DATA_V2_DIR / "manifest.json",
        output_dir=config.EVALUATION_V2_DIR,
        report_path=config.EVALUATION_V2_DIR / "evaluation_report.json",
        time_split_path=config.EVALUATION_V2_DIR / "split_time.json",
        customer_split_path=config.EVALUATION_V2_DIR / "split_customer.json",
        has_metadata=True,
        legacy_comparison=False,        # the production models were never trained on v2
        regenerate_hint="generate it with `python data/v2/generate.py` from the project root",
    ),
}


def resolve_dataset(version: str | None = None) -> DatasetSpec:
    """None -> the default (v1). Unknown versions raise ValueError; there is no fallback."""
    if version is None:
        version = DEFAULT_DATASET_VERSION
    if version not in DATASETS:
        raise ValueError(f"unknown dataset version {version!r}; expected one of {sorted(DATASETS)}")
    return DATASETS[version]


# ---- loading -------------------------------------------------------------------------

@dataclass
class Grouping:
    """v2 only. Row-aligned episode ids and the units that must not be split."""
    episode_id: pd.Series              # per frame row, explicit fraud_episode_id (0 = legitimate)
    episodes: pd.DataFrame             # data/v2 episodes.csv
    units: pd.DataFrame                # one row per episode group (a ring = one group): span + members
    component_of: pd.Series            # customer_id -> component (households + rings)


@dataclass
class EvaluationData:
    spec: DatasetSpec
    frame: pd.DataFrame                # MODEL_FRAME_COLUMNS, canonical order
    metadata: pd.DataFrame | None      # transaction_id + GROUND_TRUTH_COLUMNS, same row order (v2)
    grouping: Grouping | None          # v2
    sha256: str
    manifest: dict | None


def _require(path: Path, spec: DatasetSpec) -> Path:
    if not path.exists():
        hint = f"; {spec.regenerate_hint}" if spec.regenerate_hint else ""
        raise FileNotFoundError(f"dataset {spec.version}: {path} not found{hint}")
    return path


def load_evaluation_data(spec: DatasetSpec) -> EvaluationData:
    from .split import canonical_order
    raw = _require(spec.features_csv, spec).read_bytes()
    df = canonical_order(pd.read_csv(spec.features_csv, keep_default_na=False) if spec.has_metadata
                         else pd.read_csv(spec.features_csv))
    expected = MODEL_FRAME_COLUMNS + (list(GROUND_TRUTH_COLUMNS) if spec.has_metadata else [])
    if sorted(df.columns) != sorted(expected):
        missing, extra = sorted(set(expected) - set(df.columns)), sorted(set(df.columns) - set(expected))
        raise ValueError(f"dataset {spec.version}: unexpected columns in {spec.features_csv.name} "
                         f"(missing {missing}, unexpected {extra})")

    frame = df[MODEL_FRAME_COLUMNS].copy()
    metadata = df[["transaction_id", *GROUND_TRUTH_COLUMNS]].copy() if spec.has_metadata else None
    grouping, manifest = None, None
    if spec.has_metadata:
        manifest = json.loads(_require(spec.manifest_path, spec).read_text())
        episodes = pd.read_csv(_require(spec.episodes_csv, spec), keep_default_na=False)
        customers = pd.read_csv(_require(spec.customers_csv, spec), keep_default_na=False)
        grouping = build_grouping(frame, metadata, episodes, customers)
    return EvaluationData(spec, frame, metadata, grouping, hashlib.sha256(raw).hexdigest(), manifest)


# ---- v2 grouping (data/v2/synth_v2/groups.py) --------------------------------------------

def _v2_groups_module():
    """Loads data/v2/synth_v2/groups.py by path (pure pandas; does not import the generator)."""
    path = config.DATA_V2_DIR / "synth_v2" / "groups.py"
    module_spec = importlib.util.spec_from_file_location("_synth_v2_groups", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def build_grouping(frame: pd.DataFrame, metadata: pd.DataFrame, episodes: pd.DataFrame,
                   customers: pd.DataFrame) -> Grouping:
    groups = _v2_groups_module()
    if not (metadata["transaction_id"].to_numpy() == frame["transaction_id"].to_numpy()).all():
        raise ValueError("metadata is not row-aligned with the model frame")
    episode_id = pd.Series(metadata["fraud_episode_id"].astype(int).to_numpy(), index=frame.index, name="episode_id")
    fraud = frame["is_fraud"].to_numpy() == 1
    if not np.array_equal(fraud, episode_id.to_numpy() > 0):
        raise ValueError("fraud_episode_id does not match is_fraud")

    spans = groups.group_spans(episodes)                 # per group: span_start/end (warning period included)
    # per-member spans, so moving a unit moves each member's own rows only
    ep = episodes.copy()
    ep["group"] = groups.episode_groups(ep).to_numpy()
    first = pd.to_datetime(ep["first_fraud_time"])
    pre = pd.to_datetime(ep["precursor_start"].replace("", None))
    ep["member_start"] = pre.fillna(first)
    ep["member_end"] = pd.to_datetime(ep["last_fraud_time"])
    members = ep.groupby("group").apply(
        lambda g: [(c, s, e) for c, s, e in zip(g["customer_id"], g["member_start"], g["member_end"])],
        include_groups=False)
    units = spans.assign(members=members).rename(columns={"span_start": "start", "span_end": "end"})
    units["n_fraud"] = ep.groupby("group")["n_fraud_transactions"].sum()
    units["is_ring"] = [g.startswith("ring:") for g in units.index]
    units = units.sort_values(["start", "end"], kind="mergesort")
    component_of = groups.customer_components(customers, episodes)
    return Grouping(episode_id, episodes, units, component_of)


def _display_path(path: Path) -> str:
    """Relative to the project root when inside it, so reports hold no machine-specific paths."""
    try:
        return path.relative_to(config.PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def dataset_info(data: EvaluationData) -> dict:
    """What the report records about the dataset."""
    spec = data.spec
    info = {
        "version": spec.version,
        "description": spec.description,
        "file": _display_path(spec.features_csv),
        "sha256": data.sha256,
        "transactions": int(len(data.frame)),
    }
    if data.manifest is not None:
        info["generator_version"] = data.manifest.get("generator_version")
        info["generator_seed"] = data.manifest.get("seed")
        info["manifest_file_sha256"] = data.manifest.get("files", {}).get(spec.features_csv.name, {}).get("sha256")
        info["matches_manifest"] = info["manifest_file_sha256"] == data.sha256
    return info
