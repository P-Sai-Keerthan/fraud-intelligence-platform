"""Writes a generated dataset and its manifest (checksums, counts, settings)."""

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from . import schema
from .config import GENERATOR_VERSION
from .generator import GeneratedData

FILES = {
    "transactions": "transactions.csv",
    "features": "transactions_with_features.csv",
    "customers": "customers.csv",
    "episodes": "episodes.csv",
    "login_failures": "login_failures.csv",
}
MANIFEST = "manifest.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def summary(data: GeneratedData) -> dict:
    tx, ep = data.transactions, data.episodes
    return {
        "customers": int(tx["customer_id"].nunique()),
        "transactions": int(len(tx)),
        "fraud_transactions": int(tx["is_fraud"].sum()),
        "fraud_rate_pct": round(float(tx["is_fraud"].mean() * 100), 3),
        "episodes": int(len(ep)),
        "episodes_by_type": {k: int(v) for k, v in ep["fraud_type"].value_counts().sort_index().items()},
        "rings": int(ep.loc[ep["fraud_ring_id"] > 0, "fraud_ring_id"].nunique()),
        "episodes_with_precursor": int((ep["precursor_start"] != "").sum()),
        "precursor_transactions": int(tx["is_precursor"].sum()),
        "households": int(data.customers.loc[data.customers["household_id"] > 0, "household_id"].nunique()),
        "first_timestamp": str(tx["timestamp"].min()),
        "last_timestamp": str(tx["timestamp"].max()),
    }


def write_dataset(data: GeneratedData, out_dir, features: pd.DataFrame | None = None, command: str = "") -> dict:
    """Writes the CSVs (+ features file if given) and manifest.json. The
    manifest has no wall-clock timestamp, so the same seed and settings give a
    byte-identical manifest."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tables = {
        "transactions": data.transactions, "customers": data.customers,
        "episodes": data.episodes, "login_failures": data.login_failures,
    }
    if features is not None:
        tables["features"] = features
    files = {}
    for key, frame in tables.items():
        path = out / FILES[key]
        frame.to_csv(path, index=False, lineterminator="\n")
        files[FILES[key]] = {"rows": int(len(frame)), "columns": list(frame.columns), "sha256": _sha256(path)}
    manifest = {
        "generator": "data/v2/synth_v2",
        "generator_version": GENERATOR_VERSION,
        "command": command,
        "seed": data.config.seed,
        "config": _jsonable(data.config.to_dict()),
        "metadata_columns": schema.METADATA_COLUMNS,
        "note": "metadata_columns are ground truth / analysis aids and must never be model features",
        "summary": summary(data),
        "files": files,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    }
    (out / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
