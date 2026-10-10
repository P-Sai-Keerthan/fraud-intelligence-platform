"""
Builds the v2 features file with the UNCHANGED production feature code.

Ground-truth metadata never passes through feature engineering: only the v1
raw columns are handed to build_point_features, and the metadata is joined
back afterwards (by transaction_id) at the END of the table.
"""

import sys
from pathlib import Path

import pandas as pd

from . import schema

BACKEND_DIR = Path(__file__).resolve().parents[3] / "backend"


def _feature_engineering():
    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))
    from app.features import feature_engineering
    return feature_engineering


def build_features_table(transactions: pd.DataFrame) -> pd.DataFrame:
    fe = _feature_engineering()
    assert list(fe.FEATURE_COLUMNS) == schema.V1_FEATURE_COLUMNS, "production feature list changed"
    raw = transactions[schema.V1_RAW_COLUMNS].copy()
    feat = fe.build_point_features(raw)
    feat["timestamp"] = pd.to_datetime(feat["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    feat = feat[schema.V1_FEATURES_CSV_COLUMNS]
    meta = transactions[["transaction_id"] + schema.METADATA_COLUMNS]
    out = feat.merge(meta, on="transaction_id", how="left", validate="one_to_one")
    assert len(out) == len(transactions)
    return out[schema.FEATURES_CSV_COLUMNS]
