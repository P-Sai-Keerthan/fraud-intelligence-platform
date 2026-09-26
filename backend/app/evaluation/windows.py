"""
Past-only LSTM windows.

For every transaction t of a customer that has at least SEQUENCE_LENGTH
earlier transactions, the window is the SEQUENCE_LENGTH transactions that
come immediately before t in (timestamp, transaction_id) order -- the same
input the production pipeline builds at scoring time. The target is
is_fraud of t itself, which is never part of its own window. A customer's
first SEQUENCE_LENGTH transactions have no window and are not evaluated
(none of them is fraud in the current data).
"""

import numpy as np
import pandas as pd

from ..config import SEQUENCE_LENGTH
from ..features.feature_engineering import FEATURE_COLUMNS


def build_windows(df: pd.DataFrame, seq_len: int = SEQUENCE_LENGTH):
    """df must be in canonical order (split.canonical_order).
    Returns (X [n, seq_len, n_features], y [n], target_index [n]) where
    target_index is the df row of each window's target transaction."""
    values = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    labels = df["is_fraud"].to_numpy(dtype=np.float32)
    X, y, target = [], [], []
    for _, rows in df.groupby("customer_id", sort=False).indices.items():
        rows = np.asarray(rows)
        if len(rows) <= seq_len:
            continue
        # rows are contiguous and ordered because df is in canonical order
        windows = np.lib.stride_tricks.sliding_window_view(values[rows], (seq_len, values.shape[1]))[:, 0]
        X.append(windows[:-1])          # window ending just before each target
        y.append(labels[rows[seq_len:]])
        target.append(rows[seq_len:])
    return (
        np.concatenate(X).astype(np.float32),
        np.concatenate(y).astype(np.float32),
        np.concatenate(target),
    )


def window_row_indices(target_index: np.ndarray, seq_len: int = SEQUENCE_LENGTH) -> np.ndarray:
    """[n, seq_len] df row indices that make up each window (valid because a
    customer's rows are contiguous in canonical order)."""
    return target_index[:, None] - np.arange(seq_len, 0, -1)[None, :]


def assert_past_only(df: pd.DataFrame, target_index: np.ndarray, seq_len: int = SEQUENCE_LENGTH) -> None:
    """Raises AssertionError unless every window row belongs to the same
    customer as its target and comes strictly earlier in canonical order
    (timestamp <= target timestamp, ties broken by transaction_id)."""
    rows = window_row_indices(target_index, seq_len)
    cust = df["customer_id"].to_numpy()
    ts = df["timestamp"].to_numpy()
    tid = df["transaction_id"].to_numpy()
    assert (rows >= 0).all()
    assert (cust[rows] == cust[target_index][:, None]).all(), "window crosses customers"
    t_ts = ts[target_index][:, None]
    earlier = (ts[rows] < t_ts) | ((ts[rows] == t_ts) & (tid[rows] < tid[target_index][:, None]))
    assert earlier.all(), "window contains a transaction that is not before its target"
