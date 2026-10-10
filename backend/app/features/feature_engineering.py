"""
Behavioral Fraud DNA - Feature Engineering
============================================
Turns a raw transaction log into two things:

1. POINT FEATURES  (one row per transaction)
   Used by the DNN for "is THIS transaction fraudulent" classification.

2. SEQUENCE FEATURES  (one rolling window of past N transactions per customer)
   Used by the LSTM, which predicts from this history whether the NEXT
   transaction is fraudulent (the historical-risk signal).

Both share the same underlying per-transaction feature vector -- the LSTM
just consumes it as a sequence, the DNN consumes the latest single vector.
"""

import numpy as np
import pandas as pd

from ..config import (
    RAW_TRANSACTIONS_CSV, FEATURES_CSV, LSTM_X_PATH, LSTM_Y_PATH,
    LSTM_META_CSV, SEQUENCE_LENGTH as CONFIG_SEQ_LEN,
)
from .ground_truth import assert_no_ground_truth

FEATURE_COLUMNS = [
    "amount_zscore",          # how far this amount is from the customer's normal (in std devs)
    "hour_is_unusual",        # 1 if this hour is outside the customer's normal active hours
    "is_new_device",          # 1 if device never used before by this customer
    "is_new_location",        # 1 if location never used before by this customer
    "is_foreign_location",    # 1 if location is outside the customer's home country city list
    "failed_logins_24h",      # failed login attempts in the trailing 24h
    "category_is_unusual",    # 1 if merchant category is outside customer's top categories
    "txn_velocity_1h",        # number of transactions by this customer in the last hour
    "amount_pct_of_avg",      # amount as a percentage of the customer's average (captures scale)
]

# The label and the v2 ground-truth metadata columns must never be features.
assert_no_ground_truth(FEATURE_COLUMNS, "FEATURE_COLUMNS")

HOME_LOCATIONS = {
    "Hyderabad", "Mumbai", "Delhi", "Bangalore", "Chennai",
    "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Lucknow",
}

SEQUENCE_LENGTH = CONFIG_SEQ_LEN  # how many past transactions the LSTM looks at (from config)


def build_point_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds behavioral feature columns to each transaction row, computed using
    ONLY prior history for that customer (no lookahead / no leakage).
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["customer_id", "timestamp"]).reset_index(drop=True)

    out_rows = []
    for customer_id, group in df.groupby("customer_id", sort=False):
        group = group.reset_index(drop=True)
        seen_categories = {}
        amounts_hist = []
        hours_hist = []
        timestamps_hist = []

        for i, row in group.iterrows():
            amounts_arr = np.array(amounts_hist) if amounts_hist else np.array([row["amount"]])
            avg_amount = amounts_arr.mean()
            std_amount = amounts_arr.std() if len(amounts_arr) > 1 else max(avg_amount * 0.3, 1.0)
            # floor std as a fraction of the average, not just a near-zero epsilon --
            # customers with unusually consistent spending (tiny raw std) would
            # otherwise produce absurd z-scores (seen up to +/-1000+ in practice)
            # for even a moderately large transaction, saturating the DNN's
            # sigmoid and making its output look like a broken flat 100%.
            std_amount = max(std_amount, avg_amount * 0.1, 1.0)

            amount_zscore = (row["amount"] - avg_amount) / std_amount
            amount_zscore = float(np.clip(amount_zscore, -10.0, 10.0))
            amount_pct_of_avg = row["amount"] / max(avg_amount, 1e-6) * 100
            amount_pct_of_avg = float(np.clip(amount_pct_of_avg, 0.0, 1000.0))

            hour = row["timestamp"].hour
            if hours_hist:
                hour_counts = pd.Series(hours_hist).value_counts(normalize=True)
                hour_is_unusual = int(hour_counts.get(hour, 0) < 0.05)
            else:
                hour_is_unusual = 0

            cat_counts = seen_categories
            total_cat_seen = sum(cat_counts.values())
            cat_freq = cat_counts.get(row["merchant_category"], 0) / total_cat_seen if total_cat_seen > 0 else 0
            category_is_unusual = int(cat_freq < 0.05)

            is_foreign_location = int(row["location"] not in HOME_LOCATIONS)

            # transaction velocity: count of past transactions within 1h of this one
            if timestamps_hist:
                ts_arr = pd.Series(timestamps_hist)
                velocity = int(((row["timestamp"] - ts_arr).dt.total_seconds().abs() <= 3600).sum())
            else:
                velocity = 0

            out_rows.append({
                "amount_zscore": amount_zscore,
                "hour_is_unusual": hour_is_unusual,
                "is_new_device": row["is_new_device"],
                "is_new_location": row["is_new_location"],
                "is_foreign_location": is_foreign_location,
                "failed_logins_24h": row["failed_logins_24h"],
                "category_is_unusual": category_is_unusual,
                "txn_velocity_1h": velocity,
                "amount_pct_of_avg": amount_pct_of_avg,
            })

            # update rolling history AFTER computing features (no leakage)
            amounts_hist.append(row["amount"])
            hours_hist.append(hour)
            timestamps_hist.append(row["timestamp"])
            seen_categories[row["merchant_category"]] = seen_categories.get(row["merchant_category"], 0) + 1

    feat_df = pd.DataFrame(out_rows)
    # drop raw columns that we've recomputed as behavioral features, to avoid
    # duplicate column names when concatenating (is_new_device/is_new_location/
    # failed_logins_24h already exist on the raw df from the data generator)
    raw_cols_to_drop = [c for c in feat_df.columns if c in df.columns]
    base_df = df.reset_index(drop=True).drop(columns=raw_cols_to_drop)
    result = pd.concat([base_df, feat_df], axis=1)
    return result


def build_sequences(feat_df: pd.DataFrame, seq_len: int = SEQUENCE_LENGTH):
    """
    Builds sliding-window sequences of FEATURE_COLUMNS per customer, for LSTM
    input. Returns X (n_samples, seq_len, n_features), y (n_samples,) where y
    is whether the transaction AFTER the window is fraudulent: the LSTM's
    training target ("does this history predict that the next transaction
    is fraud?"). How well it does is measured in docs/EVALUATION.md; it does
    not flag the first fraud transaction of an episode.
    Also returns the index of the "current" (last) row for each sequence, so
    predictions can be mapped back to transaction_id / customer_id.
    """
    X, y, meta = [], [], []
    for customer_id, group in feat_df.groupby("customer_id", sort=False):
        group = group.reset_index(drop=True)
        values = group[FEATURE_COLUMNS].values
        fraud_labels = group["is_fraud"].values
        txn_ids = group["transaction_id"].values

        if len(group) <= seq_len:
            continue

        for i in range(seq_len, len(group)):
            window = values[i - seq_len:i]
            X.append(window)
            y.append(fraud_labels[i])
            meta.append({"customer_id": customer_id, "transaction_id": txn_ids[i]})

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    meta_df = pd.DataFrame(meta)
    return X, y, meta_df


def get_latest_sequence_for_customer(feat_df: pd.DataFrame, customer_id: str, seq_len: int = SEQUENCE_LENGTH):
    """Used at inference time: grab the most recent seq_len transactions'
    features for a customer, to feed the LSTM for a live risk score."""
    group = feat_df[feat_df["customer_id"] == customer_id].sort_values("timestamp")
    if len(group) < seq_len:
        # pad with the earliest available row if history is short (cold start)
        pad_needed = seq_len - len(group)
        if len(group) == 0:
            return None
        pad_rows = pd.concat([group.iloc[[0]]] * pad_needed, ignore_index=True)
        group = pd.concat([pad_rows, group], ignore_index=True)
    window = group[FEATURE_COLUMNS].values[-seq_len:]
    return np.array(window, dtype=np.float32)


if __name__ == "__main__":
    # Run this with:  cd backend && python -m app.features.feature_engineering
    df = pd.read_csv(RAW_TRANSACTIONS_CSV)
    print("Building point features (this loops per-customer, may take ~30-60s)...")
    feat_df = build_point_features(df)
    feat_df.to_csv(FEATURES_CSV, index=False)
    print(f"Saved {len(feat_df):,} rows with features to {FEATURES_CSV}")
    print(feat_df[FEATURE_COLUMNS + ["is_fraud"]].describe())

    X, y, meta = build_sequences(feat_df)
    print(f"\nBuilt {len(X):,} sequences of shape {X.shape[1:]}")
    print(f"Sequence-level fraud rate: {y.mean()*100:.2f}%")
    np.save(LSTM_X_PATH, X)
    np.save(LSTM_Y_PATH, y)
    meta.to_csv(LSTM_META_CSV, index=False)
