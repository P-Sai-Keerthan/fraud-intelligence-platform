"""
Shared, reproducible evaluation splits.

Primary: time-based (out-of-time) split
----------------------------------------
* Fraud episodes: a customer's fraud transactions, linked into one episode
  while consecutive fraud transactions are less than EPISODE_GAP (14 days)
  apart; legitimate transactions may be interleaved. In the current data the
  largest gap between a customer's successive fraud transactions is ~10 days
  (a ramp-up burst and the main event), so each of the 60 fraud customers has
  exactly one episode. A 7-day rule would split two of those events in two.
* Boundaries are data-driven: episodes are ordered by start time and the
  targets sit between the episodes at the SPLIT_FRACTIONS points (60% / 80% of
  episode starts). Each boundary is then moved to the midnight, within
  BOUNDARY_SEARCH of the target, that the fewest episodes straddle (ties: the
  midnight closest to the target, then the earlier one).
* Every transaction is assigned by timestamp: < b1 train, < b2 validation,
  otherwise test. An episode that still straddles a boundary is moved, whole,
  to the LATER split (together with any of that customer's legitimate
  transactions inside the episode's time span), so no episode crosses a
  boundary and the training split never contains a transaction at or after b1.

Secondary: customer-grouped split
---------------------------------
Customers are shuffled with a fixed seed, separately for customers with and
without fraud (stratified), and divided 60 / 20 / 20. A customer's whole
history, and therefore every episode, stays in one split.

Transaction order everywhere is (customer_id, timestamp, transaction_id);
transaction_id breaks ties between same-customer transactions that share a
timestamp.
"""

import hashlib
import json

import numpy as np
import pandas as pd

EPISODE_GAP = pd.Timedelta(days=14)
SPLIT_FRACTIONS = (0.6, 0.8)          # cumulative share of fraud episodes (by start time)
BOUNDARY_SEARCH = pd.Timedelta(days=14)
CUSTOMER_SPLIT_FRACTIONS = (0.6, 0.2, 0.2)
SPLIT_SEED = 42
SPLITS = ("train", "validation", "test")


def canonical_order(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by (customer, timestamp, transaction_id) and reset the index."""
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    return out.sort_values(["customer_id", "timestamp", "transaction_id"], kind="mergesort").reset_index(drop=True)


def find_episodes(df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Returns (episode_id per row, 0 for legitimate rows; episode table).
    `df` must be in canonical order."""
    episode_id = np.zeros(len(df), dtype=int)
    fraud = df[df["is_fraud"] == 1]
    next_id = 0
    for _, rows in fraud.groupby("customer_id", sort=True):
        prev_ts = None
        for idx, ts in zip(rows.index, rows["timestamp"]):
            if prev_ts is None or ts - prev_ts >= EPISODE_GAP:
                next_id += 1
            episode_id[idx] = next_id
            prev_ts = ts
    ep_series = pd.Series(episode_id, index=df.index, name="episode_id")
    fr = df.assign(episode_id=ep_series)[ep_series > 0]
    table = fr.groupby("episode_id").agg(
        customer_id=("customer_id", "first"),
        start=("timestamp", "min"),
        end=("timestamp", "max"),
        n_fraud=("is_fraud", "size"),
        first_transaction_id=("transaction_id", "first"),
    )
    return ep_series, table


def _straddling(episodes: pd.DataFrame, boundary: pd.Timestamp) -> int:
    return int(((episodes["start"] < boundary) & (episodes["end"] >= boundary)).sum())


def _best_midnight(episodes: pd.DataFrame, target: pd.Timestamp, after=None) -> pd.Timestamp:
    lo = (target - BOUNDARY_SEARCH).normalize()
    hi = (target + BOUNDARY_SEARCH).normalize()
    candidates = [d for d in pd.date_range(lo, hi, freq="D") if after is None or d > after]
    return min(candidates, key=lambda d: (_straddling(episodes, d), abs(d - target), d))


def choose_time_boundaries(episodes: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    starts = episodes["start"].sort_values().reset_index(drop=True)
    targets = []
    for frac in SPLIT_FRACTIONS:
        k = int(round(frac * len(starts)))
        targets.append(starts[k - 1] + (starts[k] - starts[k - 1]) / 2)
    b1 = _best_midnight(episodes, targets[0])
    b2 = _best_midnight(episodes, targets[1], after=b1)
    return b1, b2


def time_split(df: pd.DataFrame, episode_id: pd.Series, episodes: pd.DataFrame, b1, b2) -> tuple[pd.Series, list]:
    """Per-row split labels for the time-based split (see module docstring)."""
    ts = df["timestamp"]
    labels = pd.Series(np.where(ts < b1, "train", np.where(ts < b2, "validation", "test")), index=df.index)
    rank = {name: i for i, name in enumerate(SPLITS)}

    def split_of(t):
        return "train" if t < b1 else ("validation" if t < b2 else "test")

    moved = []
    for ep_id, ep in episodes.iterrows():
        target = split_of(ep["end"])
        if split_of(ep["start"]) == target:
            continue
        span = (df["customer_id"] == ep["customer_id"]) & (ts >= ep["start"]) & (ts <= ep["end"])
        # only ever move rows to a LATER split
        span &= labels.map(rank) < rank[target]
        labels[span] = target
        moved.append({
            "episode_id": int(ep_id), "customer_id": ep["customer_id"],
            "start": str(ep["start"]), "end": str(ep["end"]),
            "moved_to": target, "rows_moved": int(span.sum()),
        })
    return labels, moved


def customer_split(df: pd.DataFrame, seed: int = SPLIT_SEED) -> pd.Series:
    """Per-row labels for the customer-grouped split, stratified by whether the customer has fraud."""
    rng = np.random.RandomState(seed)
    has_fraud = df.groupby("customer_id")["is_fraud"].max()
    assignment = {}
    for flag in (1, 0):
        customers = np.array(sorted(has_fraud[has_fraud == flag].index))
        rng.shuffle(customers)
        n = len(customers)
        n_train = int(round(CUSTOMER_SPLIT_FRACTIONS[0] * n))
        n_val = int(round(CUSTOMER_SPLIT_FRACTIONS[1] * n))
        for i, c in enumerate(customers):
            assignment[c] = "train" if i < n_train else ("validation" if i < n_train + n_val else "test")
    return df["customer_id"].map(assignment)


def _ids_hash(ids) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def summarize(df: pd.DataFrame, labels: pd.Series, episode_id: pd.Series) -> dict:
    out = {}
    for name in SPLITS:
        part = df[labels == name]
        eps = episode_id[labels == name]
        out[name] = {
            "transactions": int(len(part)),
            "fraud_transactions": int(part["is_fraud"].sum()),
            "fraud_rate_pct": round(float(part["is_fraud"].mean() * 100), 3) if len(part) else 0.0,
            "fraud_episodes": int(eps[eps > 0].nunique()),
            "customers": int(part["customer_id"].nunique()),
            "first_timestamp": str(part["timestamp"].min()),
            "last_timestamp": str(part["timestamp"].max()),
            "transaction_ids_sha256": _ids_hash(part["transaction_id"]),
        }
    return out


def build_time_split(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, dict]:
    """df in canonical order -> (labels, episode_id, split definition dict)."""
    episode_id, episodes = find_episodes(df)
    b1, b2 = choose_time_boundaries(episodes)
    labels, moved = time_split(df, episode_id, episodes, b1, b2)
    definition = {
        "method": "time-based (out-of-time)",
        "order": "customer_id, timestamp, transaction_id",
        "episode_rule": f"a customer's fraud transactions linked while less than {EPISODE_GAP.days} days apart",
        "boundary_rule": (
            f"targets between the fraud episodes at {int(SPLIT_FRACTIONS[0]*100)}% and {int(SPLIT_FRACTIONS[1]*100)}% "
            f"of episode start times; moved to the midnight within +/-{BOUNDARY_SEARCH.days} days that the fewest "
            "episodes straddle; straddling episodes move whole to the later split"
        ),
        "train_before": str(b1),
        "validation_before": str(b2),
        "test_from": str(b2),
        "episodes_total": int(len(episodes)),
        "episodes_moved": moved,
        "splits": summarize(df, labels, episode_id),
    }
    return labels, episode_id, definition


def build_customer_split(df: pd.DataFrame, episode_id: pd.Series) -> tuple[pd.Series, dict]:
    labels = customer_split(df)
    definition = {
        "method": "customer-grouped",
        "rule": f"customers shuffled with seed {SPLIT_SEED}, stratified by has-fraud, split "
                f"{'/'.join(str(int(f*100)) for f in CUSTOMER_SPLIT_FRACTIONS)}; all of a customer's transactions stay together",
        "splits": summarize(df, labels, episode_id),
    }
    return labels, definition


def save_definition(path, definition: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(definition, indent=2) + "\n")
