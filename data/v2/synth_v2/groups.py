"""
Grouping keys that later evaluation steps need so ground truth never gets
split across train / validation / test incorrectly. (Step 4C-2a only
provides them; the evaluation does not use them yet.)

* Episode groups: every episode of one fraud ring forms ONE group, because
  the ring's members are attacked in a single coordinated wave with shared
  infrastructure. A non-ring episode is its own group.
* Customer components: customers linked by a household (shared device and
  home network) or by a fraud ring. A customer-grouped split must keep each
  component in one split.
"""

import pandas as pd


def episode_groups(episodes: pd.DataFrame) -> pd.Series:
    """fraud_episode_id -> group key ("ring:3" or "episode:17")."""
    keys = [f"ring:{r}" if r else f"episode:{e}"
            for e, r in zip(episodes["fraud_episode_id"], episodes["fraud_ring_id"])]
    return pd.Series(keys, index=episodes["fraud_episode_id"].to_numpy(), name="group")


def group_spans(episodes: pd.DataFrame) -> pd.DataFrame:
    """Time span of every episode group, precursor period included."""
    ep = episodes.copy()
    ep["group"] = episode_groups(ep).to_numpy()
    first = pd.to_datetime(ep["first_fraud_time"])
    pre = pd.to_datetime(ep["precursor_start"].replace("", None))
    ep["span_start"] = pre.fillna(first)
    ep["span_end"] = pd.to_datetime(ep["last_fraud_time"])
    return ep.groupby("group").agg(
        span_start=("span_start", "min"), span_end=("span_end", "max"),
        customers=("customer_id", lambda s: sorted(set(s))), episodes=("fraud_episode_id", list),
    )


def customer_components(customers: pd.DataFrame, episodes: pd.DataFrame) -> pd.Series:
    """customer_id -> component id (union of household and ring links)."""
    parent = {c: c for c in customers["customer_id"]}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(members):
        members = list(members)
        for m in members[1:]:
            a, b = find(members[0]), find(m)
            if a != b:
                parent[max(a, b)] = min(a, b)

    for _, g in customers[customers["household_id"] > 0].groupby("household_id"):
        union(g["customer_id"])
    for _, g in episodes[episodes["fraud_ring_id"] > 0].groupby("fraud_ring_id"):
        union(g["customer_id"])
    return pd.Series({c: find(c) for c in customers["customer_id"]}, name="component")
