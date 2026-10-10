"""
Ring-fraud detection, reported at three levels (Step 4C-3E).

A fraud ring is a set of customers attacked in one coordinated wave. The
table passed in has one row per scored transaction with the ground-truth
columns `fraud_ring_id`, `fraud_episode_id` and `fraud_stage`, a `seed`
column (ring ids restart in every generated dataset, so a ring is identified
by (seed, fraud_ring_id)) and one boolean alert column per model set.

* Ring level:        rings with at least one / at least two victims alerted,
                     and the time from the ring's first fraud to its first alert.
* Victim level:      victim episodes with any fraud transaction alerted, and
                     victims whose FIRST fraud transaction was alerted.
* Transaction level: ring fraud transactions caught, next to non-ring fraud.

Counts carry Wilson 95% intervals. `paired_ring_bootstrap` resamples whole
rings to put an interval on the difference between two model sets; with a
few dozen rings that interval is wide, and no ring result may be called
better unless it excludes zero.

The ground truth is used only here, to slice results. It is never a model input.
"""

from itertools import combinations

import numpy as np
import pandas as pd

from .analysis import wilson


def _rate(k: int, n: int) -> dict:
    return {"k": int(k), "n": int(n), "rate": round(k / n, 4) if n else None, "wilson95": wilson(int(k), int(n))}


def per_ring(table: pd.DataFrame, models, alert_prefix: str = "alert_") -> list:
    """One entry per (seed, ring) with the ring's fraud rows in `table`."""
    fraud = table[(table["is_fraud"] == 1) & (table["fraud_ring_id"] > 0)]
    out = []
    for (seed, ring), g in fraud.groupby(["seed", "fraud_ring_id"], sort=True):
        g = g.sort_values(["timestamp", "transaction_id"], kind="mergesort")
        first = g["fraud_stage"] == "first"
        entry = {"seed": int(seed), "ring": int(ring), "victims": int(g["fraud_episode_id"].nunique()),
                 "fraud_transactions": int(len(g)), "first_fraud_transactions": int(first.sum())}
        for m in models:
            a = g[f"{alert_prefix}{m}"].to_numpy(dtype=bool)
            hours = None
            if a.any():
                hours = round((g["timestamp"][a].iloc[0] - g["timestamp"].iloc[0]).total_seconds() / 3600, 2)
            entry[m] = {"fraud_caught": int(a.sum()),
                        "victims_alerted": int(g["fraud_episode_id"][a].nunique()),
                        "first_frauds_caught": int((a & first.to_numpy()).sum()),
                        "hours_from_first_fraud_to_first_alert": hours}
        out.append(entry)
    return out


def summarise(rings: list, table: pd.DataFrame, models, alert_prefix: str = "alert_") -> dict:
    fraud = table[table["is_fraud"] == 1]
    non_ring = fraud[fraud["fraud_ring_id"] == 0]
    out = {}
    for m in models:
        hours = [r[m]["hours_from_first_fraud_to_first_alert"] for r in rings
                 if r[m]["hours_from_first_fraud_to_first_alert"] is not None]
        out[m] = {
            "ring_level": {
                "rings_with_at_least_one_victim_alerted": _rate(sum(r[m]["victims_alerted"] >= 1 for r in rings), len(rings)),
                "rings_with_at_least_two_victims_alerted": _rate(sum(r[m]["victims_alerted"] >= 2 for r in rings), len(rings)),
                "median_hours_from_first_fraud_to_first_alert": round(float(np.median(hours)), 2) if hours else None,
            },
            "victim_level": {
                "victim_episodes_detected": _rate(sum(r[m]["victims_alerted"] for r in rings), sum(r["victims"] for r in rings)),
                "victim_first_frauds_detected": _rate(sum(r[m]["first_frauds_caught"] for r in rings),
                                                      sum(r["first_fraud_transactions"] for r in rings)),
            },
            "transaction_level": {
                "ring_fraud_caught": _rate(sum(r[m]["fraud_caught"] for r in rings), sum(r["fraud_transactions"] for r in rings)),
                "non_ring_fraud_caught": _rate(int(non_ring[f"{alert_prefix}{m}"].sum()), len(non_ring)),
            },
        }
    return out


def paired_ring_bootstrap(rings: list, models, reps: int = 2000, seed: int = 42, pairs=None) -> dict:
    """Ring-cluster bootstrap of the difference (a - b) in victim-episode detection
    rate and in ring fraud-transaction recall, for every pair of model sets
    (`pairs`: the (a, b) pairs to report; default: every combination)."""
    if not rings:
        return {}
    victims = np.array([r["victims"] for r in rings], dtype=float)
    txns = np.array([r["fraud_transactions"] for r in rings], dtype=float)
    det = {m: np.array([r[m]["victims_alerted"] for r in rings], dtype=float) for m in models}
    caught = {m: np.array([r[m]["fraud_caught"] for r in rings], dtype=float) for m in models}
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(rings), size=(reps, len(rings)))
    out = {}
    for a, b in (pairs if pairs is not None else combinations(models, 2)):
        entry = {}
        for name, num, den in (("victim_episode_detection_rate", det, victims), ("ring_fraud_transaction_recall", caught, txns)):
            d = (num[a][picks].sum(axis=1) - num[b][picks].sum(axis=1)) / den[picks].sum(axis=1)
            lo, hi = np.percentile(d, [2.5, 97.5])
            entry[name] = {"difference": round(float((num[a].sum() - num[b].sum()) / den.sum()), 4),
                           "ci95": [round(float(lo), 4), round(float(hi), 4)],
                           "excludes_zero": bool(lo > 0 or hi < 0)}
        out[f"{a} - {b}"] = entry
    return out


def shared_device_rule(frame: pd.DataFrame, metadata: pd.DataFrame, episodes: pd.DataFrame) -> dict:
    """The live shared-device rule (FraudIntelligencePipeline.detect_fraud_rings,
    unchanged) on one dataset's full history. It does not depend on the model set."""
    import threading
    from ..inference_pipeline import FraudIntelligencePipeline

    class _Histories:                        # the only state detect_fraud_rings reads
        def __init__(self, f):
            self._lock = threading.RLock()
            self.customer_histories = {c: g[["device_id"]] for c, g in f.groupby("customer_id")}

    flagged = FraudIntelligencePipeline.detect_fraud_rings(_Histories(frame))
    fraud = frame["is_fraud"].to_numpy() == 1
    fraud_devices = set(frame.loc[fraud, "device_id"])
    ring_devices = set(frame.loc[fraud & (metadata["fraud_ring_id"].to_numpy() > 0), "device_id"])
    members = episodes[episodes["fraud_ring_id"] > 0].groupby("fraud_ring_id")["customer_id"].apply(set)
    kinds = {"ring": 0, "other_fraud": 0, "legitimate_only": 0}
    for r in flagged:
        d = r["identifier"]
        kinds["ring" if d in ring_devices else ("other_fraud" if d in fraud_devices else "legitimate_only")] += 1
    found = sum(any(r["identifier"] in ring_devices and set(r["customer_ids"]) & m for r in flagged) for m in members)
    return {"devices_flagged": len(flagged), "by_kind": kinds, "rings": int(len(members)),
            "rings_with_a_flagged_ring_device": int(found)}


def ring_report(table: pd.DataFrame, models, alert_prefix: str = "alert_", reps: int = 2000, seed: int = 42,
                pairs=None) -> dict:
    rings = per_ring(table, models, alert_prefix)
    return {"rings": len(rings), "victim_episodes": int(sum(r["victims"] for r in rings)),
            "ring_fraud_transactions": int(sum(r["fraud_transactions"] for r in rings)),
            "summary": summarise(rings, table, models, alert_prefix),
            "paired_differences": paired_ring_bootstrap(rings, models, reps, seed, pairs),
            "per_ring": rings}
