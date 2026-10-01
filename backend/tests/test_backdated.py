"""Back-dated / out-of-order transactions (Step 4C-2f-1).

A transaction must get the score computed for THAT transaction at ITS
timestamp: its features from the customer's strictly earlier transactions,
its LSTM window from the 10 transactions before it, and its similarity from
the transactions before it -- wherever it falls in the customer's history.

The reference for "the score of that transaction" is the score it gets when
it is the newest transaction of a history that contains exactly the
customer's earlier transactions (the case the pipeline always handled)."""

import random

import numpy as np
import pandas as pd
import pytest

from app import inference_pipeline
from app.features.feature_engineering import FEATURE_COLUMNS, build_point_features
from app.inference_pipeline import RAW_COLUMNS_FOR_FEATURES

from conftest import NORMAL_TXN, SUSPICIOUS_TXN

SCORE_KEYS = ("risk_score", "fraud_probability", "alert_level", "similarity_pct", "deviation_pct", "amount",
              "device_id", "location", "reasons")


def _seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)


def _reference(pipeline, txn):
    """Score txn against only the customer's transactions strictly before it,
    then restore the pipeline's histories exactly."""
    snapshot = dict(pipeline.customer_histories)
    cid = txn["customer_id"]
    hist = pipeline.customer_histories.get(cid)
    ts = pd.Timestamp(txn["timestamp"])
    if hist is not None and len(hist):
        earlier = hist[pd.to_datetime(hist["timestamp"]) < ts][RAW_COLUMNS_FOR_FEATURES]
        pipeline.customer_histories[cid] = build_point_features(earlier) if len(earlier) else earlier.iloc[:0]
    _seed()
    try:
        return pipeline.score_transaction(dict(txn))
    finally:
        pipeline.customer_histories.clear()
        pipeline.customer_histories.update(snapshot)


def _score(pipeline, txn):
    _seed()
    return pipeline.score_transaction(dict(txn))


def _same_score(a, b):
    for k in SCORE_KEYS:
        assert a[k] == b[k], (k, a[k], b[k])


def _history(pipeline, cid):
    return pipeline.customer_histories[cid]


def _assert_chronological_and_unique(hist):
    ts = pd.to_datetime(hist["timestamp"])
    assert ts.is_monotonic_increasing
    assert hist["transaction_id"].is_unique


def _assert_history_matches_replay(pipeline, cid, scored_rows):
    """The live history equals a from-scratch chronological replay of the seed
    plus the scored transactions (what a restart rebuilds from the database)."""
    live = _history(pipeline, cid).reset_index(drop=True)
    snapshot = dict(pipeline.customer_histories)
    try:
        pipeline.restore_scored_history(scored_rows)
        replay = _history(pipeline, cid).reset_index(drop=True)
    finally:
        pipeline.customer_histories.clear()
        pipeline.customer_histories.update(snapshot)
    assert list(live["transaction_id"]) == list(replay["transaction_id"])
    for col in FEATURE_COLUMNS:
        np.testing.assert_allclose(live[col].to_numpy(float), replay[col].to_numpy(float), err_msg=col)


def _row(out):
    return {k: out[k] for k in ("transaction_id", "customer_id", "timestamp", "amount", "merchant_category",
                                "device_id", "location", "failed_logins_24h")}


# 1 --------------------------------------------------------------------------------------------------

def test_newest_transaction_unchanged(pipeline):
    before = len(_history(pipeline, "CUST_0001"))
    txn = dict(NORMAL_TXN)
    ref = _reference(pipeline, txn)
    out = _score(pipeline, txn)
    _same_score(out, ref)
    hist = _history(pipeline, "CUST_0001")
    assert len(hist) == before + 1
    assert hist["transaction_id"].iloc[-1] == out["transaction_id"]
    _assert_chronological_and_unique(hist)


# 2 --------------------------------------------------------------------------------------------------

def test_transaction_inserted_between_two_existing(pipeline):
    hist = _history(pipeline, "CUST_0001")
    ts = pd.to_datetime(hist["timestamp"])
    i = len(hist) - 20
    between = ts.iloc[i] + (ts.iloc[i + 1] - ts.iloc[i]) / 2
    assert ts.iloc[i] < between < ts.iloc[i + 1]
    txn = {**SUSPICIOUS_TXN, "timestamp": between.isoformat()}
    ref = _reference(pipeline, txn)
    out = _score(pipeline, txn)
    _same_score(out, ref)
    new = _history(pipeline, "CUST_0001")
    pos = int(np.flatnonzero(new["transaction_id"].to_numpy() == out["transaction_id"])[0])
    assert pos == i + 1                                  # exactly between the two
    assert len(new) == len(hist) + 1
    _assert_chronological_and_unique(new)
    _assert_history_matches_replay(pipeline, "CUST_0001", [_row(out)])


# 3 --------------------------------------------------------------------------------------------------

def test_transaction_earlier_than_latest(pipeline):
    """The 4C-2e-e reproduction: a 03:00 suspicious transaction scored after a
    13:30 normal one on the same day must be scored as itself."""
    alone = _reference(pipeline, SUSPICIOUS_TXN)
    normal = _score(pipeline, NORMAL_TXN)                      # 13:30
    susp = _score(pipeline, SUSPICIOUS_TXN)                    # 03:00, earlier than the 13:30 row
    _same_score(susp, alone)
    assert susp["alert_level"] == "Critical Risk"
    assert (susp["fraud_probability"], susp["risk_score"]) != (normal["fraud_probability"], normal["risk_score"])
    hist = _history(pipeline, "CUST_0001")
    assert list(hist["transaction_id"].iloc[-2:]) == [susp["transaction_id"], normal["transaction_id"]]
    _assert_history_matches_replay(pipeline, "CUST_0001", [_row(normal), _row(susp)])


def test_backdated_via_api_returns_its_own_score(client, pipeline):
    snapshot = dict(pipeline.customer_histories)
    ref = client.post("/predict", json=dict(SUSPICIOUS_TXN)).json()
    pipeline.customer_histories.clear()
    pipeline.customer_histories.update(snapshot)
    later = client.post("/predict", json=dict(NORMAL_TXN)).json()
    back = client.post("/predict", json=dict(SUSPICIOUS_TXN)).json()
    assert back["fraud_probability"] == ref["fraud_probability"]
    assert back["risk_score"] == ref["risk_score"]
    assert back["alert_level"] == ref["alert_level"] == "Critical Risk"
    assert back["fraud_probability"] != later["fraud_probability"]
    # both are persisted with their own scores
    timeline = {t["transaction_id"]: t for t in client.get("/customer/CUST_0001/history").json()["timeline"]}
    assert timeline[back["transaction_id"]]["fraud_probability"] == back["fraud_probability"]
    assert timeline[later["transaction_id"]]["fraud_probability"] == later["fraud_probability"]


# 4 --------------------------------------------------------------------------------------------------

class _FixedUUID:
    def __init__(self, hexes):
        self.hexes = list(hexes)

    def __call__(self):
        class U:
            pass
        u = U()
        u.hex = self.hexes.pop(0) if len(self.hexes) > 1 else self.hexes[0]
        return u


def test_duplicate_transaction_id_is_never_created(pipeline, monkeypatch):
    first = _score(pipeline, NORMAL_TXN)
    taken = first["transaction_id"][4:].lower() + "000000"          # uuid hex whose prefix repeats that id
    fresh = "abcdef0123" + "0" * 22
    monkeypatch.setattr(inference_pipeline.uuid, "uuid4", _FixedUUID([taken, taken, fresh]))
    second = _score(pipeline, {**NORMAL_TXN, "timestamp": "2026-07-11T13:30:00"})
    assert second["transaction_id"] == "TXN_ABCDEF0123"
    assert _history(pipeline, "CUST_0001")["transaction_id"].is_unique


def test_duplicate_transaction_id_fails_clearly(pipeline, monkeypatch):
    first = _score(pipeline, NORMAL_TXN)
    taken = first["transaction_id"][4:].lower() + "000000"
    monkeypatch.setattr(inference_pipeline.uuid, "uuid4", _FixedUUID([taken]))
    before = len(_history(pipeline, "CUST_0001"))
    with pytest.raises(RuntimeError, match="unique transaction id"):
        pipeline.score_transaction({**NORMAL_TXN, "timestamp": "2026-07-11T13:30:00"})
    assert len(_history(pipeline, "CUST_0001")) == before           # nothing was added


def test_replayed_duplicates_are_skipped(pipeline):
    out = _score(pipeline, {**SUSPICIOUS_TXN, "timestamp": "2026-06-01T03:00:00"})
    row = _row(out)
    n = pipeline.restore_scored_history([row, dict(row)])
    assert n == 1
    assert _history(pipeline, "CUST_0001")["transaction_id"].is_unique


# 5 --------------------------------------------------------------------------------------------------

def test_multiple_backdated_transactions(pipeline):
    cid = "CUST_0003"
    hist = _history(pipeline, cid)
    ts = pd.to_datetime(hist["timestamp"])
    rng = np.random.default_rng(4)
    picks = sorted(rng.choice(np.arange(15, len(hist) - 1), size=6, replace=False))
    times = [ts.iloc[i] + (ts.iloc[i + 1] - ts.iloc[i]) / 3 for i in picks]
    rng.shuffle(times)
    rows = []
    for k, t in enumerate(times):
        txn = {"customer_id": cid, "amount": [900.0, 60000.0][k % 2], "merchant_category": ["grocery", "electronics"][k % 2],
               "device_id": ["DEV_0003_A", f"DEV_NEW_{k}"][k % 2], "location": ["Delhi", "Lagos"][k % 2],
               "failed_logins_24h": [0, 3][k % 2], "timestamp": t.isoformat()}
        ref = _reference(pipeline, txn)
        out = _score(pipeline, txn)
        _same_score(out, ref)
        rows.append(_row(out))
    new = _history(pipeline, cid)
    assert len(new) == len(hist) + len(times)
    _assert_chronological_and_unique(new)
    _assert_history_matches_replay(pipeline, cid, rows)


def test_backdated_new_device_updates_later_rows(pipeline):
    """A device first used by a back-dated transaction is no longer 'new' for
    the later transaction that used it -- as if processed in time order."""
    cid = "CUST_0004"
    late = _score(pipeline, {**NORMAL_TXN, "customer_id": cid, "device_id": "DEV_LATER", "location": "Mumbai",
                             "timestamp": "2026-07-20T12:00:00"})
    hist = _history(pipeline, cid)
    assert int(hist.loc[hist["transaction_id"] == late["transaction_id"], "is_new_device"].iloc[0]) == 1
    early = _score(pipeline, {**NORMAL_TXN, "customer_id": cid, "device_id": "DEV_LATER", "location": "Mumbai",
                              "timestamp": "2026-07-15T12:00:00"})
    hist = _history(pipeline, cid)
    flags = dict(zip(hist["transaction_id"], hist["is_new_device"].astype(int)))
    assert flags[early["transaction_id"]] == 1 and flags[late["transaction_id"]] == 0
    _assert_history_matches_replay(pipeline, cid, [_row(late), _row(early)])


# 6 --------------------------------------------------------------------------------------------------

def test_returned_score_belongs_to_the_requested_transaction(pipeline):
    """Score a batch of transactions in a scrambled time order: every response
    equals that transaction's own reference score, and its echoed fields are its own."""
    cid = "CUST_0005"
    base = pd.Timestamp("2026-07-12T10:00:00")
    txns = [{"customer_id": cid, "amount": float(a), "merchant_category": c, "device_id": d, "location": l,
             "failed_logins_24h": f, "timestamp": (base + pd.Timedelta(hours=h)).isoformat()}
            for a, c, d, l, f, h in [(1200, "grocery", "DEV_0005_A", "Kolkata", 0, 30),
                                     (75000, "electronics", "DEV_X1", "Lagos", 5, 2),
                                     (1500, "fuel", "DEV_0005_A", "Kolkata", 0, 50),
                                     (40000, "jewelry", "DEV_X2", "Dubai", 2, 10),
                                     (900, "grocery", "DEV_0005_A", "Kolkata", 0, 20)]]
    for txn in txns:
        ref = _reference(pipeline, txn)
        out = _score(pipeline, txn)
        _same_score(out, ref)
        assert out["timestamp"] == txn["timestamp"] or pd.Timestamp(out["timestamp"]) == pd.Timestamp(txn["timestamp"])
        hist = _history(pipeline, cid)
        row = hist[hist["transaction_id"] == out["transaction_id"]]
        assert len(row) == 1
        assert row["amount"].iloc[0] == txn["amount"] and row["device_id"].iloc[0] == txn["device_id"]
        assert pd.Timestamp(row["timestamp"].iloc[0]) == pd.Timestamp(txn["timestamp"])
