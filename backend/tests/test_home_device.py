"""90-day home device (Step 4C-2f-1; design: docs/step4c2e-retraining-design.md §5).

home_device = the most frequent device_id among the customer's transactions in
(anchor - 90 days, anchor], anchor = the customer's newest transaction, ties
broken alphabetically. Fallback to the whole history when that window holds
fewer than 5 transactions or the history spans less than 90 days.
home_location keeps the whole-history rule (the design left it to be confirmed)."""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from app.features.feature_engineering import build_point_features
from app.inference_pipeline import HOME_DEVICE_MIN_TRANSACTIONS, HOME_DEVICE_WINDOW, home_device_window

ANCHOR = datetime(2026, 7, 1, 12, 0)


def _history(spec, cid="CUST_HD"):
    """spec: list of (days_before_anchor, device, count[, spread]); `count` transactions,
    1 minute apart (spread=True, the default) or all at exactly that time (spread=False)."""
    rows = []
    for days, device, count, *rest in spec:
        spread = rest[0] if rest else True
        for j in range(count):
            rows.append({"customer_id": cid, "transaction_id": f"T_{len(rows):05d}",
                         "timestamp": ANCHOR - timedelta(days=days) + timedelta(minutes=j if spread else 0),
                         "amount": 1000.0, "merchant_category": "grocery", "device_id": device,
                         "location": "Pune" if device != "DEV_TRAVEL" else "Delhi", "failed_logins_24h": 0,
                         "is_new_device": 0, "is_new_location": 0, "is_fraud": 0})
    df = pd.DataFrame(rows).sort_values("timestamp")
    df["is_new_device"] = (~df["device_id"].duplicated()).astype(int)
    df["is_new_location"] = (~df["location"].duplicated()).astype(int)
    return build_point_features(df)


def _daily(device, start_days_ago, end_days_ago, per_day=1):
    return [(d, device, per_day) for d in range(start_days_ago, end_days_ago - 1, -1)]


@pytest.fixture
def profile(pipeline):
    def get(spec):
        pipeline.customer_histories["CUST_HD"] = _history(spec)
        return pipeline.get_customer_profile("CUST_HD")
    return get


def _whole_history_device(spec):
    counts = {}
    for _, d, c, *_ in spec:
        counts[d] = counts.get(d, 0) + c
    top = max(counts.values())
    return sorted(d for d, c in counts.items() if c == top)[0]


def test_constants():
    assert HOME_DEVICE_WINDOW == pd.Timedelta(days=90) and HOME_DEVICE_MIN_TRANSACTIONS == 5


def test_more_than_90_days_stable_device(profile):
    spec = _daily("DEV_A", 200, 0)
    assert profile(spec)["home_device"] == "DEV_A"


def test_upgrade_inside_the_window(profile):
    # old phone for 160 days (2/day), new phone for the last 40 days (3/day):
    # whole history says old, the last 90 days say new
    spec = _daily("DEV_OLD", 200, 41, 2) + _daily("DEV_NEW", 40, 0, 3)
    assert _whole_history_device(spec) == "DEV_OLD"
    assert profile(spec)["home_device"] == "DEV_NEW"


def test_upgrade_inside_the_window_not_yet_dominant(profile):
    # new phone only for the last 10 days: the old phone still dominates the window
    spec = _daily("DEV_OLD", 200, 11, 1) + _daily("DEV_NEW", 10, 0, 1)
    assert profile(spec)["home_device"] == "DEV_OLD"


def test_upgrade_before_the_window(profile):
    # upgraded 120 days ago: the window holds only the new phone, the whole history the old one
    spec = _daily("DEV_OLD", 300, 121, 3) + _daily("DEV_NEW", 120, 0, 1)
    assert _whole_history_device(spec) == "DEV_OLD"
    assert profile(spec)["home_device"] == "DEV_NEW"


def test_old_burst_outside_the_window_is_ignored(profile):
    spec = [(150, "DEV_BURST", 300)] + _daily("DEV_A", 200, 0, 1)
    assert _whole_history_device(spec) == "DEV_BURST"
    assert profile(spec)["home_device"] == "DEV_A"


def test_less_than_90_days_uses_whole_history(profile):
    spec = _daily("DEV_OLD", 60, 21, 1) + _daily("DEV_NEW", 20, 0, 1)
    p = profile(spec)
    assert p["home_device"] == _whole_history_device(spec) == "DEV_OLD"


@pytest.mark.parametrize("span_days, expected", [(89, "DEV_EARLY"), (90, "DEV_LATE"), (91, "DEV_LATE")])
def test_around_90_days(profile, span_days, expected):
    """12 DEV_EARLY transactions, all exactly `span_days` before the anchor, then 10
    DEV_LATE ones ending at the anchor. Under 90 days of history the whole history
    decides (DEV_EARLY, 12 > 10). From 90 days on the window (anchor-90d, anchor]
    applies; it excludes transactions exactly 90 days old, so DEV_LATE wins."""
    spec = [(span_days, "DEV_EARLY", 12, False)] + [(d, "DEV_LATE", 1) for d in range(10)]
    assert _whole_history_device(spec) == "DEV_EARLY"
    assert profile(spec)["home_device"] == expected


def test_window_boundary_is_half_open(pipeline):
    h = _history([(90, "DEV_EDGE", 1), (89, "DEV_IN", 1), (0, "DEV_IN", 5, False)])
    w = home_device_window(h)
    assert "DEV_EDGE" not in set(w["device_id"]) and len(w) == 6


def test_sparse_window_falls_back_to_whole_history(profile):
    # 200 days of history, but only 3 transactions in the last 90 days
    spec = _daily("DEV_OLD", 200, 100, 1) + [(30, "DEV_NEW", 3)]
    assert profile(spec)["home_device"] == "DEV_OLD"


def test_tie_in_window_is_alphabetical(profile):
    spec = _daily("DEV_OLD", 200, 111, 1) + [(10, "DEV_B", 4), (20, "DEV_A", 4)]
    assert profile(spec)["home_device"] == "DEV_A"


def test_no_history(pipeline, client):
    assert pipeline.get_customer_profile("CUST_NO_HISTORY") is None
    assert client.get("/customer/CUST_NO_HISTORY/profile").status_code == 404


def test_location_keeps_whole_history_rule(profile):
    spec = [(150, "DEV_TRAVEL", 300)] + _daily("DEV_A", 200, 0, 1)      # DEV_TRAVEL rows are in Delhi
    p = profile(spec)
    assert p["home_device"] == "DEV_A" and p["home_location"] == "Delhi"


def test_batch_default_uses_90_day_device(client, pipeline):
    spec = _daily("DEV_OLD", 300, 121, 3) + _daily("DEV_NEW", 120, 0, 1)
    pipeline.customer_histories["CUST_HD"] = _history(spec)
    csv = "customer_id,amount,merchant_category\nCUST_HD,1000,grocery\n"
    r = client.post("/predict/batch", files={"file": ("b.csv", csv.encode(), "text/csv")})
    assert r.status_code == 200
    hist = pipeline.customer_histories["CUST_HD"]
    assert hist["device_id"].iloc[-1] == "DEV_NEW"


def test_seed_customers_unchanged(pipeline):
    """v1: the 90-day rule gives the same device as the whole-history rule for all 500."""
    for cid in pipeline.known_customer_ids():
        assert pipeline.get_customer_profile(cid)["home_device"] == cid.replace("CUST_", "DEV_") + "_A"
