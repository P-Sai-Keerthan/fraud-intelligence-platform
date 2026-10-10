"""Cold-start sensitivity of hour_is_unusual / category_is_unusual (Step
4C-2f-2, analysis only -- the behaviour is pinned here, not changed).

Current rule in build_point_features: a value (hour of day, or merchant
category) is "unusual" when its share of the customer's EARLIER transactions
is below 5%. So with n earlier transactions:

* n = 0: hour_is_unusual = 0 (no hours seen), category_is_unusual = 1
  (placeholder; the pipeline imputes both at n = 0 since 4C-2f-1);
* 1 <= n <= 20: any value not seen before is unusual (0/n < 5%), and any value
  seen at least once is not (1/n >= 5%) -- so with fewer than 20 earlier
  transactions an unseen hour/category is always flagged, however varied the
  customer's behaviour;
* n >= 21: a value seen once is also unusual (1/21 < 5%)."""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from app.features.feature_engineering import build_point_features


def _features(n_prior, hour_new, cat_new, repeat_first=1):
    """n_prior earlier transactions at 10:00 in 'grocery' (the first `repeat_first` of them at 22:00 in
    'travel'), then one transaction at hour_new / cat_new."""
    rows = []
    t0 = datetime(2026, 1, 1)
    for i in range(n_prior):
        odd = i < repeat_first
        rows.append({"customer_id": "C", "transaction_id": f"T{i:03d}",
                     "timestamp": t0 + timedelta(days=i, hours=22 if odd else 10),
                     "amount": 1000.0, "merchant_category": "travel" if odd else "grocery",
                     "device_id": "D", "location": "Pune", "failed_logins_24h": 0,
                     "is_new_device": int(i == 0), "is_new_location": int(i == 0), "is_fraud": 0})
    rows.append({"customer_id": "C", "transaction_id": "TNEW",
                 "timestamp": t0 + timedelta(days=n_prior, hours=hour_new), "amount": 1000.0,
                 "merchant_category": cat_new, "device_id": "D", "location": "Pune", "failed_logins_24h": 0,
                 "is_new_device": 0, "is_new_location": 0, "is_fraud": 0})
    f = build_point_features(pd.DataFrame(rows))
    last = f[f["transaction_id"] == "TNEW"].iloc[0]
    return int(last["hour_is_unusual"]), int(last["category_is_unusual"])


def test_no_prior_transactions():
    assert _features(0, 10, "grocery") == (0, 1)


@pytest.mark.parametrize("n", [1, 2, 3, 5, 9, 10, 15, 20, 21, 40])
def test_unseen_value_is_always_unusual(n):
    assert _features(n, 3, "jewelry") == (1, 1)


@pytest.mark.parametrize("n", [1, 2, 3, 5, 9, 10, 15, 20])
def test_value_seen_once_is_normal_up_to_20_prior(n):
    # hour 22 / 'travel' appear exactly once among the earlier transactions
    assert _features(n, 22, "travel") == (0, 0)


@pytest.mark.parametrize("n", [21, 30, 40])
def test_value_seen_once_is_unusual_from_21_prior(n):
    assert _features(n, 22, "travel") == (1, 1)


@pytest.mark.parametrize("n", [21, 40])
def test_value_seen_twice_is_normal_up_to_40_prior(n):
    assert _features(n, 22, "travel", repeat_first=2) == (0, 0)


@pytest.mark.parametrize("n", [1, 5, 9, 15])
def test_the_most_common_value_is_normal(n):
    assert _features(n, 10, "grocery", repeat_first=0) == (0, 0)
