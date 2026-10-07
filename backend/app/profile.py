"""
Customer Behavioral Context
=============================
Summarises a customer's EXISTING transaction history (seed history plus any
transactions already scored this session). Nothing here is invented or
hard-coded: every value is a statistic of the stored history.

It is used for two things:
  1. The dashboard's "Behavioral Context / What Changed?" section.
  2. Building a genuinely *typical* demo transaction for the selected
     customer (their own most common device / location / category / hour and
     their median amount), so the "Typical purchase" preset is realistic.

Robust statistics are used throughout (median / mode), so a handful of
fraudulent or unusual rows in the history cannot drag the "typical" values.
This module is read-only and never influences a prediction.
"""

from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from .config import MAX_FUTURE_TIMESTAMP_DAYS

# Same 5% rule feature_engineering.py uses for hour_is_unusual / category_is_unusual,
# so "preferred hours" here agree with the model's own definition of "usual".
USUAL_SHARE = 0.05
TOP_N = 3
MAX_KNOWN_LISTED = 200   # cap on the full device/location lists returned
MIN_HISTORY_FOR_TYPICAL = 5
TYPICAL_MINUTE = 30


def _count_share(series: pd.Series, top_n: Optional[int] = TOP_N, min_share: float = 0.0):
    counts = series.value_counts()
    total = int(counts.sum())
    out = []
    for value, count in counts.items():
        share = count / total
        if share < min_share:
            continue
        out.append({"value": str(value), "count": int(count), "share": round(float(share), 4)})
        if top_n is not None and len(out) >= top_n:
            break
    return out


def _typical_timestamp(typical_hour: int, latest: datetime, now: datetime) -> Optional[str]:
    """Most recent occurrence of the customer's usual hour that is not in the
    future; if that is not later than the customer's latest recorded
    transaction, step forward a day at a time so repeated typical scans stay in
    chronological order. Returns None if no admissible slot exists."""
    slot = now.replace(hour=typical_hour, minute=TYPICAL_MINUTE, second=0, microsecond=0)
    if slot > now:
        slot -= timedelta(days=1)
    limit = now + timedelta(days=MAX_FUTURE_TIMESTAMP_DAYS)
    if latest >= limit:
        return None
    while slot <= latest:
        slot += timedelta(days=1)
    if slot > limit:
        return None
    return slot.isoformat(timespec="seconds")


def build_customer_profile(customer_id: str, history: Optional[pd.DataFrame], now: Optional[datetime] = None) -> dict:
    now = now or datetime.now()
    empty = {
        "customer_id": customer_id, "n_transactions": 0,
        "first_transaction_at": None, "last_transaction_at": None,
        "typical_amount": None, "amount_p25": None, "amount_p75": None,
        "primary_device": None, "devices": [], "known_devices": [],
        "primary_location": None, "locations": [], "known_locations": [],
        "top_categories": [],
        "typical_hour": None, "preferred_hours": [], "hour_distribution": [],
        "transactions_per_week": None, "typical_scenario": None,
    }
    if history is None or len(history) == 0:
        return empty

    ts = pd.to_datetime(history["timestamp"])
    amounts = history["amount"].astype(float)
    amounts = amounts[np.isfinite(amounts)]
    n = int(len(history))

    devices = _count_share(history["device_id"])
    locations = _count_share(history["location"])
    categories = _count_share(history["merchant_category"], top_n=None, min_share=USUAL_SHARE)

    hour_share = ts.dt.hour.value_counts(normalize=True)
    hour_distribution = [round(float(hour_share.get(h, 0.0)), 4) for h in range(24)]
    preferred_hours = [h for h in range(24) if hour_share.get(h, 0.0) >= USUAL_SHARE]
    typical_hour = int(hour_share.idxmax())

    first_ts, last_ts = ts.min().to_pydatetime(), ts.max().to_pydatetime()
    # transactions per week *in which the customer was active*: unlike n / total span,
    # it does not shrink when a later transaction lands after a long gap
    active_weeks = int(ts.dt.to_period("W").nunique())
    per_week = round(n / active_weeks, 2) if active_weeks >= 2 else None

    typical_amount = round(float(amounts.median()), 2) if len(amounts) else None

    profile = dict(empty)
    profile.update({
        "n_transactions": n,
        "first_transaction_at": first_ts.isoformat(timespec="seconds"),
        "last_transaction_at": last_ts.isoformat(timespec="seconds"),
        "typical_amount": typical_amount,
        "amount_p25": round(float(amounts.quantile(0.25)), 2) if len(amounts) else None,
        "amount_p75": round(float(amounts.quantile(0.75)), 2) if len(amounts) else None,
        "primary_device": devices[0]["value"] if devices else None,
        "devices": devices,
        "known_devices": sorted(map(str, history["device_id"].dropna().unique()))[:MAX_KNOWN_LISTED],
        "primary_location": locations[0]["value"] if locations else None,
        "locations": locations,
        "known_locations": sorted(map(str, history["location"].dropna().unique()))[:MAX_KNOWN_LISTED],
        "top_categories": categories,
        "typical_hour": typical_hour,
        "preferred_hours": preferred_hours,
        "hour_distribution": hour_distribution,
        "transactions_per_week": per_week,
    })

    if n >= MIN_HISTORY_FOR_TYPICAL and typical_amount is not None and devices and locations and categories:
        profile["typical_scenario"] = {
            "amount": typical_amount,
            "merchant_category": categories[0]["value"],
            "device_id": devices[0]["value"],
            "location": locations[0]["value"],
            "failed_logins_24h": 0,
            "timestamp": _typical_timestamp(typical_hour, last_ts, now),
        }
    return profile
