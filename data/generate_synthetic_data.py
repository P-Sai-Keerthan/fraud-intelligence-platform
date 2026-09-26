"""
Synthetic Banking Transaction Data Generator
=============================================
Generates realistic customer transaction histories with:
- Normal behavioral patterns per customer (spending, timing, device, location)
- Fraud episodes for ~12% of customers: a lower-severity "ramp-up" week
  followed by a high-severity fraud week. The ramp-up was intended to give
  the LSTM a "risk escalation before fraud" signal. Both weeks' transactions
  are labeled is_fraud=1, though, so the data has no legitimate-labeled
  lead-up period to learn an early warning from (see docs/EVALUATION.md).
- Sudden anomalies for point-in-time fraud detection (so the DNN has a real
  signal to learn "this single transaction looks wrong" from)

WHY THIS EXISTS:
Real bank data is never public. This generator lets you build, train, and
demo the ENTIRE pipeline right now. When you get access to a real dataset
(PaySim, IEEE-CIS Fraud Detection, or your own bank's data), swap out this
CSV for that one -- as long as the column names match (see OUTPUT SCHEMA
below), nothing else in the project needs to change.

OUTPUT SCHEMA (data/transactions.csv):
    customer_id         str   e.g. "CUST_0001"
    transaction_id       str   unique id
    timestamp            datetime
    amount                float
    merchant_category     str   one of MERCHANT_CATEGORIES
    device_id             str   device fingerprint used for this txn
    location              str   city/region used for this txn
    failed_logins_24h     int   failed login attempts in the last 24h
    is_new_device         int   1 if device not seen before for this customer
    is_new_location        int   1 if location not seen before for this customer
    is_fraud              int   ground truth label (1 = fraudulent transaction)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

RNG_SEED = 42
np.random.seed(RNG_SEED)
random.seed(RNG_SEED)

N_CUSTOMERS = 500
DAYS_OF_HISTORY = 180
MERCHANT_CATEGORIES = [
    "grocery", "electronics", "travel", "dining", "utilities",
    "entertainment", "fashion", "healthcare", "fuel", "online_retail",
]
LOCATIONS = [
    "Hyderabad", "Mumbai", "Delhi", "Bangalore", "Chennai",
    "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Lucknow",
]
FOREIGN_LOCATIONS = ["Lagos", "Kyiv", "Manila", "Bucharest", "Jakarta"]


def make_customer_profile(customer_idx):
    """Each customer gets a stable 'normal' behavioral fingerprint."""
    return {
        "customer_id": f"CUST_{customer_idx:04d}",
        "home_location": random.choice(LOCATIONS),
        "primary_device": f"DEV_{customer_idx:04d}_A",
        "secondary_device": f"DEV_{customer_idx:04d}_B",
        "avg_amount": np.random.uniform(300, 4000),
        "amount_std": np.random.uniform(50, 400),
        "preferred_hours": sorted(random.sample(range(24), k=random.randint(3, 6))),
        "preferred_categories": random.sample(MERCHANT_CATEGORIES, k=random.randint(3, 5)),
        "avg_txns_per_week": np.random.uniform(3, 12),
    }


def generate_normal_transaction(profile, ts):
    hour = random.choice(profile["preferred_hours"])
    ts = ts.replace(hour=hour, minute=random.randint(0, 59))
    amount = max(10, np.random.normal(profile["avg_amount"], profile["amount_std"]))
    category = random.choice(profile["preferred_categories"])
    device = profile["primary_device"] if random.random() < 0.85 else profile["secondary_device"]
    location = profile["home_location"]
    return {
        "customer_id": profile["customer_id"],
        "timestamp": ts,
        "amount": round(amount, 2),
        "merchant_category": category,
        "device_id": device,
        "location": location,
        "failed_logins_24h": np.random.poisson(0.1),
        "is_fraud": 0,
    }


def generate_fraud_transaction(profile, ts, severity=1.0):
    """Fraud transactions deviate on multiple axes at once, scaled by severity
    (0->1), so a fraud episode can start with a milder ramp-up week before
    the high-severity week. Every transaction generated here is labeled
    is_fraud=1, ramp-up included."""
    hour = random.choice([h for h in range(24) if h not in profile["preferred_hours"]])
    ts = ts.replace(hour=hour, minute=random.randint(0, 59))
    amount = profile["avg_amount"] * (3 + 5 * severity) + np.random.uniform(0, 1000)
    category = random.choice(MERCHANT_CATEGORIES)
    use_new_device = random.random() < (0.4 + 0.5 * severity)
    device = f"DEV_UNKNOWN_{random.randint(1000,9999)}" if use_new_device else profile["primary_device"]
    use_foreign = random.random() < (0.3 + 0.5 * severity)
    location = random.choice(FOREIGN_LOCATIONS) if use_foreign else profile["home_location"]
    return {
        "customer_id": profile["customer_id"],
        "timestamp": ts,
        "amount": round(amount, 2),
        "merchant_category": category,
        "device_id": device,
        "location": location,
        "failed_logins_24h": np.random.poisson(2 + 4 * severity),
        "is_fraud": 1,
    }


def generate_dataset():
    all_rows = []
    start_date = datetime.now() - timedelta(days=DAYS_OF_HISTORY)

    # ~12% of customers will experience a fraud event with a lead-up ramp
    fraud_customer_idxs = set(
        random.sample(range(N_CUSTOMERS), k=int(N_CUSTOMERS * 0.12))
    )

    for idx in range(N_CUSTOMERS):
        profile = make_customer_profile(idx)
        n_weeks = DAYS_OF_HISTORY // 7
        day_cursor = start_date

        will_have_fraud = idx in fraud_customer_idxs
        # pick a random week (not too early, not the very last) for the fraud event
        fraud_week = random.randint(n_weeks // 3, n_weeks - 2) if will_have_fraud else -1

        for week in range(n_weeks):
            n_txns_this_week = np.random.poisson(profile["avg_txns_per_week"])
            for _ in range(max(1, n_txns_this_week)):
                ts = day_cursor + timedelta(
                    days=random.randint(0, 6), hours=random.randint(0, 23)
                )

                if will_have_fraud and week == fraud_week - 1:
                    # ramp-up week: milder anomalies, also labeled fraud (is_fraud=1)
                    row = generate_fraud_transaction(profile, ts, severity=random.uniform(0.15, 0.4))
                elif will_have_fraud and week == fraud_week:
                    # confirmed fraud event(s): high severity
                    row = generate_fraud_transaction(profile, ts, severity=random.uniform(0.7, 1.0))
                else:
                    row = generate_normal_transaction(profile, ts)

                all_rows.append(row)
            day_cursor += timedelta(days=7)

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["customer_id", "timestamp"]).reset_index(drop=True)

    # derive is_new_device / is_new_location by looking at each customer's history in order
    seen_devices, seen_locations = {}, {}
    new_device_flags, new_location_flags = [], []
    for _, row in df.iterrows():
        cid = row["customer_id"]
        devices = seen_devices.setdefault(cid, set())
        locations = seen_locations.setdefault(cid, set())
        new_device_flags.append(int(row["device_id"] not in devices))
        new_location_flags.append(int(row["location"] not in locations))
        devices.add(row["device_id"])
        locations.add(row["location"])
    df["is_new_device"] = new_device_flags
    df["is_new_location"] = new_location_flags

    df.insert(1, "transaction_id", [f"TXN_{i:07d}" for i in range(len(df))])
    return df


if __name__ == "__main__":
    df = generate_dataset()
    out_path = "data/transactions.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df):,} transactions for {df['customer_id'].nunique()} customers")
    print(f"Fraud rate: {df['is_fraud'].mean()*100:.2f}%")
    print(f"Saved to {out_path}")
    print(df.head())
