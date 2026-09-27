"""
Column contracts for v2 output files.

The v1 columns come first, with the same names, order and meaning as v1.
Everything v2 adds is appended at the end and is METADATA: ground truth and
analysis aids that must never be used as a model feature.
"""

# data/transactions.csv in v1 (see data/generate_synthetic_data.py)
V1_RAW_COLUMNS = [
    "customer_id", "transaction_id", "timestamp", "amount", "merchant_category",
    "device_id", "location", "failed_logins_24h", "is_fraud", "is_new_device", "is_new_location",
]

# data/transactions_with_features.csv in v1: the raw columns that
# build_point_features keeps, followed by the nine feature columns.
V1_FEATURES_BASE_COLUMNS = [
    "customer_id", "transaction_id", "timestamp", "amount", "merchant_category",
    "device_id", "location", "is_fraud",
]
V1_FEATURE_COLUMNS = [
    "amount_zscore", "hour_is_unusual", "is_new_device", "is_new_location", "is_foreign_location",
    "failed_logins_24h", "category_is_unusual", "txn_velocity_1h", "amount_pct_of_avg",
]
V1_FEATURES_CSV_COLUMNS = V1_FEATURES_BASE_COLUMNS + V1_FEATURE_COLUMNS

# Appended by v2. Must stay identical to app.features.ground_truth.GROUND_TRUTH_COLUMNS
# (a test checks this).
METADATA_COLUMNS = [
    "merchant_id",        # merchant the transaction was made at
    "network_id",         # network (IP / ASN proxy) the transaction came from
    "fraud_type",         # archetype of the fraud episode, "none" for legitimate rows
    "fraud_episode_id",   # explicit episode id, 0 for legitimate rows
    "fraud_stage",        # "first" / "subsequent" for fraud rows, "none" otherwise
    "fraud_ring_id",      # ring id for ring fraud rows, 0 otherwise
    "is_precursor",       # 1 = legitimate row inside an episode's warning period (still is_fraud=0)
    "legit_context",      # why a legitimate row may look unusual ("travel_domestic|big_purchase"), "none"
    "customer_segment",   # behavioural segment of the customer
]

RAW_COLUMNS = V1_RAW_COLUMNS + METADATA_COLUMNS
FEATURES_CSV_COLUMNS = V1_FEATURES_CSV_COLUMNS + METADATA_COLUMNS

CUSTOMER_COLUMNS = [
    "customer_id", "customer_segment", "home_city", "household_id", "primary_device",
    "secondary_device", "upgrade_device", "upgrade_date", "has_foreign_trip", "n_domestic_trips",
]
EPISODE_COLUMNS = [
    "fraud_episode_id", "customer_id", "fraud_type", "fraud_ring_id", "precursor_start",
    "first_fraud_time", "last_fraud_time", "n_fraud_transactions",
]
LOGIN_FAILURE_COLUMNS = ["customer_id", "timestamp", "source"]

FRAUD_STAGES = ("none", "first", "subsequent")
LEGIT_CONTEXTS = (
    "travel_domestic", "travel_foreign", "device_upgrade", "borrowed_device", "household_device",
    "big_purchase", "small_purchase", "new_category", "burst", "forgot_password", "public_network", "vpn",
)
LOGIN_FAILURE_SOURCES = ("typo", "forgot_password", "credential_attack", "fraud_session")
