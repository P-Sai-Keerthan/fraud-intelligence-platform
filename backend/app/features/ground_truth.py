"""
Columns that are ground truth or analysis metadata, never model inputs.

The v2 synthetic data (data/v2/) appends these columns to the transaction
files. They describe how a row was generated (its fraud episode, ring,
archetype, whether it is a precursor, why a legitimate row looks unusual)
and would leak the label if used as features. `is_fraud` is the label itself.

feature_engineering checks its FEATURE_COLUMNS against this list at import
time, and the test suite checks the DNN input columns too.
"""

LABEL_COLUMN = "is_fraud"

# must match data/v2/synth_v2/schema.py METADATA_COLUMNS (checked by a test)
GROUND_TRUTH_COLUMNS = (
    "merchant_id",
    "network_id",
    "fraud_type",
    "fraud_episode_id",
    "fraud_stage",
    "fraud_ring_id",
    "is_precursor",
    "legit_context",
    "customer_segment",
)

FORBIDDEN_FEATURE_COLUMNS = frozenset(GROUND_TRUTH_COLUMNS) | {LABEL_COLUMN}


def assert_no_ground_truth(columns, where: str = "model features") -> None:
    """Raises ValueError if any label / ground-truth column is among `columns`."""
    leaked = sorted(set(columns) & FORBIDDEN_FEATURE_COLUMNS)
    if leaked:
        raise ValueError(f"ground-truth columns must not be used as {where}: {leaked}")
