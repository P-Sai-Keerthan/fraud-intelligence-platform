"""
Behavioral Similarity Score
=============================
Measures how closely a customer's CURRENT transaction matches their normal
behavioral profile.

IMPORTANT DESIGN NOTE: a naive cosine similarity on raw feature values does
NOT work well here, because features live on very different scales (e.g.
"amount_pct_of_avg" ~ 0-500 vs "is_new_device" which is 0/1). The large-scale
feature dominates the cosine angle and the score stays near 100% regardless
of actual behavior. Instead, we z-score each feature by the customer's own
historical mean/std (the same idea already used for amount_zscore), take the
average magnitude of deviation across all dimensions, and map that onto a
0-100 similarity scale with an exponential decay. This keeps every feature's
contribution comparable and the result behaves intuitively:
    - identical to normal behavior      -> ~100% similarity
    - moderately different behavior     -> mid-range similarity
    - wildly different (fraud-like)     -> low similarity / high deviation
"""

import numpy as np

from ..config import SEQUENCE_LENGTH

# A baseline (mean and standard deviation of the customer's earlier feature vectors) needs enough
# earlier transactions to mean anything. Below this many, the score is NOT computed: comparing a
# transaction with an empty or one-row baseline (zeros / forced unit std) produced values such as
# "98 % deviation" for an ordinary first purchase (audit finding R-05). The constant equals the
# cold-start threshold used by the rest of the pipeline (MIN_PRIOR_TRANSACTIONS = SEQUENCE_LENGTH).
MIN_HISTORY_FOR_SIMILARITY = SEQUENCE_LENGTH
STATUS_OK = "ok"
STATUS_INSUFFICIENT_HISTORY = "insufficient_history"

# controls how fast similarity decays with average z-deviation; larger DECAY_K
# = more forgiving (slower decay). Tuned so ~2 std devs of average deviation
# lands around 50% similarity.
DECAY_K = 2.9


def compute_similarity(
    current_vector: np.ndarray,
    historical_mean_vector: np.ndarray,
    historical_std_vector: np.ndarray,
) -> dict:
    """
    current_vector: raw feature vector for the current transaction
    historical_mean_vector: per-feature mean from the customer's own history
    historical_std_vector: per-feature std from the customer's own history
    """
    current = np.asarray(current_vector, dtype=np.float64).flatten()
    mean = np.asarray(historical_mean_vector, dtype=np.float64).flatten()
    std = np.asarray(historical_std_vector, dtype=np.float64).flatten()
    std = np.where(std <= 1e-6, 1.0, std)  # avoid divide-by-zero for constant features

    z = (current - mean) / std
    avg_abs_deviation = float(np.mean(np.abs(z)))

    similarity_pct = round(100 * np.exp(-avg_abs_deviation / DECAY_K), 2)
    deviation_pct = round(100 - similarity_pct, 2)

    return {
        "similarity_pct": similarity_pct,
        "deviation_pct": deviation_pct,
        "avg_z_deviation": round(avg_abs_deviation, 3),
    }


def behavioral_similarity(current_vector, prior_features: np.ndarray) -> dict:
    """Similarity of the current transaction to the customer's own earlier transactions
    (`prior_features`: one row per earlier transaction, strictly before the current one).

    Returns {"similarity_pct", "deviation_pct", "similarity_status", "history_transactions"}.
    With fewer than MIN_HISTORY_FOR_SIMILARITY earlier transactions the two scores are None
    and the status is "insufficient_history": a missing score is never replaced by 0 or 100.
    The formula (compute_similarity) is unchanged."""
    prior = np.asarray(prior_features, dtype=np.float64)
    n_prior = int(len(prior))
    if n_prior < MIN_HISTORY_FOR_SIMILARITY:
        return {"similarity_pct": None, "deviation_pct": None,
                "similarity_status": STATUS_INSUFFICIENT_HISTORY, "history_transactions": n_prior}
    result = compute_similarity(current_vector, prior.mean(axis=0), prior.std(axis=0))
    return {"similarity_pct": result["similarity_pct"], "deviation_pct": result["deviation_pct"],
            "similarity_status": STATUS_OK, "history_transactions": n_prior}


def historical_profile_for_customer(feat_df, customer_id: str, feature_columns, up_to_index=None):
    """
    Computes (mean, std) of the given feature_columns for a customer's
    history (optionally only up to a certain row index, to avoid leakage in
    offline evaluation). Returns two numpy arrays in the same column order.
    """
    group = feat_df[feat_df["customer_id"] == customer_id]
    if up_to_index is not None:
        group = group.iloc[:up_to_index]
    if len(group) == 0:
        return np.zeros(len(feature_columns)), np.ones(len(feature_columns))
    mean = group[feature_columns].mean().values
    std = group[feature_columns].std().fillna(1.0).values if len(group) > 1 else np.ones(len(feature_columns))
    return mean, std


if __name__ == "__main__":
    # quick sanity check using a customer with a stable behavioral profile
    normal_mean = np.array([0.1, 0, 0, 0, 0, 0.2, 0, 1, 100])
    normal_std = np.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.4, 0.1, 0.5, 15])

    current_normal = np.array([0.15, 0, 0, 0, 0, 0.3, 0, 1, 98])
    current_mild_drift = np.array([1.2, 0, 0, 0, 0, 1.0, 0, 2, 140])
    current_fraud = np.array([8.0, 1, 1, 1, 1, 5, 1, 3, 450])

    print("Normal-looking current transaction vs profile:")
    print(compute_similarity(current_normal, normal_mean, normal_std))

    print("\nMild-drift current transaction vs profile:")
    print(compute_similarity(current_mild_drift, normal_mean, normal_std))

    print("\nFraud-looking current transaction vs profile:")
    print(compute_similarity(current_fraud, normal_mean, normal_std))
