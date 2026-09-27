"""
Generator v2 settings.

Everything that shapes the data lives here, so the manifest can record the
exact configuration a dataset was generated with. The rates below are
generator PARAMETERS describing plausible behaviour. They are not tuned to
reach any model score.
"""

from dataclasses import asdict, dataclass, field, replace

# 2.0.1 (Step 4C-2b): every foreign city has legitimate travellers, every trip has at least
# one transaction, and the manifest no longer records the output directory.
GENERATOR_VERSION = "2.0.1"

# ---- fixed vocabulary (same values as v1) --------------------------------------

MERCHANT_CATEGORIES = [
    "grocery", "electronics", "travel", "dining", "utilities",
    "entertainment", "fashion", "healthcare", "fuel", "online_retail",
]

# Home and domestic-travel cities. Exactly v1's list, which is also the list
# the feature `is_foreign_location` treats as domestic, so the meaning of that
# feature is unchanged on v2 data.
DOMESTIC_CITIES = [
    "Hyderabad", "Mumbai", "Delhi", "Bangalore", "Chennai",
    "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Lucknow",
]
# v1's five foreign cities plus common, benign travel destinations. Both
# legitimate travellers and fraudsters appear in both groups.
FOREIGN_CITIES_V1 = ["Lagos", "Kyiv", "Manila", "Bucharest", "Jakarta"]
FOREIGN_CITIES_TRAVEL = ["Dubai", "Singapore", "London", "Bangkok"]
FOREIGN_CITIES = FOREIGN_CITIES_TRAVEL + FOREIGN_CITIES_V1

# ---- customer segments ---------------------------------------------------------
# peaks: (hour, relative weight). amount_median: range of the customer's median
# spend (INR). weekly: range of transactions per week. p_domestic_trip /
# p_foreign_trip: probability the customer travels at least once.

SEGMENTS = {
    "salaried_commuter": dict(share=0.40, peaks=[(9, 1.0), (13, 0.8), (20, 1.0)], amount_median=(600, 3500),
                              weekly=(4, 10), p_domestic_trip=0.25, p_foreign_trip=0.04),
    "night_shift": dict(share=0.10, peaks=[(23, 1.0), (2, 0.8), (6, 0.5)], amount_median=(300, 2000),
                        weekly=(3, 9), p_domestic_trip=0.15, p_foreign_trip=0.02),
    "student": dict(share=0.15, peaks=[(12, 0.8), (18, 1.0), (23, 0.8)], amount_median=(150, 900),
                    weekly=(5, 14), p_domestic_trip=0.20, p_foreign_trip=0.02),
    "business_traveller": dict(share=0.10, peaks=[(7, 0.8), (13, 0.8), (21, 1.0)], amount_median=(2000, 8000),
                               weekly=(5, 12), p_domestic_trip=0.80, p_foreign_trip=0.35),
    "retiree": dict(share=0.10, peaks=[(9, 1.0), (11, 0.9), (16, 0.7)], amount_median=(400, 2500),
                    weekly=(3, 7), p_domestic_trip=0.20, p_foreign_trip=0.03),
    "household": dict(share=0.15, peaks=[(10, 0.9), (18, 1.0), (21, 0.7)], amount_median=(800, 4000),
                      weekly=(4, 11), p_domestic_trip=0.25, p_foreign_trip=0.04),
}

# ---- fraud archetypes ----------------------------------------------------------
# share: share of fraud EPISODES. n_txns / span_hours: size and duration of the
# fraud part of an episode. p_* are per-transaction probabilities unless marked
# per-episode. amount: multiple of the customer's median spend. Ring transactions
# take their time from the ring's coordinated attack wave, not from an
# off-hours probability. No indicator probability is 0 or 1 in any archetype,
# so no single signal is certain.

ARCHETYPES = {
    "account_takeover": dict(
        share=0.20, n_txns=(2, 8), span_hours=(6, 72),
        p_new_device=0.80, p_foreign=0.25, p_other_city=0.15, p_off_hours=0.50,
        amount=(1.0, 6.0), p_category_shift=0.60, shift_categories=["electronics", "online_retail", "travel"],
        p_attack_logins=0.70, attack_failures=(2, 8), p_fraud_network=0.80,
        p_precursor=0.50,
    ),
    "stolen_card_online": dict(
        share=0.20, n_txns=(3, 10), span_hours=(12, 120),
        p_new_device=0.60, p_foreign=0.25, p_other_city=0.15, p_off_hours=0.40,
        amount=(0.8, 4.0), p_category_shift=0.70,
        shift_categories=["electronics", "online_retail", "entertainment", "travel"],
        p_attack_logins=0.10, attack_failures=(1, 3), p_fraud_network=0.60,
        p_precursor=0.0,
    ),
    "card_testing_cashout": dict(
        share=0.15, n_probes=(3, 10), probe_span_hours=(1, 6), probe_amount_inr=(1.0, 150.0),
        p_cashout=0.50, n_cashout=(1, 2), cashout_delay_hours=(0.5, 24), amount=(3.0, 10.0),
        p_new_device=0.70, p_foreign=0.30, p_other_city=0.10, p_off_hours=0.50,
        p_category_shift=0.80, shift_categories=["online_retail", "entertainment"],
        p_attack_logins=0.20, attack_failures=(1, 4), p_fraud_network=0.70,
        p_precursor=0.30,
    ),
    "device_takeover": dict(
        share=0.10, n_txns=(1, 4), span_hours=(1, 24),
        p_new_device=0.10, p_foreign=0.05, p_other_city=0.10, p_off_hours=0.30,
        amount=(2.0, 10.0), p_category_shift=0.50, shift_categories=["electronics", "online_retail", "travel"],
        p_attack_logins=0.30, attack_failures=(1, 4), p_fraud_network=0.30,
        p_precursor=0.0,
    ),
    "high_value_single": dict(
        share=0.10, n_txns=(1, 2), span_hours=(0.1, 12),
        p_new_device=0.50, p_foreign=0.15, p_other_city=0.15, p_off_hours=0.40,
        amount=(5.0, 20.0), p_category_shift=0.80, shift_categories=["electronics", "travel", "fashion"],
        p_attack_logins=0.30, attack_failures=(1, 5), p_fraud_network=0.50,
        p_precursor=0.0,
    ),
    "normal_looking": dict(
        share=0.10, n_txns=(1, 4), span_hours=(12, 96),
        p_new_device=0.15, p_foreign=0.03, p_other_city=0.07, p_off_hours=0.10,
        amount=(0.8, 2.5), p_category_shift=0.30, shift_categories=["online_retail", "fashion", "electronics"],
        p_attack_logins=0.05, attack_failures=(1, 2), p_fraud_network=0.10,
        p_precursor=0.0,
    ),
    "ring": dict(
        share=0.15, n_txns=(1, 4), span_hours=(0.5, 8),
        p_ring_device=0.60, p_ring_network=0.70, p_ring_merchant=0.50, p_ring_location=0.60,
        amount=(1.0, 5.0), p_category_shift=0.50,
        p_attack_logins=0.40, attack_failures=(1, 5),
        p_precursor=0.30,
    ),
}


@dataclass(frozen=True)
class GeneratorConfig:
    seed: int = 42
    n_customers: int = 500
    start_date: str = "2026-01-12"          # fixed; never derived from the clock
    days: int = 176                          # 2026-01-12 .. 2026-07-06 inclusive (same span as v1)

    # fraud volume: episodes = round(episodes_per_customer * n_customers)
    episodes_per_customer: float = 0.22
    repeat_victim_share: float = 0.10        # share of (non-ring) victims who get a second episode
    min_days_between_episodes: int = 30
    first_episode_day: int = 21              # leave some history before any fraud

    # rings
    ring_size: tuple = (3, 6)
    min_rings: int = 2
    ring_wave_max_hours: float = 72.0        # a whole ring attack fits in <= 3 days

    # precursor period before the first fraud (only for archetypes with p_precursor > 0)
    precursor_days: tuple = (1, 7)
    precursor_attack_bursts: tuple = (1, 3)
    precursor_failures_per_burst: tuple = (2, 6)

    # legitimate behaviour (life events and anomalies)
    household_share: float = SEGMENTS["household"]["share"]
    household_size: tuple = (2, 3)
    p_household_device_use: tuple = (0.20, 0.35)
    p_secondary_device: float = 0.55
    secondary_device_use: tuple = (0.10, 0.20)
    p_device_upgrade: float = 0.25
    upgrade_overlap_days: int = 14
    p_borrowed_device: float = 0.08
    domestic_trips: tuple = (1, 3)
    domestic_trip_days: tuple = (2, 7)
    foreign_trip_days: tuple = (4, 10)
    p_benign_foreign_destination: float = 0.80
    cover_foreign_destinations: bool = True    # every foreign city gets >= 1 legitimate traveller (no fraud-only city)
    hour_peak_width: tuple = (0.7, 1.0)        # std-dev (hours) of each daily activity peak
    hour_uniform_floor: tuple = (0.03, 0.08)   # share of a customer's activity spread over all 24 hours
    amount_sigma: tuple = (0.25, 0.60)         # lognormal sigma around the customer's median
    p_big_purchase: float = 0.015
    big_purchase_multiple: tuple = (3.0, 15.0)
    p_small_purchase: float = 0.05
    small_purchase_inr: tuple = (10.0, 200.0)
    p_new_category: float = 0.03
    p_burst_day: float = 0.01
    burst_extra_txns: tuple = (2, 5)
    salary_days: tuple = (1, 2, 3)
    salary_rate_uplift: float = 1.4
    forgot_password_per_day: float = 0.012     # per customer-day
    forgot_password_failures: tuple = (1, 4)
    typo_failures_per_day: float = 0.015       # isolated single failed logins
    p_public_network: float = 0.05
    p_carrier_network: float = 0.40
    p_vpn_user: float = 0.05                   # legitimate customers who route some traffic through VPN/proxy exits
    vpn_use: float = 0.30
    carrier_networks_per_city: int = 3
    public_networks_per_city: int = 3
    merchants_per_category: int = 40
    fraud_proxy_networks: int = 25

    segments: dict = field(default_factory=lambda: SEGMENTS)
    archetypes: dict = field(default_factory=lambda: ARCHETYPES)

    def with_overrides(self, **kw) -> "GeneratorConfig":
        return replace(self, **kw)

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_CONFIG = GeneratorConfig()
