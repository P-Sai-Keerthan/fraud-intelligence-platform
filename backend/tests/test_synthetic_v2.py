"""Synthetic data generator v2 (data/v2/synth_v2).

The tests generate small samples in memory or in a temp directory. They
never write into data/v2 and never generate the full 500-customer dataset
(the 500-customer check uses the plan, which creates no transactions).
"""

import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.config import FEATURES_CSV, PROJECT_ROOT
from app.features.feature_engineering import FEATURE_COLUMNS, build_point_features, build_sequences
from app.features.ground_truth import FORBIDDEN_FEATURE_COLUMNS, GROUND_TRUTH_COLUMNS, assert_no_ground_truth

V2_DIR = PROJECT_ROOT / "data" / "v2"
sys.path.insert(0, str(V2_DIR))

from synth_v2 import DEFAULT_CONFIG, build_plan, generate  # noqa: E402
from synth_v2 import schema  # noqa: E402
from synth_v2.config import ARCHETYPES, DOMESTIC_CITIES, FOREIGN_CITIES, MERCHANT_CATEGORIES  # noqa: E402
from synth_v2.features import build_features_table  # noqa: E402
from synth_v2.groups import customer_components, episode_groups, group_spans  # noqa: E402
from synth_v2.output import FILES, MANIFEST, write_dataset  # noqa: E402

SAMPLE_CUSTOMERS = 100
NEUTRAL_DEVICE = re.compile(r"^DEV_[0-9A-F]{8}$")
ORIGINAL_DEVICE = re.compile(r"^DEV_\d{4}_[AB]$")


@pytest.fixture(scope="module")
def sample():
    return generate(DEFAULT_CONFIG.with_overrides(n_customers=SAMPLE_CUSTOMERS))


@pytest.fixture(scope="module")
def tx(sample):
    df = sample.transactions.copy()
    df["ts"] = pd.to_datetime(df["timestamp"])
    return df


@pytest.fixture(scope="module")
def features(sample):
    return build_features_table(sample.transactions)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---- configuration, size and determinism --------------------------------------------

def test_default_config_is_fixed_and_has_500_customers():
    cfg = DEFAULT_CONFIG
    assert cfg.n_customers == 500 and cfg.seed == 42
    assert (cfg.start_date, cfg.days) == ("2026-01-12", 176)
    plan = build_plan(cfg)                                 # plan only: no transactions are generated
    assert [c.customer_id for c in plan.customers] == [f"CUST_{i:04d}" for i in range(500)]
    planned = pd.Series([e.archetype for e in plan.episodes]).value_counts()
    assert set(planned.index) == set(ARCHETYPES)
    total = len(plan.episodes)
    for name, spec in ARCHETYPES.items():                  # quotas follow the configured shares
        assert abs(planned[name] / total - spec["share"]) < 0.03, name


def test_generator_never_reads_the_clock():
    for path in (V2_DIR / "synth_v2").glob("*.py"):
        src = path.read_text()
        assert "datetime.now" not in src and "date.today" not in src and "time.time" not in src, path.name


def test_same_seed_gives_identical_output(tmp_path):
    cfg = DEFAULT_CONFIG.with_overrides(n_customers=30)
    a, b = generate(cfg), generate(cfg)
    for name in ("transactions", "customers", "episodes", "login_failures"):
        pd.testing.assert_frame_equal(getattr(a, name), getattr(b, name))
    write_dataset(a, tmp_path / "a")
    write_dataset(b, tmp_path / "b")
    for fname in [FILES["transactions"], FILES["customers"], FILES["episodes"], FILES["login_failures"], MANIFEST]:
        assert _sha(tmp_path / "a" / fname) == _sha(tmp_path / "b" / fname), fname
    other = generate(cfg.with_overrides(seed=7))
    assert not other.transactions.equals(a.transactions)


def test_cli_writes_dataset_and_manifest_checksums(tmp_path):
    import generate as cli
    default_out = V2_DIR / "transactions.csv"
    before = default_out.stat().st_mtime_ns if default_out.exists() else None
    cli.main(["--customers", "12", "--out", str(tmp_path), "--skip-features"])
    after = default_out.stat().st_mtime_ns if default_out.exists() else None
    assert before == after                                 # nothing written to data/v2
    import json
    manifest = json.loads((tmp_path / MANIFEST).read_text())
    assert manifest["summary"]["customers"] == 12
    assert manifest["config"]["start_date"] == "2026-01-12"
    for fname, info in manifest["files"].items():
        assert _sha(tmp_path / fname) == info["sha256"]


# ---- schema -------------------------------------------------------------------------------

def test_columns_and_order(sample, features):
    assert schema.RAW_COLUMNS == schema.V1_RAW_COLUMNS + schema.METADATA_COLUMNS
    assert list(sample.transactions.columns) == schema.RAW_COLUMNS
    assert list(features.columns) == schema.V1_FEATURES_CSV_COLUMNS + schema.METADATA_COLUMNS
    # the v1 names/order are exactly those of the committed v1 features file
    v1_header = list(pd.read_csv(FEATURES_CSV, nrows=0).columns)
    assert v1_header == schema.V1_FEATURES_CSV_COLUMNS
    assert schema.V1_FEATURE_COLUMNS == list(FEATURE_COLUMNS)
    assert list(schema.METADATA_COLUMNS) == list(GROUND_TRUTH_COLUMNS)


def test_v1_columns_keep_their_meaning(tx, sample):
    assert tx.groupby("customer_id")["ts"].apply(lambda s: s.is_monotonic_increasing and s.is_unique).all()
    assert tx["ts"].min() >= pd.Timestamp("2026-01-12") and tx["ts"].max() < pd.Timestamp("2026-07-07")
    assert list(tx["transaction_id"]) == [f"TXN_{i:07d}" for i in range(len(tx))]
    assert tx.equals(tx.sort_values(["customer_id", "ts"]))
    assert set(tx["is_fraud"]) <= {0, 1}
    assert set(tx["merchant_category"]) <= set(MERCHANT_CATEGORIES)
    assert set(tx["location"]) <= set(DOMESTIC_CITIES) | set(FOREIGN_CITIES)
    assert (tx["amount"] > 0).all()
    assert (tx["is_new_device"] == (~tx.duplicated(["customer_id", "device_id"])).astype(int)).all()
    assert (tx["is_new_location"] == (~tx.duplicated(["customer_id", "location"])).astype(int)).all()
    assert tx["customer_id"].nunique() == SAMPLE_CUSTOMERS


# ---- labels and episodes ----------------------------------------------------------------

def test_labels_are_consistent(tx, sample):
    fraud = tx["is_fraud"] == 1
    assert (fraud == (tx["fraud_episode_id"] > 0)).all()
    assert (fraud == (tx["fraud_type"] != "none")).all()
    assert (fraud == tx["fraud_stage"].isin(["first", "subsequent"])).all()
    assert (tx.loc[~fraud, "fraud_stage"] == "none").all()
    assert (tx.loc[fraud, "legit_context"] == "none").all()
    assert (tx.loc[tx["fraud_ring_id"] > 0, "fraud_type"] == "ring").all()
    assert (tx.loc[tx["fraud_type"] == "ring", "fraud_ring_id"] > 0).all()

    fr = tx[fraud]
    assert (fr.groupby("fraud_episode_id")["customer_id"].nunique() == 1).all()
    assert (fr.groupby("fraud_episode_id")["fraud_type"].nunique() == 1).all()
    firsts = fr[fr["fraud_stage"] == "first"]
    assert firsts["fraud_episode_id"].is_unique
    assert set(firsts["fraud_episode_id"]) == set(fr["fraud_episode_id"])
    earliest = fr.groupby("fraud_episode_id")["ts"].min()
    assert (firsts.set_index("fraud_episode_id")["ts"] == earliest.loc[firsts["fraud_episode_id"]]).all()

    ep = sample.episodes.set_index("fraud_episode_id")
    assert list(ep.index) == list(range(1, len(ep) + 1))
    assert (fr.groupby("fraud_episode_id").size() == ep["n_fraud_transactions"]).all()
    assert (earliest == pd.to_datetime(ep["first_fraud_time"])).all()


def test_only_unauthorized_transactions_are_fraud(tx, sample):
    fraud = tx["is_fraud"] == 1
    # every fraud row belongs to a generated (unauthorized) episode, nothing else is labelled
    assert set(tx.loc[fraud, "fraud_episode_id"]) == set(sample.episodes["fraud_episode_id"])
    # warning-period rows stay legitimate and come strictly before the episode's first fraud
    pre = tx[tx["is_precursor"] == 1]
    assert len(pre) > 0 and (pre["is_fraud"] == 0).all() and (pre["fraud_type"] == "none").all()
    ep = sample.episodes[sample.episodes["precursor_start"] != ""]
    for _, e in ep.iterrows():
        rows = pre[pre["customer_id"] == e["customer_id"]]
        rows = rows[(rows["ts"] >= pd.Timestamp(e["precursor_start"])) & (rows["ts"] < pd.Timestamp(e["first_fraud_time"]))]
        own = tx[(tx["customer_id"] == e["customer_id"]) & (tx["ts"] >= pd.Timestamp(e["precursor_start"]))
                 & (tx["ts"] < pd.Timestamp(e["first_fraud_time"]))]
        assert (own["is_fraud"] == 0).all() and (own["is_precursor"] == 1).all() and len(rows) == len(own)
    # unusual-but-legitimate rows exist and are labelled 0
    assert ((tx["legit_context"] != "none") & ~fraud).sum() > 0


def test_all_fraud_archetypes_exist(sample, tx):
    assert set(sample.episodes["fraud_type"]) == set(ARCHETYPES)
    assert set(tx.loc[tx["is_fraud"] == 1, "fraud_type"]) == set(ARCHETYPES)
    probes = tx[(tx["fraud_type"] == "card_testing_cashout")]
    assert (probes["amount"] <= 150).any()                # small test charges


# ---- rings and shared infrastructure ---------------------------------------------------

def test_fraud_rings_have_intentional_shared_infrastructure(tx, sample):
    ring_rows = tx[tx["fraud_ring_id"] > 0]
    rings = ring_rows.groupby("fraud_ring_id")
    assert rings.ngroups >= DEFAULT_CONFIG.min_rings
    for ring_id, rows in rings:
        members = rows["customer_id"].nunique()
        assert members >= DEFAULT_CONFIG.ring_size[0], ring_id
        shared = set()
        for col in ("device_id", "network_id", "merchant_id"):
            per_value = rows.groupby(col)["customer_id"].nunique()
            shared |= set(per_value[per_value >= 2].index)
        assert shared, f"ring {ring_id} has no infrastructure shared across members"
        span = rows["ts"].max() - rows["ts"].min()
        assert span <= pd.Timedelta(hours=DEFAULT_CONFIG.ring_wave_max_hours), ring_id
    # not every ring transaction carries the ring's shared device
    dev_members = ring_rows.groupby("device_id")["customer_id"].nunique()
    assert (ring_rows["device_id"].map(dev_members) == 1).any()
    # fraud devices are never shared between different rings or unrelated episodes (no accidental collisions)
    fr = tx[tx["is_fraud"] == 1]
    fraud_only = set(fr["device_id"]) - set(tx.loc[tx["is_fraud"] == 0, "device_id"])
    for dev in fraud_only:
        rows = fr[fr["device_id"] == dev]
        if rows["customer_id"].nunique() > 1:
            assert rows["fraud_ring_id"].nunique() == 1 and rows["fraud_ring_id"].iloc[0] > 0, dev


def test_legitimate_shared_infrastructure_exists(tx, sample):
    legit = tx[tx["is_fraud"] == 0]
    shared_devices = legit.groupby("device_id")["customer_id"].nunique()
    shared_devices = shared_devices[shared_devices >= 2]
    assert len(shared_devices) > 0                                    # households / borrowed phones
    assert sample.customers["household_id"].gt(0).sum() >= 2
    shared_networks = legit.groupby("network_id")["customer_id"].nunique()
    assert (shared_networks >= 2).sum() > 0                           # carriers, public wi-fi, VPN exits
    # the production ring rule (a device used by 2+ customers) would also flag purely legitimate devices
    all_dev = tx.groupby("device_id").agg(customers=("customer_id", "nunique"), fraud=("is_fraud", "max"))
    flagged = all_dev[all_dev["customers"] >= 2]
    assert (flagged["fraud"] == 0).any() and (flagged["fraud"] == 1).any()


# ---- warning signs overlap ------------------------------------------------------------

def _indicators(f: pd.DataFrame) -> dict:
    return {
        "hour_is_unusual": f["hour_is_unusual"] == 1,
        "is_new_device": f["is_new_device"] == 1,
        "is_new_location": f["is_new_location"] == 1,
        "is_foreign_location": f["is_foreign_location"] == 1,
        "category_is_unusual": f["category_is_unusual"] == 1,
        "failed_logins_24h>0": f["failed_logins_24h"] > 0,
        "txn_velocity_1h>0": f["txn_velocity_1h"] > 0,
        "amount_pct_of_avg>=300": f["amount_pct_of_avg"] >= 300,
    }


def test_warning_signs_occur_in_both_classes(features):
    fraud = features["is_fraud"] == 1
    for name, flag in _indicators(features).items():
        assert flag[fraud].any(), f"{name} never occurs on fraud"
        assert flag[~fraud].any(), f"{name} never occurs on legitimate transactions"
    # fraud that looks normal: no warning sign at all
    none = ~np.logical_or.reduce(list(_indicators(features).values()))
    assert (none & fraud).any()


def test_no_single_feature_perfectly_identifies_fraud(features):
    fraud = features["is_fraud"] == 1
    for col in FEATURE_COLUMNS:
        f, l = features.loc[fraud, col], features.loc[~fraud, col]
        assert not f.min() > l.max(), f"{col}: every fraud value is above every legitimate value"
        assert not f.max() < l.min(), f"{col}: every fraud value is below every legitimate value"
        assert not ((f == 1).all() and (l == 0).all()), col
    for name, flag in _indicators(features).items():
        assert not (flag == fraud).all(), name
    # v1's two-condition rule (unusual hour AND >= 188% of average) no longer separates the classes
    rule = (features["hour_is_unusual"] == 1) & (features["amount_pct_of_avg"] >= 188)
    assert features.loc[rule, "is_fraud"].mean() < 1 and rule[fraud].mean() < 1


def test_warning_period_signals_also_occur_for_legitimate_customers(tx, sample):
    pre = tx[tx["is_precursor"] == 1]
    assert (pre["failed_logins_24h"] > 0).any()            # credential attacks show up before the fraud
    victims = set(sample.episodes["customer_id"])
    innocent = tx[~tx["customer_id"].isin(victims)]
    assert (innocent["failed_logins_24h"] >= 2).any()      # forgotten passwords give the same pattern
    assert (innocent["legit_context"].str.contains("forgot_password")).any()


def test_legitimate_unusual_behaviour_is_generated(tx):
    contexts = set(tx.loc[tx["is_fraud"] == 0, "legit_context"].str.split("|").explode())
    assert set(schema.LEGIT_CONTEXTS) <= contexts
    legit = tx[tx["is_fraud"] == 0]
    assert (legit["location"].isin(FOREIGN_CITIES)).any()
    assert (legit["location"].isin(DOMESTIC_CITIES) & legit["legit_context"].str.contains("travel_domestic")).any()


# ---- failed logins ------------------------------------------------------------------------

def test_failed_logins_24h_counts_the_preceding_24_hours(tx, sample):
    ev = sample.login_failures.copy()
    ev["ts"] = pd.to_datetime(ev["timestamp"])
    by_c = {c: np.sort(g["ts"].to_numpy()) for c, g in ev.groupby("customer_id")}
    day = np.timedelta64(24, "h")
    checked = 0
    for c, rows in tx.groupby("customer_id"):
        et = by_c.get(c, np.array([], dtype="datetime64[ns]"))
        for ts, got in zip(rows["ts"].to_numpy(), rows["failed_logins_24h"].to_numpy()):
            expected = int(((et >= ts - day) & (et < ts)).sum())
            assert got == expected, (c, ts)
            checked += 1
    assert checked == len(tx)
    assert set(ev["source"]) == set(schema.LOGIN_FAILURE_SOURCES)
    # two transactions of one customer within an hour share most of their 24h window
    assert (tx["failed_logins_24h"] > 0).mean() < 0.2


def test_failed_login_window_boundaries():
    from synth_v2.generator import DAY, count_preceding
    t = 10 * DAY
    events = np.array([t - DAY - 1, t - DAY, t - 3600, t - 1, t, t + 5])
    assert count_preceding(events, np.array([t]))[0] == 3        # [t-24h, t): t-24h, t-1h, t-1s
    assert count_preceding(events, np.array([t + 6]))[0] == 4    # window [t+6s-24h, t+6s): t-1h, t-1s, t, t+5s
    assert count_preceding(np.array([], dtype=int), np.array([t]))[0] == 0


# ---- device ids --------------------------------------------------------------------------

def test_no_dev_unknown_giveaway(tx):
    assert not tx["device_id"].str.contains("UNKNOWN").any()
    assert tx["device_id"].map(lambda d: bool(NEUTRAL_DEVICE.match(d) or ORIGINAL_DEVICE.match(d))).all()
    fraud_devices = set(tx.loc[tx["is_fraud"] == 1, "device_id"]) - set(tx.loc[tx["is_fraud"] == 0, "device_id"])
    legit_new = set(tx.loc[(tx["is_fraud"] == 0), "device_id"]) - {d for d in tx["device_id"] if ORIGINAL_DEVICE.match(d)}
    assert fraud_devices and legit_new
    # fraud devices and legitimate non-original devices share one id format
    assert all(NEUTRAL_DEVICE.match(d) for d in fraud_devices) and all(NEUTRAL_DEVICE.match(d) for d in legit_new)


# ---- ground truth never reaches the models ---------------------------------------------

def test_ground_truth_never_enters_model_features(sample, features):
    from app.models.dnn_model import DNN_INPUT_COLUMNS
    assert not set(FEATURE_COLUMNS) & FORBIDDEN_FEATURE_COLUMNS
    assert not set(DNN_INPUT_COLUMNS) & FORBIDDEN_FEATURE_COLUMNS
    with pytest.raises(ValueError):
        assert_no_ground_truth(list(FEATURE_COLUMNS) + ["fraud_type"])

    # the model matrices are identical whether or not the metadata is present or scrambled
    subset_ids = sorted(features["customer_id"].unique())[:15]
    feat = features[features["customer_id"].isin(subset_ids)].reset_index(drop=True)
    scrambled = feat.copy()
    rng = np.random.default_rng(0)
    for col in schema.METADATA_COLUMNS:
        scrambled[col] = rng.permutation(scrambled[col].to_numpy())
    X1, y1, _ = build_sequences(feat)
    X2, y2, _ = build_sequences(scrambled)
    X3, y3, _ = build_sequences(feat.drop(columns=schema.METADATA_COLUMNS))
    assert np.array_equal(X1, X2) and np.array_equal(X1, X3) and np.array_equal(y1, y3)

    # feature values do not depend on metadata being passed through feature engineering
    raw = sample.transactions[sample.transactions["customer_id"].isin(subset_ids)]
    direct = build_point_features(raw)            # raw WITH metadata columns
    assert np.allclose(direct[FEATURE_COLUMNS].to_numpy(float), feat[FEATURE_COLUMNS].to_numpy(float))


# ---- evaluation grouping ----------------------------------------------------------------

def test_ring_and_household_groups_cannot_cross_evaluation_splits(sample, tx):
    ep = sample.episodes
    groups = episode_groups(ep)
    ring_eps = ep[ep["fraud_ring_id"] > 0]
    for ring_id, rows in ring_eps.groupby("fraud_ring_id"):
        assert groups.loc[rows["fraud_episode_id"]].nunique() == 1          # a ring is one group
    solo = ep[ep["fraud_ring_id"] == 0]
    assert groups.loc[solo["fraud_episode_id"]].is_unique

    spans = group_spans(ep)
    for g, row in spans.iterrows():
        rows = tx[tx["fraud_episode_id"].isin(row["episodes"])]
        assert rows["ts"].min() >= row["span_start"] and rows["ts"].max() <= row["span_end"]

    comp = customer_components(sample.customers, ep)
    for _, rows in ring_eps.groupby("fraud_ring_id"):
        assert comp.loc[rows["customer_id"]].nunique() == 1
    hh = sample.customers[sample.customers["household_id"] > 0]
    for _, rows in hh.groupby("household_id"):
        assert comp.loc[rows["customer_id"]].nunique() == 1

    # a component-level split keeps every ring and household inside one split
    rng = np.random.default_rng(1)
    split_of = {c: rng.choice(["train", "validation", "test"]) for c in sorted(set(comp))}
    labels = tx["customer_id"].map(comp).map(split_of)
    assert (tx.assign(s=labels)[tx["fraud_ring_id"] > 0].groupby("fraud_ring_id")["s"].nunique() == 1).all()
    assert (tx.assign(s=labels)[tx["fraud_episode_id"] > 0].groupby("fraud_episode_id")["s"].nunique() == 1).all()
