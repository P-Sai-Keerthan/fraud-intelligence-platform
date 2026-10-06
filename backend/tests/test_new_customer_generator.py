"""The generator's new-customer extension (data/v2/synth_v2, version 2.1.0).

Off by default: with the two settings at 0 the generator must produce exactly
the 2.0.1 files. The checksums below were recorded from the 2.0.1 generator
before the extension was added. Everything is generated in memory or in a
temp directory; nothing is written to data/.
"""

import hashlib
import json
import sys

import pandas as pd
import pytest

from app.config import PROJECT_ROOT
from app.features.feature_engineering import FEATURE_COLUMNS
from app.features.ground_truth import GROUND_TRUTH_COLUMNS

V2_DIR = PROJECT_ROOT / "data" / "v2"
sys.path.insert(0, str(V2_DIR))

from synth_v2 import DEFAULT_CONFIG, generate  # noqa: E402
from synth_v2 import schema  # noqa: E402
from synth_v2.config import GENERATOR_VERSION, NEW_CUSTOMER_FIELDS, NEW_CUSTOMER_GENERATOR_VERSION  # noqa: E402
from synth_v2.features import build_features_table  # noqa: E402

# SHA-256 of the files written by generator 2.0.1 (`--skip-features`), recorded before the extension existed
V201 = {
    (42, 60): {
        "transactions.csv": "d5eebd3af699a6217cf10ab460feb183024c7d832b64977e57792722f3c1b176",
        "customers.csv": "5998cac00dc53b36cf44434f27f4f6c31701707553835d0f700e0ef52f7a37d6",
        "episodes.csv": "72a1ae29dc5fda3f41a0bba6ada148becdaab130e9956d0e5d5c8b06b5b35161",
        "login_failures.csv": "234b03afda9e707dcf5acff28ac210f166a74a7f9392fcc816e83b1d989fa579",
    },
    (7, 80): {
        "transactions.csv": "2e900aca7f817bbfe43069e9a247f15dfa840ea3d9f239cbe126647ffc5682a3",
        "customers.csv": "be8a8c903b269a82ffbfa86f46ded8105274ef40223b61fe63f978144f80a17e",
        "episodes.csv": "e185217c9bff278e19ce6ee83f1436804505bc504e459d8993bf616fcc07ad1d",
        "login_failures.csv": "10dc84a5334400fc10f71e884de77210ff6eb6e80e6852266a3b3be8f246ca64",
    },
}
V201_CONFIG_KEYS = 52            # settings recorded in a 2.0.1 manifest

BASE = DEFAULT_CONFIG.with_overrides(seed=7, n_customers=80)
EXT = BASE.with_overrides(late_joiner_share=0.25, new_customer_fraud_episodes=8)


@pytest.fixture(scope="module")
def base():
    return generate(BASE)


@pytest.fixture(scope="module")
def ext():
    return generate(EXT)


def _prior(tx: pd.DataFrame) -> pd.Series:
    return tx.groupby("customer_id").cumcount()


# ---- off by default: exactly generator 2.0.1 --------------------------------------------------------

def test_extension_is_off_by_default():
    assert DEFAULT_CONFIG.late_joiner_share == 0.0 and DEFAULT_CONFIG.new_customer_fraud_episodes == 0
    assert not DEFAULT_CONFIG.new_customer_extension
    assert GENERATOR_VERSION == "2.0.1" and DEFAULT_CONFIG.generator_version == "2.0.1"
    recorded = DEFAULT_CONFIG.to_dict()
    assert not set(recorded) & set(NEW_CUSTOMER_FIELDS) and len(recorded) == V201_CONFIG_KEYS
    assert schema.EPISODE_COLUMNS[-1] == "n_fraud_transactions" and "join_date" not in schema.CUSTOMER_COLUMNS


@pytest.mark.parametrize("seed, customers", sorted(V201))
def test_default_output_is_byte_identical_to_2_0_1(seed, customers, tmp_path):
    import generate as cli
    cli.main(["--seed", str(seed), "--customers", str(customers), "--out", str(tmp_path), "--skip-features"])
    for fname, sha in V201[(seed, customers)].items():
        assert hashlib.sha256((tmp_path / fname).read_bytes()).hexdigest() == sha, fname
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["generator_version"] == "2.0.1"
    assert manifest["command"] == f"python data/v2/generate.py --seed {seed} --customers {customers} --skip-features"
    assert len(manifest["config"]) == V201_CONFIG_KEYS and not set(manifest["config"]) & set(NEW_CUSTOMER_FIELDS)
    assert "late_joiners" not in manifest["summary"]
    assert manifest["files"]["episodes.csv"]["columns"] == schema.EPISODE_COLUMNS
    assert manifest["files"]["customers.csv"]["columns"] == schema.CUSTOMER_COLUMNS


def test_tracked_seed_42_manifest_still_describes_the_default_generator():
    manifest = json.loads((V2_DIR / "manifest.json").read_text())
    assert manifest["generator_version"] == DEFAULT_CONFIG.generator_version
    assert manifest["config"] == json.loads(json.dumps(DEFAULT_CONFIG.to_dict()))


# ---- the extension -----------------------------------------------------------------------------

def test_extension_reports_its_version_and_settings():
    assert EXT.new_customer_extension and EXT.generator_version == NEW_CUSTOMER_GENERATOR_VERSION == "2.1.0"
    assert {k: EXT.to_dict()[k] for k in NEW_CUSTOMER_FIELDS} == {
        "late_joiner_share": 0.25, "new_customer_fraud_episodes": 8, "late_join_days": (14, 140)}
    assert BASE.with_overrides(late_joiner_share=0.1).generator_version == "2.1.0"


def test_new_customer_episodes_start_within_the_first_ten_transactions(ext):
    ep, tx = ext.episodes, ext.transactions
    assert list(ep.columns) == schema.EPISODE_COLUMNS + schema.NEW_CUSTOMER_EPISODE_COLUMNS
    assert list(ext.customers.columns) == schema.CUSTOMER_COLUMNS + schema.NEW_CUSTOMER_CUSTOMER_COLUMNS
    new = ep[ep["prior_transactions_at_first_fraud"] < 10]
    assert len(new) == 8 and (new["fraud_ring_id"] == 0).all() and (new["precursor_start"] == "").all()
    assert new["prior_transactions_at_first_fraud"].between(0, 9).all()
    assert new["customer_id"].is_unique
    # the recorded count is the real one: the customer's earlier rows at the episode's first fraud row
    first_rows = tx[tx["fraud_stage"] == "first"].assign(n_prior=_prior(tx)).set_index("fraud_episode_id")
    for _, e in ep.iterrows():
        assert first_rows.loc[e["fraud_episode_id"], "n_prior"] == e["prior_transactions_at_first_fraud"]
    # no other fraud lands in a customer's first ten transactions
    early_fraud = tx[(tx["is_fraud"] == 1) & (_prior(tx) < 10)]
    assert len(early_fraud) >= 8 and set(early_fraud["fraud_episode_id"]) <= set(new["fraud_episode_id"])


def test_late_joiners_have_no_activity_before_joining_and_no_other_fraud(base, ext):
    cust = ext.customers
    start = cust["join_date"].min()
    late = cust[cust["join_date"] > start]
    assert len(late) == round(0.25 * 80)
    first_seen = ext.transactions.groupby("customer_id")["timestamp"].min()
    new = ext.episodes[ext.episodes["prior_transactions_at_first_fraud"] < 10]
    for _, c in late.iterrows():
        if c["customer_id"] in set(new["customer_id"]):      # a fraud that is the first transaction may precede the first legitimate one
            legit_first = ext.transactions.query("customer_id == @c.customer_id and is_fraud == 0")["timestamp"].min()
            assert legit_first >= c["join_date"]
        else:
            assert first_seen[c["customer_id"]] >= c["join_date"]
    assert pd.Timestamp(late["join_date"].min()) >= pd.Timestamp("2026-01-12") + pd.Timedelta(days=14)
    assert pd.Timestamp(late["join_date"].max()) <= pd.Timestamp("2026-01-12") + pd.Timedelta(days=140)
    assert set(new["customer_id"]) <= set(late["customer_id"])
    assert not set(late["customer_id"]) & set(base.episodes["customer_id"])       # chosen among customers without fraud


def test_everything_else_is_exactly_the_base_dataset(base, ext):
    late = set(ext.customers.loc[ext.customers["join_date"] > ext.customers["join_date"].min(), "customer_id"])
    # the original episodes are unchanged
    key = ["customer_id", "fraud_type", "fraud_ring_id", "precursor_start", "first_fraud_time", "last_fraud_time", "n_fraud_transactions"]
    old = ext.episodes[ext.episodes["prior_transactions_at_first_fraud"] >= 10]
    assert sorted(map(tuple, old[key].to_numpy())) == sorted(map(tuple, base.episodes[key].to_numpy()))
    # every other customer's transactions are identical (ids are global row numbers, so they are not compared)
    cols = [c for c in schema.RAW_COLUMNS if c not in ("transaction_id", "fraud_episode_id")]
    a = base.transactions[~base.transactions["customer_id"].isin(late)][cols].reset_index(drop=True)
    b = ext.transactions[~ext.transactions["customer_id"].isin(late)][cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(base.customers, ext.customers[schema.CUSTOMER_COLUMNS])
    # late joiners only lose their history before joining
    assert len(ext.transactions) < len(base.transactions) + int(ext.episodes["n_fraud_transactions"].sum())


def test_extension_is_deterministic(ext):
    again = generate(EXT)
    for name in ("transactions", "customers", "episodes", "login_failures"):
        pd.testing.assert_frame_equal(getattr(ext, name), getattr(again, name))


def test_extension_adds_no_transaction_column_and_no_model_feature(ext):
    assert list(ext.transactions.columns) == schema.RAW_COLUMNS
    features = build_features_table(ext.transactions)
    assert list(features.columns) == schema.FEATURES_CSV_COLUMNS
    new_columns = set(schema.NEW_CUSTOMER_EPISODE_COLUMNS + schema.NEW_CUSTOMER_CUSTOMER_COLUMNS)
    assert not new_columns & set(features.columns) and not new_columns & set(FEATURE_COLUMNS)
    assert schema.METADATA_COLUMNS == list(GROUND_TRUTH_COLUMNS)


def test_invalid_extension_settings_are_refused():
    with pytest.raises(ValueError, match="at least as many late"):
        generate(BASE.with_overrides(late_joiner_share=0.05, new_customer_fraud_episodes=10))
    with pytest.raises(ValueError, match="no fraud planned"):
        generate(BASE.with_overrides(late_joiner_share=0.99))
    with pytest.raises(ValueError, match="late_join_days"):
        generate(BASE.with_overrides(late_joiner_share=0.1, late_join_days=(0, 500)))


def test_cli_records_the_extension(tmp_path):
    import generate as cli
    cli.main(["--seed", "7", "--customers", "80", "--out", str(tmp_path), "--skip-features",
              "--late-joiner-share", "0.25", "--new-customer-fraud-episodes", "8"])
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["generator_version"] == "2.1.0"
    assert manifest["command"] == ("python data/v2/generate.py --seed 7 --customers 80 --late-joiner-share 0.25 "
                                   "--new-customer-fraud-episodes 8 --skip-features")
    assert manifest["config"]["late_joiner_share"] == 0.25 and manifest["config"]["new_customer_fraud_episodes"] == 8
    assert manifest["summary"]["late_joiners"] == 20 and manifest["summary"]["new_customer_fraud_episodes"] == 8
    assert sum(manifest["summary"]["new_customer_fraud_episodes_by_prior_transactions"].values()) == 8
    assert manifest["files"]["episodes.csv"]["columns"][-1] == "prior_transactions_at_first_fraud"
    for fname, info in manifest["files"].items():
        assert hashlib.sha256((tmp_path / fname).read_bytes()).hexdigest() == info["sha256"]
