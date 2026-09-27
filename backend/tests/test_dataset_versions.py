"""Dataset-version selection for the evaluation (app/evaluation/datasets.py) and
the v2 grouping it feeds into the splits. v2 is tested on a small generated
sample in a temp directory, so these tests do not need data/v2 to exist.
Nothing here trains a model or writes into data/ or models/."""

import json
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from app import config
from app.evaluation import datasets
from app.evaluation.datasets import (
    DEFAULT_DATASET_VERSION, MODEL_FRAME_COLUMNS, dataset_info, load_evaluation_data, resolve_dataset,
)
from app.evaluation.run import evaluate_split, feature_version, main as run_main
from app.evaluation.split import SPLITS, build_customer_split, build_time_split, canonical_order
from app.evaluation.windows import assert_past_only, build_windows
from app.features.feature_engineering import FEATURE_COLUMNS
from app.features.ground_truth import GROUND_TRUTH_COLUMNS

PRODUCTION_FEATURES = [
    "amount_zscore", "hour_is_unusual", "is_new_device", "is_new_location", "is_foreign_location",
    "failed_logins_24h", "category_is_unusual", "txn_velocity_1h", "amount_pct_of_avg",
]


@pytest.fixture(scope="module")
def v2_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("v2_eval_sample")
    subprocess.run([sys.executable, str(config.DATA_V2_DIR / "generate.py"), "--customers", "80", "--out", str(out)],
                   check=True, capture_output=True)
    return out


@pytest.fixture(scope="module")
def v2(v2_dir):
    return load_evaluation_data(resolve_dataset("v2").with_data_dir(v2_dir))


@pytest.fixture(scope="module")
def v2_splits(v2):
    labels, episode_id, definition = build_time_split(v2.frame, v2.grouping)
    return labels, episode_id, definition


@pytest.fixture(scope="module")
def v1():
    return load_evaluation_data(resolve_dataset("v1"))


# ---- dataset selection ------------------------------------------------------------------

def test_default_is_v1():
    assert DEFAULT_DATASET_VERSION == "v1"
    assert resolve_dataset() is resolve_dataset("v1") is resolve_dataset(None)


def test_v1_resolves_to_the_existing_paths():
    spec = resolve_dataset("v1")
    assert spec.features_csv == config.FEATURES_CSV
    assert spec.report_path == config.EVAL_REPORT_PATH              # the file GET /metrics serves
    assert spec.time_split_path == config.EVAL_TIME_SPLIT_PATH
    assert spec.customer_split_path == config.EVAL_CUSTOMER_SPLIT_PATH
    assert spec.models_dir == config.EVALUATION_DIR / "time_split"
    assert not spec.has_metadata and spec.legacy_comparison


def test_v2_resolves_to_data_v2_and_its_own_output():
    spec = resolve_dataset("v2")
    assert spec.features_csv == config.DATA_V2_DIR / "transactions_with_features.csv"
    assert spec.episodes_csv == config.DATA_V2_DIR / "episodes.csv"
    assert spec.output_dir == config.EVALUATION_V2_DIR
    # v2 output can never replace the v1 report, splits or evaluation models
    v1 = resolve_dataset("v1")
    for attr in ("features_csv", "report_path", "time_split_path", "customer_split_path", "models_dir"):
        assert getattr(spec, attr) != getattr(v1, attr), attr
    assert spec.has_metadata and not spec.legacy_comparison


@pytest.mark.parametrize("bad", ["v3", "V2", "", "latest", "2"])
def test_invalid_version_fails_clearly(bad):
    with pytest.raises(ValueError, match="unknown dataset version"):
        resolve_dataset(bad)
    with pytest.raises(ValueError, match="unknown dataset version"):
        run_main(dataset_version=bad)                     # fails before loading data or TensorFlow work


def test_cli_rejects_unknown_dataset():
    r = subprocess.run([sys.executable, "-m", "app.evaluation.run", "--dataset", "v3"],
                       cwd=config.BACKEND_DIR, capture_output=True, text=True)
    assert r.returncode != 0 and "invalid choice" in r.stderr


def test_missing_v2_files_raise_instead_of_falling_back(tmp_path):
    spec = resolve_dataset("v2").with_data_dir(tmp_path)
    with pytest.raises(FileNotFoundError, match="generate.py"):
        load_evaluation_data(spec)


def test_output_dir_override_moves_every_output(tmp_path):
    spec = resolve_dataset("v1").with_output_dir(tmp_path)
    assert spec.features_csv == config.FEATURES_CSV
    for attr in ("report_path", "time_split_path", "customer_split_path", "models_dir"):
        assert tmp_path in getattr(spec, attr).parents or getattr(spec, attr).parent == tmp_path, attr


# ---- feature isolation --------------------------------------------------------------------

def test_feature_set_is_the_production_nine():
    assert list(FEATURE_COLUMNS) == PRODUCTION_FEATURES
    assert feature_version()["columns"] == PRODUCTION_FEATURES and feature_version()["count"] == 9


def test_v2_model_frame_has_no_metadata(v2):
    assert list(v2.frame.columns) == MODEL_FRAME_COLUMNS
    assert not set(v2.frame.columns) & set(GROUND_TRUTH_COLUMNS)
    X, y, target = build_windows(v2.frame)
    assert X.shape[1:] == (10, 9)
    assert_past_only(v2.frame, target)


def test_v1_model_frame_is_the_v1_file_unchanged(v1):
    assert list(v1.frame.columns) == MODEL_FRAME_COLUMNS
    assert v1.metadata is None and v1.grouping is None
    reference = canonical_order(pd.read_csv(config.FEATURES_CSV))
    pd.testing.assert_frame_equal(v1.frame, reference)


def test_evaluate_split_refuses_a_frame_with_metadata(v2):
    leaky = v2.frame.assign(fraud_type=v2.metadata["fraud_type"].to_numpy())
    labels = pd.Series("train", index=leaky.index)
    with pytest.raises(ValueError, match="ground-truth"):
        evaluate_split(leaky, labels, v2.grouping.episode_id, build_windows(v2.frame), log=lambda m: None)


def test_unexpected_columns_are_rejected(v2_dir, tmp_path):
    bad = tmp_path / "bad"
    shutil.copytree(v2_dir, bad)
    f = pd.read_csv(bad / "transactions_with_features.csv", keep_default_na=False)
    f["leak"] = f["is_fraud"]
    f.to_csv(bad / "transactions_with_features.csv", index=False)
    with pytest.raises(ValueError, match="unexpected columns"):
        load_evaluation_data(resolve_dataset("v2").with_data_dir(bad))


# ---- metadata kept separately ----------------------------------------------------------------

def test_metadata_is_kept_row_aligned_for_analysis(v2):
    m = v2.metadata
    assert list(m.columns) == ["transaction_id", *GROUND_TRUTH_COLUMNS]
    assert (m["transaction_id"].to_numpy() == v2.frame["transaction_id"].to_numpy()).all()
    fraud = v2.frame["is_fraud"] == 1
    assert (m.loc[fraud, "fraud_type"] != "none").all() and (m.loc[~fraud, "fraud_type"] == "none").all()
    assert (m.loc[fraud, "fraud_episode_id"] > 0).all()
    assert m["fraud_ring_id"].max() > 0 and m["is_precursor"].sum() > 0


def test_metadata_does_not_change_model_input(v2_dir, v2, tmp_path):
    scrambled = tmp_path / "scrambled"
    shutil.copytree(v2_dir, scrambled)
    f = pd.read_csv(scrambled / "transactions_with_features.csv", keep_default_na=False)
    rng = np.random.default_rng(0)
    for col in ("fraud_type", "legit_context", "customer_segment", "merchant_id", "network_id", "fraud_stage"):
        f[col] = rng.permutation(f[col].to_numpy())
    f.to_csv(scrambled / "transactions_with_features.csv", index=False)
    other = load_evaluation_data(resolve_dataset("v2").with_data_dir(scrambled))
    pd.testing.assert_frame_equal(other.frame, v2.frame)
    np.testing.assert_array_equal(build_windows(other.frame)[0], build_windows(v2.frame)[0])


def test_dataset_info_records_version_and_checksums(v1, v2):
    info = dataset_info(v1)
    assert info["version"] == "v1" and info["transactions"] == len(v1.frame)
    import hashlib
    assert info["sha256"] == hashlib.sha256(config.FEATURES_CSV.read_bytes()).hexdigest()
    info2 = dataset_info(v2)
    assert info2["version"] == "v2" and info2["matches_manifest"] is True
    assert info2["generator_version"] == v2.manifest["generator_version"]


# ---- grouping ---------------------------------------------------------------------------------

def test_v2_episodes_come_from_the_dataset(v2, v2_splits):
    _, episode_id, definition = v2_splits
    assert (episode_id.to_numpy() == v2.metadata["fraud_episode_id"].to_numpy()).all()
    assert definition["episodes_total"] == len(v2.grouping.episodes)
    rings = v2.grouping.episodes["fraud_ring_id"]
    n_groups = int((rings == 0).sum() + rings[rings > 0].nunique())
    assert definition["episode_groups_total"] == n_groups == len(v2.grouping.units)


def test_ring_episodes_and_warning_periods_stay_in_one_split(v2, v2_splits):
    labels, episode_id, d = v2_splits
    meta = v2.metadata
    ring = meta["fraud_ring_id"]
    per_ring = labels[ring > 0].groupby(ring[ring > 0]).nunique()
    assert len(per_ring) >= 2 and (per_ring == 1).all()
    per_episode = labels[episode_id > 0].groupby(episode_id[episode_id > 0]).nunique()
    assert (per_episode == 1).all()
    # a warning period is in the same split as the fraud it precedes
    ep = v2.grouping.episodes
    frame = v2.frame
    for _, e in ep[ep["precursor_start"] != ""].iterrows():
        rows = (frame["customer_id"] == e["customer_id"]) & (frame["timestamp"] >= pd.Timestamp(e["precursor_start"])) \
            & (frame["timestamp"] <= pd.Timestamp(e["last_fraud_time"]))
        assert labels[rows].nunique() == 1


def test_grouped_time_split_is_chronological_and_moves_only_later(v2, v2_splits):
    labels, _, d = v2_splits
    ts = v2.frame["timestamp"]
    b1, b2 = pd.Timestamp(d["train_before"]), pd.Timestamp(d["validation_before"])
    assert ts[labels == "train"].max() < b1
    by_time = np.where(ts < b1, 0, np.where(ts < b2, 1, 2))
    rank = labels.map({s: i for i, s in enumerate(SPLITS)}).to_numpy()
    assert (rank >= by_time).all()
    assert int((rank != by_time).sum()) == sum(m["rows_moved"] for m in d["episodes_moved"])
    assert all(d["splits"][s]["fraud_episodes"] > 0 for s in SPLITS)


def test_customer_split_keeps_rings_and_households_together(v2_dir, v2, v2_splits):
    _, episode_id, _ = v2_splits
    labels, d = build_customer_split(v2.frame, episode_id, v2.grouping)
    frame = v2.frame
    assert (labels.groupby(frame["customer_id"]).nunique() == 1).all()
    split_of = labels.groupby(frame["customer_id"]).first()
    ep = v2.grouping.episodes
    for _, members in ep[ep["fraud_ring_id"] > 0].groupby("fraud_ring_id")["customer_id"]:
        assert split_of.loc[members].nunique() == 1
    customers = pd.read_csv(v2_dir / "customers.csv")
    households = customers[customers["household_id"] > 0].groupby("household_id")["customer_id"]
    assert households.ngroups > 0
    for _, members in households:
        assert split_of.loc[members].nunique() == 1
    assert d["components"] == v2.grouping.component_of.nunique() < frame["customer_id"].nunique()
    assert all(d["splits"][s]["fraud_episodes"] > 0 for s in SPLITS)


def test_v1_splits_are_unchanged(v1):
    labels, episode_id, definition = build_time_split(v1.frame)
    assert definition == json.loads(config.EVAL_TIME_SPLIT_PATH.read_text())
    assert definition["episode_rule"].startswith("a customer's fraud transactions linked")
    _, cust_def = build_customer_split(v1.frame, episode_id)
    assert cust_def == json.loads(config.EVAL_CUSTOMER_SPLIT_PATH.read_text())


def test_metrics_endpoint_still_reads_the_v1_report():
    from app.models import evaluate
    assert evaluate.EVAL_REPORT_PATH == resolve_dataset("v1").report_path == config.EVAL_REPORT_PATH
    assert "v2" not in str(evaluate.EVAL_REPORT_PATH)
    assert datasets.DATASETS["v2"].report_path.parent == config.EVALUATION_V2_DIR
