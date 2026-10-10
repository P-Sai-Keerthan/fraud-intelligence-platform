"""Data sanity audit for generated v2 data (data/v2/synth_v2/sanity.py) and the
generator fixes found by it in Step 4C-2b. Small samples only, written to a
temp directory; nothing is written to data/v2."""

import shutil
import sys

import pandas as pd
import pytest

from app.config import PROJECT_ROOT

V2_DIR = PROJECT_ROOT / "data" / "v2"
sys.path.insert(0, str(V2_DIR))

import generate as cli  # noqa: E402
from synth_v2 import DEFAULT_CONFIG, build_plan, generate  # noqa: E402
from synth_v2.config import FOREIGN_CITIES  # noqa: E402
from synth_v2.sanity import audit, render  # noqa: E402

SECTIONS = ["## Verdict", "## 1. Files and reproducibility", "## 2. Dataset structure", "## 3. Class balance",
            "## 4. Warning-sign overlap", "## 5. Legitimate unusual behaviour", "## 6. Fraud archetypes",
            "## 7. Fraud rings", "## 8. Failed-login correctness", "## 9. Metadata / leakage audit",
            "## 10. Giveaway checks", "## 11. Integrity checks", "## 12. Production feature compatibility"]


@pytest.fixture(scope="module")
def sample_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("v2_sample")
    cli.main(["--customers", "60", "--out", str(out)])
    return out


@pytest.fixture(scope="module")
def results(sample_dir):
    return audit(sample_dir, recompute_features=True)


def test_sample_has_no_anomalies(results):
    assert results["anomalies"] == []
    assert results["failed_logins"]["mismatches"] == 0
    assert results["compatibility"]["max_abs_difference_vs_features_file"] < 1e-6
    assert results["leakage"]["forbidden_in_feature_columns"] == []
    assert results["leakage"]["forbidden_in_dnn_columns"] == []


def test_report_renders_every_section_and_is_reproducible(results, sample_dir):
    import json
    manifest = json.loads((sample_dir / "manifest.json").read_text())
    text = render(results, manifest)
    for heading in SECTIONS:
        assert heading in text, heading
    assert "No anomalies found" in text
    assert render(audit(sample_dir, recompute_features=False), manifest).split("## 12.")[0] == text.split("## 12.")[0]


def test_audit_detects_planted_problems(sample_dir, tmp_path):
    bad = tmp_path / "bad"
    shutil.copytree(sample_dir, bad)
    tx = pd.read_csv(bad / "transactions.csv", keep_default_na=False)
    legit_idx = tx.index[tx["is_fraud"] == 0]
    tx.loc[legit_idx[0], "is_precursor"] = 1                         # warning row outside any window
    tx.loc[legit_idx[1], "failed_logins_24h"] += 3                    # wrong 24h count
    fraud_idx = tx.index[tx["is_fraud"] == 1]
    tx.loc[fraud_idx[0], "device_id"] = "DEV_UNKNOWN_1234"            # v1 giveaway
    tx.to_csv(bad / "transactions.csv", index=False, lineterminator="\n")
    found = " | ".join(audit(bad, recompute_features=False)["anomalies"])
    assert "warning_rows_not_before_their_fraud" in found
    assert "failed_logins_24h" in found
    assert "DEV_UNKNOWN" in found


def test_manifest_is_identical_for_different_output_directories(tmp_path):
    cli.main(["--customers", "12", "--out", str(tmp_path / "a"), "--skip-features"])
    cli.main(["--customers", "12", "--out", str(tmp_path / "b"), "--skip-features"])
    assert (tmp_path / "a" / "manifest.json").read_bytes() == (tmp_path / "b" / "manifest.json").read_bytes()


def test_no_city_appears_only_on_fraud():
    # a small population has fewer natural foreign travellers than foreign cities;
    # the generator adds travellers so every foreign city has legitimate visits
    data = generate(DEFAULT_CONFIG.with_overrides(n_customers=40))
    tx = data.transactions
    legit_cities = set(tx.loc[tx["is_fraud"] == 0, "location"])
    assert set(FOREIGN_CITIES) <= legit_cities
    assert set(tx.loc[tx["is_fraud"] == 1, "location"]) <= legit_cities


def test_every_planned_trip_has_a_transaction():
    cfg = DEFAULT_CONFIG.with_overrides(n_customers=40)
    plan, data = build_plan(cfg), generate(cfg)
    tx = data.transactions.assign(ts=pd.to_datetime(data.transactions["timestamp"]))
    start = pd.Timestamp(cfg.start_date)
    for c in plan.customers:
        rows = tx[(tx["customer_id"] == c.customer_id) & (tx["is_fraud"] == 0)]
        for t0, t1, city, kind in c.trips:
            inside = rows[(rows["ts"] >= start + pd.Timedelta(seconds=t0)) & (rows["ts"] < start + pd.Timedelta(seconds=t1))]
            assert len(inside) > 0 and (inside["location"] == city).all(), (c.customer_id, kind, city)
