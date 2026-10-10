"""Customer history survives an application restart.

Scored transactions are persisted to the database; on startup the lifespan
hook replays them into the pipeline's in-memory histories (seed data + scored
transactions), which the models use for features such as "is this a new
device" and for customer profiles and fraud rings.

A restart is simulated by putting the in-memory histories back to what a
freshly started process has (seed data only) and then running the app's
lifespan startup again. One test does a full restart with a brand-new
pipeline instance that reloads the models from disk.
"""

from datetime import timedelta

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.inference_pipeline as pipeline_module
from app.features.feature_engineering import FEATURE_COLUMNS
from app.main import app
from conftest import BASE_TIME


def _post(client, customer_id, when, device, location, amount=3800, category="fashion"):
    r = client.post("/predict", json={
        "customer_id": customer_id,
        "amount": amount,
        "merchant_category": category,
        "device_id": device,
        "location": location,
        "timestamp": when.isoformat(),
    })
    assert r.status_code == 200
    return r.json()


def _simulate_restart(pipeline):
    """In-memory state of a freshly started process, then the startup lifespan."""
    pipeline.customer_histories.clear()
    pipeline.customer_histories.update(pipeline._seed_histories)
    with TestClient(app):
        pass


def _seed_len(pipeline, customer_id):
    seed = pipeline._seed_histories.get(customer_id)
    return 0 if seed is None else len(seed)


def test_restart_restores_scored_transactions(client, pipeline):
    posted = [_post(client, "CUST_0001", BASE_TIME + timedelta(hours=i), "DEV_0001_A", "Pune") for i in range(3)]
    live = pipeline.customer_histories["CUST_0001"]
    assert len(live) == _seed_len(pipeline, "CUST_0001") + 3

    _simulate_restart(pipeline)

    restored = pipeline.customer_histories["CUST_0001"]
    assert len(restored) == len(live)
    assert list(restored["transaction_id"].tail(3)) == [p["transaction_id"] for p in posted]


def test_restored_features_match_live_history(client, pipeline):
    _post(client, "CUST_0001", BASE_TIME, "DEV_0001_A", "Pune")
    _post(client, "CUST_0001", BASE_TIME + timedelta(minutes=20), "DEV_TABLET_7", "Pune", amount=9000)
    _post(client, "CUST_0001", BASE_TIME + timedelta(hours=2), "DEV_TABLET_7", "Mumbai", category="travel")
    live = pipeline.customer_histories["CUST_0001"].copy()

    _simulate_restart(pipeline)

    restored = pipeline.customer_histories["CUST_0001"]
    assert list(restored["transaction_id"]) == list(live["transaction_id"])
    np.testing.assert_allclose(
        restored[FEATURE_COLUMNS].to_numpy(dtype=float),
        live[FEATURE_COLUMNS].to_numpy(dtype=float),
    )


def test_restored_history_is_used_by_next_score(client, pipeline):
    # a device first seen in a transaction scored BEFORE the restart must not
    # count as "new" for a transaction scored AFTER it
    _post(client, "CUST_0001", BASE_TIME, "DEV_NEWPHONE_1", "Pune")
    assert pipeline.customer_histories["CUST_0001"].iloc[-1]["is_new_device"] == 1

    _simulate_restart(pipeline)
    _post(client, "CUST_0001", BASE_TIME + timedelta(hours=1), "DEV_NEWPHONE_1", "Pune")

    assert pipeline.customer_histories["CUST_0001"].iloc[-1]["is_new_device"] == 0


def test_repeated_restarts_do_not_duplicate(client, pipeline):
    for i in range(2):
        _post(client, "CUST_0002", BASE_TIME + timedelta(hours=i), "DEV_0002_A", "Chennai")
    expected = _seed_len(pipeline, "CUST_0002") + 2

    for _ in range(3):
        _simulate_restart(pipeline)
    # also run the startup again WITHOUT resetting memory first
    with TestClient(app):
        pass

    history = pipeline.customer_histories["CUST_0002"]
    assert len(history) == expected
    assert history["transaction_id"].is_unique


def test_restore_skips_duplicate_rows(pipeline):
    seed = pipeline._seed_histories["CUST_0001"]
    row = {
        "transaction_id": "TXN_DUPLICATE01", "customer_id": "CUST_0001",
        "timestamp": BASE_TIME, "amount": 100.0, "merchant_category": "fashion",
        "device_id": "DEV_0001_A", "location": "Pune", "failed_logins_24h": 0,
    }
    already_in_seed = {**row, "transaction_id": seed["transaction_id"].iloc[0]}

    restored = pipeline.restore_scored_history([row, dict(row), already_in_seed])

    assert restored == 1
    history = pipeline.customer_histories["CUST_0001"]
    assert len(history) == len(seed) + 1
    assert history["transaction_id"].is_unique


def test_restore_with_no_scored_rows_is_seed_only(client, pipeline):
    _post(client, "CUST_0001", BASE_TIME, "DEV_0001_A", "Pune")
    # restoring from an empty set of scored rows resets memory to seed data only
    assert pipeline.restore_scored_history([]) == 0
    assert len(pipeline.customer_histories["CUST_0001"]) == _seed_len(pipeline, "CUST_0001")


def test_out_of_order_scores_restored_in_time_order(client, pipeline):
    late = _post(client, "CUST_0003", BASE_TIME + timedelta(days=2), "DEV_0003_A", "Kolkata")
    early = _post(client, "CUST_0003", BASE_TIME + timedelta(days=1), "DEV_0003_A", "Kolkata")

    _simulate_restart(pipeline)

    tail = list(pipeline.customer_histories["CUST_0003"]["transaction_id"].tail(2))
    assert tail == [early["transaction_id"], late["transaction_id"]]


def test_customer_not_in_seed_data_is_restored(client, pipeline):
    _post(client, "CUST_BRAND_NEW", BASE_TIME, "DEV_BN_1", "Delhi")
    _simulate_restart(pipeline)

    assert "CUST_BRAND_NEW" in pipeline.known_customer_ids()
    profile = client.get("/customer/CUST_BRAND_NEW/profile").json()
    assert (profile["home_device"], profile["home_location"], profile["n_transactions"]) == ("DEV_BN_1", "Delhi", 1)


def test_profile_and_history_endpoints_unchanged_by_restart(client, pipeline):
    for i in range(3):
        _post(client, "CUST_0001", BASE_TIME + timedelta(hours=i), "DEV_0001_B", "Pune")
    profile_before = client.get("/customer/CUST_0001/profile").json()
    history_before = client.get("/customer/CUST_0001/history", params={"limit": 2}).json()

    _simulate_restart(pipeline)

    assert client.get("/customer/CUST_0001/profile").json() == profile_before
    assert client.get("/customer/CUST_0001/history", params={"limit": 2}).json() == history_before


def test_fraud_ring_survives_restart(client, pipeline):
    for cid in ("CUST_0010", "CUST_0011", "CUST_0012"):
        _post(client, cid, BASE_TIME, "DEV_SHARED_RESTART", "Mumbai", amount=5000, category="electronics")

    _simulate_restart(pipeline)

    rings = client.get("/fraud-rings", params={"min_customers": 3}).json()["rings"]
    ring = next(r for r in rings if r["identifier"] == "DEV_SHARED_RESTART")
    assert ring["customer_ids"] == ["CUST_0010", "CUST_0011", "CUST_0012"]


@pytest.mark.slow
def test_full_restart_with_new_pipeline_instance(client, pipeline):
    """A real cold start: a brand-new pipeline loads the models and seed data
    from disk, then the lifespan rebuilds its history from the database."""
    posted = [_post(client, "CUST_0001", BASE_TIME + timedelta(hours=i), "DEV_0001_A", "Pune") for i in range(2)]
    expected_ids = list(pipeline.customer_histories["CUST_0001"]["transaction_id"])

    original = pipeline_module._pipeline_instance
    try:
        pipeline_module._pipeline_instance = fresh = pipeline_module.FraudIntelligencePipeline()
        assert len(fresh.customer_histories["CUST_0001"]) == len(expected_ids) - 2  # seed only
        with TestClient(app) as restarted:
            assert list(fresh.customer_histories["CUST_0001"]["transaction_id"]) == expected_ids
            assert restarted.get("/customer/CUST_0001/profile").json()["n_transactions"] == len(expected_ids)
            timeline = restarted.get("/customer/CUST_0001/history").json()["timeline"]
            assert [p["transaction_id"] for p in timeline] == [posted[1]["transaction_id"], posted[0]["transaction_id"]]
    finally:
        pipeline_module._pipeline_instance = original
