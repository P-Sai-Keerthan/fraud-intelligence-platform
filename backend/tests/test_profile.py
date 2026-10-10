"""GET /customer/{customer_id}/profile -- a customer's home device and city."""

import pandas as pd
import pytest

from app.config import FEATURES_CSV
from conftest import BASE_TIME


@pytest.mark.parametrize(
    "customer_id, device, city",
    [("CUST_0001", "DEV_0001_A", "Pune"), ("CUST_0002", "DEV_0002_A", "Chennai")],
)
def test_profile_known_customer(client, customer_id, device, city):
    r = client.get(f"/customer/{customer_id}/profile")
    assert r.status_code == 200
    body = r.json()
    assert body == {
        "customer_id": customer_id,
        "home_device": device,
        "home_location": city,
        "n_transactions": body["n_transactions"],
    }
    assert body["n_transactions"] > 0


def test_profile_matches_seed_data_for_every_customer(pipeline):
    # the seed generator gives every customer a primary device DEV_nnnn_A and
    # one home city; the profile must recover both for all 500 customers
    seed = pd.read_csv(FEATURES_CSV, usecols=["customer_id", "device_id", "location"])
    for customer_id, group in seed.groupby("customer_id"):
        profile = pipeline.get_customer_profile(customer_id)
        assert profile["home_device"] == customer_id.replace("CUST_", "DEV_") + "_A"
        assert profile["home_location"] == group["location"].value_counts().idxmax()
        assert profile["n_transactions"] == len(group)


def test_profile_is_not_hardcoded_to_hyderabad(pipeline):
    cities = {pipeline.get_customer_profile(c)["home_location"] for c in pipeline.known_customer_ids()}
    assert len(cities) > 1


def test_profile_unknown_customer_404(client):
    r = client.get("/customer/NOBODY/profile")
    assert r.status_code == 404
    assert "No transaction history" in r.json()["detail"]


def test_profile_does_not_create_customer(client, pipeline):
    client.get("/customer/NOBODY/profile")
    assert "NOBODY" not in pipeline.customer_histories


def test_profile_counts_scored_transactions(client, normal_txn):
    before = client.get("/customer/CUST_0001/profile").json()["n_transactions"]
    client.post("/predict", json=normal_txn)
    after = client.get("/customer/CUST_0001/profile").json()
    assert after["n_transactions"] == before + 1
    assert after["home_device"] == "DEV_0001_A"


def test_profile_for_new_customer_after_first_scan(client):
    client.post("/predict", json={
        "customer_id": "CUST_FRESH",
        "amount": 900,
        "merchant_category": "grocery",
        "device_id": "DEV_FRESH_1",
        "location": "Delhi",
        "timestamp": BASE_TIME.isoformat(),
    })
    body = client.get("/customer/CUST_FRESH/profile").json()
    assert (body["home_device"], body["home_location"], body["n_transactions"]) == ("DEV_FRESH_1", "Delhi", 1)


def test_normal_transaction_with_profile_defaults_is_not_new_or_foreign(client, pipeline):
    profile = client.get("/customer/CUST_0001/profile").json()
    r = client.post("/predict", json={
        "customer_id": "CUST_0001",
        "amount": 3800,
        "merchant_category": "fashion",
        "device_id": profile["home_device"],
        "location": profile["home_location"],
        "timestamp": BASE_TIME.isoformat(),
    })
    assert r.status_code == 200
    last = pipeline.customer_histories["CUST_0001"].iloc[-1]
    assert (last["is_new_device"], last["is_new_location"], last["is_foreign_location"]) == (0, 0, 0)
    assert r.json()["alert_level"] == "Low Risk"
