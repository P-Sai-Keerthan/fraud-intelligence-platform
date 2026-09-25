"""GET /customer/{customer_id}/history -- fraud evolution timeline."""

from datetime import timedelta

import pytest

from conftest import BASE_TIME


def _post_at(client, txn, when):
    r = client.post("/predict", json={**txn, "timestamp": when.isoformat()})
    assert r.status_code == 200
    return r.json()


def test_history_404_before_any_scoring(client):
    # seed/training history is NOT served by this endpoint -- only scored txns
    r = client.get("/customer/CUST_0001/history")
    assert r.status_code == 404
    assert "No SCORED transactions" in r.json()["detail"]


def test_history_404_for_unknown_customer(client):
    assert client.get("/customer/NOBODY/history").status_code == 404


def test_history_returns_scored_transactions_in_time_order(client, normal_txn, suspicious_txn):
    first = _post_at(client, normal_txn, BASE_TIME)
    second = _post_at(client, suspicious_txn, BASE_TIME + timedelta(hours=1))

    r = client.get("/customer/CUST_0001/history")
    assert r.status_code == 200
    body = r.json()
    assert body["customer_id"] == "CUST_0001"
    assert body["n_transactions"] == 2
    timeline = body["timeline"]
    assert [p["transaction_id"] for p in timeline] == [first["transaction_id"], second["transaction_id"]]
    for point, pred in zip(timeline, (first, second)):
        assert set(point) == {"transaction_id", "timestamp", "risk_score", "fraud_probability", "alert_level"}
        assert point["risk_score"] == pred["risk_score"]
        assert point["fraud_probability"] == pred["fraud_probability"]
        assert point["alert_level"] == pred["alert_level"]


def test_history_is_per_customer(client, normal_txn):
    _post_at(client, normal_txn, BASE_TIME)
    other = {**normal_txn, "customer_id": "CUST_0002", "device_id": "DEV_0002_A", "location": "Chennai"}
    _post_at(client, other, BASE_TIME)

    assert client.get("/customer/CUST_0001/history").json()["n_transactions"] == 1
    assert client.get("/customer/CUST_0002/history").json()["n_transactions"] == 1


def test_history_limit_caps_results(client, normal_txn):
    for i in range(3):
        _post_at(client, normal_txn, BASE_TIME + timedelta(hours=i))
    body = client.get("/customer/CUST_0001/history", params={"limit": 2}).json()
    assert body["n_transactions"] == 2


@pytest.mark.xfail(
    strict=True,
    reason="Known bug (plan step 2): history orders ascending then applies the limit, "
           "so it returns the OLDEST N scored transactions instead of the latest N.",
)
def test_history_limit_returns_most_recent(client, normal_txn):
    posted = [_post_at(client, normal_txn, BASE_TIME + timedelta(hours=i)) for i in range(3)]
    body = client.get("/customer/CUST_0001/history", params={"limit": 2}).json()
    ids = [p["transaction_id"] for p in body["timeline"]]
    assert ids == [posted[1]["transaction_id"], posted[2]["transaction_id"]]
