"""GET /customer/{customer_id}/history -- fraud evolution timeline."""

from datetime import timedelta

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


def test_history_returns_scored_transactions_newest_first(client, normal_txn, suspicious_txn):
    first = _post_at(client, normal_txn, BASE_TIME)
    second = _post_at(client, suspicious_txn, BASE_TIME + timedelta(hours=1))

    r = client.get("/customer/CUST_0001/history")
    assert r.status_code == 200
    body = r.json()
    assert body["customer_id"] == "CUST_0001"
    assert body["n_transactions"] == 2
    timeline = body["timeline"]
    assert [p["transaction_id"] for p in timeline] == [second["transaction_id"], first["transaction_id"]]
    for point, pred in zip(timeline, (second, first)):
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


def test_history_limit_returns_most_recent(client, normal_txn):
    posted = [_post_at(client, normal_txn, BASE_TIME + timedelta(hours=i)) for i in range(3)]
    body = client.get("/customer/CUST_0001/history", params={"limit": 2}).json()
    ids = [p["transaction_id"] for p in body["timeline"]]
    # the latest two, newest first
    assert ids == [posted[2]["transaction_id"], posted[1]["transaction_id"]]


def test_history_limit_one_is_newest(client, normal_txn):
    posted = [_post_at(client, normal_txn, BASE_TIME + timedelta(hours=i)) for i in range(3)]
    body = client.get("/customer/CUST_0001/history", params={"limit": 1}).json()
    assert body["n_transactions"] == 1
    assert body["timeline"][0]["transaction_id"] == posted[-1]["transaction_id"]


def test_history_newest_selected_by_timestamp_not_insertion_order(client, normal_txn):
    # score a later transaction first, then an earlier one
    later = _post_at(client, normal_txn, BASE_TIME + timedelta(days=2))
    earlier = _post_at(client, normal_txn, BASE_TIME + timedelta(days=1))
    body = client.get("/customer/CUST_0001/history", params={"limit": 1}).json()
    assert body["timeline"][0]["transaction_id"] == later["transaction_id"]
    full = client.get("/customer/CUST_0001/history").json()
    assert [p["transaction_id"] for p in full["timeline"]] == [later["transaction_id"], earlier["transaction_id"]]


def test_history_timeline_is_descending_by_timestamp(client, normal_txn):
    for i in (2, 0, 3, 1):  # score out of time order
        _post_at(client, normal_txn, BASE_TIME + timedelta(hours=i))
    stamps = [p["timestamp"] for p in client.get("/customer/CUST_0001/history").json()["timeline"]]
    assert len(stamps) == 4
    assert stamps == sorted(stamps, reverse=True)


def test_history_limit_larger_than_available(client, normal_txn):
    for i in range(2):
        _post_at(client, normal_txn, BASE_TIME + timedelta(hours=i))
    body = client.get("/customer/CUST_0001/history", params={"limit": 50}).json()
    assert body["n_transactions"] == 2
