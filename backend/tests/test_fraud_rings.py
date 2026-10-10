"""GET /fraud-rings -- cross-customer shared-device detection."""

from conftest import BASE_TIME


def test_fraud_rings_response_shape(client):
    r = client.get("/fraud-rings")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == len(body["rings"])
    assert body["count"] > 0  # the seed data contains shared DEV_UNKNOWN_* devices
    for ring in body["rings"]:
        assert set(ring) == {"ring_type", "identifier", "customer_ids", "transaction_count"}
        assert ring["ring_type"] == "shared_device"
        assert len(ring["customer_ids"]) >= 2
        assert ring["customer_ids"] == sorted(set(ring["customer_ids"]))
        assert ring["transaction_count"] >= len(ring["customer_ids"])


def test_fraud_rings_sorted_by_size_then_activity(client):
    rings = client.get("/fraud-rings").json()["rings"]
    keys = [(len(r["customer_ids"]), r["transaction_count"]) for r in rings]
    assert keys == sorted(keys, reverse=True)


def test_min_customers_filter(client):
    all_rings = client.get("/fraud-rings", params={"min_customers": 2}).json()
    big_rings = client.get("/fraud-rings", params={"min_customers": 3}).json()
    assert big_rings["count"] <= all_rings["count"]
    assert all(len(r["customer_ids"]) >= 3 for r in big_rings["rings"])


def test_new_shared_device_creates_ring(client):
    device = "DEV_RING_TEST_0001"
    customers = ["CUST_0010", "CUST_0011", "CUST_0012"]
    for cid in customers:
        r = client.post("/predict", json={
            "customer_id": cid,
            "amount": 5000,
            "merchant_category": "electronics",
            "device_id": device,
            "location": "Mumbai",
            "timestamp": BASE_TIME.isoformat(),
        })
        assert r.status_code == 200

    rings = client.get("/fraud-rings", params={"min_customers": 3}).json()["rings"]
    ring = next((r for r in rings if r["identifier"] == device), None)
    assert ring is not None
    assert ring["customer_ids"] == customers
    assert ring["transaction_count"] == 3
    # largest ring should now be first
    assert rings[0]["identifier"] == device


def test_single_customer_device_is_not_a_ring(client):
    device = "DEV_SOLO_TEST_0001"
    client.post("/predict", json={
        "customer_id": "CUST_0010",
        "amount": 5000,
        "merchant_category": "electronics",
        "device_id": device,
        "location": "Mumbai",
        "timestamp": BASE_TIME.isoformat(),
    })
    rings = client.get("/fraud-rings").json()["rings"]
    assert all(r["identifier"] != device for r in rings)


def test_fraud_rings_state_is_isolated_between_tests(client):
    # the ring created in test_new_shared_device_creates_ring must not leak
    rings = client.get("/fraud-rings").json()["rings"]
    assert all(r["identifier"] != "DEV_RING_TEST_0001" for r in rings)


def test_fraud_rings_invalid_param(client):
    assert client.get("/fraud-rings", params={"min_customers": "x"}).status_code == 422
