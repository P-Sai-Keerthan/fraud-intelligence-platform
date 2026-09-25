"""GET /health and GET /customers."""


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_customers_default_limit(client):
    r = client.get("/customers")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 500  # total known customers in the seed data
    assert len(body["customer_ids"]) == 50  # default limit
    assert body["customer_ids"] == sorted(body["customer_ids"])
    assert body["customer_ids"][0] == "CUST_0000"


def test_customers_custom_limit(client):
    r = client.get("/customers", params={"limit": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 500  # count is the total, not the page size
    assert body["customer_ids"] == ["CUST_0000", "CUST_0001", "CUST_0002"]


def test_customers_invalid_limit(client):
    r = client.get("/customers", params={"limit": "abc"})
    assert r.status_code == 422


def test_unknown_route_404(client):
    assert client.get("/does-not-exist").status_code == 404
