"""POST /predict/batch -- CSV upload scoring."""

import pytest
from fastapi.testclient import TestClient

from app.main import app

ALERT_LEVELS = ["Low Risk", "Medium Risk", "High Risk", "Critical Risk"]


def _upload(client, csv_text, filename="batch.csv"):
    return client.post(
        "/predict/batch",
        files={"file": (filename, csv_text.encode("utf-8"), "text/csv")},
    )


FULL_CSV = (
    "customer_id,amount,merchant_category,device_id,location,failed_logins_24h\n"
    "CUST_0002,3500,fuel,DEV_0002_A,Chennai,0\n"
    "CUST_0003,90000,electronics,DEV_UNKNOWN_4242,Lagos,5\n"
    "CUST_0004,2500,grocery,DEV_0004_A,Mumbai,0\n"
)


def test_batch_full_columns(client):
    r = _upload(client, FULL_CSV)
    assert r.status_code == 200
    body = r.json()

    assert body["count"] == 3
    assert set(body["summary"]) == set(ALERT_LEVELS)
    assert sum(body["summary"].values()) == 3

    results = body["results"]
    assert [row["customer_id"] for row in results] == ["CUST_0002", "CUST_0003", "CUST_0004"]
    for row in results:
        assert set(row) == {"transaction_id", "customer_id", "amount", "risk_score", "fraud_probability", "alert_level"}
        assert row["alert_level"] in ALERT_LEVELS
        assert 0 <= row["fraud_probability"] <= 100
    assert results[1]["amount"] == 90000
    # the obviously fraudulent row must be flagged regardless of time of day
    assert results[1]["alert_level"] in {"High Risk", "Critical Risk"}
    assert results[1]["fraud_probability"] > max(results[0]["fraud_probability"], results[2]["fraud_probability"])

    # summary agrees with the per-row alert levels
    for level in ALERT_LEVELS:
        assert body["summary"][level] == sum(1 for row in results if row["alert_level"] == level)


def test_batch_results_are_persisted(client):
    _upload(client, FULL_CSV)
    for cid in ("CUST_0002", "CUST_0003", "CUST_0004"):
        r = client.get(f"/customer/{cid}/history")
        assert r.status_code == 200
        assert r.json()["n_transactions"] == 1


def test_batch_minimal_required_columns(client):
    r = _upload(client, "customer_id,amount,merchant_category\nCUST_0005,1500,grocery\nCUST_0006,2000,fuel\n")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2
    assert sum(body["summary"].values()) == 2


def test_batch_blank_optional_values(client):
    csv_text = (
        "customer_id,amount,merchant_category,device_id,location,failed_logins_24h\n"
        "CUST_0007,1800,grocery,,,\n"
    )
    r = _upload(client, csv_text)
    assert r.status_code == 200
    assert r.json()["count"] == 1


def test_batch_header_only_returns_empty_result(client):
    r = _upload(client, "customer_id,amount,merchant_category\n")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 0
    assert body["results"] == []
    assert sum(body["summary"].values()) == 0


def test_batch_missing_required_columns(client):
    r = _upload(client, "customer_id,amount\nCUST_0001,100\n")
    assert r.status_code == 400
    assert "merchant_category" in r.json()["detail"]


def test_batch_unrelated_columns(client):
    r = _upload(client, "a,b\n1,2\n")
    assert r.status_code == 400
    detail = r.json()["detail"]
    for col in ("customer_id", "amount", "merchant_category"):
        assert col in detail


def test_batch_empty_file(client):
    r = _upload(client, "")
    assert r.status_code == 400
    assert "Could not parse CSV" in r.json()["detail"]


def test_batch_no_file(client):
    assert client.post("/predict/batch").status_code == 422


def test_batch_rejected_rows_not_persisted(client):
    _upload(client, "customer_id,amount\nCUST_0001,100\n")
    assert client.get("/customer/CUST_0001/history").status_code == 404


@pytest.mark.xfail(
    strict=True,
    reason="Known bug (plan step 2): a non-numeric amount raises an unhandled "
           "ValueError and the endpoint returns 500 instead of a 400 validation error.",
)
def test_batch_non_numeric_amount_is_client_error():
    with TestClient(app, raise_server_exceptions=False) as c:
        r = _upload(c, "customer_id,amount,merchant_category\nCUST_0001,abc,grocery\n")
    assert r.status_code == 400
