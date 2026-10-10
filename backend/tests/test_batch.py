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


# ---- invalid row values: whole batch rejected with a 400 naming row + field ----

def _upload_raw(csv_text):
    # raise_server_exceptions=False so an unhandled error shows up as a 500
    # response instead of propagating out of the test client
    with TestClient(app, raise_server_exceptions=False) as c:
        return _upload(c, csv_text)


def test_batch_non_numeric_amount_is_client_error():
    r = _upload_raw("customer_id,amount,merchant_category\nCUST_0001,abc,grocery\n")
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert "row 1 (customer_id=CUST_0001)" in detail
    assert "amount must be a number, got 'abc'" in detail


HEADER = "customer_id,amount,merchant_category,device_id,location,failed_logins_24h\n"


@pytest.mark.parametrize(
    "row, expected",
    [
        ("CUST_0001,abc,grocery,,,0", "amount must be a number, got 'abc'"),
        ("CUST_0001,12abc,grocery,,,0", "amount must be a number, got '12abc'"),
        ("CUST_0001,1_000,grocery,,,0", "amount must be a number, got '1_000'"),
        ("CUST_0001,inf,grocery,,,0", "amount must be a number, got 'inf'"),
        ("CUST_0001,nan,grocery,,,0", "amount must be a number, got 'nan'"),
        ("CUST_0001,1e999,grocery,,,0", "amount must be a finite number, got '1e999'"),
        ("CUST_0001,,grocery,,,0", "amount is required"),
        ("CUST_0001,0,grocery,,,0", "amount must be greater than 0, got '0'"),
        ("CUST_0001,-50,grocery,,,0", "amount must be greater than 0, got '-50'"),
        ("CUST_0001,100,grocery,,,many", "failed_logins_24h must be a whole number >= 0, got 'many'"),
        ("CUST_0001,100,grocery,,,-1", "failed_logins_24h must be a whole number >= 0, got '-1'"),
        ("CUST_0001,100,grocery,,,2.5", "failed_logins_24h must be a whole number >= 0, got '2.5'"),
        (",100,grocery,,,0", "customer_id is required"),
        ("CUST_0001,100,,,,0", "merchant_category is required"),
    ],
    ids=[
        "amount_text", "amount_trailing_text", "amount_underscore", "amount_inf", "amount_nan", "amount_overflow",
        "amount_blank", "amount_zero", "amount_negative",
        "logins_text", "logins_negative", "logins_fraction",
        "customer_blank", "category_blank",
    ],
)
def test_batch_invalid_row_values(row, expected):
    r = _upload_raw(HEADER + row + "\n")
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert "row 1" in detail
    assert expected in detail


def test_batch_reports_every_invalid_row_with_row_numbers():
    csv_text = HEADER + (
        "CUST_0001,100,grocery,,,0\n"      # row 1 valid
        "CUST_0002,abc,fuel,,,0\n"         # row 2 bad amount
        "CUST_0003,200,dining,,,x\n"       # row 3 bad logins
        "CUST_0004,-1,grocery,,,-2\n"      # row 4 two problems
    )
    r = _upload_raw(csv_text)
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert "Invalid CSV data in 3 row(s); nothing was scored." in detail
    assert "row 1" not in detail
    assert "row 2 (customer_id=CUST_0002): amount must be a number, got 'abc'" in detail
    assert "row 3 (customer_id=CUST_0003): failed_logins_24h must be a whole number >= 0, got 'x'" in detail
    assert "row 4 (customer_id=CUST_0004): amount must be greater than 0, got '-1'; failed_logins_24h" in detail


def test_batch_error_list_is_capped():
    csv_text = "customer_id,amount,merchant_category\n" + "CUST_0001,abc,grocery\n" * 25
    detail = _upload_raw(csv_text).json()["detail"]
    assert "Invalid CSV data in 25 row(s)" in detail
    assert "row 20 " in detail and "row 21 " not in detail
    assert "...and 5 more invalid row(s)" in detail


def test_batch_with_invalid_row_scores_and_saves_nothing(client, pipeline):
    before = {cid: len(pipeline.customer_histories[cid]) for cid in ("CUST_0002", "CUST_0003")}
    csv_text = HEADER + "CUST_0002,3500,fuel,,,0\nCUST_0003,abc,dining,,,0\n"
    assert _upload(client, csv_text).status_code == 400
    # the valid first row must not have been scored or persisted either
    assert {cid: len(pipeline.customer_histories[cid]) for cid in before} == before
    assert client.get("/customer/CUST_0002/history").status_code == 404


def test_batch_accepts_whole_number_float_logins(client):
    # spreadsheets often export integers as "4.0"; that is still a whole number
    r = _upload(client, HEADER + "CUST_0001,100,grocery,,,4.0\n")
    assert r.status_code == 200


# ---- blank device/location defaults to the customer's real home device/city ----

def test_batch_blank_device_and_location_use_customer_profile(client, pipeline):
    csv_text = HEADER + "CUST_0001,3800,fashion,,,0\nCUST_0002,3500,fuel,,,0\n"
    r = _upload(client, csv_text)
    assert r.status_code == 200
    for cid, device, city in (("CUST_0001", "DEV_0001_A", "Pune"), ("CUST_0002", "DEV_0002_A", "Chennai")):
        last = pipeline.customer_histories[cid].iloc[-1]
        assert last["device_id"] == device
        assert last["location"] == city
        assert last["is_new_device"] == 0
        assert last["is_new_location"] == 0
        assert last["is_foreign_location"] == 0


def test_batch_minimal_columns_use_customer_profile(client, pipeline):
    r = _upload(client, "customer_id,amount,merchant_category\nCUST_0001,3800,fashion\n")
    assert r.status_code == 200
    last = pipeline.customer_histories["CUST_0001"].iloc[-1]
    assert (last["device_id"], last["location"]) == ("DEV_0001_A", "Pune")


def test_batch_explicit_device_and_location_are_kept(client, pipeline):
    r = _upload(client, HEADER + "CUST_0001,85000,electronics,DEV_UNKNOWN_9999,Lagos,4\n")
    assert r.status_code == 200
    last = pipeline.customer_histories["CUST_0001"].iloc[-1]
    assert (last["device_id"], last["location"]) == ("DEV_UNKNOWN_9999", "Lagos")
    assert last["is_new_device"] == 1
    assert last["is_foreign_location"] == 1
    assert r.json()["results"][0]["alert_level"] in {"High Risk", "Critical Risk"}


def test_batch_partial_override_keeps_given_value(client, pipeline):
    # device given, location blank -> only the location falls back to home
    r = _upload(client, HEADER + "CUST_0001,3800,fashion,DEV_0001_B,,0\n")
    assert r.status_code == 200
    last = pipeline.customer_histories["CUST_0001"].iloc[-1]
    assert (last["device_id"], last["location"]) == ("DEV_0001_B", "Pune")


def test_batch_unknown_customer_needs_device_and_location():
    r = _upload_raw(HEADER + "CUST_NEW_1,500,grocery,,,0\n")
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert "row 1 (customer_id=CUST_NEW_1)" in detail
    assert "device_id and location required" in detail


def test_batch_unknown_customer_with_device_and_location(client):
    r = _upload(client, HEADER + "CUST_NEW_2,500,grocery,DEV_NEW_2,Chennai,0\n")
    assert r.status_code == 200
    assert r.json()["count"] == 1
