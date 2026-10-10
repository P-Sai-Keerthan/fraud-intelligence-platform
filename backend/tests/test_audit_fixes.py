"""Regression tests for the defects found in the project audit (reports/BUG_AND_RISK_REGISTER.md).

Each test names the finding it guards. Before the fixes, every "rejected" case below was either
accepted and stored (corrupting the customer's history), or answered with a 500.
"""

import json

import numpy as np
import pandas as pd
import pytest
from sqlalchemy.orm import Session

from app import main as app_main
from app.db import database as db_module
from app.db import models as db_models
from app.features.feature_engineering import FEATURE_COLUMNS


def _db_rows():
    s = db_module.SessionLocal()
    try:
        return s.query(db_models.Transaction).count()
    finally:
        s.close()


def _raw_post(client, body: str):
    return client.post("/predict", content=body, headers={"Content-Type": "application/json"})


# ---- F-01 / F-02: non-finite and absurd numbers -------------------------------------------------

@pytest.mark.parametrize("amount", ["Infinity", "-Infinity", "NaN", "2e9", "1e308"])
def test_non_finite_or_absurd_amount_is_rejected_with_422_and_nothing_is_stored(client, pipeline, normal_txn, amount):
    before = len(pipeline.customer_histories["CUST_0001"])
    body = json.dumps(normal_txn).replace(json.dumps(normal_txn["amount"]), amount)
    r = _raw_post(client, body)
    assert r.status_code == 422, r.text          # NaN used to be a 500, Infinity was stored
    assert _db_rows() == 0
    assert len(pipeline.customer_histories["CUST_0001"]) == before
    r.json()                                      # the error body itself is valid JSON


def test_a_valid_large_amount_is_still_accepted(client, normal_txn):
    r = client.post("/predict", json=dict(normal_txn, amount=5_000_000.0))
    assert r.status_code == 200


def test_failed_logins_has_an_upper_bound(client, normal_txn):
    assert client.post("/predict", json=dict(normal_txn, failed_logins_24h=10**30)).status_code == 422
    assert client.post("/predict", json=dict(normal_txn, failed_logins_24h=10_001)).status_code == 422
    assert client.post("/predict", json=dict(normal_txn, failed_logins_24h=10_000)).status_code == 200


# ---- F-03: blank and over-long identifiers -------------------------------------------------------

@pytest.mark.parametrize("field", ["customer_id", "merchant_category", "device_id", "location"])
@pytest.mark.parametrize("value", ["", "   ", "x" * 65])
def test_blank_or_overlong_text_fields_are_rejected(client, pipeline, normal_txn, field, value):
    n_customers = len(pipeline.customer_histories)
    r = client.post("/predict", json=dict(normal_txn, **{field: value}))
    assert r.status_code == 422
    assert len(pipeline.customer_histories) == n_customers      # no phantom customer
    assert _db_rows() == 0


def test_surrounding_whitespace_is_stripped(client, normal_txn):
    r = client.post("/predict", json=dict(normal_txn, customer_id="  CUST_0001 "))
    assert r.status_code == 200
    assert r.json()["customer_id"] == "CUST_0001"


# ---- F-04: timezone-aware timestamps -------------------------------------------------------------

@pytest.mark.parametrize("stamp, expected", [
    ("2026-07-10T13:30:00Z", "2026-07-10T13:30:00"),
    ("2026-07-10T19:00:00+05:30", "2026-07-10T13:30:00"),
])
def test_timezone_aware_timestamp_is_accepted_and_converted_to_utc(client, normal_txn, stamp, expected):
    r = client.post("/predict", json=dict(normal_txn, timestamp=stamp))
    assert r.status_code == 200, r.text          # used to be a 500
    assert r.json()["timestamp"] == expected


# ---- F-05: memory and database must not drift ----------------------------------------------------

def test_remove_transaction_restores_the_exact_previous_history(pipeline, normal_txn):
    from app.features.feature_engineering import build_point_features
    from app.inference_pipeline import RAW_COLUMNS_FOR_FEATURES
    before = pipeline.customer_histories["CUST_0001"].copy()
    # reference: the history as live scoring defines it, i.e. every row recomputed with the
    # current feature code (the committed v1 seed CSV was produced by an older feature version,
    # see docs/step4c2f-1-v1-feature-version-audit.md, so its stored values are not the reference)
    expected = build_point_features(before[RAW_COLUMNS_FOR_FEATURES])
    result = pipeline.score_transaction(dict(normal_txn, timestamp=pd.Timestamp(normal_txn["timestamp"])))
    assert len(pipeline.customer_histories["CUST_0001"]) == len(before) + 1
    assert pipeline.remove_transaction("CUST_0001", result["transaction_id"]) is True
    after = pipeline.customer_histories["CUST_0001"]
    assert list(after["transaction_id"]) == list(before["transaction_id"])
    np.testing.assert_allclose(after[FEATURE_COLUMNS].to_numpy(float), expected[FEATURE_COLUMNS].to_numpy(float), atol=1e-9)
    assert pipeline.remove_transaction("CUST_0001", result["transaction_id"]) is False      # already gone


def test_remove_transaction_of_a_brand_new_customer_removes_the_customer(pipeline, normal_txn):
    txn = dict(normal_txn, customer_id="CUST_NEW_AUDIT", device_id="D1", location="L1")
    result = pipeline.score_transaction(txn)
    assert "CUST_NEW_AUDIT" in pipeline.customer_histories
    assert pipeline.remove_transaction("CUST_NEW_AUDIT", result["transaction_id"]) is True
    assert "CUST_NEW_AUDIT" not in pipeline.customer_histories


def test_a_failed_commit_leaves_no_trace_in_memory(client, pipeline, normal_txn, monkeypatch):
    before = pipeline.customer_histories["CUST_0001"].copy()

    def boom(self):
        raise RuntimeError("simulated database failure")

    monkeypatch.setattr(Session, "commit", boom)
    with pytest.raises(RuntimeError, match="simulated"):
        client.post("/predict", json=normal_txn)
    monkeypatch.undo()
    assert _db_rows() == 0
    assert list(pipeline.customer_histories["CUST_0001"]["transaction_id"]) == list(before["transaction_id"])


def test_a_failed_batch_commit_removes_every_scored_row_from_memory(client, pipeline, monkeypatch):
    csv = "customer_id,amount,merchant_category,device_id,location\n" + "\n".join(
        f"CUST_000{i},100,grocery,DEV_000{i}_A,Pune" for i in (1, 2, 3)) + "\n"
    before = {c: list(pipeline.customer_histories[c]["transaction_id"]) for c in ("CUST_0001", "CUST_0002", "CUST_0003")}

    def boom(self):
        raise RuntimeError("simulated database failure")

    monkeypatch.setattr(Session, "commit", boom)
    with pytest.raises(RuntimeError, match="simulated"):
        client.post("/predict/batch", files={"file": ("b.csv", csv, "text/csv")})
    monkeypatch.undo()
    assert _db_rows() == 0
    for c, ids in before.items():
        assert list(pipeline.customer_histories[c]["transaction_id"]) == ids


# ---- F-06: query-parameter bounds ----------------------------------------------------------------

@pytest.mark.parametrize("url", [
    "/customers?limit=-1", "/customers?limit=0", "/customer/CUST_0001/history?limit=0",
    "/customer/CUST_0001/history?limit=-5", "/fraud-rings?min_customers=0",
])
def test_out_of_range_query_parameters_are_rejected(client, url):
    assert client.get(url).status_code == 422


def test_customers_limit_returns_exactly_that_many(client):
    body = client.get("/customers?limit=7").json()
    assert len(body["customer_ids"]) == 7 and body["count"] >= 500


# ---- F-07: PDF download filename -----------------------------------------------------------------

@pytest.mark.parametrize("tid", ['TXN"; evil=1', "TXN_é中\U0001F600", "TXN\r\nX-Injected: 1", "../../etc/passwd"])
def test_pdf_filename_cannot_break_or_inject_into_the_header(client, normal_txn, tid):
    pred = client.post("/predict", json=normal_txn).json()
    r = client.post("/report/pdf", json=dict(pred, transaction_id=tid))
    assert r.status_code == 200, r.text            # a non-Latin-1 id used to be a 500
    disposition = r.headers["content-disposition"]
    assert disposition.count('"') == 2 and "\n" not in disposition and "/" not in disposition and ";" not in disposition.split("filename=")[1]
    assert "X-Injected" not in r.headers


# ---- F-08: batch limits --------------------------------------------------------------------------

def test_batch_row_limit(client, monkeypatch):
    monkeypatch.setattr(app_main, "MAX_BATCH_ROWS", 2)
    csv = "customer_id,amount,merchant_category,device_id,location\n" + "CUST_0001,100,grocery,DEV_0001_A,Pune\n" * 3
    r = client.post("/predict/batch", files={"file": ("b.csv", csv, "text/csv")})
    assert r.status_code == 413 and "limit is 2" in r.json()["detail"]
    assert _db_rows() == 0


def test_batch_size_limit(client, monkeypatch):
    monkeypatch.setattr(app_main, "MAX_BATCH_BYTES", 50)
    csv = "customer_id,amount,merchant_category,device_id,location\n" + "CUST_0001,100,grocery,DEV_0001_A,Pune\n" * 3
    r = client.post("/predict/batch", files={"file": ("b.csv", csv, "text/csv")})
    assert r.status_code == 413
    assert _db_rows() == 0


def test_batch_rows_follow_the_same_limits_as_predict(client):
    header = "customer_id,amount,merchant_category,device_id,location,failed_logins_24h\n"
    for row, fragment in [
        ("CUST_0001,2000000000,grocery,DEV_0001_A,Pune,0", "must not exceed"),
        ("CUST_0001,100,grocery,DEV_0001_A,Pune,10001", "failed_logins_24h must not exceed"),
        (f"{'C' * 65},100,grocery,DEV_0001_A,Pune,0", "customer_id is longer than 64"),
    ]:
        r = client.post("/predict/batch", files={"file": ("b.csv", header + row + "\n", "text/csv")})
        assert r.status_code == 400 and fragment in r.json()["detail"], r.text


# ---- F-09: health check --------------------------------------------------------------------------

def test_health_is_ok_when_the_database_is_reachable(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_health_reports_503_when_the_database_is_unreachable(client, monkeypatch):
    class Broken:
        def connect(self):
            raise RuntimeError("database down")

    monkeypatch.setattr(app_main, "engine", Broken())
    r = client.get("/health")
    assert r.status_code == 503 and r.json()["status"] != "ok"
