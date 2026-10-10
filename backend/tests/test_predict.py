"""POST /predict -- single-transaction scoring."""

import re

import numpy as np
import pytest

ALERT_LEVELS = {"Low Risk", "Medium Risk", "High Risk", "Critical Risk"}


def _assert_prediction_shape(body, sent):
    for key in ("customer_id", "merchant_category", "device_id", "location", "failed_logins_24h"):
        assert body[key] == sent[key]
    assert body["amount"] == pytest.approx(sent["amount"])
    assert re.fullmatch(r"TXN_[0-9A-F]{10}", body["transaction_id"])
    assert 0 <= body["risk_score"] <= 100
    assert 0 <= body["fraud_probability"] <= 100
    assert 0 <= body["similarity_pct"] <= 100
    assert body["similarity_pct"] + body["deviation_pct"] == pytest.approx(100, abs=0.02)
    assert body["alert_level"] in ALERT_LEVELS
    assert isinstance(body["reasons"], list)


def test_normal_transaction_is_low_risk(client, normal_txn):
    r = client.post("/predict", json=normal_txn)
    assert r.status_code == 200
    body = r.json()
    _assert_prediction_shape(body, normal_txn)

    assert body["alert_level"] == "Low Risk"
    assert body["fraud_probability"] < 5
    # reasons are only surfaced at >= 5% fraud probability
    assert body["reasons"] == []


def test_suspicious_transaction_is_critical_with_reasons(client, suspicious_txn):
    r = client.post("/predict", json=suspicious_txn)
    assert r.status_code == 200
    body = r.json()
    _assert_prediction_shape(body, suspicious_txn)

    assert body["alert_level"] == "Critical Risk"
    assert body["fraud_probability"] >= 80
    assert body["fraud_probability"] <= 99.9  # capped: never claims certainty

    reasons = body["reasons"]
    assert 1 <= len(reasons) <= 4
    assert all(r_["shap_value"] > 0 for r_ in reasons)
    shap_values = [r_["shap_value"] for r_ in reasons]
    assert shap_values == sorted(shap_values, reverse=True)
    for r_ in reasons:
        assert set(r_) == {"feature", "display_name", "shap_value"}
    names = {r_["display_name"] for r_ in reasons}
    assert names & {"New Device", "Foreign Location", "Amount Far Above Average"}


def test_suspicious_scores_higher_than_normal(client, normal_txn, suspicious_txn):
    normal = client.post("/predict", json=normal_txn).json()
    suspicious = client.post("/predict", json=suspicious_txn).json()
    assert suspicious["fraud_probability"] > normal["fraud_probability"]
    assert suspicious["similarity_pct"] < normal["similarity_pct"]


def _replay(client, pipeline, txn, seed=None):
    """Score txn from the current history snapshot, then roll the history back."""
    snapshot = dict(pipeline.customer_histories)
    if seed is not None:
        np.random.seed(seed)
    body = client.post("/predict", json=txn).json()
    pipeline.customer_histories.clear()
    pipeline.customer_histories.update(snapshot)
    return body


def test_model_scores_are_deterministic_for_same_state(client, pipeline, suspicious_txn):
    # the LSTM/DNN/similarity outputs don't depend on any RNG
    first = _replay(client, pipeline, suspicious_txn)
    second = _replay(client, pipeline, suspicious_txn)
    for key in ("risk_score", "fraud_probability", "alert_level", "similarity_pct", "deviation_pct"):
        assert first[key] == second[key]
    assert first["transaction_id"] != second["transaction_id"]


def test_shap_reasons_reproducible_with_fixed_seed(client, pipeline, suspicious_txn):
    # SHAP GradientExplainer is stochastic; with a fixed seed it is exactly reproducible
    first = _replay(client, pipeline, suspicious_txn, seed=42)
    second = _replay(client, pipeline, suspicious_txn, seed=42)
    assert first["reasons"] == second["reasons"]


def test_prediction_updates_customer_history_in_memory(client, pipeline, normal_txn):
    before = len(pipeline.customer_histories["CUST_0001"])
    client.post("/predict", json=normal_txn)
    assert len(pipeline.customer_histories["CUST_0001"]) == before + 1


def test_unknown_customer_cold_start(client, pipeline):
    txn = {
        "customer_id": "CUST_BRAND_NEW",
        "amount": 1200,
        "merchant_category": "grocery",
        "device_id": "DEV_NEW_1",
        "location": "Chennai",
        "timestamp": "2026-07-10T12:00:00",
    }
    r = client.post("/predict", json=txn)
    assert r.status_code == 200
    body = r.json()
    assert body["customer_id"] == "CUST_BRAND_NEW"
    assert body["alert_level"] in ALERT_LEVELS
    assert "CUST_BRAND_NEW" in pipeline.customer_histories


def test_failed_logins_defaults_to_zero(client, normal_txn):
    normal_txn.pop("failed_logins_24h")
    r = client.post("/predict", json=normal_txn)
    assert r.status_code == 200
    assert r.json()["failed_logins_24h"] == 0


def test_timestamp_is_optional(client, normal_txn):
    normal_txn.pop("timestamp")
    r = client.post("/predict", json=normal_txn)
    assert r.status_code == 200
    assert r.json()["timestamp"]


@pytest.mark.parametrize(
    "override, missing",
    [
        ({"amount": 0}, None),
        ({"amount": -100}, None),
        ({"amount": "lots"}, None),
        ({"failed_logins_24h": -1}, None),
        ({"failed_logins_24h": "many"}, None),
        ({"timestamp": "not-a-date"}, None),
        ({}, "customer_id"),
        ({}, "amount"),
        ({}, "merchant_category"),
        ({}, "device_id"),
        ({}, "location"),
    ],
    ids=[
        "amount_zero", "amount_negative", "amount_string",
        "failed_logins_negative", "failed_logins_string", "bad_timestamp",
        "missing_customer_id", "missing_amount", "missing_merchant_category",
        "missing_device_id", "missing_location",
    ],
)
def test_invalid_input_rejected(client, normal_txn, override, missing):
    payload = {**normal_txn, **override}
    if missing:
        payload.pop(missing)
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_invalid_input_is_not_persisted(client, normal_txn):
    client.post("/predict", json={**normal_txn, "amount": -1})
    assert client.get("/customer/CUST_0001/history").status_code == 404


def test_non_json_body_rejected(client):
    r = client.post("/predict", content=b"not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 422
