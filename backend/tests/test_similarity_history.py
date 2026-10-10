"""Behavioral Similarity with little or no history (audit finding R-05, workstream A1).

A customer with fewer than 10 earlier transactions has no meaningful behavioral baseline. The API must then return
`similarity_pct = deviation_pct = null` with `similarity_status = "insufficient_history"` -- never 0, 100 or an
invented number -- and customers with enough history must keep getting the unchanged score.
"""

from datetime import datetime, timedelta

import numpy as np
import pytest

from app.db import database as db_module
from app.db import models as db_models
from app.models.similarity import (
    MIN_HISTORY_FOR_SIMILARITY, STATUS_INSUFFICIENT_HISTORY, STATUS_OK, behavioral_similarity, compute_similarity,
)

START = datetime(2026, 8, 1, 12, 0, 0)


def _post(client, customer, k, **over):
    body = {"customer_id": customer, "amount": 1500.0, "merchant_category": "grocery", "device_id": "D1",
            "location": "Pune", "failed_logins_24h": 0, "timestamp": (START + timedelta(days=k)).isoformat()}
    body.update(over)
    r = client.post("/predict", json=body)
    assert r.status_code == 200, r.text
    return r.json()


# ---- unit level -----------------------------------------------------------------------------------

def test_threshold_equals_the_cold_start_threshold():
    from app.inference_pipeline import MIN_PRIOR_TRANSACTIONS
    assert MIN_HISTORY_FOR_SIMILARITY == MIN_PRIOR_TRANSACTIONS == 10


@pytest.mark.parametrize("n_prior", [0, 1, 2, 9])
def test_too_little_history_gives_none_not_a_number(n_prior):
    prior = np.random.default_rng(0).normal(size=(n_prior, 9))
    out = behavioral_similarity(np.zeros(9), prior)
    assert out["similarity_pct"] is None and out["deviation_pct"] is None
    assert out["similarity_status"] == STATUS_INSUFFICIENT_HISTORY and out["history_transactions"] == n_prior


@pytest.mark.parametrize("n_prior", [10, 11, 200])
def test_enough_history_uses_the_unchanged_formula(n_prior):
    prior = np.random.default_rng(1).normal(size=(n_prior, 9))
    current = np.random.default_rng(2).normal(size=9)
    out = behavioral_similarity(current, prior)
    ref = compute_similarity(current, prior.mean(axis=0), prior.std(axis=0))
    assert out["similarity_status"] == STATUS_OK and out["history_transactions"] == n_prior
    assert out["similarity_pct"] == ref["similarity_pct"] and out["deviation_pct"] == ref["deviation_pct"]


# ---- API level -------------------------------------------------------------------------------------

def test_new_customer_gets_null_similarity_with_a_status(client):
    body = _post(client, "CUST_SIM_NEW", 0)
    assert body["similarity_pct"] is None and body["deviation_pct"] is None
    assert body["similarity_status"] == "insufficient_history" and body["history_transactions"] == 0
    # the rest of the response is still a normal, complete prediction
    assert 0 <= body["fraud_probability"] <= 100 and body["alert_level"] in {"Low Risk", "Medium Risk", "High Risk", "Critical Risk"}


def test_status_flips_exactly_when_the_tenth_earlier_transaction_exists(client):
    for k in range(MIN_HISTORY_FOR_SIMILARITY + 2):
        body = _post(client, "CUST_SIM_RAMP", k)
        assert body["history_transactions"] == k
        if k < MIN_HISTORY_FOR_SIMILARITY:
            assert body["similarity_pct"] is None and body["similarity_status"] == "insufficient_history", k
        else:
            assert body["similarity_status"] == "ok" and 0 <= body["similarity_pct"] <= 100, k
            assert body["similarity_pct"] + body["deviation_pct"] == pytest.approx(100, abs=0.02)


def test_seeded_customer_keeps_a_numeric_score(client, normal_txn):
    body = client.post("/predict", json=normal_txn).json()
    assert body["similarity_status"] == "ok" and body["history_transactions"] >= 10
    assert 0 <= body["similarity_pct"] <= 100


def test_back_dated_transaction_before_all_history_has_no_baseline(client, normal_txn):
    body = client.post("/predict", json=dict(normal_txn, timestamp="2025-01-01T10:00:00")).json()
    assert body["history_transactions"] == 0 and body["similarity_pct"] is None
    assert body["similarity_status"] == "insufficient_history"


def test_null_similarity_is_stored_as_null_not_zero(client):
    body = _post(client, "CUST_SIM_DB", 0)
    s = db_module.SessionLocal()
    try:
        row = s.get(db_models.Transaction, body["transaction_id"])
        assert row.similarity_pct is None and row.deviation_pct is None
    finally:
        s.close()


def test_pdf_report_states_the_missing_baseline_instead_of_a_number(client):
    from tests.test_report import _extract_text
    body = _post(client, "CUST_SIM_PDF", 0)
    text = _extract_text(client.post("/report/pdf", json=body).content)
    assert "n/a" in text and "fewer than 10 earlier transactions" in " ".join(text.split())
    assert "100.0%" not in text.split("Behavioral Similarity")[1].split("Explainable")[0]
