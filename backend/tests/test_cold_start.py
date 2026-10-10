"""Cold start (Step 4C-2f-1): customers with fewer than SEQUENCE_LENGTH (10)
earlier transactions.

Every model in every model set was trained only on transactions with at
least 10 earlier transactions: LSTM windows are 10 real earlier rows, and the
DNNs' training rows are the windows' targets. So for a customer with fewer
than 10 earlier transactions:

* the LSTM is not run on padded windows (padding never occurred in training
  and Masking is inert); risk_score is instead the training mean of
  risk_score (scaled input 0), for model sets that take risk_score;
* for a customer's first transaction, the baseline-relative features (which
  compare the transaction with the customer's own history, and which the
  feature builder sets to placeholder values when there is no history) are
  treated as missing: scaled input 0, the training mean;
* baseline-free features (foreign location, failed logins, velocity) are
  always used as observed, so a suspicious first transaction still scores
  high -- nothing is hard-coded to Low.
* No transactions are fabricated; the response schema is unchanged; the
  pipeline reports the state in result["history_context"] (not part of the API).
"""

import random
from datetime import datetime, timedelta

import numpy as np
import pytest

from app import config
from app.inference_pipeline import (BASELINE_RELATIVE_FEATURES, MIN_PRIOR_TRANSACTIONS, FraudIntelligencePipeline)
from app.features.feature_engineering import FEATURE_COLUMNS

REAL = all((config.CANDIDATES_DIR / "v2" / c / "manifest.json").exists() for c in ("dnn_lstm", "dnn_only"))

NORMAL = {"amount": 2000.0, "merchant_category": "grocery", "device_id": "DEV_NEWCUST_A", "location": "Chennai",
          "failed_logins_24h": 0}
SUSPICIOUS = {"amount": 85000.0, "merchant_category": "electronics", "device_id": "DEV_UNKNOWN_7777",
              "location": "Lagos", "failed_logins_24h": 4}
T0 = datetime(2026, 8, 1, 12, 0)


class Recording:
    def __init__(self, model):
        self.model, self.inputs = model, []

    def predict(self, x, **kw):
        self.inputs.append(np.array(x))
        return self.model.predict(x, **kw)


def _grow(p, cid, k):
    """k earlier, consistent transactions for a new customer (scored through the pipeline)."""
    amounts = np.random.default_rng(3).normal(2000, 250, 40).round(2)
    for i in range(k):
        p.score_transaction({**NORMAL, "amount": float(amounts[i]), "customer_id": cid,
                             "timestamp": T0 + timedelta(days=i, minutes=17 * i % 90)})


def _score(p, cid, k, txn, record=True):
    random.seed(0)
    np.random.seed(0)
    real_dnn, real_lstm = p.dnn_model, p.lstm_model
    p.dnn_model = Recording(real_dnn)
    if real_lstm is not None:
        p.lstm_model = Recording(real_lstm)
    try:
        out = p.score_transaction({**txn, "customer_id": cid, "timestamp": T0 + timedelta(days=k, minutes=30)})
        dnn_in = p.dnn_model.inputs
        lstm_in = p.lstm_model.inputs if real_lstm is not None else None
    finally:
        p.dnn_model, p.lstm_model = real_dnn, real_lstm
    return out, dnn_in, lstm_in


@pytest.fixture(scope="module")
def candidate_pipelines():
    if not REAL:
        pytest.skip("trained v2 candidates are not present locally")
    return {n: FraudIntelligencePipeline(model_set=n) for n in ("v2_dnn_lstm", "v2_dnn_only")}


def _pipelines(pipeline, request, name):
    return pipeline if name == "production" else request.getfixturevalue("candidate_pipelines")[name]


KS = [0, 1, 2, 5, 9, 10, 15]
SETS = ["production", "v2_dnn_lstm", "v2_dnn_only"]


def test_minimum_history_is_the_sequence_length():
    assert MIN_PRIOR_TRANSACTIONS == config.SEQUENCE_LENGTH == 10
    assert set(BASELINE_RELATIVE_FEATURES) == {"amount_zscore", "hour_is_unusual", "is_new_device", "is_new_location",
                                              "category_is_unusual", "amount_pct_of_avg"}
    assert set(FEATURE_COLUMNS) - set(BASELINE_RELATIVE_FEATURES) == {"is_foreign_location", "failed_logins_24h",
                                                                      "txn_velocity_1h"}


@pytest.mark.parametrize("name", SETS)
@pytest.mark.parametrize("k", KS)
def test_cold_start_handling(pipeline, request, name, k):
    p = _pipelines(pipeline, request, name)
    cid = f"CUST_COLD_{name}_{k}"
    snapshot = dict(p.customer_histories)
    try:
        _grow(p, cid, k)
        out, dnn_in, lstm_in = _score(p, cid, k, NORMAL)
        ctx = out["history_context"]
        uses_lstm = p.model_set.uses_lstm
        cold = k < MIN_PRIOR_TRANSACTIONS

        # no fabricated history: exactly k earlier rows + this one
        assert len(p.customer_histories[cid]) == k + 1
        assert ctx["prior_transactions"] == k
        assert ctx["cold_start"] is cold
        assert ctx["lstm_used"] is (uses_lstm and not cold)

        # the LSTM is never run on a padded / empty window
        if uses_lstm:
            assert len(lstm_in) == (0 if cold else 1)
            if not cold:
                assert lstm_in[0].shape == (1, config.SEQUENCE_LENGTH, len(FEATURE_COLUMNS))

        cols = p.model_set.dnn_input_columns
        x = dnn_in[0][0]
        expected_imputed = (list(BASELINE_RELATIVE_FEATURES) if k == 0 else []) + \
                           (["risk_score"] if uses_lstm and cold else [])
        assert sorted(ctx["imputed_inputs"]) == sorted(expected_imputed)
        for c in expected_imputed:
            assert x[cols.index(c)] == 0.0                       # training mean, scaled
        if uses_lstm and cold:
            assert out["risk_score"] == round(float(p.dnn_mean[cols.index("risk_score")]), 2)
        if not uses_lstm:
            assert out["risk_score"] == out["fraud_probability"]
        # imputed (unobserved) inputs are never offered as reasons
        assert not {r["feature"] for r in out["reasons"]} & set(expected_imputed)

        # with full history the scaled inputs are the observed ones (unchanged behaviour)
        if not cold:
            hist = p.customer_histories[cid]
            feats = hist[FEATURE_COLUMNS].to_numpy(np.float32)[-1]
            raw = np.append(feats, out["risk_score"]) if uses_lstm else feats
            np.testing.assert_array_equal(x, np.clip((raw.astype(np.float32) - p.dnn_mean) / p.dnn_std, -6, 6))
    finally:
        p.customer_histories.clear()
        p.customer_histories.update(snapshot)


@pytest.mark.parametrize("name", SETS)
def test_first_transaction_still_uses_observed_risk_signals(pipeline, request, name):
    """Not hard-coded: a first transaction from a foreign city with failed logins
    scores higher than an ordinary first transaction."""
    p = _pipelines(pipeline, request, name)
    snapshot = dict(p.customer_histories)
    try:
        normal, _, _ = _score(p, f"CUST_FIRST_N_{name}", 0, NORMAL)
        susp, _, _ = _score(p, f"CUST_FIRST_S_{name}", 0, SUSPICIOUS)
    finally:
        p.customer_histories.clear()
        p.customer_histories.update(snapshot)
    assert susp["fraud_probability"] > normal["fraud_probability"]
    assert {r["feature"] for r in susp["reasons"]} <= {"is_foreign_location", "failed_logins_24h", "txn_velocity_1h"}


def test_seeded_customers_are_never_cold(pipeline):
    assert min(len(h) for h in pipeline.customer_histories.values()) >= MIN_PRIOR_TRANSACTIONS


def test_api_schema_unchanged_for_cold_start(client):
    r = client.post("/predict", json={**NORMAL, "customer_id": "CUST_BRAND_NEW", "timestamp": "2026-08-01T12:00:00"})
    assert r.status_code == 200
    body = r.json()
    assert "history_context" not in body
    assert set(body) == {"transaction_id", "customer_id", "timestamp", "amount", "merchant_category", "device_id",
                         "location", "failed_logins_24h", "risk_score", "fraud_probability", "alert_level",
                         "similarity_pct", "deviation_pct", "reasons"}
