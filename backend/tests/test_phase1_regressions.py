"""
Phase 1 regression tests: one focused test group per audited Critical/High fix.
They exercise the real API with the real shipped models (nothing is mocked
except the SHAP backend in the pure label-logic tests).
"""
import io
import json
from datetime import datetime, timedelta

import numpy as np
import pytest

from app.config import BATCH_MAX_ROWS, MAX_TRANSACTION_AMOUNT
from app.models.shap_explainer import FraudExplainer, DNN_INPUT_COLUMNS

HIGH = {"High Risk", "Critical Risk"}


def scenario_payload(profile, **override):
    """The Typical-purchase payload exactly as the dashboard builds it from the profile."""
    s = dict(profile["typical_scenario"])
    payload = {
        "customer_id": profile["customer_id"],
        "amount": s["amount"], "merchant_category": s["merchant_category"],
        "device_id": s["device_id"], "location": s["location"],
        "failed_logins_24h": s["failed_logins_24h"],
    }
    if s.get("timestamp"):
        payload["timestamp"] = s["timestamp"]
    payload.update(override)
    return payload


def snapshot(pipeline):
    """Cheap fingerprint of every customer history (row counts + last transaction id)."""
    return {
        cid: (len(h), h["transaction_id"].iloc[-1] if len(h) else None)
        for cid, h in pipeline.customer_histories.items()
    }


# ----------------------------------------------------------------------- C1
def test_typical_purchase_is_derived_from_history_and_scores_low(client, take_customers):
    customers = take_customers(20)
    levels, probs = [], []
    for cid in customers:
        profile = client.get(f"/customer/{cid}/profile").json()
        scenario = profile["typical_scenario"]
        assert scenario is not None
        # values come from the customer's OWN history, in the dataset's real device format
        assert scenario["device_id"] in {d["value"] for d in profile["devices"]}
        assert scenario["device_id"].startswith("DEV_") and "CUST" not in scenario["device_id"]
        assert scenario["location"] == profile["primary_location"]
        assert scenario["amount"] == profile["typical_amount"]
        assert datetime.fromisoformat(scenario["timestamp"]).hour in profile["preferred_hours"]
        r = client.post("/predict", json=scenario_payload(profile))
        assert r.status_code == 200, r.text
        levels.append(r.json()["alert_level"])
        probs.append(r.json()["fraud_probability"])
    assert levels.count("Low Risk") >= 18, (levels, probs)       # overwhelmingly Low
    assert not any(l in HIGH for l in levels), (levels, probs)    # never systematically High/Critical


# ------------------------------------------------- suspicious presets still work
def test_suspicious_patterns_still_score_high(client, take_customers):
    preset, new_dev_foreign, all_signals = take_customers(5), take_customers(5), take_customers(5)
    for cid in preset:   # the dashboard's existing "Suspicious pattern" preset (sent at "now")
        r = client.post("/predict", json={"customer_id": cid, "amount": 75000, "merchant_category": "electronics",
                                          "device_id": "DEV_UNKNOWN_9999", "location": "Lagos", "failed_logins_24h": 4})
        assert r.json()["alert_level"] in HIGH, r.json()
    for cid in new_dev_foreign:
        p = client.get(f"/customer/{cid}/profile").json()
        r = client.post("/predict", json=scenario_payload(p, device_id="DEV_UNKNOWN_5555", location="Lagos"))
        assert r.json()["alert_level"] in HIGH, r.json()
    for cid in all_signals:
        p = client.get(f"/customer/{cid}/profile").json()
        bad_hour = next(h for h in range(24) if h not in p["preferred_hours"])
        ts = datetime.fromisoformat(p["typical_scenario"]["timestamp"]).replace(hour=bad_hour)
        r = client.post("/predict", json=scenario_payload(
            p, amount=p["typical_amount"] * 10, device_id="DEV_UNKNOWN_7777", location="Lagos",
            failed_logins_24h=5, timestamp=ts.isoformat()))
        assert r.json()["alert_level"] in HIGH, r.json()


# ----------------------------------------------------------------------- C2
VALID = {"customer_id": "CUST_0100", "amount": 1500, "merchant_category": "grocery",
         "device_id": "DEV_0100_A", "location": "Hyderabad", "failed_logins_24h": 0}

BAD_BODIES = {
    "NaN": json.dumps({**VALID, "amount": float("nan")}),            # json.dumps emits NaN
    "Infinity": json.dumps({**VALID, "amount": float("inf")}),
    "1e400": '{"customer_id":"CUST_0100","amount":1e400,"merchant_category":"grocery","device_id":"d","location":"l"}',
    "negative amount": json.dumps({**VALID, "amount": -500}),
    "zero amount": json.dumps({**VALID, "amount": 0}),
    "amount over bound": json.dumps({**VALID, "amount": MAX_TRANSACTION_AMOUNT * 10}),
    "1e308": json.dumps({**VALID, "amount": 1e308}),
    "negative failed logins": json.dumps({**VALID, "failed_logins_24h": -1}),
    "huge failed logins": json.dumps({**VALID, "failed_logins_24h": 10 ** 9}),
    "huge device_id": json.dumps({**VALID, "device_id": "D" * 100_000}),
    "huge customer_id": json.dumps({**VALID, "customer_id": "C" * 100_000}),
    "huge location": json.dumps({**VALID, "location": "L" * 100_000}),
    "empty customer_id": json.dumps({**VALID, "customer_id": ""}),
    "missing amount": json.dumps({k: v for k, v in VALID.items() if k != "amount"}),
    "missing device": json.dumps({k: v for k, v in VALID.items() if k != "device_id"}),
    "wrong type amount": json.dumps({**VALID, "amount": "abc"}),
    "wrong type logins": json.dumps({**VALID, "failed_logins_24h": "x"}),
    "malformed json": '{"customer_id": ',
    "malformed timestamp": json.dumps({**VALID, "timestamp": "yesterday"}),
    "timestamp year 1999": json.dumps({**VALID, "timestamp": "1999-01-01T00:00:00"}),
    "timestamp far future": json.dumps({**VALID, "timestamp": "2099-01-01T00:00:00"}),
}


@pytest.mark.parametrize("name", list(BAD_BODIES))
def test_invalid_input_is_rejected_cleanly_and_mutates_nothing(client, pipeline, name):
    before_hist = snapshot(pipeline)
    before_cust = set(pipeline.customer_histories)
    r = client.post("/predict", content=BAD_BODIES[name], headers={"content-type": "application/json"})
    assert 400 <= r.status_code < 500, (name, r.status_code, r.text[:200])
    body = r.json()                       # must be valid JSON with a clear message
    assert "detail" in body
    assert len(r.text) < 5000             # offending input is not echoed back
    assert snapshot(pipeline) == before_hist
    assert set(pipeline.customer_histories) == before_cust


def test_unknown_customer_creates_no_history_when_request_is_invalid(client, pipeline):
    before = set(pipeline.customer_histories)
    r = client.post("/predict", json={**VALID, "customer_id": "BRAND_NEW_CUSTOMER", "amount": -1})
    assert r.status_code == 422
    assert "BRAND_NEW_CUSTOMER" not in pipeline.customer_histories
    assert set(pipeline.customer_histories) == before


# ----------------------------------------------------------- H6 / history integrity
def test_valid_invalid_valid_sequence_does_not_alter_baseline(client, pipeline, take_customers):
    cid = take_customers(1)[0]
    p = client.get(f"/customer/{cid}/profile").json()
    t1 = datetime.fromisoformat(p["typical_scenario"]["timestamp"])
    assert client.post("/predict", json=scenario_payload(p, timestamp=t1.isoformat())).status_code == 200
    history_after_first = pipeline.customer_histories[cid].copy()

    for body in (BAD_BODIES["Infinity"], BAD_BODIES["NaN"], BAD_BODIES["1e400"], BAD_BODIES["negative amount"]):
        r = client.post("/predict", content=body.replace("CUST_0100", cid), headers={"content-type": "application/json"})
        assert r.status_code == 422
    assert pipeline.customer_histories[cid].equals(history_after_first)   # byte-for-byte unchanged

    t2 = t1 + timedelta(hours=1)
    payload2 = scenario_payload(p, timestamp=t2.isoformat())
    r = client.post("/predict", json=payload2)
    assert r.status_code == 200
    # the second valid result must equal what the pipeline computes from the clean history
    expected, _ = pipeline._compute({**payload2, "timestamp": t2}, history_after_first)
    got = r.json()
    for key in ("risk_score", "fraud_probability", "similarity_pct", "deviation_pct", "alert_level"):
        assert got[key] == expected[key], (key, got[key], expected[key])


# ----------------------------------------------------------------------- H9
def test_back_dated_transaction_is_scored_against_only_the_history_before_it(client, pipeline, take_customers):
    cid = take_customers(1)[0]
    p = client.get(f"/customer/{cid}/profile").json()
    day = datetime.fromisoformat(p["typical_scenario"]["timestamp"]).replace(minute=0)
    later, earlier = day.replace(hour=20), day.replace(hour=5)
    assert client.post("/predict", json=scenario_payload(p, timestamp=later.isoformat())).status_code == 200
    hist_with_later = pipeline.customer_histories[cid].copy()
    hist_without_later = hist_with_later[hist_with_later["timestamp"] < later].reset_index(drop=True)

    attack = scenario_payload(p, amount=p["typical_amount"] * 10, device_id="DEV_UNKNOWN_7777", location="Lagos",
                              failed_logins_24h=5, timestamp=earlier.isoformat())
    r = client.post("/predict", json=attack)
    assert r.status_code == 200
    got = r.json()
    expected, _ = pipeline._compute({**attack, "timestamp": earlier}, hist_without_later)
    for key in ("risk_score", "fraud_probability", "similarity_pct", "alert_level"):
        assert got[key] == expected[key], (key, got[key], expected[key])
    assert got["alert_level"] in HIGH
    assert any(rs["feature"] in ("is_new_device", "is_foreign_location", "is_new_location") for rs in got["reasons"])


def test_timezone_aware_and_malformed_timestamps(client, take_customers):
    cid = take_customers(1)[0]
    p = client.get(f"/customer/{cid}/profile").json()
    ok = client.post("/predict", json=scenario_payload(p, timestamp="2026-10-07T10:00:00Z"))
    assert ok.status_code == 200, ok.text                       # was a 500
    assert client.post("/predict", json=scenario_payload(p, timestamp="not-a-date")).status_code == 422


# ----------------------------------------------------------------------- H7
def _csv(rows, header="customer_id,amount,merchant_category,device_id,location,failed_logins_24h"):
    return (header + "\n" + "\n".join(rows) + "\n").encode()


def _post_csv(client, data):
    return client.post("/predict/batch", files={"file": ("t.csv", io.BytesIO(data), "text/csv")})


@pytest.mark.parametrize("n", [5, 10, 20])
def test_batch_sizes(client, take_customers, n):
    ids = take_customers(n)
    rows = []
    for cid in ids:
        p = client.get(f"/customer/{cid}/profile").json()
        rows.append(f"{cid},{p['typical_amount']},{p['typical_scenario']['merchant_category']},,,0")   # blank device/location
    r = _post_csv(client, _csv(rows))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] == n and sum(body["summary"].values()) == n
    assert set(body) == {"count", "summary", "results"}                         # schema unchanged
    assert set(body["results"][0]) == {"transaction_id", "customer_id", "amount", "risk_score", "fraud_probability", "alert_level"}


def test_batch_is_all_or_nothing_and_reports_bad_rows(client, pipeline, take_customers):
    good1, good2 = take_customers(2)
    before = snapshot(pipeline)
    rows = [f"{good1},1500,grocery,,,0", f"{good2},NaN,grocery,,,0", f"{good2},-5,grocery,,,0",
            f"{good2},inf,grocery,,,0", f"{good2},1e400,grocery,,,0", f"{good2},abc,grocery,,,0",
            f"{good2},100,grocery,,,-3", f"{good2},100,,,,0"]
    r = _post_csv(client, _csv(rows))
    assert r.status_code == 422, r.text
    assert "row 3" in r.json()["detail"] and "nothing was scored" in r.json()["detail"]
    assert snapshot(pipeline) == before          # the one valid row was NOT scored either


def test_batch_oversized_and_malformed_files(client, pipeline, take_customers):
    cid = take_customers(1)[0]
    before = snapshot(pipeline)
    r = _post_csv(client, _csv([f"{cid},100,grocery,,,0"] * (BATCH_MAX_ROWS + 1)))
    assert r.status_code == 413
    assert _post_csv(client, b"").status_code == 400
    assert _post_csv(client, b"customer_id,amount\nCUST_0001,1\n").status_code == 400
    assert _post_csv(client, bytes(range(256)) * 8).status_code == 400
    assert snapshot(pipeline) == before


# ----------------------------------------------------------------------- H4
class _StubShap:
    def __init__(self, values):
        self.values = np.array(values, dtype=float)

    def shap_values(self, x):
        return self.values.reshape(1, -1, 1)


def _explain(shap_values, **raw):
    ex = FraudExplainer.__new__(FraudExplainer)
    ex.explainer = _StubShap(shap_values)
    base = dict(amount_zscore=0.0, hour_is_unusual=0, is_new_device=0, is_new_location=0, is_foreign_location=0,
                failed_logins_24h=0, category_is_unusual=0, txn_velocity_1h=0, amount_pct_of_avg=100.0, risk_score=10.0)
    base.update(raw)
    raw_vec = np.array([base[c] for c in DNN_INPUT_COLUMNS], dtype=float)
    return ex.explain(np.zeros((1, 10)), raw_values=raw_vec)


def _shap(**contrib):
    return [contrib.get(c, 0.0) for c in DNN_INPUT_COLUMNS]


def test_shap_labels_follow_actual_feature_state():
    # velocity is 0 but has a positive SHAP value -> must NOT be called "High Transaction Velocity"
    out = _explain(_shap(txn_velocity_1h=0.4, hour_is_unusual=0.3), hour_is_unusual=1, txn_velocity_1h=0)
    assert [r["feature"] for r in out] == ["hour_is_unusual"]
    assert all("High Transaction Velocity" != r["display_name"] for r in out)
    # velocity genuinely high -> label allowed
    out = _explain(_shap(txn_velocity_1h=0.4), txn_velocity_1h=3)
    assert out[0]["display_name"] == "High Transaction Velocity"
    # risk score 17 is not "elevated"; 60 is
    assert _explain(_shap(risk_score=0.2), risk_score=17.0)[0]["display_name"] != "Elevated Behavioral Risk Score"
    assert _explain(_shap(risk_score=0.2), risk_score=60.0)[0]["display_name"] == "Elevated Behavioral Risk Score"
    # nothing anomalous at all -> neutral wording only, never an alarming label
    out = _explain(_shap(is_new_device=0.2, txn_velocity_1h=0.1))
    assert out and all(r["display_name"] in ("Device (previously used)", "Transaction velocity (normal)") for r in out)
    # no positive contribution -> no reasons
    assert _explain(_shap(is_new_device=-0.2)) == []


def test_reasons_returned_by_the_api_never_name_an_inactive_feature(client, take_customers):
    for cid in take_customers(8):
        p = client.get(f"/customer/{cid}/profile").json()
        bad_hour = next(h for h in range(24) if h not in p["preferred_hours"])
        ts = datetime.fromisoformat(p["typical_scenario"]["timestamp"]).replace(hour=bad_hour)
        r = client.post("/predict", json=scenario_payload(p, timestamp=ts.isoformat())).json()
        for reason in r["reasons"]:
            if reason["feature"] == "txn_velocity_1h":   # these customers have no transaction in the previous hour
                assert reason["display_name"] != "High Transaction Velocity"


# ------------------------------------------------------------------- H3 / H8 / misc
def test_metrics_include_pr_auc_fpr_and_framing(client):
    m = client.get("/metrics").json()
    assert "synthetic" in m["evaluation_context"]["dataset"].lower()
    assert m["evaluation_context"]["scores_are_calibrated_probabilities"] is False
    for key in ("lstm_risk_predictor", "dnn_fraud_classifier"):
        d = m[key]
        cm = d["confusion_matrix"]
        assert 0.0 <= d["pr_auc"] <= 1.0
        assert d["false_positive_rate"] == pytest.approx(cm["false_positive"] / (cm["false_positive"] + cm["true_negative"]), abs=1e-4)
        assert {"precision", "recall", "f1_score", "auc_roc"} <= set(d)      # existing keys preserved


def test_profile_endpoint_shape_and_404(client, take_customers):
    cid = take_customers(1)[0]
    p = client.get(f"/customer/{cid}/profile").json()
    assert p["n_transactions"] >= 30 and len(p["hour_distribution"]) == 24
    assert abs(sum(p["hour_distribution"]) - 1.0) < 0.01
    assert p["primary_device"] and p["primary_location"] and p["preferred_hours"]
    assert client.get("/customer/NO_SUCH_CUSTOMER/profile").status_code == 404


def test_core_endpoints_and_pdf(client, take_customers):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/customers").json()["count"] >= 500
    cid = take_customers(1)[0]
    p = client.get(f"/customer/{cid}/profile").json()
    pred = client.post("/predict", json=scenario_payload(p)).json()
    assert {"transaction_id", "risk_score", "fraud_probability", "alert_level", "similarity_pct", "deviation_pct", "reasons"} <= set(pred)
    assert client.get(f"/customer/{cid}/history").json()["n_transactions"] >= 1
    # Phase 2: the report request carries only the transaction id (everything printed comes from the server's record)
    pdf = client.post("/report/pdf", json={"transaction_id": pred["transaction_id"]})
    assert pdf.status_code == 200 and pdf.content[:5] == b"%PDF-"
