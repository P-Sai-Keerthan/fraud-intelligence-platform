"""POST /report/pdf -- downloadable explanation report."""

import io
import re

import pytest

from app.report import build_pdf_report


def _extract_text(pdf_bytes):
    pypdf = pytest.importorskip("pypdf")
    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def test_pdf_report_for_suspicious_prediction(client, suspicious_txn):
    prediction = client.post("/predict", json=suspicious_txn).json()
    r = client.post("/report/pdf", json=prediction)
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    disposition = r.headers["content-disposition"]
    assert disposition.startswith("attachment")
    assert f'fraud_report_{prediction["transaction_id"]}.pdf' in disposition
    assert r.content.startswith(b"%PDF")
    assert r.content.rstrip().endswith(b"%%EOF")


def test_pdf_report_for_normal_prediction_without_reasons(client, normal_txn):
    prediction = client.post("/predict", json=normal_txn).json()
    assert prediction["reasons"] == []
    r = client.post("/report/pdf", json=prediction)
    assert r.status_code == 200
    assert r.content.startswith(b"%PDF")


def test_pdf_report_contains_prediction_details(client, suspicious_txn):
    prediction = client.post("/predict", json=suspicious_txn).json()
    text = _extract_text(client.post("/report/pdf", json=prediction).content)
    compact = re.sub(r"\s+", "", text)
    assert prediction["transaction_id"] in compact
    assert "CUST_0001" in compact
    assert "CriticalRisk" in compact


def test_pdf_report_rejects_a_body_without_a_transaction_id(client):
    assert client.post("/report/pdf", json={}).status_code == 422
    assert client.post("/report/pdf", json={"customer_id": "CUST_0001"}).status_code == 422


def test_pdf_report_for_an_unknown_transaction_is_404(client):
    r = client.post("/report/pdf", json={"transaction_id": "TXN_X"})
    assert r.status_code == 404 and "TXN_X" in r.json()["detail"]


@pytest.mark.parametrize("bad", ['TXN"; evil=1', "TXN_\u00e9\u4e2d", "TXN\r\nX: 1", "../../etc/passwd", "A" * 65, ""])
def test_pdf_report_rejects_malformed_transaction_ids(client, bad):
    assert client.post("/report/pdf", json={"transaction_id": bad}).status_code == 422


def test_pdf_report_only_needs_the_transaction_id(client, normal_txn):
    pred = client.post("/predict", json=normal_txn).json()
    r = client.post("/report/pdf", json={"transaction_id": pred["transaction_id"]})
    assert r.status_code == 200 and r.content.startswith(b"%PDF")


# ---- R-04: the printed values come from the database, never from the request ----------------------------

def _flat(text):
    return re.sub(r"\s+", " ", text)


def test_client_supplied_scores_cannot_change_the_report(client, normal_txn):
    pred = client.post("/predict", json=normal_txn).json()
    honest = _flat(_extract_text(client.post("/report/pdf", json={"transaction_id": pred["transaction_id"]}).content))
    forged = dict(pred, risk_score=99.9, fraud_probability=99.9, alert_level="Critical Risk", similarity_pct=1.0,
                  deviation_pct=99.0, amount=999999.0, customer_id="CUST_9999", location="Atlantis", device_id="FORGED",
                  reasons=[{"feature": "x", "display_name": "Forged reason", "shap_value": 9.9}])
    tampered = _flat(_extract_text(client.post("/report/pdf", json=forged).content))
    assert tampered == honest                                    # byte-for-byte the same text
    for marker in ("Forged reason", "Atlantis", "FORGED", "CUST_9999", "999,999", "99.9 / 100"):
        assert marker not in tampered
    assert "Low Risk" in tampered and f"{pred['fraud_probability']:.1f} / 100" in tampered


def test_report_prints_the_stored_values(client, suspicious_txn):
    pred = client.post("/predict", json=suspicious_txn).json()
    text = _flat(_extract_text(client.post("/report/pdf", json={"transaction_id": pred["transaction_id"]}).content))
    assert f"{pred['risk_score']:.1f} / 100" in text and f"{pred['fraud_probability']:.1f} / 100" in text
    assert pred["alert_level"].replace(" ", "") in text.replace(" ", "")
    for reason in pred["reasons"]:
        assert reason["display_name"] in text and f"{reason['shap_value']:+.3f}" in text


def test_report_explanation_survives_a_round_trip_through_the_database(client, suspicious_txn):
    pred = client.post("/predict", json=suspicious_txn).json()
    assert pred["reasons"]                                       # the suspicious transaction has reasons
    from app.db import database as dbm, models as models_
    s = dbm.SessionLocal()
    try:
        row = s.get(models_.Transaction, pred["transaction_id"])
        import json as _json
        assert _json.loads(row.reasons_json) == pred["reasons"]
    finally:
        s.close()


def test_report_for_a_row_without_a_stored_explanation_says_so_instead_of_inventing_one(client, normal_txn):
    pred = client.post("/predict", json=normal_txn).json()
    from app.db import database as dbm, models as models_
    s = dbm.SessionLocal()
    try:
        s.get(models_.Transaction, pred["transaction_id"]).reasons_json = None     # a row from before the column
        s.commit()
    finally:
        s.close()
    text = _flat(_extract_text(client.post("/report/pdf", json={"transaction_id": pred["transaction_id"]}).content))
    assert "was not stored for this transaction" in text
    assert "No significant fraud-indicating factors" not in text


def test_build_pdf_report_directly():
    pdf = build_pdf_report({
        "transaction_id": "TXN_UNIT00001",
        "customer_id": "CUST_0001",
        "timestamp": "2026-07-10T13:30:00",
        "amount": 1000.0,
        "merchant_category": "grocery",
        "device_id": "DEV_0001_A",
        "location": "Pune",
        "failed_logins_24h": 0,
        "risk_score": 10.0,
        "fraud_probability": 1.0,
        "alert_level": "Low Risk",
        "similarity_pct": 90.0,
        "deviation_pct": 10.0,
        "reasons": [],
    })
    assert isinstance(pdf, bytes)
    assert pdf.startswith(b"%PDF")
