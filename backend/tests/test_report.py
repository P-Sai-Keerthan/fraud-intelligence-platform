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


def test_pdf_report_rejects_incomplete_body(client):
    r = client.post("/report/pdf", json={"transaction_id": "TXN_X"})
    assert r.status_code == 422


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
