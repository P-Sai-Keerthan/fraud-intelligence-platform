"""
Fraud Intelligence Platform - FastAPI Backend
================================================
Run with (from the backend/ directory):
    uvicorn app.main:app --reload --port 8000

Endpoints:
    POST /predict                     score a single transaction
    POST /predict/batch                score a CSV of transactions at once
    GET  /customers                    list known customer IDs (for demo/testing)
    GET  /customer/{customer_id}/profile   a customer's usual (home) device and city
    GET  /customer/{customer_id}/history   fraud evolution timeline for a customer
    GET  /fraud-rings                   customers linked by a shared device/identifier
    GET  /metrics                       held-out test-set model performance (precision/recall/F1/AUC-ROC)
    POST /report/pdf                    downloadable PDF explanation report for one prediction
    GET  /health                        basic health check
"""

import io
import math
import re

import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import desc

from .db.database import engine, get_db, Base
from .db import models as db_models
from .schemas import (
    TransactionInput, PredictionResponse, ExplanationReason,
    CustomerHistoryResponse, TimelinePoint, CustomerProfile,
    FraudRingsResponse, BatchPredictionResponse,
)
from .inference_pipeline import get_pipeline
from .models.evaluate import evaluate_all
from .report import build_pdf_report

# create DB tables on startup if they don't exist
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Explainable Fraud Intelligence Platform API",
    description="Behavioral Fraud DNA, real-time fraud detection, and explainable AI for banking transactions.",
    version="1.0.0",
)

# allow the React dashboard (running on a different port during development) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your actual frontend origin before deploying publicly
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def load_models_on_startup():
    # forces the (potentially slow) model-loading step to happen once at
    # startup rather than on the first incoming request
    get_pipeline()


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/customers")
def list_customers(limit: int = 50):
    pipeline = get_pipeline()
    ids = pipeline.known_customer_ids()
    return {"count": len(ids), "customer_ids": ids[:limit]}


def _persist_transaction(db: Session, result: dict):
    # persist to DB so it shows up in the customer's fraud evolution timeline
    db_txn = db_models.Transaction(
        transaction_id=result["transaction_id"],
        customer_id=result["customer_id"],
        timestamp=result["timestamp"],
        amount=result["amount"],
        merchant_category=result["merchant_category"],
        device_id=result["device_id"],
        location=result["location"],
        failed_logins_24h=result["failed_logins_24h"],
        risk_score=result["risk_score"],
        fraud_probability=result["fraud_probability"],
        similarity_pct=result["similarity_pct"],
        deviation_pct=result["deviation_pct"],
        alert_level=result["alert_level"],
    )
    db.add(db_txn)


@app.post("/predict", response_model=PredictionResponse)
def predict_transaction(txn: TransactionInput, db: Session = Depends(get_db)):
    pipeline = get_pipeline()
    result = pipeline.score_transaction(txn.model_dump())
    _persist_transaction(db, result)
    db.commit()

    return PredictionResponse(
        transaction_id=result["transaction_id"],
        customer_id=result["customer_id"],
        timestamp=result["timestamp"],
        amount=result["amount"],
        merchant_category=result["merchant_category"],
        device_id=result["device_id"],
        location=result["location"],
        failed_logins_24h=result["failed_logins_24h"],
        risk_score=result["risk_score"],
        fraud_probability=result["fraud_probability"],
        alert_level=result["alert_level"],
        similarity_pct=result["similarity_pct"],
        deviation_pct=result["deviation_pct"],
        reasons=[ExplanationReason(**r) for r in result["reasons"]],
    )


@app.get("/customer/{customer_id}/profile", response_model=CustomerProfile)
def get_customer_profile(customer_id: str):
    """The customer's usual device and home city, derived from their own
    transaction history. The dashboard uses these as the defaults for a
    normal transaction, and batch scoring uses them for blank device_id /
    location cells."""
    profile = get_pipeline().get_customer_profile(customer_id)
    if profile is None:
        raise HTTPException(
            status_code=404,
            detail=f"No transaction history for '{customer_id}', so there is no home device or city to report.",
        )
    return CustomerProfile(**profile)


@app.get("/customer/{customer_id}/history", response_model=CustomerHistoryResponse)
def get_customer_history(customer_id: str, limit: int = 100, db: Session = Depends(get_db)):
    # the `limit` MOST RECENT scored transactions, newest first
    rows = (
        db.query(db_models.Transaction)
        .filter(db_models.Transaction.customer_id == customer_id)
        .order_by(desc(db_models.Transaction.timestamp))
        .limit(limit)
        .all()
    )
    if not rows:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No SCORED transactions found for '{customer_id}'. "
                "This endpoint only shows transactions that went through /predict "
                "in this session, not the raw seed/training history. "
                "Try POSTing a transaction for this customer first."
            ),
        )

    timeline = [
        TimelinePoint(
            transaction_id=r.transaction_id,
            timestamp=r.timestamp,
            risk_score=r.risk_score,
            fraud_probability=r.fraud_probability,
            alert_level=r.alert_level,
        )
        for r in rows
    ]
    return CustomerHistoryResponse(customer_id=customer_id, n_transactions=len(timeline), timeline=timeline)


@app.get("/fraud-rings", response_model=FraudRingsResponse)
def get_fraud_rings(min_customers: int = 2):
    """Customers who share a device (or other identifier) that a single
    legitimate customer would never plausibly share with another -- a
    strong signal of an organized fraud ring rather than one customer
    behaving oddly on their own."""
    pipeline = get_pipeline()
    rings = pipeline.detect_fraud_rings(min_customers=min_customers)
    return FraudRingsResponse(count=len(rings), rings=rings)


@app.get("/metrics")
def get_metrics(refresh: bool = False):
    """Held-out test-set performance for both trained models (precision,
    recall, F1, AUC-ROC, confusion matrix) -- computed once and cached,
    pass ?refresh=true to force recomputation."""
    return evaluate_all(force_refresh=refresh)


_DECIMAL_RE = re.compile(r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?")
_WHOLE_NUMBER_RE = re.compile(r"\d+(\.0*)?")
_MAX_REPORTED_ROW_ERRORS = 20


def _validate_batch_rows(df: pd.DataFrame, pipeline) -> tuple[list[dict], list[str]]:
    """Checks every CSV row up front and returns (transactions, errors).
    Values are read as raw strings and are never coerced: a bad value is an
    error naming its row and field. Rows are numbered from 1 (the first row
    after the header). A blank device_id / location falls back to that
    customer's usual device / home city (GET /customer/{id}/profile)."""
    has_device = "device_id" in df.columns
    has_location = "location" in df.columns
    has_logins = "failed_logins_24h" in df.columns

    txns, errors = [], []
    for row_number, (_, row) in enumerate(df.iterrows(), start=1):
        customer_id = row["customer_id"].strip()
        where = f"row {row_number}" + (f" (customer_id={customer_id})" if customer_id else "")
        row_errors = []

        if not customer_id:
            row_errors.append("customer_id is required")

        raw_amount = row["amount"].strip()
        amount = None
        if not raw_amount:
            row_errors.append("amount is required")
        elif not _DECIMAL_RE.fullmatch(raw_amount):
            row_errors.append(f"amount must be a number, got '{raw_amount}'")
        else:
            amount = float(raw_amount)
            if not math.isfinite(amount):
                row_errors.append(f"amount must be a finite number, got '{raw_amount}'")
            elif amount <= 0:
                row_errors.append(f"amount must be greater than 0, got '{raw_amount}'")

        merchant_category = row["merchant_category"].strip()
        if not merchant_category:
            row_errors.append("merchant_category is required")

        failed_logins = 0
        raw_logins = row["failed_logins_24h"].strip() if has_logins else ""
        if raw_logins:
            if not _WHOLE_NUMBER_RE.fullmatch(raw_logins):
                row_errors.append(f"failed_logins_24h must be a whole number >= 0, got '{raw_logins}'")
            else:
                failed_logins = int(float(raw_logins))

        device_id = row["device_id"].strip() if has_device else ""
        location = row["location"].strip() if has_location else ""
        if customer_id and (not device_id or not location):
            profile = pipeline.get_customer_profile(customer_id)
            if profile is None:
                missing = [f for f, v in (("device_id", device_id), ("location", location)) if not v]
                row_errors.append(
                    f"{' and '.join(missing)} required: this customer has no transaction "
                    "history to take a home device/city from"
                )
            else:
                device_id = device_id or profile["home_device"]
                location = location or profile["home_location"]

        if row_errors:
            errors.append(f"{where}: " + "; ".join(row_errors))
        else:
            txns.append({
                "customer_id": customer_id,
                "amount": amount,
                "merchant_category": merchant_category,
                "device_id": device_id,
                "location": location,
                "failed_logins_24h": failed_logins,
            })
    return txns, errors


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Score every row of an uploaded CSV in one call. Required columns:
    customer_id, amount, merchant_category. Optional: device_id, location,
    failed_logins_24h. The whole file is validated before anything is
    scored: if any row is invalid, nothing is scored or saved and the 400
    response lists the bad rows."""
    pipeline = get_pipeline()
    contents = await file.read()
    try:
        # read every cell as a raw string so invalid values surface as
        # validation errors instead of being silently coerced by pandas
        df = pd.read_csv(io.BytesIO(contents), dtype=str, keep_default_na=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    required_cols = {"customer_id", "amount", "merchant_category"}
    missing = required_cols - set(df.columns)
    if missing:
        raise HTTPException(status_code=400, detail=f"CSV is missing required columns: {sorted(missing)}")

    txns, errors = _validate_batch_rows(df, pipeline)
    if errors:
        shown = errors[:_MAX_REPORTED_ROW_ERRORS]
        more = len(errors) - len(shown)
        detail = (
            f"Invalid CSV data in {len(errors)} row(s); nothing was scored. "
            + " | ".join(shown)
            + (f" | ...and {more} more invalid row(s)" if more else "")
        )
        raise HTTPException(status_code=400, detail=detail)

    results = []
    summary = {"Low Risk": 0, "Medium Risk": 0, "High Risk": 0, "Critical Risk": 0}
    for txn in txns:
        result = pipeline.score_transaction(txn)
        _persist_transaction(db, result)
        summary[result["alert_level"]] = summary.get(result["alert_level"], 0) + 1
        results.append({
            "transaction_id": result["transaction_id"],
            "customer_id": result["customer_id"],
            "amount": result["amount"],
            "risk_score": result["risk_score"],
            "fraud_probability": result["fraud_probability"],
            "alert_level": result["alert_level"],
        })

    db.commit()
    return BatchPredictionResponse(count=len(results), summary=summary, results=results)


@app.post("/report/pdf")
def get_pdf_report(prediction: PredictionResponse):
    """Generate a downloadable PDF explaining a single prediction result --
    takes exactly what POST /predict returns, so the frontend can request
    a report for whatever is currently on screen."""
    pdf_bytes = build_pdf_report(prediction.model_dump())
    filename = f"fraud_report_{prediction.transaction_id}.pdf"
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
