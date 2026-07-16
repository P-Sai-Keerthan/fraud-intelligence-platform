"""
Fraud Intelligence Platform - FastAPI Backend
================================================
Run with (from the backend/ directory):
    uvicorn app.main:app --reload --port 8000

Endpoints:
    POST /predict                     score a single transaction
    POST /predict/batch                score a CSV of transactions at once
    GET  /customers                    list known customer IDs (for demo/testing)
    GET  /customer/{customer_id}/history   fraud evolution timeline for a customer
    GET  /fraud-rings                   customers linked by a shared device/identifier
    GET  /metrics                       held-out test-set model performance (precision/recall/F1/AUC-ROC)
    POST /report/pdf                    downloadable PDF explanation report for one prediction
    GET  /health                        basic health check
"""

import io

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
    CustomerHistoryResponse, TimelinePoint,
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


@app.get("/customer/{customer_id}/history", response_model=CustomerHistoryResponse)
def get_customer_history(customer_id: str, limit: int = 100, db: Session = Depends(get_db)):
    rows = (
        db.query(db_models.Transaction)
        .filter(db_models.Transaction.customer_id == customer_id)
        .order_by(db_models.Transaction.timestamp)
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


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Score every row of an uploaded CSV in one call. Required columns:
    customer_id, amount, merchant_category. Optional: device_id, location,
    failed_logins_24h."""
    pipeline = get_pipeline()
    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    required_cols = {"customer_id", "amount", "merchant_category"}
    missing = required_cols - set(df.columns)
    if missing:
        raise HTTPException(status_code=400, detail=f"CSV is missing required columns: {sorted(missing)}")

    results = []
    summary = {"Low Risk": 0, "Medium Risk": 0, "High Risk": 0, "Critical Risk": 0}
    for _, row in df.iterrows():
        customer_id = str(row["customer_id"])
        device_id = str(row["device_id"]) if "device_id" in df.columns and pd.notna(row.get("device_id")) else ""
        location = str(row["location"]) if "location" in df.columns and pd.notna(row.get("location")) else ""
        txn = {
            "customer_id": customer_id,
            "amount": float(row["amount"]),
            "merchant_category": str(row["merchant_category"]),
            # an omitted device/location isn't itself suspicious -- fall back
            # to this customer's presumed home device/city, same convention
            # the dashboard's manual scan form uses, so a minimal CSV
            # (just customer_id/amount/category) doesn't get misread as
            # "every row uses a brand-new device in a foreign city"
            "device_id": device_id or f"DEV_{customer_id}_A",
            "location": location or "Hyderabad",
            "failed_logins_24h": int(row["failed_logins_24h"]) if "failed_logins_24h" in df.columns and pd.notna(row.get("failed_logins_24h")) else 0,
        }
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
