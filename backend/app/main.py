"""
Fraud Intelligence Platform - FastAPI Backend
================================================
Run with (from the backend/ directory):
    uvicorn app.main:app --reload --port 8000

Endpoints:
    POST /predict                     score a single transaction
    GET  /customers                    list known customer IDs (for demo/testing)
    GET  /customer/{customer_id}/history   fraud evolution timeline for a customer
    GET  /health                        basic health check
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import desc

from .db.database import engine, get_db, Base
from .db import models as db_models
from .schemas import (
    TransactionInput, PredictionResponse, ExplanationReason,
    CustomerHistoryResponse, TimelinePoint,
)
from .inference_pipeline import get_pipeline

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


@app.post("/predict", response_model=PredictionResponse)
def predict_transaction(txn: TransactionInput, db: Session = Depends(get_db)):
    pipeline = get_pipeline()
    result = pipeline.score_transaction(txn.model_dump())

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
    db.commit()

    return PredictionResponse(
        transaction_id=result["transaction_id"],
        customer_id=result["customer_id"],
        timestamp=result["timestamp"],
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
