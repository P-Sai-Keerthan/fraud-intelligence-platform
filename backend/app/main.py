"""
Fraud Intelligence Platform - FastAPI Backend
================================================
Run with (from the backend/ directory):
    uvicorn app.main:app --reload --port 8000

Endpoints:
    POST /predict                     score a single transaction
    POST /predict/batch                score a CSV of transactions at once (all-or-nothing, max rows enforced)
    GET  /customers                    list known customer IDs (for demo/testing)
    GET  /customer/{customer_id}/history   fraud evolution timeline for a customer
    GET  /customer/{customer_id}/profile   behavioral context derived from the customer's own history
    GET  /fraud-rings                   customers linked by a shared device/identifier
    GET  /metrics                       held-out test-set model performance (precision/recall/F1/ROC-AUC/PR-AUC/FPR)
    POST /report/pdf                    PDF report for one prediction, built from the SERVER's record (body: {"transaction_id": ...})
    GET  /health                        basic health check
"""

import io
import threading

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Path, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from sqlalchemy import desc

from .db.database import engine, get_db, Base
from .db import models as db_models
from .schemas import (
    TransactionInput, PredictionResponse, ExplanationReason,
    CustomerHistoryResponse, TimelinePoint, CustomerProfileResponse,
    FraudRingsResponse, BatchPredictionResponse, ReportRequest,
)
from .batch import parse_batch_csv, BatchError
from .config import (
    BATCH_MAX_BYTES, MAX_ID_LENGTH, MAX_JSON_BODY_BYTES, MAX_REPORT_BODY_BYTES, MAX_BATCH_BODY_BYTES,
)
from .security import BodySizeLimitMiddleware, cors_settings, enforce_rate_limit, rate_limit
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

# Request-body ceilings, enforced before any parsing (added first => runs inside CORS,
# so even the 413 response carries the CORS headers the dashboard needs to read it).
app.add_middleware(
    BodySizeLimitMiddleware,
    limits={
        "/predict": MAX_JSON_BODY_BYTES,
        "/report/pdf": MAX_REPORT_BODY_BYTES,
        "/predict/batch": MAX_BATCH_BODY_BYTES,
    },
    default_limit=MAX_JSON_BODY_BYTES,
)

# CORS: an explicit allow-list of frontend origins (config.CORS_ORIGINS / the CORS_ORIGINS
# environment variable) -- never "*" together with credentials. See security.cors_settings.
app.add_middleware(CORSMiddleware, **cors_settings())

# only one batch may run at a time: it is the one request type that can occupy the CPU for tens of seconds
_batch_gate = threading.BoundedSemaphore(1)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """Same {"detail": [...]} shape FastAPI uses by default, but WITHOUT echoing
    the offending input back: NaN/Infinity can't be JSON-encoded (that used to
    turn a bad request into a 500) and a huge string shouldn't be reflected."""
    detail = [
        {"type": err.get("type"), "loc": list(err.get("loc", [])), "msg": err.get("msg")}
        for err in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": detail})


@app.on_event("startup")
def load_models_on_startup():
    # forces the (potentially slow) model-loading step to happen once at
    # startup rather than on the first incoming request
    get_pipeline()


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/customers")
def list_customers(limit: int = Query(50, ge=1, le=1000)):
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


@app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(rate_limit("predict"))])
def predict_transaction(txn: TransactionInput, db: Session = Depends(get_db)):
    pipeline = get_pipeline()

    def persist(result: dict):
        # runs BEFORE the in-memory history is updated: if the DB write fails the
        # customer history is left exactly as it was
        _persist_transaction(db, result)
        db.commit()

    try:
        result = pipeline.score_transaction(txn.model_dump(), persist=persist)
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(status_code=503, detail="Could not record the transaction; no state was changed.")

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
        history_status=result["history_status"],
        history_transactions=result["history_transactions"],
        reasons=[ExplanationReason(**r) for r in result["reasons"]],
    )


@app.get("/customer/{customer_id}/history", response_model=CustomerHistoryResponse)
def get_customer_history(
    customer_id: str = Path(..., min_length=1, max_length=MAX_ID_LENGTH),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
):
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


@app.get("/customer/{customer_id}/profile", response_model=CustomerProfileResponse)
def get_customer_profile(customer_id: str = Path(..., min_length=1, max_length=MAX_ID_LENGTH)):
    """Behavioral context computed ONLY from the customer's existing history
    (typical amount, usual device/location/hours, ...) plus a genuinely typical
    transaction built from those values. Read-only: never affects a prediction."""
    profile = get_pipeline().customer_profile(customer_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"Unknown customer '{customer_id}'.")
    return profile


@app.get("/fraud-rings", response_model=FraudRingsResponse, dependencies=[Depends(rate_limit("rings"))])
def get_fraud_rings(min_customers: int = Query(2, ge=2, le=1000)):
    """Customers who share a device (or other identifier) that a single
    legitimate customer would never plausibly share with another -- a
    strong signal of an organized fraud ring rather than one customer
    behaving oddly on their own."""
    pipeline = get_pipeline()
    rings = pipeline.detect_fraud_rings(min_customers=min_customers)
    return FraudRingsResponse(count=len(rings), rings=rings)


@app.get("/metrics")
def get_metrics(request: Request, refresh: bool = False):
    """Held-out test-set performance for both trained models (precision,
    recall, F1, AUC-ROC, confusion matrix) -- computed once and cached,
    pass ?refresh=true to force recomputation (rate-limited: it re-scores the whole test set)."""
    enforce_rate_limit(request, "metrics")
    if refresh:
        enforce_rate_limit(request, "metrics_refresh")
    return evaluate_all(force_refresh=refresh)


@app.post("/predict/batch", response_model=BatchPredictionResponse, dependencies=[Depends(rate_limit("batch"))])
async def predict_batch(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Score every row of an uploaded CSV in one call. Required columns:
    customer_id, amount, merchant_category. Optional: device_id, location,
    failed_logins_24h, timestamp.

    All-or-nothing: every row is validated first (same rules as POST /predict);
    if any row is invalid, or the file is larger than the limits, a 4xx is
    returned and NOTHING is scored. The blocking model inference runs in a
    worker thread so the API stays responsive (e.g. /health) during a batch."""
    pipeline = get_pipeline()
    contents = await file.read(BATCH_MAX_BYTES + 1)
    if len(contents) > BATCH_MAX_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large (maximum {BATCH_MAX_BYTES // 1000} KB).")

    try:
        txns, _ = await run_in_threadpool(parse_batch_csv, contents, pipeline)
    except BatchError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)

    def persist_all(results: list):
        for r in results:
            _persist_transaction(db, r)
        db.commit()

    # a batch is the only request that can occupy the CPU for tens of seconds: allow one at a time
    if not _batch_gate.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail="Another batch is already being scored; try again when it finishes.",
            headers={"Retry-After": "10"},
        )
    try:
        results_full = await run_in_threadpool(pipeline.score_batch, txns, persist_all)
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(status_code=503, detail="Could not record the transactions; no state was changed.")
    finally:
        _batch_gate.release()

    summary = {"Low Risk": 0, "Medium Risk": 0, "High Risk": 0, "Critical Risk": 0}
    results = []
    for result in results_full:
        summary[result["alert_level"]] = summary.get(result["alert_level"], 0) + 1
        results.append({
            "transaction_id": result["transaction_id"],
            "customer_id": result["customer_id"],
            "amount": result["amount"],
            "risk_score": result["risk_score"],
            "fraud_probability": result["fraud_probability"],
            "alert_level": result["alert_level"],
        })
    return BatchPredictionResponse(count=len(results), summary=summary, results=results)


@app.post("/report/pdf", dependencies=[Depends(rate_limit("report"))])
def get_pdf_report(req: ReportRequest):
    """Generate a downloadable PDF explaining a single prediction. The request
    carries ONLY the transaction id returned by POST /predict; every value in the
    report (scores, alert level, explanation, ...) comes from the server's own
    record of that prediction, never from the browser. The filename is derived
    from the validated id, so it can't be tampered with either."""
    result = get_pipeline().get_recent_result(req.transaction_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail="No such prediction is available on the server (it may predate a restart); scan the transaction again.",
        )
    pdf_bytes = build_pdf_report(result)
    filename = f"fraud_report_{req.transaction_id}.pdf"
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
