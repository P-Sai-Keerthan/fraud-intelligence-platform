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
    GET  /metrics                       evaluation metrics of the loaded model set, labelled with it
    GET  /metrics/report                full evaluation report of the loaded model set
    GET  /model-info                    metadata of the loaded model set (version, data, features, thresholds)
    POST /report/pdf                    downloadable PDF explanation report for one prediction
    GET  /health                        basic health check
"""

import io
import json
import math
import os
import re
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, Query, Request, UploadFile, File
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import desc, text

from .config import cors_allow_origins
from .db.database import engine, get_db, Base, SessionLocal
from .db import models as db_models
from .schemas import (
    TransactionInput, PredictionResponse, ExplanationReason,
    CustomerHistoryResponse, TimelinePoint, CustomerProfile,
    FraudRingsResponse, BatchPredictionResponse, ReportRequest, MAX_ID_LENGTH, MAX_AMOUNT, MAX_FAILED_LOGINS,
)
from .inference_pipeline import get_pipeline
from .db.migrations import ensure_schema
from .model_metadata import candidate_evaluation, downstream_evaluation, final_holdout_evaluation
from .models.evaluate import EvaluationReportMissing, evaluate_all, load_report
from .report import build_pdf_report


def _restore_history_from_db(pipeline) -> int:
    """Replays every persisted scored transaction into the pipeline's
    in-memory customer histories (see FraudIntelligencePipeline.restore_scored_history)."""
    db = SessionLocal()
    try:
        rows = [
            {
                "transaction_id": r.transaction_id,
                "customer_id": r.customer_id,
                "timestamp": r.timestamp,
                "amount": r.amount,
                "merchant_category": r.merchant_category,
                "device_id": r.device_id,
                "location": r.location,
                "failed_logins_24h": r.failed_logins_24h,
            }
            for r in db.query(db_models.Transaction).all()
        ]
    finally:
        db.close()
    return pipeline.restore_scored_history(rows)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # create DB tables if they don't exist
    Base.metadata.create_all(bind=engine)
    # add columns introduced after the table was first created (model-set provenance)
    added = ensure_schema(engine)
    if added:
        print(f"[startup] Added database column(s): {', '.join(added)} (existing rows keep NULL).")
    # forces the (potentially slow) model-loading step to happen once at
    # startup rather than on the first incoming request
    pipeline = get_pipeline()
    # bring back what the models knew about each customer before the last restart
    restored = _restore_history_from_db(pipeline)
    print(f"[startup] Restored {restored} previously scored transaction(s) from the database.")
    yield


def docs_config(env=None) -> dict:
    """Interactive API documentation (/docs, /redoc, /openapi.json) is on by default for the local demo.
    API_DOCS=off removes all three, for deployments that should not advertise the API surface."""
    env = os.environ if env is None else env
    if str(env.get("API_DOCS", "on")).strip().lower() in ("off", "0", "false", "no"):
        return {"docs_url": None, "redoc_url": None, "openapi_url": None}
    return {}


app = FastAPI(
    title="Explainable Fraud Intelligence Platform API",
    description="Behavioral Fraud DNA, real-time fraud detection, and explainable AI for banking transactions.",
    version="1.0.0",
    lifespan=lifespan,
    **docs_config(),
)

# browser origins allowed to call this API directly -- configured with the
# CORS_ALLOW_ORIGINS environment variable (see config.py / .env.example)
CORS_ALLOW_ORIGINS = cors_allow_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    # an allow-any-origin wildcard must not be combined with credentials
    allow_credentials="*" not in CORS_ALLOW_ORIGINS,
    # only what the dashboard uses (JSON and multipart POSTs, GETs); a wildcard here would also permit
    # DELETE/PUT and arbitrary request headers from any allowed origin
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
)

# a request may not declare a body larger than the batch limit plus multipart overhead. This is a cheap guard on the
# declared Content-Length only; chunked bodies and real request-size limits belong at the reverse proxy (see
# docs/deployment-security-requirements.md).
MAX_REQUEST_BYTES_OVERHEAD = 100_000


@app.middleware("http")
async def security_defaults(request: Request, call_next):
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > MAX_BATCH_BYTES + MAX_REQUEST_BYTES_OVERHEAD:
        return JSONResponse(status_code=413, content={"detail": "Request body is too large."},
                            headers={"X-Content-Type-Options": "nosniff"})
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    if request.url.path not in ("/docs", "/redoc", "/openapi.json"):
        # customer histories, scores and reports must not be kept by browser or proxy caches
        response.headers["Cache-Control"] = "no-store"
    return response


def _json_safe(value):
    """Replace non-finite floats (which JSON cannot carry) so a validation error that echoes
    the offending input can always be serialised. Without this, amount=NaN made the 422
    response itself fail and the client saw a 500 (audit fix F-02)."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    errors = [{k: _json_safe(v) for k, v in e.items() if k in ("type", "loc", "msg", "input")} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.get("/health")
def health_check():
    """Liveness plus the one dependency every request needs: a reachable database.
    The models are loaded at start-up, so a running server has them. Returns 503
    when the database cannot be queried (audit fix F-09); the body stays
    {"status": "ok"} when healthy, as before."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:
        return JSONResponse(status_code=503, content={"status": "database unavailable"})
    return {"status": "ok"}


@app.get("/customers")
def list_customers(limit: int = Query(50, ge=1, le=10000)):
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
        model_set=result["model_set"],
        model_version=result["model_version"],
        reasons_json=json.dumps(result["reasons"]),
    )
    db.add(db_txn)


@app.post("/predict", response_model=PredictionResponse)
def predict_transaction(txn: TransactionInput, db: Session = Depends(get_db)):
    pipeline = get_pipeline()
    result = pipeline.score_transaction(txn.model_dump())
    try:
        _persist_transaction(db, result)
        db.commit()
    except Exception:
        # scoring already added the transaction to the customer's in-memory history;
        # undo it so memory never holds a transaction the database lacks (F-05)
        db.rollback()
        pipeline.remove_transaction(result["customer_id"], result["transaction_id"])
        raise

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
        similarity_status=result["similarity_status"],
        history_transactions=result["history_transactions"],
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
def get_customer_history(customer_id: str, limit: int = Query(100, ge=1, le=1000), db: Session = Depends(get_db)):
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
def get_fraud_rings(min_customers: int = Query(2, ge=1)):
    """Customers who share a device (or other identifier) that a single
    legitimate customer would never plausibly share with another -- a
    strong signal of an organized fraud ring rather than one customer
    behaving oddly on their own."""
    pipeline = get_pipeline()
    rings = pipeline.detect_fraud_rings(min_customers=min_customers)
    return FraudRingsResponse(count=len(rings), rings=rings)


@app.get("/model-info")
def get_model_info():
    """Metadata of the loaded model set: name and version (content checksums),
    training dataset, feature version, whether an LSTM is used, what the scores
    mean (not calibrated probabilities) and the alert/threshold configuration."""
    return get_pipeline().model_metadata


def _is_lstm_classifier(pipeline) -> bool:
    manifest = pipeline.model_set.manifest
    return manifest is not None and manifest.get("model_set_kind") == "lstm_classifier"


def _model_context(pipeline) -> dict:
    meta = pipeline.model_metadata
    return {
        "model_set": meta["model_set"],
        "model_version": meta["model_version"],
        "model_metadata": meta,
        "alerting": meta["thresholds"],
    }


@app.get("/metrics")
def get_metrics(refresh: bool = False):
    """Evaluation metrics for the LOADED model set, labelled with it.

    v2_lstm_rf_seed14 (the default, Step 4D): downstream_evaluation -- the model-
    selection comparison on development data and the fresh hold-out confirmation,
    copied from models/evaluation/downstream/*.json (docs/model_selection_report.md).

    production (the previous default): the held-out time-split metrics of the corrected v1 evaluation
    (the same lstm_risk_predictor / dnn_fraud_classifier entries as before,
    served from models/evaluation/evaluation_report.json; ?refresh=true re-reads
    it). They evaluate evaluation copies of the production architecture on v1,
    as "evaluation" says.

    candidate model sets: the v1 report does not evaluate them, so
    lstm_risk_predictor / dnn_fraud_classifier are null, and the candidate's own
    v2 test-period metrics from models/candidates/v2/comparison.json are given
    separately under candidate_evaluation (only if that file describes exactly
    the loaded weights). Scores are not calibrated probabilities."""
    pipeline = get_pipeline()
    context = _model_context(pipeline)
    evaluation = dict(pipeline.model_metadata["evaluation"])
    if _is_lstm_classifier(pipeline):
        downstream = downstream_evaluation(pipeline.model_set)
        evaluation["available"] = downstream["available"]
        return {
            "lstm_risk_predictor": None,
            "dnn_fraud_classifier": None,
            "v1_metrics_withheld": "the v1 evaluation report evaluates the previous default (v1 LSTM -> DNN), "
                                   "not this model set",
            **context,
            "evaluation": evaluation,
            "downstream_evaluation": downstream,
        }
    if pipeline.model_set.manifest is None:
        try:
            metrics = evaluate_all(force_refresh=refresh)
            report = load_report()
        except EvaluationReportMissing as e:
            raise HTTPException(status_code=503, detail=str(e))
        evaluation.update({"applies_to_loaded_model_set": True, "report_version": report.get("report_version"),
                           "report_dataset": report.get("dataset")})
        return {**metrics, **context, "evaluation": evaluation}
    candidate = candidate_evaluation(pipeline.model_set)
    evaluation["available"] = candidate["available"]
    return {
        "lstm_risk_predictor": None,
        "dnn_fraud_classifier": None,
        "v1_metrics_withheld": "the v1 evaluation report evaluates the production architecture on v1, "
                               "not this model set",
        **context,
        "evaluation": evaluation,
        "candidate_evaluation": candidate,
        "final_holdout_evaluation": final_holdout_evaluation(pipeline.model_set),
    }


@app.get("/metrics/report")
def get_metrics_report(refresh: bool = False):
    """The complete evaluation report for the loaded model set.

    production: the v1 evaluation report (methodology, saved split definitions,
    primary and secondary results, baselines, first-fraud / episode metrics,
    legacy random-split numbers), plus model_set / model_version keys.
    candidate model sets: no v1 report; the candidate's comparison.json slice
    (candidate_evaluation) and the model metadata."""
    pipeline = get_pipeline()
    context = _model_context(pipeline)
    if _is_lstm_classifier(pipeline):
        return {**context, "v1_report_withheld": "the v1 evaluation report does not evaluate this model set",
                "downstream_evaluation": downstream_evaluation(pipeline.model_set)}
    if pipeline.model_set.manifest is None:
        try:
            report = load_report(force_refresh=refresh)
        except EvaluationReportMissing as e:
            raise HTTPException(status_code=503, detail=str(e))
        return {**report, **context}
    return {
        **context,
        "v1_report_withheld": "the v1 evaluation report evaluates the production architecture on v1, "
                              "not this model set",
        "candidate_evaluation": candidate_evaluation(pipeline.model_set),
        "final_holdout_evaluation": final_holdout_evaluation(pipeline.model_set),
    }


_DECIMAL_RE = re.compile(r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?")
_WHOLE_NUMBER_RE = re.compile(r"\d+(\.0*)?")
_MAX_REPORTED_ROW_ERRORS = 20
# Batch scoring costs roughly 0.4-0.6 s per row (measured), the dashboard waits at most 5 minutes
# and the endpoint is unauthenticated, so an unbounded upload could occupy the server for hours
# (audit fix F-08). Both limits can be raised with environment variables.
MAX_BATCH_BYTES = int(os.getenv("BATCH_MAX_BYTES", 2_000_000))
MAX_BATCH_ROWS = int(os.getenv("BATCH_MAX_ROWS", 500))


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
        elif len(customer_id) > MAX_ID_LENGTH:
            row_errors.append(f"customer_id is longer than {MAX_ID_LENGTH} characters")

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
            elif amount > MAX_AMOUNT:
                row_errors.append(f"amount must not exceed {MAX_AMOUNT:,.0f}, got '{raw_amount}'")

        merchant_category = row["merchant_category"].strip()
        if not merchant_category:
            row_errors.append("merchant_category is required")
        elif len(merchant_category) > MAX_ID_LENGTH:
            row_errors.append(f"merchant_category is longer than {MAX_ID_LENGTH} characters")

        failed_logins = 0
        raw_logins = row["failed_logins_24h"].strip() if has_logins else ""
        if raw_logins:
            if not _WHOLE_NUMBER_RE.fullmatch(raw_logins):
                row_errors.append(f"failed_logins_24h must be a whole number >= 0, got '{raw_logins}'")
            else:
                failed_logins = int(float(raw_logins))
                if failed_logins > MAX_FAILED_LOGINS:
                    row_errors.append(f"failed_logins_24h must not exceed {MAX_FAILED_LOGINS}, got '{raw_logins}'")

        device_id = row["device_id"].strip() if has_device else ""
        location = row["location"].strip() if has_location else ""
        for field_name, value in (("device_id", device_id), ("location", location)):
            if len(value) > MAX_ID_LENGTH:
                row_errors.append(f"{field_name} is longer than {MAX_ID_LENGTH} characters")
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
    contents = await file.read(MAX_BATCH_BYTES + 1)
    if len(contents) > MAX_BATCH_BYTES:
        raise HTTPException(status_code=413, detail=f"CSV is larger than {MAX_BATCH_BYTES:,} bytes; nothing was scored.")
    # parsing, validation and model scoring are CPU-bound; run them on a
    # worker thread so a large batch doesn't block the event loop (and every
    # other request) until it finishes
    return await run_in_threadpool(_score_batch_csv, contents, db)


def _score_batch_csv(contents: bytes, db: Session) -> BatchPredictionResponse:
    pipeline = get_pipeline()
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

    if len(df) > MAX_BATCH_ROWS:
        raise HTTPException(
            status_code=413,
            detail=f"CSV has {len(df):,} rows; the limit is {MAX_BATCH_ROWS:,} rows per upload. Nothing was scored.",
        )
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
    scored = []        # (customer_id, transaction_id) already added to the in-memory histories
    summary = {"Low Risk": 0, "Medium Risk": 0, "High Risk": 0, "Critical Risk": 0}
    try:
        for txn in txns:
            result = pipeline.score_transaction(txn)
            scored.append((result["customer_id"], result["transaction_id"]))
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
    except Exception:
        # all-or-nothing: nothing is saved, so nothing may stay in memory either (F-05)
        db.rollback()
        for customer_id, transaction_id in reversed(scored):
            pipeline.remove_transaction(customer_id, transaction_id)
        raise
    return BatchPredictionResponse(count=len(results), summary=summary, results=results)


def _provenance(transaction_id: str, db: Session) -> dict:
    """Which model set scored this transaction, from its stored row (4C-2f-2).
    Never inferred from the currently loaded model set."""
    row = db.get(db_models.Transaction, transaction_id)
    if row is None:
        return {"recorded": False, "model_set": None, "model_version": None,
                "reason": "transaction not found in the database"}
    if row.model_set is None:
        return {"recorded": False, "model_set": None, "model_version": None,
                "reason": "scored before model-set provenance was recorded"}
    provenance = {"recorded": True, "model_set": row.model_set, "model_version": row.model_version}
    pipeline = get_pipeline()
    if row.model_version == pipeline.model_version:
        provenance["metadata"] = pipeline.model_metadata
    return provenance


def _stored_prediction(row) -> dict:
    """The report's input, built only from the stored transaction (never from the request)."""
    return {
        "transaction_id": row.transaction_id, "customer_id": row.customer_id, "timestamp": row.timestamp,
        "amount": row.amount, "merchant_category": row.merchant_category, "device_id": row.device_id,
        "location": row.location, "failed_logins_24h": row.failed_logins_24h or 0,
        "risk_score": row.risk_score, "fraud_probability": row.fraud_probability, "alert_level": row.alert_level,
        "similarity_pct": row.similarity_pct, "deviation_pct": row.deviation_pct,
        # None = the explanation was never stored (older row); [] = stored, nothing flagged
        "reasons": None if row.reasons_json is None else json.loads(row.reasons_json),
    }


@app.post("/report/pdf")
def get_pdf_report(request: ReportRequest, db: Session = Depends(get_db)):
    """Generate a downloadable PDF explaining one scored transaction.

    The report is produced ONLY from the transaction stored by POST /predict (or the batch endpoint): the
    request supplies just the transaction id, and scores, alert level, similarity and explanation printed in the
    report are read from the database, so a client cannot alter them (audit finding R-04). Extra fields in the
    body, such as a full /predict response, are ignored. An unknown id returns 404. The model set named in the
    report is the one recorded with the transaction, not the currently loaded one."""
    row = db.get(db_models.Transaction, request.transaction_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"No scored transaction '{request.transaction_id}' was found; "
                                                    "reports are only available for transactions that were scored and saved.")
    pdf_bytes = build_pdf_report(_stored_prediction(row), _provenance(row.transaction_id, db))
    # the id comes from the client: keep only filename-safe characters, so a quote, newline or
    # non-Latin-1 character cannot break or inject into the Content-Disposition header (F-07)
    safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", row.transaction_id)[:64] or "transaction"
    filename = f"fraud_report_{safe_id}.pdf"
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
