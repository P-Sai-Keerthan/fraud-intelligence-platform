"""
Batch CSV parsing + validation
================================
Every CSV row is validated with the SAME TransactionInput model used by
POST /predict, BEFORE anything is scored. If any row is invalid, nothing is
scored and every bad row is reported (no partial processing, so a bad file
can never half-update customer histories).

Required columns: customer_id, amount, merchant_category
Optional columns: device_id, location, failed_logins_24h, timestamp
A blank device_id / location falls back to the customer's own most common
device / location from their history (not to a made-up identifier), so a
minimal CSV is scored the way the customer actually behaves.
"""

import io
from typing import List, Tuple

import pandas as pd
from pydantic import ValidationError

from .config import BATCH_MAX_ROWS
from .schemas import TransactionInput

REQUIRED_COLUMNS = ("customer_id", "amount", "merchant_category")
MAX_REPORTED_ERRORS = 10


class BatchError(Exception):
    """Carries an HTTP status + human-readable message for the endpoint."""
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def _cell(row, column: str) -> str:
    return str(row[column]).strip() if column in row.index else ""


def _short_error(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        field = ".".join(str(p) for p in err["loc"]) or "row"
        parts.append(f"{field}: {err['msg']}")
    return "; ".join(parts)


def parse_batch_csv(contents: bytes, pipeline) -> Tuple[List[dict], int]:
    """Returns (validated transaction dicts, row_count). Raises BatchError."""
    try:
        # dtype=str / keep_default_na=False: blanks stay "" and "nan"/"inf" are
        # NOT silently converted -- the validator decides what is acceptable
        df = pd.read_csv(io.BytesIO(contents), dtype=str, keep_default_na=False)
    except Exception as e:
        raise BatchError(400, f"Could not parse CSV: {e}")

    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise BatchError(400, f"CSV is missing required columns: {sorted(missing)}")

    if len(df) > BATCH_MAX_ROWS:
        raise BatchError(
            413,
            f"Batch too large: {len(df)} rows (maximum {BATCH_MAX_ROWS} per request). "
            f"Split the file and upload it in parts.",
        )

    txns: List[dict] = []
    errors: List[str] = []
    bad_rows = 0
    for i, (_, row) in enumerate(df.iterrows()):
        line_no = i + 2  # +1 for the header, +1 for 1-based numbering
        customer_id = _cell(row, "customer_id")
        payload = {
            "customer_id": customer_id,
            "amount": _cell(row, "amount"),
            "merchant_category": _cell(row, "merchant_category"),
            "device_id": _cell(row, "device_id"),
            "location": _cell(row, "location"),
            "failed_logins_24h": _cell(row, "failed_logins_24h") or "0",
        }
        ts = _cell(row, "timestamp")
        if ts:
            payload["timestamp"] = ts

        # blank device/location -> the customer's own usual one (from history)
        if not payload["device_id"] or not payload["location"]:
            profile = pipeline.customer_profile(customer_id) if customer_id else None
            if profile and profile.get("primary_device") and not payload["device_id"]:
                payload["device_id"] = profile["primary_device"]
            if profile and profile.get("primary_location") and not payload["location"]:
                payload["location"] = profile["primary_location"]

        try:
            txns.append(TransactionInput.model_validate(payload).model_dump())
        except ValidationError as exc:
            bad_rows += 1
            if len(errors) < MAX_REPORTED_ERRORS:
                errors.append(f"row {line_no} ({customer_id or 'no customer_id'}): {_short_error(exc)}")

    if bad_rows:
        more = f" (+{bad_rows - len(errors)} more invalid rows)" if bad_rows > len(errors) else ""
        raise BatchError(
            422,
            f"{bad_rows} of {len(df)} rows are invalid; nothing was scored. " + " | ".join(errors) + more,
        )
    return txns, len(df)
