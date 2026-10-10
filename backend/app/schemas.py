from datetime import datetime, timezone
from typing import Annotated, List, Optional

from pydantic import BaseModel, Field, StringConstraints, field_validator

# Input limits for POST /predict (audit fix F-01..F-04). A value outside these bounds is
# rejected with 422 instead of being stored: a non-finite or absurd amount would otherwise
# corrupt the customer's behavioural history (mean/std become inf/NaN) permanently.
MAX_AMOUNT = 1_000_000_000.0          # 100 crore INR; far above any amount in the data (max ~1e6)
MAX_FAILED_LOGINS = 10_000
MAX_ID_LENGTH = 64
# Back-dated transactions are a supported feature, but a timestamp of year 1 or 9999 is a client error, and a far-future
# transaction would become the "newest" anchor of the customer's home-device window (audit finding R-06).
MIN_TIMESTAMP = datetime(2000, 1, 1)
MAX_TIMESTAMP = datetime(2100, 1, 1)

# identifiers and labels: surrounding whitespace is removed, blank values and over-long values are rejected
ShortText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_ID_LENGTH)]


class TransactionInput(BaseModel):
    customer_id: ShortText = Field(..., example="CUST_0001")
    amount: float = Field(..., gt=0, le=MAX_AMOUNT, allow_inf_nan=False, example=15000.0)
    merchant_category: ShortText = Field(..., example="electronics")
    device_id: ShortText = Field(..., example="DEV_UNKNOWN_1234")
    location: ShortText = Field(..., example="Lagos")
    failed_logins_24h: int = Field(0, ge=0, le=MAX_FAILED_LOGINS, example=3)
    timestamp: Optional[datetime] = Field(
        None, description="Defaults to now if omitted. A timezone-aware value is converted to UTC; histories are stored without a timezone. Must lie between 2000-01-01 and 2100-01-01.")

    @field_validator("timestamp")
    @classmethod
    def _timestamp_to_naive_utc(cls, value):
        # the stored histories use naive timestamps: a tz-aware value (e.g. "...Z") made scoring fail with a 500
        if value is not None and value.tzinfo is not None:
            value = value.astimezone(timezone.utc).replace(tzinfo=None)
        if value is not None and not (MIN_TIMESTAMP <= value < MAX_TIMESTAMP):
            raise ValueError(f"timestamp must be between {MIN_TIMESTAMP.date()} and {MAX_TIMESTAMP.date()}")
        return value


class ExplanationReason(BaseModel):
    feature: str
    display_name: str
    shap_value: float


class PredictionResponse(BaseModel):
    transaction_id: str
    customer_id: str
    timestamp: datetime

    amount: float
    merchant_category: str
    device_id: str
    location: str
    failed_logins_24h: int

    risk_score: float = Field(..., description="0-100. production / v2_dnn_lstm: LSTM risk score from the customer's previous transactions; v2_dnn_only: repeats the DNN fraud score (no LSTM). See GET /model-info")
    fraud_probability: float = Field(..., description="0-100, DNN fraud score (capped at 99.9); a model score, not a calibrated probability")
    alert_level: str = Field(..., description="Low Risk / Medium Risk / High Risk / Critical Risk")

    similarity_pct: Optional[float] = Field(None, description="0-100, how closely this matches the customer's normal behavior. null (never 0 or 100) when similarity_status is 'insufficient_history'")
    deviation_pct: Optional[float] = Field(None, description="100 - similarity_pct; null together with similarity_pct")
    similarity_status: str = Field("ok", description="'ok', or 'insufficient_history' when the customer has fewer than 10 earlier transactions and no baseline exists")
    history_transactions: int = Field(0, ge=0, description="number of the customer's earlier transactions the baseline is built from")

    reasons: List[ExplanationReason] = Field(default_factory=list, description="Top SHAP-derived reasons, empty if transaction looks normal")


class ReportRequest(BaseModel):
    """Body of POST /report/pdf. Only the transaction id is used: every value printed in the report is read
    from the stored transaction. Any other field a client sends (for example a whole /predict response, which
    the dashboard used to send) is ignored."""
    transaction_id: Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_.-]{1,64}$")]


class FraudRing(BaseModel):
    ring_type: str = Field(..., description="e.g. 'shared_device'")
    identifier: str = Field(..., description="the shared device_id (or other identifier) linking these customers")
    customer_ids: List[str]
    transaction_count: int


class FraudRingsResponse(BaseModel):
    count: int
    rings: List[FraudRing]


class BatchPredictionResult(BaseModel):
    transaction_id: str
    customer_id: str
    amount: float
    risk_score: float
    fraud_probability: float
    alert_level: str


class BatchPredictionResponse(BaseModel):
    count: int
    summary: dict
    results: List[BatchPredictionResult]


class CustomerProfile(BaseModel):
    customer_id: str
    home_device: str = Field(..., description="the customer's most frequently used device_id in the 90 days up to their newest transaction (whole history if less than 90 days or fewer than 5 transactions in that window)")
    home_location: str = Field(..., description="the customer's most frequent transaction city")
    n_transactions: int = Field(..., description="transactions in the customer's history (seed + scored)")


class TimelinePoint(BaseModel):
    transaction_id: str
    timestamp: datetime
    risk_score: float
    fraud_probability: float
    alert_level: str


class CustomerHistoryResponse(BaseModel):
    customer_id: str
    n_transactions: int
    timeline: List[TimelinePoint]
