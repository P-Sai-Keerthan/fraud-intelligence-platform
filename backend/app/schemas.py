from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Optional, List
from datetime import datetime, timedelta

from .config import (
    MAX_TRANSACTION_AMOUNT, MAX_FAILED_LOGINS, MAX_ID_LENGTH, MAX_TEXT_LENGTH,
    MAX_FUTURE_TIMESTAMP_DAYS, MIN_TIMESTAMP_YEAR,
)


class TransactionInput(BaseModel):
    """Validated at the API boundary: nothing that fails here ever reaches
    feature engineering, the customer history, the models, or the database.
    The same model validates every batch-CSV row (see app/batch.py)."""
    model_config = ConfigDict(str_strip_whitespace=True)

    customer_id: str = Field(..., min_length=1, max_length=MAX_ID_LENGTH, example="CUST_0001")
    # allow_inf_nan=False rejects NaN / Infinity / 1e400 (parsed as inf) explicitly
    amount: float = Field(..., gt=0, le=MAX_TRANSACTION_AMOUNT, allow_inf_nan=False, example=15000.0)
    merchant_category: str = Field(..., min_length=1, max_length=MAX_TEXT_LENGTH, example="electronics")
    device_id: str = Field(..., min_length=1, max_length=MAX_ID_LENGTH, example="DEV_UNKNOWN_1234")
    location: str = Field(..., min_length=1, max_length=MAX_TEXT_LENGTH, example="Lagos")
    failed_logins_24h: int = Field(0, ge=0, le=MAX_FAILED_LOGINS, example=3)
    timestamp: Optional[datetime] = Field(
        None,
        description=(
            "Defaults to now if omitted. Timezone-aware values are converted to the server's "
            "local time (the same naive-local convention used everywhere else). "
            "Earlier-than-latest timestamps are allowed and scored against only the history "
            "that precedes them."
        ),
    )

    @field_validator("timestamp")
    @classmethod
    def _normalise_and_bound_timestamp(cls, value):
        if value is None:
            return value
        if value.tzinfo is not None:
            value = value.astimezone().replace(tzinfo=None)
        if value.year < MIN_TIMESTAMP_YEAR:
            raise ValueError(f"timestamp must be in {MIN_TIMESTAMP_YEAR} or later")
        if value > datetime.now() + timedelta(days=MAX_FUTURE_TIMESTAMP_DAYS):
            raise ValueError(f"timestamp may not be more than {MAX_FUTURE_TIMESTAMP_DAYS} days in the future")
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

    risk_score: float = Field(..., description="0-100 temporal risk score from the LSTM, computed from the customer's previous transactions (a model score, not a calibrated probability)")
    fraud_probability: float = Field(..., description="0-100 model fraud risk score from the DNN (field name kept for API compatibility; NOT a calibrated probability)")
    alert_level: str = Field(..., description="Low Risk / Medium Risk / High Risk / Critical Risk")

    similarity_pct: float = Field(..., description="0-100, how closely this matches the customer's normal behavior")
    deviation_pct: float

    reasons: List[ExplanationReason] = Field(default_factory=list, description="Top SHAP-derived reasons, empty if transaction looks normal")


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


class CountShare(BaseModel):
    value: str
    count: int
    share: float = Field(..., description="Fraction of this customer's history (0-1)")


class TypicalScenario(BaseModel):
    """Values for a genuinely typical purchase, derived only from this
    customer's own history (never hard-coded)."""
    amount: float
    merchant_category: str
    device_id: str
    location: str
    failed_logins_24h: int
    timestamp: Optional[str] = Field(None, description="ISO timestamp at the customer's most common hour")


class CustomerProfileResponse(BaseModel):
    """Behavioral context computed from the customer's existing transaction
    history. Read-only; does not affect any prediction."""
    customer_id: str
    n_transactions: int
    first_transaction_at: Optional[str] = None
    last_transaction_at: Optional[str] = None
    typical_amount: Optional[float] = Field(None, description="Median amount (robust to outliers)")
    amount_p25: Optional[float] = None
    amount_p75: Optional[float] = None
    primary_device: Optional[str] = None
    devices: List[CountShare] = Field(default_factory=list)
    known_devices: List[str] = Field(default_factory=list, description="Every distinct device in the history (capped)")
    primary_location: Optional[str] = None
    locations: List[CountShare] = Field(default_factory=list)
    known_locations: List[str] = Field(default_factory=list, description="Every distinct location in the history (capped)")
    top_categories: List[CountShare] = Field(default_factory=list)
    typical_hour: Optional[int] = None
    preferred_hours: List[int] = Field(default_factory=list, description="Hours holding >=5% of the history (same rule as the hour_is_unusual feature)")
    hour_distribution: List[float] = Field(default_factory=list, description="24 values: share of history per hour of day")
    transactions_per_week: Optional[float] = None
    typical_scenario: Optional[TypicalScenario] = None
