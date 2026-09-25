from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class TransactionInput(BaseModel):
    customer_id: str = Field(..., example="CUST_0001")
    amount: float = Field(..., gt=0, example=15000.0)
    merchant_category: str = Field(..., example="electronics")
    device_id: str = Field(..., example="DEV_UNKNOWN_1234")
    location: str = Field(..., example="Lagos")
    failed_logins_24h: int = Field(0, ge=0, example=3)
    timestamp: Optional[datetime] = Field(None, description="Defaults to now if omitted")


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

    risk_score: float = Field(..., description="0-100, from LSTM behavioral risk model")
    fraud_probability: float = Field(..., description="0-100%, from DNN real-time classifier")
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


class CustomerProfile(BaseModel):
    customer_id: str
    home_device: str = Field(..., description="the customer's most frequently used device_id")
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
