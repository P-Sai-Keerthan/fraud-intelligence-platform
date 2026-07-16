from sqlalchemy import Column, String, Float, Integer, DateTime, Boolean
from datetime import datetime, timezone

from .database import Base


class Transaction(Base):
    """Stores every transaction the platform has scored, so we can build the
    Fraud Evolution Timeline and let the dashboard look up customer history."""
    __tablename__ = "transactions"

    transaction_id = Column(String, primary_key=True, index=True)
    customer_id = Column(String, index=True, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    amount = Column(Float, nullable=False)
    merchant_category = Column(String)
    device_id = Column(String)
    location = Column(String)
    failed_logins_24h = Column(Integer, default=0)

    # scored outputs
    risk_score = Column(Float)                 # 0-100, from LSTM
    fraud_probability = Column(Float)           # 0-100, from DNN
    similarity_pct = Column(Float)               # 0-100, behavioral similarity
    deviation_pct = Column(Float)
    alert_level = Column(String)                 # Low / Medium / High / Critical
    is_fraud_actual = Column(Boolean, nullable=True)  # ground truth, if known (for demo/eval)
