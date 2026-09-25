"""
Shared pytest fixtures for the backend test suite.

Isolation strategy
------------------
* Database: DATABASE_URL is pointed at a throwaway SQLite file in a temp
  directory BEFORE the app is imported (app.db.database reads it at import
  time), so tests never touch backend/fraud_platform.db. Every row is
  deleted after each test.
* Model state: the inference pipeline is a process-wide singleton that
  keeps each customer's behavioral history in memory and appends to it on
  every scored transaction. Loading the models takes several seconds, so
  the pipeline is shared across the session, and its history dict is
  snapshotted before each test and restored afterwards. score_transaction
  replaces a customer's DataFrame rather than mutating it in place, so a
  shallow snapshot is sufficient.
* Determinism: tests pass explicit timestamps wherever the endpoint
  accepts one, because the hour of day feeds the "unusual hour" feature.
  NumPy/Python RNGs are re-seeded before every test because
  shap.GradientExplainer samples background rows and interpolation points
  at random, so SHAP values (and which top-4 reasons are shown) vary
  between otherwise identical calls unless the RNG is seeded.
"""

import os
import random
import shutil
import tempfile
from datetime import datetime

# ---- must run before anything imports the app --------------------------------
_TMP_DIR = tempfile.mkdtemp(prefix="fraud_platform_tests_")
os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(_TMP_DIR, "test.db").replace("\\", "/")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # silence TensorFlow C++ startup noise

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.db import database as db_module  # noqa: E402
from app.db import models as db_models  # noqa: E402
from app.inference_pipeline import get_pipeline  # noqa: E402


# A timestamp after the end of the seed history (which ends 2026-07-06), so a
# scored transaction is always the newest row in the customer's history.
# 13:30 is CUST_0001's most common transaction hour in the seed data.
BASE_TIME = datetime(2026, 7, 10, 13, 30, 0)

# CUST_0001's real profile in the seed data: primary device DEV_0001_A,
# home city Pune, average spend ~3,800, fashion is their top category.
NORMAL_TXN = {
    "customer_id": "CUST_0001",
    "amount": 3800.0,
    "merchant_category": "fashion",
    "device_id": "DEV_0001_A",
    "location": "Pune",
    "failed_logins_24h": 0,
    "timestamp": BASE_TIME.isoformat(),
}

SUSPICIOUS_TXN = {
    "customer_id": "CUST_0001",
    "amount": 85000.0,
    "merchant_category": "electronics",
    "device_id": "DEV_UNKNOWN_9999",
    "location": "Lagos",
    "failed_logins_24h": 4,
    "timestamp": BASE_TIME.replace(hour=3).isoformat(),
}


@pytest.fixture(scope="session")
def client():
    # entering the context runs the app's startup hook (loads the models once)
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def pipeline(client):
    return get_pipeline()


@pytest.fixture(autouse=True)
def _isolate_state(pipeline):
    """Seed RNGs; restore in-memory customer histories and empty the DB after every test."""
    random.seed(1234)
    np.random.seed(1234)
    snapshot = dict(pipeline.customer_histories)
    yield
    pipeline.customer_histories.clear()
    pipeline.customer_histories.update(snapshot)

    session = db_module.SessionLocal()
    try:
        session.query(db_models.Transaction).delete()
        session.commit()
    finally:
        session.close()


@pytest.fixture
def normal_txn():
    return dict(NORMAL_TXN)


@pytest.fixture
def suspicious_txn():
    return dict(SUSPICIOUS_TXN)


def pytest_sessionfinish(session, exitstatus):
    db_module.engine.dispose()  # release the SQLite file handle (needed on Windows)
    shutil.rmtree(_TMP_DIR, ignore_errors=True)
