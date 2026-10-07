"""
Shared fixtures for the Phase 1 regression tests.

Run from the backend/ folder:
    pip install -r requirements.txt -r requirements-dev.txt
    python -m pytest tests -q

The tests start the real app (real trained models, real seed history), so the
first start takes ~1 minute (TensorFlow import + model load). They use a
throw-away SQLite file, never the developer's backend/fraud_platform.db.
"""
import os
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# must be set BEFORE the app modules are imported. The suite fires hundreds of requests from one
# client IP in a couple of minutes, so the default per-IP budgets are lifted here; the rate-limit
# test re-enables a tiny budget for one bucket explicitly.
for _name in ("PREDICT", "BATCH", "REPORT", "METRICS", "METRICS_REFRESH", "RINGS"):
    os.environ.setdefault(f"RATE_LIMIT_{_name}_PER_MIN", "1000000")

_TMP_DB_DIR = tempfile.mkdtemp(prefix="fraud_tests_")
os.environ["DATABASE_URL"] = "sqlite:///" + str(Path(_TMP_DB_DIR) / "test.db").replace("\\", "/")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(scope="session")
def client():
    from app.main import app
    with TestClient(app) as c:   # `with` runs the startup hook that loads the models
        yield c


@pytest.fixture(scope="session")
def pipeline(client):
    from app.inference_pipeline import get_pipeline
    return get_pipeline()


@pytest.fixture(scope="session")
def clean_customers(pipeline):
    """Customers whose stored history contains no labelled fraud, in a fixed order."""
    ids = [
        cid for cid, h in sorted(pipeline.customer_histories.items())
        if "is_fraud" in h.columns and int(h["is_fraud"].sum()) == 0 and len(h) >= 30
    ]
    assert len(ids) > 100
    return ids


@pytest.fixture(scope="session")
def take_customers(clean_customers):
    """take_customers(n) -> n clean customers no other test has used yet
    (scoring mutates history, so every test gets its own customers)."""
    state = {"i": 0}

    def _take(n):
        start = state["i"]
        state["i"] += n
        assert state["i"] <= len(clean_customers), "tests ran out of unused clean customers"
        return clean_customers[start:start + n]

    return _take


@pytest.fixture(scope="session")
def live_server(client):
    """The real app on a real socket (uvicorn in a background thread) for the
    concurrency tests, which need genuinely simultaneous requests."""
    import uvicorn
    from app.main import app

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 60
    while not server.started and time.time() < deadline:
        time.sleep(0.1)
    assert server.started, "live test server did not start"
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=10)
