"""Batch scoring must not block the event loop, and concurrent scoring must
not lose transactions.

The test client dispatches every request onto one shared event loop, so if
the batch endpoint ran its CPU-bound work on the loop itself, a /health call
made during a batch would have to wait for the whole batch to finish.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

from conftest import BASE_TIME

ROW_DELAY = 0.1  # seconds added per scored row to make the batch reliably slow
N_ROWS = 15      # -> batch takes >= 1.5 s


def _slow_scoring(monkeypatch, pipeline, started: threading.Event):
    original = pipeline.score_transaction

    def slow(txn):
        started.set()
        time.sleep(ROW_DELAY)
        return original(txn)

    monkeypatch.setattr(pipeline, "score_transaction", slow)


def _batch_csv(customer_id, n):
    rows = "".join(f"{customer_id},{1000 + i},fashion,DEV_0001_A,Pune,0\n" for i in range(n))
    return "customer_id,amount,merchant_category,device_id,location,failed_logins_24h\n" + rows


def test_batch_does_not_block_other_requests(client, pipeline, monkeypatch):
    started = threading.Event()
    _slow_scoring(monkeypatch, pipeline, started)

    with ThreadPoolExecutor(max_workers=1) as pool:
        batch = pool.submit(
            client.post, "/predict/batch",
            files={"file": ("big.csv", _batch_csv("CUST_0001", N_ROWS).encode(), "text/csv")},
        )
        assert started.wait(timeout=30), "batch never started scoring"

        t0 = time.perf_counter()
        health = client.get("/health")
        health_latency = time.perf_counter() - t0

        assert health.status_code == 200
        assert not batch.done(), "batch finished before /health returned -- test is not exercising overlap"
        assert health_latency < 0.5, f"/health waited {health_latency:.2f}s behind the batch"

        r = batch.result(timeout=120)

    # response format and results are unchanged
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == N_ROWS
    assert sum(body["summary"].values()) == N_ROWS
    assert [row["amount"] for row in body["results"]] == [1000 + i for i in range(N_ROWS)]


def test_invalid_batch_still_rejected_off_loop(client):
    r = client.post(
        "/predict/batch",
        files={"file": ("bad.csv", b"customer_id,amount,merchant_category\nCUST_0001,abc,grocery\n", "text/csv")},
    )
    assert r.status_code == 400
    assert "row 1 (customer_id=CUST_0001): amount must be a number, got 'abc'" in r.json()["detail"]


def test_concurrent_batch_and_single_scoring_keep_every_transaction(client, pipeline):
    before = len(pipeline.customer_histories["CUST_0001"])
    n_single = 8

    def single(i):
        return client.post("/predict", json={
            "customer_id": "CUST_0001", "amount": 2000 + i, "merchant_category": "fashion",
            "device_id": "DEV_0001_A", "location": "Pune",
            "timestamp": (BASE_TIME + timedelta(minutes=i)).isoformat(),
        })

    with ThreadPoolExecutor(max_workers=6) as pool:
        batch = pool.submit(
            client.post, "/predict/batch",
            files={"file": ("b.csv", _batch_csv("CUST_0001", 10).encode(), "text/csv")},
        )
        singles = list(pool.map(single, range(n_single)))
        batch_response = batch.result(timeout=120)

    assert batch_response.status_code == 200
    assert all(r.status_code == 200 for r in singles)

    history = pipeline.customer_histories["CUST_0001"]
    assert len(history) == before + 10 + n_single
    assert history["transaction_id"].is_unique
    persisted = client.get("/customer/CUST_0001/history", params={"limit": 1000}).json()
    assert persisted["n_transactions"] == 10 + n_single
