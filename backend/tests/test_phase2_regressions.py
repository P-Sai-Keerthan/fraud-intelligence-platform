"""
Phase 2 regression tests (medium-priority fixes): cold start, CORS, API abuse
protection, PDF security, batch concurrency, back-dated history consistency.
Real app, real shipped models, throw-away database.
"""
import json
import struct
import threading
import time
import zlib
from datetime import datetime, timedelta

import httpx
import pytest

from app.config import BATCH_MAX_BYTES, ESTABLISHED_HISTORY_MIN
from app.inference_pipeline import RAW_COLUMNS_FOR_FEATURES, with_chronological_flags
from app.security import cors_settings, rate_limiter
from app.config import _env_list

HISTORY_BASED = {"amount_zscore", "hour_is_unusual", "is_new_device", "is_new_location",
                 "category_is_unusual", "amount_pct_of_avg", "risk_score"}
T0 = datetime(2026, 9, 1, 14, 30)


def tx(cid, day=0, **kw):
    t = {"customer_id": cid, "amount": 1500.0, "merchant_category": "grocery", "device_id": "DEV_COLD_A",
         "location": "Hyderabad", "failed_logins_24h": 0, "timestamp": (T0 + timedelta(days=day)).isoformat()}
    t.update(kw)
    return t


# ================================================================= M1 cold start
def test_unknown_customer_is_flagged_cold_start_and_claims_no_history(client, pipeline):
    cid = "COLD_UNKNOWN_1"
    assert cid not in pipeline.customer_histories
    assert client.get(f"/customer/{cid}/profile").status_code == 404          # unknown customers are identified as such

    r = client.post("/predict", json=tx(cid))
    assert r.status_code == 200, r.text
    p = r.json()
    assert p["history_status"] == "none" and p["history_transactions"] == 0
    # nothing that would require history is claimed
    assert p["risk_score"] is None and p["similarity_pct"] is None and p["deviation_pct"] is None
    # ...and the explanation never names a history-based factor (they were not available)
    assert not ({r_["feature"] for r_ in p["reasons"]} & HISTORY_BASED)
    # an ordinary first transaction (home-city list, no failed logins) is not flagged on the strength of
    # "new device / new location / unusual category" artefacts that an EMPTY history would otherwise produce
    assert p["alert_level"] == "Low Risk", p

    # after that first valid transaction the customer exists, but still has only "limited" history
    prof = client.get(f"/customer/{cid}/profile").json()
    assert prof["n_transactions"] == 1 and prof["history_status"] == "limited"
    assert prof["typical_scenario"] is None                                     # no invented "typical" behaviour


def test_cold_start_still_uses_genuine_transaction_level_signals(client):
    plain = client.post("/predict", json=tx("COLD_UNKNOWN_2A")).json()
    p = client.post("/predict", json=tx("COLD_UNKNOWN_2B", location="Lagos", failed_logins_24h=5)).json()
    assert p["history_status"] == "none" and p["risk_score"] is None and p["similarity_pct"] is None
    # foreign location + failed logins are genuinely available signals, so they DO move the score...
    assert p["fraud_probability"] > plain["fraud_probability"] + 10, (p["fraud_probability"], plain["fraud_probability"])
    # ...but they are the only things offered as reasons (history-based factors are unavailable). Note the
    # sensitivity is limited by design: the DNN relies mostly on history-based features that do not exist yet.
    assert p["reasons"] and {r["feature"] for r in p["reasons"]} <= {"is_foreign_location", "failed_logins_24h", "txn_velocity_1h"}


def test_cold_start_progression_none_limited_established(client):
    cid = "COLD_PROGRESS_1"
    seen = []
    for i in range(ESTABLISHED_HISTORY_MIN + 2):          # 12 transactions
        p = client.post("/predict", json=tx(cid, day=i)).json()
        seen.append(p)
    for i, p in enumerate(seen):
        assert p["history_transactions"] == i
        expected = "none" if i == 0 else ("limited" if i < ESTABLISHED_HISTORY_MIN else "established")
        assert p["history_status"] == expected, (i, p["history_status"])
        if expected == "established":
            assert isinstance(p["risk_score"], float) and isinstance(p["similarity_pct"], float)
        else:   # the LSTM is never fed a padded window, and similarity needs a real baseline
            assert p["risk_score"] is None and p["similarity_pct"] is None and p["deviation_pct"] is None
    assert [p["history_status"] for p in seen].count("limited") == ESTABLISHED_HISTORY_MIN - 1

    # downstream endpoints cope with the nulls: timeline (DB rows with NULL risk) and the PDF
    hist = client.get(f"/customer/{cid}/history")
    assert hist.status_code == 200
    assert [pt["risk_score"] for pt in hist.json()["timeline"]][:ESTABLISHED_HISTORY_MIN] == [None] * ESTABLISHED_HISTORY_MIN
    pdf = client.post("/report/pdf", json={"transaction_id": seen[0]["transaction_id"]})
    assert pdf.status_code == 200 and pdf.content[:5] == b"%PDF-"
    assert client.get(f"/customer/{cid}/profile").json()["history_status"] == "established"


def test_established_customer_keeps_full_behavioral_outputs(client, take_customers):
    cid = take_customers(1)[0]
    s = client.get(f"/customer/{cid}/profile").json()
    assert s["history_status"] == "established"
    p = client.post("/predict", json={"customer_id": cid, **{k: s["typical_scenario"][k] for k in
                    ("amount", "merchant_category", "device_id", "location", "failed_logins_24h", "timestamp")}}).json()
    assert p["history_status"] == "established" and p["history_transactions"] >= 30
    assert isinstance(p["risk_score"], float) and isinstance(p["similarity_pct"], float)


def test_batch_with_unknown_customer_returns_null_temporal_risk(client):
    csv = ("customer_id,amount,merchant_category,device_id,location,failed_logins_24h\n"
           "COLD_BATCH_1,1500,grocery,DEV_COLD_B,Hyderabad,0\n").encode()
    r = client.post("/predict/batch", files={"file": ("t.csv", csv, "text/csv")})
    assert r.status_code == 200, r.text
    assert r.json()["results"][0]["risk_score"] is None
    # blank device/location for a customer with no history to default from is an invalid row, not a made-up device
    bad = client.post("/predict/batch", files={"file": ("t.csv", b"customer_id,amount,merchant_category\nCOLD_BATCH_2,100,grocery\n", "text/csv")})
    assert bad.status_code == 422


# ===================================================================== M2 CORS
def test_cors_only_allows_listed_origins(client):
    ok = client.options("/predict", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST",
                                              "Access-Control-Request-Headers": "content-type"})
    assert ok.status_code == 200
    assert ok.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert "access-control-allow-credentials" not in ok.headers            # credentials are never enabled

    evil = client.options("/predict", headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"})
    assert evil.status_code == 400 and "access-control-allow-origin" not in evil.headers

    simple = client.get("/health", headers={"Origin": "http://evil.example"})
    assert simple.status_code == 200 and "access-control-allow-origin" not in simple.headers
    assert client.get("/health", headers={"Origin": "http://127.0.0.1:5173"}).headers["access-control-allow-origin"] == "http://127.0.0.1:5173"


def test_cors_configuration_never_combines_wildcard_with_credentials():
    assert "*" not in cors_settings()["allow_origins"]
    assert cors_settings()["allow_credentials"] is False
    wild = cors_settings(["*"])
    assert wild["allow_origins"] == ["*"] and wild["allow_credentials"] is False
    assert _env_list("NOT_SET_ANYWHERE_XYZ", ["http://a"]) == ["http://a"]


# ============================================================ M3 abuse protection
def test_oversized_bodies_are_rejected_before_parsing(client):
    big = json.dumps({"customer_id": "CUST_0001", "amount": 5, "merchant_category": "x" * 200_000,
                      "device_id": "d", "location": "l"})
    r = client.post("/predict", content=big, headers={"content-type": "application/json"})
    assert r.status_code == 413 and "too large" in r.json()["detail"]

    def chunks():                       # no Content-Length: sent chunked
        for _ in range(50):
            yield b"x" * 4096
    assert client.post("/predict", content=chunks(), headers={"content-type": "application/json"}).status_code == 413

    assert client.post("/report/pdf", content=b"{" + b" " * 10_000 + b"}", headers={"content-type": "application/json"}).status_code == 413
    huge_csv = b"customer_id,amount,merchant_category\n" + b"A,1,b\n" * 250_000
    assert len(huge_csv) > BATCH_MAX_BYTES + 65_536
    assert client.post("/predict/batch", files={"file": ("t.csv", huge_csv, "text/csv")}).status_code == 413
    assert client.get("/health").status_code == 200


def test_oversized_upload_over_a_real_socket_gets_a_readable_413_not_a_reset(live_server):
    """Over TCP the server must answer 413 while the client is still uploading. uvicorn closes the socket as
    soon as a response completes before the request body has, so the middleware drains (a bounded amount of)
    the rejected body BEFORE replying. Without that, a stdlib client sees ConnectionReset/Aborted on a
    large share of attempts (measured: 3 of 8)."""
    import urllib.error
    import urllib.request
    import uuid

    boundary = "----t" + uuid.uuid4().hex
    head = ('--%s\r\nContent-Disposition: form-data; name="file"; filename="t.csv"\r\nContent-Type: text/csv\r\n\r\n' % boundary).encode()
    body = head + b"x" * 1_200_000 + ("\r\n--%s--\r\n" % boundary).encode()
    outcomes = []
    for _ in range(8):
        req = urllib.request.Request(live_server + "/predict/batch", data=body, method="POST",
                                     headers={"Content-Type": "multipart/form-data; boundary=" + boundary})
        try:
            urllib.request.urlopen(req, timeout=30)
            outcomes.append(200)
        except urllib.error.HTTPError as e:
            e.read()                                   # the body must be readable too, not just the status line
            outcomes.append(e.code)
        except Exception as e:                         # a connection reset/abort lands here
            outcomes.append(type(e).__name__)
    assert outcomes == [413] * 8, outcomes

    # a 1 MB JSON body sent to a 16 KB-limit endpoint (e.g. a megabyte-long device_id) must also get a clean 413
    huge = json.dumps({"customer_id": "C", "amount": 1, "merchant_category": "g", "device_id": "D" * 1_000_000, "location": "l"}).encode()
    outcomes = []
    for _ in range(8):
        req = urllib.request.Request(live_server + "/predict", data=huge, method="POST", headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=30)
            outcomes.append(200)
        except urllib.error.HTTPError as e:
            e.read()
            outcomes.append(e.code)
        except Exception as e:
            outcomes.append(type(e).__name__)
    assert outcomes == [413] * 8, outcomes
    assert httpx.get(live_server + "/health", timeout=10).status_code == 200          # server unaffected


def test_rate_limit_applies_to_expensive_endpoints_and_sends_retry_after(client):
    original = rate_limiter.limits["predict"]
    try:
        rate_limiter.configure("predict", 3, 60)
        rate_limiter.reset()
        # invalid bodies: cheap (422) but they still spend the budget -- no inference is needed to be throttled
        statuses = [client.post("/predict", json={"customer_id": ""}).status_code for _ in range(5)]
        assert statuses == [422, 422, 422, 429, 429], statuses
        r = client.post("/predict", json={"customer_id": ""})
        assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1
        assert client.get("/health").status_code == 200               # cheap endpoints are not limited
    finally:
        rate_limiter.limits["predict"] = original
        rate_limiter.reset()


def test_query_parameters_are_bounded(client):
    assert client.get("/customers?limit=0").status_code == 422
    assert client.get("/customers?limit=-5").status_code == 422
    assert client.get("/customers?limit=100000").status_code == 422
    assert client.get("/fraud-rings?min_customers=0").status_code == 422
    assert client.get("/customer/CUST_0001/history?limit=0").status_code == 422
    assert client.get("/customers?limit=5").json()["customer_ids"] == client.get("/customers?limit=5").json()["customer_ids"]


# ================================================================ M4 PDF security
def _scan(client, cid="PDF_CUST_1", **kw):
    r = client.post("/predict", json=tx(cid, **kw))
    assert r.status_code == 200, r.text
    return r.json()


def test_pdf_normal_report_is_valid_and_filename_is_server_derived(client):
    p = _scan(client, "PDF_CUST_1")
    r = client.post("/report/pdf", json={"transaction_id": p["transaction_id"]})
    assert r.status_code == 200 and r.content[:5] == b"%PDF-" and r.content.rstrip().endswith(b"%%EOF")
    assert r.headers["content-type"] == "application/pdf"
    assert r.headers["content-disposition"] == f'attachment; filename="fraud_report_{p["transaction_id"]}.pdf"'


MALFORMED_REPORT_BODIES = {
    "path traversal id": {"transaction_id": "../../etc/passwd"},
    "quote / header injection id": {"transaction_id": 'TXN_0123456789"; evil=1; x="'},
    "CRLF id": {"transaction_id": "TXN_0123456789\r\nSet-Cookie: a=b"},
    "non-hex id": {"transaction_id": "TXN_ZZZZZZZZZZ"},
    "too long id": {"transaction_id": "TXN_0123456789ABCDEF"},
    "numeric id": {"transaction_id": 12345},
    "missing id": {},
    "markup in alert_level": {"transaction_id": "TXN_0123456789", "alert_level": "<img src='file:///etc/passwd'/>"},
    "invalid risk score": {"transaction_id": "TXN_0123456789", "risk_score": 99999},
    "unexpected string field": {"transaction_id": "TXN_0123456789", "reasons": [{"display_name": "<b>x"}]},
    "file/resource reference": {"transaction_id": "file:///C:/Windows/win.ini"},
}


@pytest.mark.parametrize("name", list(MALFORMED_REPORT_BODIES))
def test_pdf_rejects_malformed_or_tampered_requests_with_4xx(client, name):
    r = client.post("/report/pdf", json=MALFORMED_REPORT_BODIES[name])
    assert 400 <= r.status_code < 500, (name, r.status_code, r.text[:200])
    assert "detail" in r.json()


def test_pdf_for_unknown_transaction_is_404_and_tampered_copy_of_a_real_report_is_rejected(client):
    assert client.post("/report/pdf", json={"transaction_id": "TXN_0000000000"}).status_code == 404
    p = _scan(client, "PDF_CUST_2")
    tampered = {**p, "alert_level": "<img src='x'/>", "fraud_probability": 1.0}      # the old "echo the whole prediction" body
    assert client.post("/report/pdf", json=tampered).status_code == 422


def _tiny_png() -> bytes:
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + b"\xff\x00\x00" * 4 for _ in range(4))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 4, 4, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def test_user_controlled_text_can_never_make_reportlab_embed_a_local_file(client, tmp_path, monkeypatch):
    from app.report import build_pdf_report
    (tmp_path / "secret.png").write_bytes(_tiny_png())
    monkeypatch.chdir(tmp_path)
    attack = "<img src='secret.png' width='9' height='9'/>"                       # a real, readable image file
    assert len(attack) <= 64                                                         # passes /predict validation

    # CONTROL: the attack is real -- unescaped, ReportLab embeds the file's pixels into the PDF
    from reportlab.platypus import SimpleDocTemplate, Paragraph
    from reportlab.lib.styles import getSampleStyleSheet
    import io
    buf = io.BytesIO()
    SimpleDocTemplate(buf).build([Paragraph(attack, getSampleStyleSheet()["BodyText"])])
    assert b"/Subtype /Image" in buf.getvalue()

    # through the real endpoint, the same text is just text
    p = _scan(client, "PDF_CUST_3", device_id=attack, location="<b>Bold</b> & <a href='http://x'>x</a>")
    r = client.post("/report/pdf", json={"transaction_id": p["transaction_id"]})
    assert r.status_code == 200 and r.content[:5] == b"%PDF-"
    assert b"/Subtype /Image" not in r.content
    # and directly in the generator, with every user-controlled field attacking at once
    pdf = build_pdf_report({"transaction_id": "TXN_1", "customer_id": attack, "alert_level": attack, "device_id": attack,
                            "location": attack, "merchant_category": attack, "amount": 5, "risk_score": None,
                            "fraud_probability": 50.0, "similarity_pct": None, "deviation_pct": None,
                            "reasons": [{"display_name": attack, "shap_value": 0.1}], "history_status": "none",
                            "history_transactions": 0, "timestamp": attack})
    assert pdf[:5] == b"%PDF-" and b"/Subtype /Image" not in pdf


# ======================================================= M5 batch concurrency
def _batch_csv(client, customers):
    rows = []
    for cid in customers:
        pr = client.get(f"/customer/{cid}/profile").json()
        rows.append(f"{cid},{pr['typical_amount']},{pr['typical_scenario']['merchant_category']},,,0")
    return ("customer_id,amount,merchant_category,device_id,location,failed_logins_24h\n" + "\n".join(rows) + "\n").encode()


def test_batch_does_not_block_health_or_unrelated_customers(client, live_server, take_customers, pipeline):
    batch_customers = take_customers(40)
    other = take_customers(1)[0]
    data = _batch_csv(client, batch_customers)
    before = {c: len(pipeline.customer_histories[c]) for c in batch_customers + [other]}
    prof = client.get(f"/customer/{other}/profile").json()
    other_body = {"customer_id": other, **{k: prof["typical_scenario"][k] for k in
                  ("amount", "merchant_category", "device_id", "location", "failed_logins_24h", "timestamp")}}

    batch_info = {}

    def run_batch():
        t = time.time()
        r = httpx.post(live_server + "/predict/batch", files={"file": ("t.csv", data, "text/csv")}, timeout=300)
        batch_info.update(status=r.status_code, seconds=time.time() - t, end=time.time())

    th = threading.Thread(target=run_batch)
    th.start()
    time.sleep(1.0)                                    # the batch is now running
    assert th.is_alive(), "batch finished too quickly to test concurrency"

    health = []
    for _ in range(8):
        t = time.time()
        assert httpx.get(live_server + "/health", timeout=20).status_code == 200
        health.append(time.time() - t)
        time.sleep(0.15)

    t = time.time()
    r = httpx.post(live_server + "/predict", json=other_body, timeout=120)
    other_seconds, other_end = time.time() - t, time.time()
    batch_was_still_running = th.is_alive()
    th.join()

    assert r.status_code == 200 and batch_info["status"] == 200
    assert max(health) < 2.0, health                                  # event loop stays responsive
    assert batch_was_still_running and other_end < batch_info["end"]  # the unrelated customer did NOT wait for the whole batch
    assert other_seconds < batch_info["seconds"] * 0.8, (other_seconds, batch_info["seconds"])
    # every customer's history grew by exactly one transaction: nothing lost, nothing duplicated
    assert {c: len(pipeline.customer_histories[c]) - before[c] for c in before} == {c: 1 for c in before}


def test_second_batch_is_rejected_while_one_is_running(client, live_server, take_customers):
    data = _batch_csv(client, take_customers(20))
    first = {}

    def run_first():
        first["r"] = httpx.post(live_server + "/predict/batch", files={"file": ("t.csv", data, "text/csv")}, timeout=300)

    th = threading.Thread(target=run_first)
    th.start()
    time.sleep(1.0)
    assert th.is_alive(), "first batch finished too quickly to test the gate"
    valid_second = ("customer_id,amount,merchant_category,device_id,location\n"
                    "GATE_TEST_CUSTOMER,100,grocery,DEV_G,Hyderabad\n").encode()
    second = httpx.post(live_server + "/predict/batch", files={"file": ("t.csv", valid_second, "text/csv")}, timeout=60)
    th.join()
    assert second.status_code == 429 and int(second.headers["retry-after"]) >= 1
    assert first["r"].status_code == 200
    # once the first batch is done the gate is free again
    again = httpx.post(live_server + "/predict/batch", files={"file": ("t.csv", valid_second, "text/csv")}, timeout=60)
    assert again.status_code == 200


def test_parallel_requests_for_one_customer_lose_no_updates(client, live_server, take_customers, pipeline):
    cid = take_customers(1)[0]
    prof = client.get(f"/customer/{cid}/profile").json()
    n0 = len(pipeline.customer_histories[cid])
    base = datetime.fromisoformat(prof["typical_scenario"]["timestamp"])
    statuses, ids = [], []

    def one(i):
        body = {"customer_id": cid, **{k: prof["typical_scenario"][k] for k in ("amount", "merchant_category", "device_id", "location", "failed_logins_24h")},
                "timestamp": (base + timedelta(minutes=i)).isoformat()}
        r = httpx.post(live_server + "/predict", json=body, timeout=120)
        statuses.append(r.status_code)
        ids.append(r.json().get("transaction_id"))

    threads = [threading.Thread(target=one, args=(i,)) for i in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert statuses == [200] * 8 and len(set(ids)) == 8
    h = pipeline.customer_histories[cid]
    assert len(h) == n0 + 8 and h["timestamp"].is_monotonic_increasing
    assert client.get(f"/customer/{cid}/history").json()["n_transactions"] == 8     # DB agrees


# ===================================================== M6 back-dated consistency
def _flags(pipeline, cid):
    h = pipeline.customer_histories[cid]
    return [(row.timestamp.day, int(row.is_new_device), int(row.is_new_location)) for row in h.itertuples()]


def test_back_dated_insert_rewrites_later_rows_flags_chronologically(client, pipeline):
    cid = "CHRONO_1"
    # Day 1: device A / city X.  Day 3: device B / city Y.
    client.post("/predict", json=tx(cid, day=0, device_id="DEV_A", location="Pune"))
    client.post("/predict", json=tx(cid, day=2, device_id="DEV_B", location="Delhi"))
    assert _flags(pipeline, cid) == [(1, 1, 1), (3, 1, 1)]          # B / Delhi are new on day 3

    # Day 2 arrives LATE with device B / Delhi: it, not day 3, is now the first time they were seen
    p = client.post("/predict", json=tx(cid, day=1, device_id="DEV_B", location="Delhi")).json()
    assert p["history_transactions"] == 1                           # only day 1 precedes it chronologically
    assert _flags(pipeline, cid) == [(1, 1, 1), (2, 1, 1), (3, 0, 0)]
    h = pipeline.customer_histories[cid]
    assert h["timestamp"].is_monotonic_increasing

    # the spec's variant: Day 2 inserted later with a THIRD device C: day 3 (B) stays new, day 2 (C) is new
    cid2 = "CHRONO_2"
    client.post("/predict", json=tx(cid2, day=0, device_id="DEV_A", location="Pune"))
    client.post("/predict", json=tx(cid2, day=2, device_id="DEV_B", location="Delhi"))
    client.post("/predict", json=tx(cid2, day=1, device_id="DEV_C", location="Jaipur"))
    assert _flags(pipeline, cid2) == [(1, 1, 1), (2, 1, 1), (3, 1, 1)]


def test_chronological_flag_definition_matches_every_seed_customer():
    """with_chronological_flags() must reproduce the flags the dataset generator stored for every
    one of the 500 seed customers, so re-deriving them on insert cannot disagree with the training data."""
    import pandas as pd
    from app.config import FEATURES_CSV
    seed = pd.read_csv(FEATURES_CSV, parse_dates=["timestamp"])
    assert seed["customer_id"].nunique() == 500
    bad = []
    for cid, g in seed.groupby("customer_id"):
        g = g.sort_values("timestamp", kind="stable").reset_index(drop=True)
        re = with_chronological_flags(g[RAW_COLUMNS_FOR_FEATURES])
        if not ((re["is_new_device"].values == g["is_new_device"].values).all()
                and (re["is_new_location"].values == g["is_new_location"].values).all()):
            bad.append(cid)
    assert not bad, bad[:5]


# ================================================ M3: expensive endpoint is computed once
def test_metrics_are_not_computed_by_several_concurrent_requests(client, monkeypatch):
    from app.models import evaluate
    real_load = evaluate._load_sequences
    calls = []

    def counting_load():
        calls.append(1)
        time.sleep(0.5)                 # make the race window wide enough to be reliable
        return real_load()

    monkeypatch.setattr(evaluate, "_cache", None)
    monkeypatch.setattr(evaluate, "_load_sequences", counting_load)
    results = []
    threads = [threading.Thread(target=lambda: results.append(evaluate.evaluate_all())) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(calls) == 1, f"evaluation ran {len(calls)} times for 4 simultaneous requests"
    assert len(results) == 4 and all(r is results[0] for r in results)
    # simultaneous FORCED refreshes collapse into one recomputation too
    calls.clear()
    threads = [threading.Thread(target=lambda: evaluate.evaluate_all(force_refresh=True)) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(calls) == 1
