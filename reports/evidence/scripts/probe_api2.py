"""Follow-up probes (throwaway DB). usage: probe_api2.py <backend_dir> <tmp_db_path>"""
import json, os, sys, time
backend, dbpath = sys.argv[1], sys.argv[2]
os.environ["DATABASE_URL"] = f"sqlite:///{dbpath}"
sys.path.insert(0, backend); os.chdir(backend)
from fastapi.testclient import TestClient
from app.main import app, _restore_history_from_db
from app.inference_pipeline import get_pipeline
from app.db.database import SessionLocal
from app.db import models as m

def n_db(cid):
    db = SessionLocal()
    try: return db.query(m.Transaction).filter(m.Transaction.customer_id == cid).count()
    finally: db.close()

def post(c, body, raw=False):
    kw = dict(content=body, headers={"Content-Type": "application/json"}) if raw else dict(json=body)
    return c.post("/predict", **kw)

with TestClient(app, raise_server_exceptions=False) as c:
    P = get_pipeline()
    base = dict(amount=1500, merchant_category="grocery", device_id="DEV_0003_A", location="Mumbai", failed_logins_24h=0)

    print("== 1. 500 on a huge integer: does the in-memory history get ahead of the database?")
    cid = "CUST_0006"; seed_n = len(P._seed_histories[cid]); print("   seed rows", seed_n, "| memory rows", len(P.customer_histories[cid]), "| db rows", n_db(cid))
    r = post(c, dict(base, customer_id=cid, failed_logins_24h=10**30)); print("   POST failed_logins_24h=10**30 ->", r.status_code)
    print("   after: memory rows", len(P.customer_histories[cid]), "(seed", seed_n, ") | db rows", n_db(cid))
    r = post(c, dict(base, customer_id=cid)); print("   next normal POST ->", r.status_code, "| memory rows", len(P.customer_histories[cid]), "| db rows", n_db(cid))
    h = P.customer_histories[cid]; print("   phantom row present in memory history (failed_logins_24h > 1e18):", bool((h['failed_logins_24h'].astype(float) > 1e18).any()))

    print("\n== 2. amount=Infinity: stored, and what happens to that customer afterwards?")
    cid = "CUST_0012"
    r = c.post("/predict", content=json.dumps(dict(base, customer_id=cid)).replace('"amount": 1500', '"amount": Infinity'), headers={"Content-Type": "application/json"}); print("   POST amount=Infinity ->", r.status_code, "| body amount:", r.json().get("amount") if r.status_code == 200 else r.text[:80])
    db = SessionLocal(); row = db.query(m.Transaction).filter(m.Transaction.customer_id == cid).first(); print("   stored amount in DB:", row.amount if row else None); db.close()
    for k in range(2):
        r = post(c, dict(base, customer_id=cid, device_id="DEV_0012_A")); print(f"   next normal POST #{k+1} ->", r.status_code, r.text[:120].replace("\n", " "))
    h = P.customer_histories[cid]; print("   history amount_zscore values (last 3):", [str(x) for x in h["amount_zscore"].tail(3)])
    print("   -- simulate a server restart: rebuild histories from the database")
    try:
        n = _restore_history_from_db(P); print("   restore ok, replayed", n, "rows")
    except Exception as e:
        print("   RESTORE RAISED:", type(e).__name__, str(e)[:150])
    r = post(c, dict(base, customer_id=cid, device_id="DEV_0012_A")); print("   POST after restore ->", r.status_code, r.text[:120].replace("\n", " "))
    r = c.get(f"/customer/{cid}/history"); print("   GET history ->", r.status_code, r.text[:140].replace("\n", " "))

    print("\n== 3. timestamp extremes")
    for ts in ("0001-01-01T00:00:00", "9999-12-31T23:59:59", "2262-04-12T00:00:00", "1677-01-01T00:00:00"):
        r = post(c, dict(base, customer_id="CUST_0013", timestamp=ts)); print(f"   {ts} -> {r.status_code}", r.text[:90].replace("\n", " ") if r.status_code != 200 else "")

    print("\n== 4. a single normal /predict, timed (machine may be busy with the test suite)")
    ts = []
    for k in range(5):
        t0 = time.time(); r = post(c, dict(base, customer_id="CUST_0014")); ts.append(time.time() - t0)
    print("   5 sequential /predict calls (s):", [round(x, 2) for x in ts])

    print("\n== 5. model-info thresholds (what the alert bands are)")
    mi = c.get("/model-info").json(); print("  ", json.dumps(mi.get("thresholds"))[:900])
