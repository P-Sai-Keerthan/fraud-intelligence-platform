"""Black-box probes of the real FastAPI app (in-process TestClient) against a THROWAWAY sqlite file.
usage: probe_api.py <backend_dir> <tmp_db_path>
Nothing here touches the project's database or model files."""
import json, os, sys, io, time
backend, dbpath = sys.argv[1], sys.argv[2]
os.environ["DATABASE_URL"] = f"sqlite:///{dbpath}"
sys.path.insert(0, backend); os.chdir(backend)
from fastapi.testclient import TestClient
from app.main import app
from app.inference_pipeline import get_pipeline
from app.db.database import SessionLocal
from app.db import models as m

def n_db(cid=None):
    db = SessionLocal()
    try:
        q = db.query(m.Transaction)
        if cid is not None: q = q.filter(m.Transaction.customer_id == cid)
        return q.count()
    finally: db.close()

def n_mem(cid):
    h = get_pipeline().customer_histories.get(cid)
    return 0 if h is None else len(h)

def show(title, r, body=True):
    t = r.text[:230].replace("\n", " ")
    print(f"[{r.status_code}] {title}" + (f"  -> {t}" if body else ""))

with TestClient(app, raise_server_exceptions=False) as c:
    P = get_pipeline()
    print("model_set:", P.model_set.name, "| customers in memory:", len(P.customer_histories))
    base = dict(customer_id="CUST_0001", amount=1500, merchant_category="grocery", device_id="DEV_0001_A", location="Mumbai", failed_logins_24h=0)

    print("\n== A. documented scenarios (README curl + dashboard quick scenarios)")
    r = c.post("/predict", json=dict(customer_id="CUST_0001", amount=85000, merchant_category="electronics", device_id="DEV_UNKNOWN_9999", location="Lagos", failed_logins_24h=4))
    show("README suspicious example", r, body=False)
    if r.status_code == 200:
        j = r.json(); print("    alert:", j["alert_level"], "| risk_score", j["risk_score"], "| fraud_score", j["fraud_probability"], "| similarity", j["similarity_pct"], "| reasons:", [x["display_name"] for x in j["reasons"]])
    r = c.get("/customer/CUST_0001/profile"); prof = r.json(); print("    profile:", prof)
    r = c.post("/predict", json=dict(base, device_id=prof["home_device"], location=prof["home_location"]))
    j = r.json(); print(f"[{r.status_code}] typical purchase -> alert {j.get('alert_level')} fraud_score {j.get('fraud_probability')} similarity {j.get('similarity_pct')} reasons {[x['display_name'] for x in j.get('reasons', [])]}")

    print("\n== B. input-validation probes on /predict")
    def post_raw(body, title):
        r = c.post("/predict", content=body, headers={"Content-Type": "application/json"}); show(title, r); return r
    before_db, before_mem = n_db("CUST_0002"), n_mem("CUST_0002")
    post_raw(json.dumps(dict(base, customer_id="CUST_0002", timestamp="2026-07-01T10:00:00Z")), "tz-aware timestamp (ISO 'Z')")
    print("    CUST_0002 db rows", before_db, "->", n_db("CUST_0002"), "| memory rows", before_mem, "->", n_mem("CUST_0002"))
    post_raw(json.dumps(dict(base, customer_id="CUST_0002", timestamp="2026-07-01T10:00:00+05:30")), "tz-aware timestamp (+05:30)")
    r = c.post("/predict", content='{"customer_id":"CUST_0003","amount":Infinity,"merchant_category":"grocery","device_id":"d","location":"Pune"}', headers={"Content-Type": "application/json"})
    show("amount = Infinity (non-standard JSON)", r)
    print("    CUST_0003 db rows", n_db("CUST_0003"), "| memory rows", n_mem("CUST_0003"))
    r = c.post("/predict", content='{"customer_id":"CUST_0004","amount":NaN,"merchant_category":"grocery","device_id":"d","location":"Pune"}', headers={"Content-Type": "application/json"}); show("amount = NaN", r)
    post_raw(json.dumps(dict(base, customer_id="CUST_0005", amount=1e308)), "amount = 1e308")
    print("    CUST_0005 db rows", n_db("CUST_0005"), "| memory rows", n_mem("CUST_0005"))
    post_raw(json.dumps(dict(base, customer_id="CUST_0006", failed_logins_24h=10**30)), "failed_logins_24h = 10**30")
    print("    CUST_0006 db rows", n_db("CUST_0006"), "| memory rows", n_mem("CUST_0006"))
    post_raw(json.dumps(dict(base, customer_id="")), "customer_id = '' (empty)")
    post_raw(json.dumps(dict(base, customer_id="   ")), "customer_id = whitespace")
    post_raw(json.dumps(dict(base, customer_id="X" * 100000, device_id="d", location="Pune")), "customer_id 100,000 chars")
    post_raw(json.dumps(dict(base, merchant_category="")), "merchant_category = ''")
    post_raw(json.dumps(dict(base, device_id="", location="")), "device_id/location empty strings")
    post_raw(json.dumps(dict(base, timestamp="1900-01-01T00:00:00")), "timestamp year 1900")
    post_raw(json.dumps(dict(base, timestamp="2999-01-01T00:00:00")), "timestamp year 2999")
    post_raw("{not json", "malformed JSON")
    post_raw(json.dumps(dict(base, amount="12abc")), "amount non-numeric string")
    post_raw(json.dumps(dict(base, amount=-5)), "amount negative")
    print("    customers now known:", len(get_pipeline().customer_histories), "(started with 500 seed customers)")

    print("\n== C. query-parameter bounds")
    r = c.get("/customers?limit=-1"); show("GET /customers?limit=-1", r, body=False)
    if r.status_code == 200: print("    count/len:", r.json()["count"], len(r.json()["customer_ids"]))
    show("GET /customers?limit=100000000", c.get("/customers?limit=100000000"), body=False)
    show("GET history?limit=-1", c.get("/customer/CUST_0001/history?limit=-1"), body=False)
    show("GET profile for unknown id", c.get("/customer/NOPE/profile"))
    show("GET fraud-rings?min_customers=0", c.get("/fraud-rings?min_customers=0"), body=False)

    print("\n== D. PDF endpoint")
    ok = c.post("/predict", json=dict(base, customer_id="CUST_0007")).json()
    r = c.post("/report/pdf", json=ok); print(f"[{r.status_code}] PDF for a real prediction: {len(r.content)} bytes, type {r.headers.get('content-type')}")
    forged = dict(ok, fraud_probability=99.9, alert_level="Critical Risk", reasons=[{"feature": "x", "display_name": "Forged reason", "shap_value": 9.9}])
    r = c.post("/report/pdf", json=forged); print(f"[{r.status_code}] forged scores for a REAL transaction_id accepted: {r.status_code == 200}")
    r = c.post("/report/pdf", json=dict(ok, transaction_id="TXN_é中\U0001F600")); show("transaction_id with non-latin-1 chars", r, body=False)
    r = c.post("/report/pdf", json=dict(ok, transaction_id='TXN"; evil=1')); print(f"[{r.status_code}] transaction_id containing a quote -> Content-Disposition: {r.headers.get('content-disposition')}")
    r = c.post("/report/pdf", json=dict(ok, transaction_id="A" * 5000)); show("transaction_id 5000 chars", r, body=False)

    print("\n== E. batch upload")
    def batch(csv_bytes, title):
        r = c.post("/predict/batch", files={"file": ("t.csv", csv_bytes, "text/csv")}); show(title, r); return r
    batch(b"customer_id,amount,merchant_category\n", "header only (0 rows)")
    batch(b"\xef\xbb\xbfcustomer_id,amount,merchant_category\nCUST_0010,100,grocery\n", "UTF-8 BOM header")
    batch(b"Customer_ID,Amount,Merchant_Category\nCUST_0010,100,grocery\n", "capitalised column names")
    batch(b"customer_id,amount,merchant_category,amount\nCUST_0010,100,grocery,5\n", "duplicate column")
    batch(b"", "empty file")
    batch(b"customer_id,amount,merchant_category\n" + b"CUST_0010,100,grocery\n" * 3, "3 valid rows")
    NROWS = int(os.environ.get("BATCH_ROWS", 40))
    big = b"customer_id,amount,merchant_category,device_id,location\n" + b"CUST_0011,100,grocery,DEV_0011_A,Delhi\n" * NROWS
    t0 = time.time(); r = batch(big, f"{NROWS} valid rows"); print(f"    {NROWS} rows took {time.time()-t0:.1f}s")
    print("    CUST_0011 db rows:", n_db("CUST_0011"), "| memory rows:", n_mem("CUST_0011"))

    print("\n== F. consistency: DB rows vs in-memory histories for every customer touched")
    db = SessionLocal(); import collections
    cnt = collections.Counter(r.customer_id for r in db.query(m.Transaction.customer_id).all()); db.close()
    seed = {k: len(v) for k, v in P._seed_histories.items()}
    bad = {k: (v, n_mem(k) - seed.get(k, 0)) for k, v in cnt.items() if n_mem(k) - seed.get(k, 0) != v}
    print("    customers where (scored rows in DB) != (rows added in memory):", bad)
    # sparse-history similarity
    r = c.post("/predict", json=dict(base, customer_id="CUST_BRAND_NEW", device_id="DEV_X", location="Pune")); j = r.json()
    print("\nbrand-new customer, ordinary first purchase -> similarity", j.get("similarity_pct"), "deviation", j.get("deviation_pct"), "alert", j.get("alert_level"))
    print("\nhealth:", c.get("/health").json())
