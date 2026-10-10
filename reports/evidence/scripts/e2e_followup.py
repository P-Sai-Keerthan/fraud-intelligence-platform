import json, sys, time, urllib.request
from playwright.sync_api import sync_playwright
OUT = sys.argv[1]
def post(path, body):
    r = urllib.request.Request("http://localhost:8000" + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=120) as x: return x.status, json.loads(x.read())
    except urllib.error.HTTPError as e: return e.code, json.loads(e.read())
log = {}
# real API, brand-new customer: first transaction
st, new_pred = post("/predict", {"customer_id": "E2E_NEW_001", "amount": 42.5, "merchant_category": "grocery", "device_id": "D_E2E", "location": "Pune", "failed_logins_24h": 0, "timestamp": "2026-09-01T10:00:00"})
log["new_customer_api"] = {"status": st, "similarity_pct": new_pred.get("similarity_pct"), "deviation_pct": new_pred.get("deviation_pct"),
                           "similarity_status": new_pred.get("similarity_status"), "history_transactions": new_pred.get("history_transactions")}
with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/opt/pw-browsers/chromium-1194/chrome-linux/chrome", args=["--no-sandbox"])
    pg = b.new_page(viewport={"width": 1440, "height": 1000})
    errs = []; pg.on("console", lambda m: errs.append(m.text[:200]) if m.type == "error" else None)
    posts = []; pg.on("request", lambda r: posts.append((r.url.split("/api")[-1], r.post_data)) if r.method == "POST" and "/api/" in r.url else None)
    pg.goto("http://localhost:4173/", wait_until="networkidle")
    pg.wait_for_selector("#customer-select option[value='CUST_0001']", state="attached", timeout=60000)
    pg.select_option("#customer-select", "CUST_0001"); time.sleep(1)
    # A) real scan for an established customer: numbers shown, no insufficient state
    pg.get_by_role("button", name="Scan Transaction").click(); pg.wait_for_selector("text=Transaction verdict", timeout=60000); time.sleep(2)
    log["established_insufficient_visible"] = pg.locator("[data-testid=similarity-insufficient]").count()
    log["established_meter_aria"] = pg.locator("[role=img]").evaluate_all("els => els.map(e => e.getAttribute('aria-label')).filter(Boolean).slice(0,6)")
    pg.screenshot(path=f"{OUT}/e2e2_01_established.png", full_page=True)
    # B) PDF: request body must carry only the transaction id
    with pg.expect_download(timeout=60000) as dl:
        pg.get_by_role("button", name="Download Report").click()
    dl.value.save_as(f"{OUT}/e2e2_report.pdf")
    rep = [x for x in posts if x[0].startswith("/report/pdf")]
    log["report_request_body"] = rep[-1][1] if rep else None
    # C) new-customer payload (real API output) rendered by the real UI
    pg.route("**/api/predict", lambda route: route.fulfill(status=200, content_type="application/json", body=json.dumps(new_pred)))
    pg.get_by_role("button", name="Scan Transaction").click(); time.sleep(3)
    log["new_customer_insufficient_visible"] = pg.locator("[data-testid=similarity-insufficient]").count()
    log["new_customer_behavioral_note_visible"] = pg.locator("[data-testid=behavioral-insufficient]").count()
    log["new_customer_text"] = pg.locator("[data-testid=similarity-insufficient]").first.inner_text() if log["new_customer_insufficient_visible"] else None
    pg.screenshot(path=f"{OUT}/e2e2_02_new_customer.png", full_page=True)
    # D) API validation error is readable
    pg.unroute("**/api/predict")
    pg.route("**/api/predict", lambda route: route.fulfill(status=422, content_type="application/json",
        body=json.dumps({"detail": [{"type": "greater_than", "loc": ["body", "amount"], "msg": "Input should be greater than 0", "input": -5}]})))
    pg.get_by_role("button", name="Scan Transaction").click(); time.sleep(2)
    body_text = pg.inner_text("body")
    log["error_message_has_object_object"] = "[object Object]" in body_text
    log["error_message_mentions_amount"] = "amount" in body_text.lower() and "greater than 0" in body_text
    pg.screenshot(path=f"{OUT}/e2e2_03_validation_error.png", full_page=True)
    log["console_errors"] = errs[:5]
    b.close()
print(json.dumps(log, indent=1))
