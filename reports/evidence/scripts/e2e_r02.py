import json, sys, time
from playwright.sync_api import sync_playwright
OUT = sys.argv[1]; log = {}
with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/opt/pw-browsers/chromium-1194/chrome-linux/chrome", args=["--no-sandbox"])
    pg = b.new_page(viewport={"width": 1440, "height": 1000})
    pg.goto("http://localhost:4173/", wait_until="networkidle")
    pg.wait_for_selector("#customer-select option[value='CUST_0001']", state="attached", timeout=60000)
    pg.select_option("#customer-select", "CUST_0001"); time.sleep(1)
    resp = {}
    with pg.expect_response(lambda r: r.url.endswith("/predict") and r.request.method == "POST", timeout=60000) as ri:
        pg.get_by_role("button", name="Scan Transaction").click()
    resp["predict"] = ri.value.json()
    pg.wait_for_selector("text=Transaction verdict", timeout=60000); time.sleep(3)
    log["typical"] = {k: resp["predict"][k] for k in ("fraud_probability", "alert_level", "risk_score")}
    log["typical_banner_text"] = pg.locator("p.text-xs").first.inner_text()[:160]
    pg.screenshot(path=f"{OUT}/e2e3_typical.png", full_page=True)
    pg.get_by_role("button", name="Suspicious Pattern").click()
    with pg.expect_response(lambda r: r.url.endswith("/predict") and r.request.method == "POST", timeout=60000) as ri:
        pg.get_by_role("button", name="Scan Transaction").click()
    resp["predict"] = ri.value.json(); time.sleep(5)
    log["suspicious"] = {k: resp["predict"][k] for k in ("fraud_probability", "alert_level")}
    pg.screenshot(path=f"{OUT}/e2e3_suspicious.png", full_page=True)
    pg.get_by_role("tab", name="Model Performance").click(); time.sleep(4)
    txt = pg.inner_text("body")
    log["perf_tab_has_validated_badge"] = "validated cut-offs" in txt
    log["perf_tab_has_legacy_badge"] = "legacy fixed bands" in txt
    log["perf_tab_has_evidence_text"] = "false alerts per 1,000" in txt
    pg.screenshot(path=f"{OUT}/e2e3_model_performance.png", full_page=True)
    b.close()
print(json.dumps(log, indent=1))
