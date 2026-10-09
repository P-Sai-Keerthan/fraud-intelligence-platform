"""
Step 4D: hand-built sanity transactions, scored by the live pipeline.

Eight scenarios (A-H) on one seed customer and five ordinary purchases by five
other customers. Every scenario starts from the same untouched seed history, at
the customer's own most common hour, after the end of the seed data. The exact
inputs and outputs are written to models/evaluation/downstream/sanity_scenarios.json.

    cd backend
    python -m app.evaluation.downstream_sanity
"""

import argparse
import json
from datetime import datetime

import pandas as pd

from .. import config
from ..inference_pipeline import FraudIntelligencePipeline
from .downstream import OUTPUT_DIR, _write_json

CUSTOMER = "CUST_0376"
NORMAL_CUSTOMERS = ("CUST_0062", "CUST_0082", "CUST_0318", "CUST_0464", "CUST_0329")
DAY = datetime(2026, 7, 10)


def profile(customer_id: str) -> dict:
    df = pd.read_csv(config.FEATURES_CSV, parse_dates=["timestamp"])
    g = df[df["customer_id"] == customer_id]
    return {"home_device": g["device_id"].mode().iloc[0], "home_city": g["location"].mode().iloc[0],
            "usual_category": g["merchant_category"].mode().iloc[0],
            "median_amount": round(float(g["amount"].median()), -1),
            "usual_hour": int(g["timestamp"].dt.hour.mode().iloc[0]),
            "cities_used": sorted(g["location"].unique().tolist()),
            "seed_transactions": int(len(g))}


def scenarios() -> list:
    p = profile(CUSTOMER)
    ts = DAY.replace(hour=p["usual_hour"], minute=15).isoformat()
    base = {"customer_id": CUSTOMER, "amount": p["median_amount"], "merchant_category": p["usual_category"],
            "device_id": p["home_device"], "location": p["home_city"], "failed_logins_24h": 0, "timestamp": ts}
    other_city = next(c for c in ("Mumbai", "Delhi", "Chennai", "Pune", "Kolkata") if c not in p["cities_used"])
    out = [
        ("A", "Normal low-risk purchase", {}),
        ("B", "Moderate behavioral deviation (3x usual amount)", {"amount": 3 * p["median_amount"]}),
        ("C", "New device only", {"device_id": "DEV_NEW_A001"}),
        ("D", "New domestic location only", {"location": other_city}),
        ("E", "Foreign location only", {"location": "Singapore"}),
        ("F", "High amount only (Rs. 85,000)", {"amount": 85000.0}),
        ("G", "Multiple suspicious signals", {"amount": 85000.0, "merchant_category": "electronics",
                                              "device_id": "DEV_UNKNOWN_7731", "location": "Singapore"}),
        ("H", "Failed-login attack pattern", {"failed_logins_24h": 5, "device_id": "DEV_UNKNOWN_8842",
                                              "amount": 25000.0, "merchant_category": "electronics"}),
    ]
    rows = [{"id": i, "scenario": name, "input": {**base, **change}} for i, name, change in out]
    for k, cid in enumerate(NORMAL_CUSTOMERS, start=1):
        q = profile(cid)
        rows.append({"id": f"N{k}", "scenario": f"Normal purchase, {cid}",
                     "input": {"customer_id": cid, "amount": q["median_amount"], "merchant_category": q["usual_category"],
                               "device_id": q["home_device"], "location": q["home_city"], "failed_logins_24h": 0,
                               "timestamp": DAY.replace(hour=q["usual_hour"], minute=15).isoformat()}})
    return rows, p


def run(model_sets=("v2_lstm_rf_seed14", "production", "v2_dnn_lstm_seed14")) -> dict:
    rows, p = scenarios()
    results = {}
    for name in model_sets:
        pipe = FraudIntelligencePipeline(model_set=name)
        seed = dict(pipe.customer_histories)
        out = []
        for r in rows:
            pipe.customer_histories.clear()
            pipe.customer_histories.update(seed)          # every scenario starts from the seed history
            res = pipe.score_transaction(dict(r["input"]))
            out.append({"id": r["id"], "risk_score": res["risk_score"], "fraud_score": res["fraud_probability"],
                        "alert_level": res["alert_level"], "similarity_pct": res["similarity_pct"],
                        "reasons": [{"feature": x["display_name"], "shap_value": round(x["shap_value"], 4)}
                                    for x in res["reasons"]],
                        "lstm_used": res["history_context"]["lstm_used"]})
        results[name] = {"model_version": pipe.model_version, "results": out}
    report = {"step": "4D sanity scenarios", "customer": CUSTOMER, "customer_profile": p,
              "note": "each scenario scored from the untouched seed history at the customer's most common hour; "
                      "scores are model scores (0-100), not calibrated probabilities",
              "scenarios": rows, "results": results}
    _write_json(report, OUTPUT_DIR / "sanity_scenarios.json")
    return report


def main(argv=None):
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args(argv)
    r = run()
    for name, block in r["results"].items():
        print(f"== {name} ({block['model_version']})")
        for s, x in zip(r["scenarios"], block["results"]):
            print(f"  {s['id']:>2} {s['scenario']:<48} risk {x['risk_score']:6.2f}  fraud {x['fraud_score']:6.2f}  {x['alert_level']}")


if __name__ == "__main__":
    main()
