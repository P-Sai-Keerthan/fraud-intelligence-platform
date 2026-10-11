"""
R-02 impact of the alert-level change, from committed reports (read only) and, optionally, the Experiment 3 score files.

    python reports/evidence/scripts/r02_alert_band_impact.py [--exp3-scores <dir>] > reports/evidence/output/r02_alert_band_impact.json

Nothing is fitted or chosen here: the cut-offs are the frozen validation-period values recorded in the 4D selection record.
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
FINAL = ROOT / "backend/models/evaluation/downstream/final_holdout_report.json"
RECORD = ROOT / "backend/models/evaluation/downstream/selection_record.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp3-scores")
    args = ap.parse_args()
    final = json.loads(FINAL.read_text())
    cut = json.loads(RECORD.read_text())["deploy"]["cutoffs"]
    prim = final["final"]["primary"]
    sel = prim["models"]["selected"]
    dist = prim["score_distribution"]["selected"]
    out = {
        "cutoffs_frozen_0_1": cut,
        "final_holdout_501_505": {
            "population": "customers with at least 10 earlier transactions",
            "legacy_bands_share_of_fraud_by_level": dist["fraud"]["application_alert_bands_25_50_80"],
            "legacy_bands_share_of_legitimate_by_level": dist["legitimate"]["application_alert_bands_25_50_80"],
            "validated_alert_tier": {k: sel["metrics"][k]["value"] for k in ("recall", "precision", "legit_alerts_per_1000")},
            "validated_critical_tier": {"recall": sel["metrics"]["critical_recall"]["value"], "precision": sel["metrics"]["critical_precision"]["value"],
                                        "legit_alerts_per_1000": sel["metrics"]["critical_legit_alerts_per_1000"]["value"]},
        },
    }
    if args.exp3_scores:
        import pandas as pd
        d = pd.concat([pd.read_csv(Path(args.exp3_scores) / f"scores_seed_{s}.csv.gz", usecols=["is_fraud", "n_prior", "default"]) for s in (9303, 9304, 9305)])
        d = d[d.n_prior >= 10]
        y, s = d.is_fraud.to_numpy().astype(bool), d.default.to_numpy() * 100
        def tier(t):
            m = s >= t
            tp, fp = int((m & y).sum()), int((m & ~y).sum())
            return {"alerts": int(m.sum()), "recall": round(tp / y.sum(), 4), "precision": round(tp / max(m.sum(), 1), 4),
                    "legit_alerts_per_1000_legit": round(1000 * fp / (~y).sum(), 3)}
        out["experiment3_test_seeds_9303_9305"] = {
            "note": "independent re-check of the frozen cut-offs on data not used to choose them; synthetic",
            "rows": int(len(d)), "fraud": int(y.sum()),
            "legacy_ge_25": tier(25), "legacy_ge_50": tier(50), "legacy_ge_80": tier(80),
            "validated_alert": tier(cut["policy_b"] * 100), "validated_critical": tier(cut["critical"] * 100),
            "fraud_shown_low_under_legacy": round(float(((s < 25) & y).sum() / y.sum()), 4),
            "fraud_shown_low_under_validated": round(float(((s < cut["policy_b"] * 100) & y).sum() / y.sum()), 4),
        }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
