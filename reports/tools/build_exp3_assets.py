"""Markdown tables for the PDF, written from experiments/validation_exp3/results/*.json (no number typed by hand).

    python reports/tools/build_exp3_assets.py .          # repository root
"""
import json
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
R = ROOT / "experiments/validation_exp3/results"
OUT = ROOT / "reports/assets"
sim, cal, abl = (json.loads((R / f"exp3_{n}.json").read_text()) for n in ("similarity", "calibration", "ablation"))
ci = lambda x, d=3: f"[{x[0]:.{d}f}, {x[1]:.{d}f}]"  # noqa: E731

# similarity
d = sim["discrimination_pooled_test"]
rows = ["| Score | ROC-AUC [95 % CI] | PR-AUC [95 % CI] |", "|---|---|---|"]
for k, label in (("s0_p10", "similarity deviation (served formula)"), ("s1_p10", "variant: std floored at 0.5"),
                 ("classifier_p10", "default classifier score"), ("combo_p10", "classifier + similarity")):
    v = d[k]
    rows.append(f"| {label} | {v['roc_auc']:.3f} {ci(v['roc_auc_ci95'])} | {v['pr_auc']:.3f} {ci(v['pr_auc_ci95'])} |")
rows += ["", "| Fraud type (against all legitimate rows) | Fraud rows | ROC-AUC [95 % CI] |", "|---|---|---|"]
rows += [f"| {k[8:].replace('_', ' ')} | {v['fraud']} | {v['roc_auc']:.3f} {ci(v['roc_auc_ci95'])} |" for k, v in d.items() if k.startswith("s0_type_")]
(OUT / "exp3_similarity.md").write_text("\n".join(rows) + "\n")

# calibration
t = cal["pooled_test_all_rows"]
rows = ["| Version | Brier | Log loss | ECE (equal mass) | Slope | Brier difference vs none [95 % CI] |", "|---|---|---|---|---|---|"]
for k in ("uncalibrated", "sigmoid", "isotonic"):
    v = t[k]
    diff = "–" if k == "uncalibrated" else (lambda p: f"{p['brier']['difference']:+.6f} {ci(p['brier']['ci95'], 6)}")(cal["paired_differences_vs_uncalibrated"][k])
    rows.append(f"| {k} | {v['brier']:.6f} | {v['log_loss']:.5f} | {v['ece_equal_mass']:.5f} | {v['calibration_slope']:.2f} | {diff} |")
(OUT / "exp3_calibration.md").write_text("\n".join(rows) + "\n")

# ablation
rows = ["| Model | PR-AUC [95 % CI] | ROC-AUC | Recall | Precision | Legit. alerts per 1,000 legit. |", "|---|---|---|---|---|---|"]
for k, v in abl["pooled_test"]["p10"].items():
    rows.append(f"| {v['label']} | {v['pr_auc']:.3f} {ci(v['pr_auc_ci95'])} | {v['roc_auc']:.3f} | {v['recall']:.3f} | {v['precision']:.3f} | {v['legit_alerts_per_1000_legit']:.2f} |")
(OUT / "exp3_ablation.md").write_text("\n".join(rows) + "\n")
print("wrote exp3_similarity.md, exp3_calibration.md, exp3_ablation.md")
