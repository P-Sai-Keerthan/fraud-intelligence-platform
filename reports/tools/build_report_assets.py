"""Builds figures and tables for the audit report from the project's committed result files.

usage:  python build_report_assets.py <project_root> <out_dir>

Every number in the generated tables is read from backend/models/evaluation/ JSON; nothing is typed in by hand,
except the latency measurements, which are listed below with their provenance (measured during the audit).
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

INK, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
BLUE, ORANGE, VIOLET = "#2a78d6", "#eb6834", "#6250d6"      # validated categorical slots 1-2-7 (light mode)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})

# ---------------------------------------------------------------- hold-out figure + tables
rep = json.loads((ROOT / "backend/models/evaluation/downstream/final_holdout_report.json").read_text())
LABEL = {"selected": "LSTM + random forest (default)", "dnn_seed14": "LSTM + DNN (seed 14)", "production": "Previous production (v1 LSTM + DNN)"}
COLOR = {"selected": BLUE, "dnn_seed14": ORANGE, "production": VIOLET}
METRICS = [("pr_auc", "PR-AUC"), ("recall", "Recall (at ~10 alerts / 1,000)"), ("first_fraud_recall", "First-fraud recall")]


def table(part):
    pr = rep[part]["primary"]
    lines = [f"| Model | PR-AUC [95% CI] | ROC-AUC | Recall [95% CI] | Precision | F1 | Legit alerts / 1,000 | First-fraud recall [95% CI] | Episodes detected |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, m in pr["models"].items():
        mt, c = m["metrics"], m["counts"]
        f = lambda k, d=3: f"{mt[k]['value']:.{d}f}"
        ci = lambda k: f"[{mt[k]['ci95'][0]:.3f}, {mt[k]['ci95'][1]:.3f}]"
        lines.append(f"| {LABEL[name]} | {f('pr_auc')} {ci('pr_auc')} | {f('roc_auc')} | {f('recall')} {ci('recall')} | {f('precision')} | {f('f1')} | "
                     f"{f('legit_alerts_per_1000', 2)} | {f('first_fraud_recall')} {ci('first_fraud_recall')} | "
                     f"{c['episodes_detected']} / {pr['fraud_episodes']} |")
    head = (f"{pr['datasets']} fresh datasets, {pr['rows']:,} transactions, {pr['fraud_transactions']:,} fraud transactions, "
            f"{pr['fraud_episodes']} fraud episodes, {pr['first_fraud_transactions']} first-fraud transactions.")
    return head, "\n".join(lines)


for part in ("final", "new_customer"):
    head, tab = table(part)
    (OUT / f"holdout_{part}.md").write_text(f"*{head}*\n\n{tab}\n")

fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.2), sharey=False)
pr = rep["final"]["primary"]["models"]
for ax, (key, title) in zip(axes, METRICS):
    for i, name in enumerate(pr):
        v = pr[name]["metrics"][key]
        ax.bar(i, v["value"], width=0.62, color=COLOR[name], edgecolor=SURFACE, linewidth=2)
        ax.errorbar(i, v["value"], yerr=[[v["value"] - v["ci95"][0]], [v["ci95"][1] - v["value"]]], color=INK, capsize=3, lw=1.2)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.set_xticks([]); ax.set_ylim(0, 1); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("value (0-1)")
fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COLOR[n]) for n in pr], labels=[LABEL[n] for n in pr], loc="lower center", ncol=3, frameon=False, fontsize=8, labelcolor=MUTED)
fig.suptitle("Fresh hold-out (seeds 501-505, synthetic data v2): the three model sets compared", x=0.01, ha="left", fontsize=10, color=INK)
fig.tight_layout(rect=(0, 0.1, 1, 0.93)); fig.savefig(OUT / "fig_holdout.png", dpi=200); plt.close(fig)

# ---------------------------------------------------------------- v2 time-split evaluation: recomputed vs reported
import gzip, pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
sc = pd.read_csv(ROOT / "backend/models/evaluation/v2/scores_time_split.csv.gz")
ev = json.loads((ROOT / "backend/models/evaluation/v2/evaluation_report.json").read_text())["primary"]
test = sc[sc.split == "test"]
rows = ["| Model | PR-AUC recomputed | PR-AUC in report | ROC-AUC recomputed | ROC-AUC in report |", "|---|---|---|---|---|"]
for name in ("lstm_risk_predictor", "dnn_fraud_classifier"):
    r = ev[name]
    rows.append(f"| {name} | {average_precision_score(test.is_fraud, test[name]):.4f} | {r['pr_auc']} | {roc_auc_score(test.is_fraud, test[name]):.4f} | {r['auc_roc']} |")
for name, r in ev["baselines"].items():
    if name in test.columns:
        rows.append(f"| {name} | {average_precision_score(test.is_fraud, test[name]):.4f} | {r['pr_auc']} | {roc_auc_score(test.is_fraud, test[name]):.4f} | {r['auc_roc']} |")
(OUT / "recompute_v2_time_split.md").write_text(
    f"*Test period: {len(test):,} windows, {int(test.is_fraud.sum())} fraud rows (from the committed `scores_time_split.csv.gz`).*\n\n" + "\n".join(rows) + "\n")

# ---------------------------------------------------------------- latency figure (audit measurement)
# build_point_features on one customer's history of n transactions, single run each, other load on the machine
# (the backend test suite was running), so absolute times are indicative.  Source: audit session, see FIX_AND_TEST_LOG.md.
N = [100, 400, 800, 1600, 3200]
SECONDS = [0.13, 0.40, 0.99, 2.26, 6.14]
fig, ax = plt.subplots(figsize=(5.4, 3.0))
ax.plot(N, SECONDS, color=BLUE, lw=2, marker="o", ms=6, mfc=BLUE, mec=SURFACE, mew=2)
ax.set_xlabel("transactions already in the customer's history"); ax.set_ylabel("feature rebuild per /predict (s)")
ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.annotate("6.1 s at 3,200 transactions", (3200, 6.14), (1150, 5.4), color=INK, fontsize=8, arrowprops=dict(arrowstyle="-", color=MUTED))
ax.set_title("Per-request cost grows with history length", fontsize=10, color=INK, loc="left")
fig.tight_layout(); fig.savefig(OUT / "fig_latency.png", dpi=200); plt.close(fig)

# ---------------------------------------------------------------- architecture diagram
fig, ax = plt.subplots(figsize=(9.4, 5.6)); ax.set_xlim(0, 100); ax.set_ylim(0, 66); ax.axis("off")


def box(x, y, w, h, text, fc="#eef3fb", ec=BLUE, bold=False, fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", fc=fc, ec=ec, lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, fontweight="bold" if bold else "normal", linespacing=1.35)


def arrow(x1, y1, x2, y2, label=None, dy=1.2):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, label, ha="center", fontsize=7, color=MUTED)


box(1, 46, 17, 12, "React dashboard\n(Vite, Tailwind,\nRecharts)", bold=True)
box(25, 46, 19, 12, "FastAPI  app/main.py\nvalidation (schemas.py)\n11 endpoints", bold=True)
box(52, 46, 22, 12, "FraudIntelligencePipeline\napp/inference_pipeline.py\n(per-customer lock)", bold=True)
box(81, 46, 18, 12, "SQLite (SQLAlchemy)\ntransactions table\nscored rows only", fc="#fdf1ea", ec=ORANGE, bold=True)
arrow(18.5, 52, 24.5, 52, "/api proxy"); arrow(44.5, 52, 51.5, 52)
ax.annotate("", (88, 58.6), (34, 58.6), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2, connectionstyle="arc3,rad=-0.07"))
ax.text(61, 64.2, "saved after scoring (commit); rolled back from memory if the commit fails", ha="center", fontsize=7, color=MUTED)
steps = [("1  Behavioral Fraud DNA\nfeature_engineering.py\n9 features, past-only", 2), ("2  LSTM risk score\n10 previous txns\n(skipped if <10 prior)", 22), ("3  Random forest\n9 features + risk score\n= Fraud Score 0-100", 42),
         ("4  Tree SHAP\ntop reasons\n(shown if score >= 5)", 62), ("5  Behavioral similarity\nz-score vs own history\nexp. decay, 0-100", 82)]
for text, x in steps:
    box(x, 22, 17, 12, text, fc="#f2f1ee", ec=MUTED)
for (t1, x1), (t2, x2) in zip(steps, steps[1:]):
    arrow(x1 + 17.3, 28, x2 - 0.3, 28)
ax.text(2, 37.5, "Scoring order inside score_transaction()", fontsize=8, color=MUTED)
ax.text(2, 35.6, "(history is rebuilt from all earlier rows on each request)", fontsize=7, color=MUTED)
arrow(70, 45.6, 70, 34.4)
box(2, 3, 18, 11, "Seed histories\n500 customers (synthetic v1)\ndata/transactions_\nwith_features.csv", fc="#eaf6f1", ec="#1baf7a", fs=7)
box(24, 3, 38, 11, "Model artifacts (read-only; SHA-256 checked before loading)\nbackend/models/candidates_downstream/\nv2/seed_14/lstm_random_forest/", fc="#eaf6f1", ec="#1baf7a", fs=7)
box(67, 3, 32, 11, "Evaluation (offline, outside the request path)\napp/evaluation/*  models/evaluation/*.json\npre-registered protocols in docs/", fc="#f7f7f5", ec=MUTED, fs=7)
arrow(11, 14.6, 11, 21.6); arrow(43, 14.6, 43, 21.6)
fig.suptitle("System architecture and scoring flow (verified against the source)", x=0.01, ha="left", fontsize=10, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(OUT / "fig_architecture.png", dpi=200); plt.close(fig)
print("wrote", sorted(p.name for p in OUT.iterdir()))
