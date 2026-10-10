"""
Experiment 3 (protocol: PROTOCOL.md, EXP3 v1 + amendment A1).

    python experiments/validation_exp3/run_experiment.py generate --out <data_dir>
    python experiments/validation_exp3/run_experiment.py score    --data <data_dir> [--scores <dir>]
    python experiments/validation_exp3/run_experiment.py analyze  [--scores <dir>]

Read only with respect to the saved models and the existing hold-outs: nothing under backend/models is
written and no seed other than 9301-9305 is generated or scored. The re-trained baselines are fitted
in memory on the seed-42 training rows and are not saved anywhere.
"""

import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app import config  # noqa: E402
from app.evaluation import holdout as h  # noqa: E402
from app.evaluation import multiseed as ms  # noqa: E402
from app.evaluation.baselines import (amount_hour_rule_scores, fit_logistic_regression,  # noqa: E402
                                      logistic_regression_scores)
from app.features.feature_engineering import FEATURE_COLUMNS  # noqa: E402
from app.models.similarity import compute_similarity  # noqa: E402

EXPERIMENT_ID = "EXP3-v1"
DEV_SEEDS = (9301, 9302)
TEST_SEEDS = (9303, 9304, 9305)
ALL_SEEDS = DEV_SEEDS + TEST_SEEDS
CUSTOMERS = 1000
BOOTSTRAP_SEED = 9101
REPS = 2000
FPR_TARGET = 0.01
MIN_PRIOR = h.MIN_PRIOR
DEFAULT_SET = "v2_lstm_rf_seed14"
SAVED_SETS = {"default": DEFAULT_SET, "dnn_lstm": "v2_dnn_lstm_seed14", "dnn_only": "v2_dnn_only", "production": "production"}
BASELINES = ("rf_nolstm", "logreg", "rule")
S1_STD_FLOOR = 0.5
RESULTS = HERE / "results"
MANIFESTS = HERE / "data_manifests"
PROTOCOL = HERE / "PROTOCOL.md"
HISTORY_BINS = (("0", 0, 0), ("1-9", 1, 9), ("10-29", 10, 29), ("30-99", 30, 99), (">=100", 100, 10 ** 9))
USED_ELSEWHERE = (set(range(11, 16)) | {42} | set(range(101, 106)) | set(range(201, 206)) | set(range(301, 306)) | set(range(401, 406))
                  | set(range(411, 416)) | set(range(501, 506)) | set(range(511, 516)) | {7101, 7102, 7103, 7201, 7202, 7203}
                  | set(range(8101, 8111)) | set(range(8201, 8206)))


def sha256_file(path) -> str:
    d = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            d.update(chunk)
    return d.hexdigest()


def check_seeds():
    from app.evaluation.selection import SPENT_SEEDS
    spent = {s for s in ALL_SEEDS if s in USED_ELSEWHERE or s in SPENT_SEEDS}
    if spent:
        raise SystemExit(f"seeds already used elsewhere: {sorted(spent)}")


# ---------------------------------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------------------------------

def cmd_generate(args):
    check_seeds()
    out = Path(args.out)
    MANIFESTS.mkdir(exist_ok=True)
    for seed in ALL_SEEDS:
        t0 = time.time()
        d = h.generate_dataset(seed, root=out, customers=CUSTOMERS)
        shutil.copyfile(d / "manifest.json", MANIFESTS / f"seed_{seed}_manifest.json")
        m = json.loads((d / "manifest.json").read_text())["summary"]
        print(f"seed {seed}: {m['transactions']:,} transactions, {m['fraud_transactions']} fraud ({time.time() - t0:.0f}s)")


# ---------------------------------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------------------------------

def similarity_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Per row, against the mean/std of the customer's EARLIER feature vectors (population std, as served).
    S0 = served formula (compute_similarity, rounded); S1 = same z-scores with the std floored at 0.5, unrounded.
    Rows with no earlier transaction get NaN. The served API shows a score only from 10 earlier transactions;
    values below that are diagnostic for the history-length analysis."""
    X = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32).astype(np.float64)
    n = len(frame)
    out = {k: np.full(n, np.nan) for k in ("dev_s0", "avgz_s0", "dev_s1", "n_const", "n_low")}
    starts = frame.groupby("customer_id", sort=False).indices
    for idx in starts.values():
        idx = np.sort(idx)
        for k in range(1, len(idx)):
            prior = X[idx[:k]]
            cur = X[idx[k]]
            mean, std = prior.mean(axis=0), prior.std(axis=0)
            r = compute_similarity(cur, mean, std)
            j = idx[k]
            out["dev_s0"][j] = r["deviation_pct"]
            out["avgz_s0"][j] = r["avg_z_deviation"]
            z1 = (cur - mean) / np.maximum(std, S1_STD_FLOOR)
            out["dev_s1"][j] = 100 - 100 * np.exp(-float(np.mean(np.abs(z1))) / 2.9)
            out["n_const"][j] = int((std <= 1e-6).sum())
            out["n_low"][j] = int((std < 0.1).sum())
    return pd.DataFrame(out)


def fit_baselines():
    """RF without the LSTM input and logistic regression, on the seed-42 training rows with >= 10 earlier transactions."""
    from sklearn.ensemble import RandomForestClassifier
    from app.evaluation import downstream
    data, labels = downstream.training_data()
    f = data.frame
    mask = (np.asarray(labels) == "train") & (h.prior_counts(f) >= MIN_PRIOR)
    X = f[FEATURE_COLUMNS].to_numpy(dtype=np.float32)[mask]
    y = f["is_fraud"].to_numpy()[mask]
    rf = RandomForestClassifier(n_estimators=200, max_features="sqrt", min_samples_leaf=5, max_depth=12,
                                class_weight=None, random_state=14, n_jobs=-1).fit(X, y)
    lr = fit_logistic_regression(X, y, seed=42)
    return rf, lr, {"training_rows": int(mask.sum()), "training_fraud": int(y.sum()),
                    "dataset_sha256": data.sha256, "rf": "RandomForest(200, sqrt, leaf 5, depth 12, seed 14)",
                    "logreg": "StandardScaler + LogisticRegression(balanced, seed 42)"}


def cmd_score(args):
    check_seeds()
    from app.model_sets import load_model_set
    data_root, scores_dir = Path(args.data), Path(args.scores)
    scores_dir.mkdir(parents=True, exist_ok=True)
    sets = {k: load_model_set(v) for k, v in SAVED_SETS.items()}
    rf, lr, base_info = fit_baselines()
    (scores_dir / "baselines_info.json").write_text(json.dumps(base_info, indent=1))
    for seed in ALL_SEEDS:
        t0 = time.time()
        data = h.load_holdout(seed, root=data_root)
        f, meta = data.frame, data.metadata
        comp = f["customer_id"].map(data.grouping.component_of)
        table = pd.DataFrame({
            "seed": seed, "customer_id": f["customer_id"].to_numpy(), "timestamp": f["timestamp"].to_numpy(),
            "is_fraud": f["is_fraud"].to_numpy().astype(int), "n_prior": h.prior_counts(f),
            "group": [f"{seed}:{c}" for c in comp],
            "fraud_type": meta["fraud_type"].to_numpy(), "episode": meta["fraud_episode_id"].astype(int).to_numpy(),
        })
        for key, ms_ in sets.items():
            table[key] = h.score_frame(f, ms_).astype(np.float64)
        F = f[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        table["rf_nolstm"] = rf.predict_proba(F)[:, 1]
        table["logreg"] = logistic_regression_scores(lr, F)
        table["rule"] = amount_hour_rule_scores(F)
        table = pd.concat([table, similarity_columns(f)], axis=1)
        path = scores_dir / f"scores_seed_{seed}.csv.gz"
        table.to_csv(path, index=False, float_format="%.9g")
        print(f"seed {seed}: {len(table):,} rows scored ({time.time() - t0:.0f}s) -> {path.name} {sha256_file(path)[:12]}")


# ---------------------------------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------------------------------

def _r(x, d=6):
    x = float(x)
    return None if not np.isfinite(x) else round(x, d)


def ci(v):
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    return [_r(x) for x in np.percentile(v, [2.5, 97.5])] if len(v) else None


class Resampler:
    """Customer-component bootstrap within each seed, pooled. Row weights = multiplicity of the row's group."""

    def __init__(self, groups: np.ndarray, seeds: np.ndarray, reps: int, seed: int):
        self.reps = reps
        uniq, self.row_group = np.unique(groups, return_inverse=True)
        group_seed = np.empty(len(uniq), dtype=np.int64)
        group_seed[self.row_group] = seeds
        rng = np.random.default_rng(seed)
        self.counts = np.zeros((reps, len(uniq)), dtype=np.int32)
        for s in np.unique(group_seed):
            cols = np.flatnonzero(group_seed == s)
            draws = rng.integers(0, len(cols), size=(reps, len(cols)))
            for r in range(reps):
                self.counts[r, cols] = np.bincount(draws[r], minlength=len(cols))

    def weights(self, r):
        return self.counts[r][self.row_group].astype(np.float64)


class RankItem:
    """Average precision and ROC AUC of one score on a row subset."""

    def __init__(self, idx, y, score):
        self.idx = idx
        self.y = y[idx].astype(np.float64)
        s = np.asarray(score)[idx]
        self.order, self.ends = h.rank_prepare(s)
        self.yo = self.y[self.order]
        self.io = idx[self.order]

    def point(self):
        return h.weighted_rank_metrics(self.yo, 1 - self.yo, self.ends)

    def boot(self, w):
        wo = w[self.io]
        return h.weighted_rank_metrics(wo * self.yo, wo * (1 - self.yo), self.ends)


class AlertItem:
    """Threshold metrics of one alert rule on a row subset."""

    def __init__(self, idx, y, alert, first):
        self.idx, self.y, self.a, self.first = idx, y[idx].astype(bool), alert[idx].astype(bool), first[idx].astype(bool)

    def sums(self, w):
        w = w[self.idx] if w is not None else np.ones(len(self.idx))
        pos, neg = self.y, ~self.y
        tp, fp = w[pos & self.a].sum(), w[neg & self.a].sum()
        fn, tn = w[pos & ~self.a].sum(), w[neg & ~self.a].sum()
        ff, ffa = w[self.first].sum(), w[self.first & self.a].sum()
        return tp, fp, fn, tn, ff, ffa

    @staticmethod
    def metrics(s):
        tp, fp, fn, tn, ff, ffa = s
        div = lambda a, b: a / b if b > 0 else np.nan  # noqa: E731
        prec, rec = div(tp, tp + fp), div(tp, tp + fn)
        return {"precision": prec, "recall": rec, "f1": div(2 * tp, 2 * tp + fp + fn),
                "legit_alerts_per_1000_legit": 1000 * div(fp, fp + tn), "first_fraud_recall": div(ffa, ff)}


def auc_binary(y, s):
    """ROC AUC with ties counted half (equals the rank statistic)."""
    y = np.asarray(y)
    order, ends = h.rank_prepare(s)
    ap, auc = h.weighted_rank_metrics(y[order].astype(float), 1.0 - y[order], ends)
    return auc


def first_fraud_flags(df: pd.DataFrame) -> np.ndarray:
    """True for the first fraud transaction (by time) of every fraud episode."""
    flag = np.zeros(len(df), dtype=bool)
    fr = df[(df["is_fraud"] == 1) & (df["episode"] > 0)]
    first = fr.sort_values("timestamp", kind="mergesort").groupby(["seed", "episode"], sort=False).head(1)
    flag[df.index.get_indexer(first.index)] = True
    return flag


# ---------------------------------------------------------------------------------------------------
# calibration helpers
# ---------------------------------------------------------------------------------------------------

EPS = 1e-6


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def cal_metrics(p, y, w=None):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    y = np.asarray(y, dtype=np.float64)
    w = np.ones(len(y)) if w is None else w
    W = w.sum()
    brier = float((w * (p - y) ** 2).sum() / W)
    ll = float(-(w * (y * np.log(p) + (1 - y) * np.log(1 - p))).sum() / W)
    return brier, ll


def ece(p, y, mode, bins=15):
    p, y = np.asarray(p, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if mode == "width":
        edges = np.linspace(0, 1, bins + 1)
        b = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    else:
        order = np.argsort(p, kind="mergesort")
        b = np.empty(len(p), dtype=int)
        b[order] = np.minimum((np.arange(len(p)) * bins) // len(p), bins - 1)
    total, e = len(p), 0.0
    for k in range(bins):
        m = b == k
        if m.any():
            e += m.sum() / total * abs(p[m].mean() - y[m].mean())
    return float(e)


def reliability(p, y, bins=15):
    order = np.argsort(p, kind="mergesort")
    rows = []
    for chunk in np.array_split(order, bins):
        if len(chunk):
            rows.append({"n": int(len(chunk)), "mean_predicted": _r(np.mean(p[chunk]), 6),
                         "observed_rate": _r(np.mean(y[chunk]), 6), "fraud": int(y[chunk].sum())})
    return rows


def slope_intercept(p, y):
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p).reshape(-1, 1), y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


# ---------------------------------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------------------------------

def load_scores(scores_dir):
    frames = {s: pd.read_csv(Path(scores_dir) / f"scores_seed_{s}.csv.gz", parse_dates=["timestamp"]) for s in ALL_SEEDS}
    return frames


def meta_block(scores_dir):
    manifests = {s: json.loads((MANIFESTS / f"seed_{s}_manifest.json").read_text()) for s in ALL_SEEDS}
    model_files = {}
    for key, name in SAVED_SETS.items():
        from app.model_sets import MODEL_SETS, model_set_directory
        spec = MODEL_SETS[name]
        d = config.MODELS_SAVED_DIR if spec.candidate is None else model_set_directory(name)
        model_files[name] = {p.name: sha256_file(p) for p in sorted(Path(d).glob("*")) if p.is_file()}
    import sklearn
    import tensorflow as tf
    return {
        "experiment": EXPERIMENT_ID, "protocol_sha256": sha256_file(PROTOCOL),
        "seeds": {"validation": list(DEV_SEEDS), "test": list(TEST_SEEDS)}, "customers_per_seed": CUSTOMERS,
        "bootstrap": {"seed": BOOTSTRAP_SEED, "reps": REPS, "unit": "customer component (ring/household whole), within seed, pooled"},
        "dataset_sha256": {str(s): m["files"]["transactions_with_features.csv"]["sha256"] for s, m in manifests.items()},
        "generator_version": manifests[ALL_SEEDS[0]].get("generator_version"),
        "generator_commands": {str(s): m.get("command") for s, m in manifests.items()},
        "model_artifact_sha256": model_files,
        "scores_sha256": {str(s): sha256_file(Path(scores_dir) / f"scores_seed_{s}.csv.gz") for s in ALL_SEEDS},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scikit_learn": sklearn.__version__, "tensorflow": tf.__version__},
        "baselines": json.loads((Path(scores_dir) / "baselines_info.json").read_text()),
    }


def cmd_analyze(args):
    scores_dir = Path(args.scores)
    RESULTS.mkdir(exist_ok=True)
    frames = load_scores(scores_dir)
    meta = meta_block(scores_dir)
    dev = pd.concat([frames[s] for s in DEV_SEEDS], ignore_index=True)
    test = pd.concat([frames[s] for s in TEST_SEEDS], ignore_index=True)
    test["first"] = first_fraud_flags(test)
    dev["first"] = first_fraud_flags(dev)

    y_test = test["is_fraud"].to_numpy()
    p10 = (test["n_prior"] >= MIN_PRIOR).to_numpy()
    p10_dev = (dev["n_prior"] >= MIN_PRIOR).to_numpy()
    meta["populations"] = {
        "dev": _population_counts(dev), "test": _population_counts(test),
        "test_by_seed": {str(s): _population_counts(test[test["seed"] == s]) for s in TEST_SEEDS},
    }
    t0 = time.time()
    rs = Resampler(test["group"].to_numpy(), test["seed"].to_numpy(), REPS, BOOTSTRAP_SEED)
    print(f"resamples ready ({time.time() - t0:.0f}s)")

    sim = analyze_similarity(dev, test, y_test, p10, p10_dev, rs)
    (RESULTS / "exp3_similarity.json").write_text(json.dumps({"meta": meta, **sim["report"]}, indent=1))
    print("similarity done")
    cal = analyze_calibration(dev, test, rs)
    (RESULTS / "exp3_calibration.json").write_text(json.dumps({"meta": meta, **cal}, indent=1))
    print("calibration done")
    abl = analyze_ablation(dev, test, p10, p10_dev, rs, sim["combo_test"], sim["combo_dev"])
    (RESULTS / "exp3_ablation.json").write_text(json.dumps({"meta": meta, **abl}, indent=1))
    print("ablation done")


def _population_counts(df):
    p10 = df["n_prior"] >= MIN_PRIOR
    return {"rows": int(len(df)), "fraud": int(df["is_fraud"].sum()), "customers": int(df["customer_id"].nunique()),
            "p10_rows": int(p10.sum()), "p10_fraud": int(df.loc[p10, "is_fraud"].sum()),
            "fraud_under_10_prior": int(df.loc[~p10, "is_fraud"].sum()),
            "fraud_episodes": int(df.loc[df["is_fraud"] == 1, ["seed", "episode"]].drop_duplicates().shape[0])}


def _summ(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return None
    q = np.percentile(x, [25, 50, 75])
    return {"n": int(len(x)), "q25": _r(q[0], 3), "median": _r(q[1], 3), "q75": _r(q[2], 3), "mean": _r(x.mean(), 3)}


def boot_rank(items: dict, rs: Resampler):
    """items: name -> RankItem. Returns name -> (ap[reps], auc[reps])."""
    ap = {k: np.empty(rs.reps) for k in items}
    auc = {k: np.empty(rs.reps) for k in items}
    for r in range(rs.reps):
        w = rs.weights(r)
        for k, it in items.items():
            ap[k][r], auc[k][r] = it.boot(w)
    return ap, auc


def analyze_similarity(dev, test, y_test, p10, p10_dev, rs):
    idx10 = np.flatnonzero(p10)
    d_s0, d_s1 = test["dev_s0"].to_numpy(), test["dev_s1"].to_numpy()
    clf = test["default"].to_numpy()
    rep = {"experiment_id": "EXP3-SIM"}

    # 1. distributions
    is_f = y_test == 1
    dist = {"all_p10": {"legit": _summ(d_s0[idx10][~is_f[idx10]]), "fraud": _summ(d_s0[idx10][is_f[idx10]])}}
    dist["by_fraud_type_p10"] = {str(t): _summ(d_s0[idx10][(test["fraud_type"].to_numpy()[idx10] == t) & is_f[idx10]])
                                 for t in sorted(test.loc[is_f & p10, "fraud_type"].unique())}
    nprior = test["n_prior"].to_numpy()
    dist["by_history_bin_diagnostic"] = {}
    for name, lo, hi in HISTORY_BINS:
        m = (nprior >= lo) & (nprior <= hi) & np.isfinite(d_s0)
        dist["by_history_bin_diagnostic"][name] = {
            "served_by_api": bool(lo >= MIN_PRIOR), "legit": _summ(d_s0[m & ~is_f]), "fraud": _summ(d_s0[m & is_f])}
    rep["distributions_deviation"] = dist

    # 2-5. discrimination (rank items, bootstrap)
    items = {"s0_p10": RankItem(idx10, y_test, d_s0), "s1_p10": RankItem(idx10, y_test, d_s1),
             "classifier_p10": RankItem(idx10, y_test, clf)}
    types = sorted(test.loc[is_f & p10, "fraud_type"].unique())
    ftype = test["fraud_type"].to_numpy()
    for t in types:       # fraud of this type against all legitimate rows
        items[f"s0_type_{t}"] = RankItem(np.flatnonzero(p10 & (~is_f | (ftype == t))), y_test, d_s0)
    for name, lo, hi in HISTORY_BINS:
        m = (nprior >= lo) & (nprior <= hi) & np.isfinite(d_s0)
        if (m & is_f).sum() >= 20:
            items[f"s0_bin_{name}"] = RankItem(np.flatnonzero(m), y_test, d_s0)
    # combination (fitted on validation seeds only)
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    dev10 = np.flatnonzero(p10_dev)
    Xd = np.column_stack([logit(dev["default"].to_numpy()), dev["dev_s0"].to_numpy()])[dev10]
    combo = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xd, dev["is_fraud"].to_numpy()[dev10])
    X_test = np.column_stack([logit(clf), d_s0])
    combo_test = np.full(len(test), np.nan)
    combo_test[idx10] = combo.predict_proba(X_test[idx10])[:, 1]
    combo_dev = np.full(len(dev), np.nan)
    combo_dev[dev10] = combo.predict_proba(np.column_stack([logit(dev["default"].to_numpy()), dev["dev_s0"].to_numpy()])[dev10])[:, 1]
    items["combo_p10"] = RankItem(idx10, y_test, combo_test)
    ap, auc = boot_rank(items, rs)
    out = {}
    for k, it in items.items():
        pap, pauc = it.point()
        out[k] = {"pr_auc": _r(pap), "pr_auc_ci95": ci(ap[k]), "roc_auc": _r(pauc), "roc_auc_ci95": ci(auc[k]),
                  "cliffs_delta": _r(2 * pauc - 1), "rows": int(len(it.idx)), "fraud": int(it.y.sum()),
                  "auc_interval_excludes_0.5": bool(ci(auc[k])[0] > 0.5 or ci(auc[k])[1] < 0.5)}
    rep["discrimination_pooled_test"] = out
    base_rate = float(y_test[idx10].mean())
    rep["prevalence_p10_test"] = _r(base_rate)

    def paired(a, b, which):
        pa, pb = items[a].point(), items[b].point()
        v = (ap if which == "pr_auc" else auc)
        j = 0 if which == "pr_auc" else 1
        diff = v[a] - v[b]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        return {"difference": _r(pa[j] - pb[j]), "ci95": [_r(lo), _r(hi)], "excludes_zero": bool(lo > 0 or hi < 0)}
    rep["incremental_value_over_classifier"] = {
        "model": "StandardScaler + LogisticRegression on [logit(clipped classifier score), deviation S0], fitted on validation seeds 9301-9302, P10 rows",
        "combo_minus_classifier_pr_auc": paired("combo_p10", "classifier_p10", "pr_auc"),
        "combo_minus_classifier_roc_auc": paired("combo_p10", "classifier_p10", "roc_auc"),
        "coefficients_standardised": {"logit_classifier": _r(combo[-1].coef_[0, 0]), "deviation_s0": _r(combo[-1].coef_[0, 1])},
        "interpretation_rule_met_discriminatory_value": bool(out["s0_p10"]["auc_interval_excludes_0.5"] and out["s0_p10"]["roc_auc"] > 0.5),
    }
    rep["s1_minus_s0"] = {"pr_auc": paired("s1_p10", "s0_p10", "pr_auc"), "roc_auc": paired("s1_p10", "s0_p10", "roc_auc")}
    from scipy.stats import spearmanr
    fm = np.flatnonzero(p10 & is_f)
    rep["within_fraud_spearman_deviation_vs_classifier"] = _r(spearmanr(d_s0[fm], clf[fm])[0])
    lm = np.flatnonzero(p10 & ~is_f)
    rep["within_legit_spearman_deviation_vs_classifier"] = _r(spearmanr(d_s0[lm], clf[lm])[0])
    rep["per_seed_roc_auc_s0_p10"] = {str(s): _r(auc_binary(y_test[idx10][test["seed"].to_numpy()[idx10] == s],
                                                           d_s0[idx10][test["seed"].to_numpy()[idx10] == s]))
                                      for s in TEST_SEEDS}

    # zero / low variance
    nc, nl = test["n_const"].to_numpy(), test["n_low"].to_numpy()
    zv = {}
    for label, m in (("p10_all", p10), ("p10_legit", p10 & ~is_f), ("p10_fraud", p10 & is_f)):
        zv[label] = {"rows": int(m.sum()), "share_with_a_zero_std_feature": _r((nc[m] >= 1).mean(), 4),
                     "share_with_a_std_below_0.1": _r((nl[m] >= 1).mean(), 4),
                     "mean_zero_std_features": _r(nc[m].mean(), 3)}
    rep["near_constant_features"] = zv
    rep["circularity_note"] = ("Fraud in this generator departs from the generator's own customer baselines on the same nine features "
                               "the similarity uses, so a high AUC shows only that the score reflects the generator's fraud rules.")
    return {"report": rep, "combo_test": combo_test, "combo_dev": combo_dev}


def analyze_calibration(dev, test, rs):
    y_dev, y = dev["is_fraud"].to_numpy(), test["is_fraud"].to_numpy()
    s_dev, s = dev["default"].to_numpy(), test["default"].to_numpy()
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    platt = LogisticRegression(C=1e6, max_iter=1000).fit(logit(s_dev).reshape(-1, 1), y_dev)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(s_dev, y_dev)
    prevalence = float(y_dev.mean())
    variants = {"uncalibrated": np.clip(s, EPS, 1 - EPS),
                "sigmoid": np.clip(platt.predict_proba(logit(s).reshape(-1, 1))[:, 1], EPS, 1 - EPS),
                "isotonic": np.clip(iso.predict(s), EPS, 1 - EPS),
                "prevalence_only": np.full(len(s), np.clip(prevalence, EPS, 1 - EPS))}
    p10 = (test["n_prior"] >= MIN_PRIOR).to_numpy()
    seeds = test["seed"].to_numpy()

    def block(mask):
        out = {}
        for k, p in variants.items():
            b, ll = cal_metrics(p[mask], y[mask])
            out[k] = {"brier": _r(b, 8), "log_loss": _r(ll, 6), "ece_equal_width": _r(ece(p[mask], y[mask], "width"), 6),
                      "ece_equal_mass": _r(ece(p[mask], y[mask], "mass"), 6)}
            if k != "prevalence_only":
                sl, ic = slope_intercept(p[mask], y[mask])
                out[k].update({"calibration_slope": _r(sl, 4), "calibration_intercept": _r(ic, 4),
                               "mean_predicted": _r(p[mask].mean(), 6), "observed_rate": _r(y[mask].mean(), 6)})
        return out
    rep = {"experiment_id": "EXP3-CAL",
           "calibrator_fit": {"data": "validation seeds 9301-9302, all rows", "rows": int(len(y_dev)), "fraud": int(y_dev.sum()),
                              "prevalence": _r(prevalence, 6), "probability_clip": EPS},
           "pooled_test_all_rows": block(np.ones(len(y), dtype=bool)), "pooled_test_p10_rows": block(p10),
           "per_seed_all_rows": {str(sd): block(seeds == sd) for sd in TEST_SEEDS},
           "reliability_uncalibrated_equal_mass_15": reliability(variants["uncalibrated"], y),
           "reliability_isotonic_equal_mass_15": reliability(variants["isotonic"], y),
           "reliability_sigmoid_equal_mass_15": reliability(variants["sigmoid"], y)}
    # bootstrap: differences of Brier / log loss (calibrated - uncalibrated), pooled test, all rows
    d = {k: {"brier": np.empty(rs.reps), "ll": np.empty(rs.reps)} for k in variants}
    for r in range(rs.reps):
        w = rs.weights(r)
        for k, p in variants.items():
            d[k]["brier"][r], d[k]["ll"][r] = cal_metrics(p, y, w)
    rep["paired_differences_vs_uncalibrated"] = {}
    for k in ("sigmoid", "isotonic", "prevalence_only"):
        row = {}
        for m, key in (("brier", "brier"), ("log_loss", "ll")):
            diff = d[k][key] - d["uncalibrated"][key]
            lo, hi = np.percentile(diff, [2.5, 97.5])
            point = rep["pooled_test_all_rows"][k][m] - rep["pooled_test_all_rows"]["uncalibrated"][m]
            row[m] = {"difference": _r(point, 8), "ci95": [_r(lo, 8), _r(hi, 8)], "excludes_zero": bool(lo > 0 or hi < 0)}
        better_each_seed = all(rep["per_seed_all_rows"][str(sd)][k]["brier"] < rep["per_seed_all_rows"][str(sd)]["uncalibrated"]["brier"]
                               and rep["per_seed_all_rows"][str(sd)][k]["log_loss"] < rep["per_seed_all_rows"][str(sd)]["uncalibrated"]["log_loss"]
                               for sd in TEST_SEEDS)
        row["lower_in_every_test_seed"] = bool(better_each_seed)
        row["protocol_rule_calibrator_useful"] = bool(row["brier"]["difference"] < 0 and row["brier"]["ci95"][1] < 0
                                                     and row["log_loss"]["difference"] < 0 and row["log_loss"]["ci95"][1] < 0
                                                     and better_each_seed)
        rep["paired_differences_vs_uncalibrated"][k] = row
    # ranking metrics before and after
    rank = {}
    for k, p in variants.items():
        ap_, auc_ = RankItem(np.arange(len(y)), y, p).point()
        rank[k] = {"pr_auc": _r(ap_), "roc_auc": _r(auc_), "distinct_score_values": int(len(np.unique(p)))}
    rep["ranking_metrics_pooled_test"] = rank
    rep["note"] = ("The served Fraud Score is this output x 100 (capped at 99.9). Nothing in this file changes serving; the score "
                   "remains a model score, not a probability.")
    return rep


def analyze_ablation(dev, test, p10, p10_dev, rs, combo_test, combo_dev):
    y_dev, y = dev["is_fraud"].to_numpy(), test["is_fraud"].to_numpy()
    models = {"default": "default (LSTM -> random forest)", "dnn_lstm": "LSTM -> DNN", "dnn_only": "DNN only (no LSTM)",
              "production": "production v1 (LSTM -> DNN, different training data)", "rf_nolstm": "random forest, no LSTM input",
              "logreg": "logistic regression (balanced)", "rule": "amount-and-hour rule",
              "sim_s0": "similarity deviation alone (S0)", "default_plus_sim": "default + similarity (logistic combination)"}
    score_t = {k: test[k].to_numpy() for k in ("default", "dnn_lstm", "dnn_only", "production", "rf_nolstm", "logreg", "rule")}
    score_d = {k: dev[k].to_numpy() for k in score_t}
    score_t["sim_s0"], score_d["sim_s0"] = test["dev_s0"].to_numpy(), dev["dev_s0"].to_numpy()
    score_t["default_plus_sim"], score_d["default_plus_sim"] = combo_test, combo_dev
    rep = {"experiment_id": "EXP3-ABL", "models": models,
           "threshold_rule": f"lowest cut-off whose false-positive rate on validation seeds 9301-9302 (rows with >= {MIN_PRIOR} earlier transactions) is <= {FPR_TARGET}; frozen before test is scored",
           "cutoffs": {}, "pooled_test": {}, "per_seed": {}, "paired": {}}
    cut = {}
    for k in models:
        sd = score_d[k][p10_dev]
        keep = np.isfinite(sd)
        cut[k] = ms.tie_safe_cutoff(dev["is_fraud"].to_numpy()[p10_dev][keep], sd[keep], FPR_TARGET)
        rep["cutoffs"][k] = cut[k]
    first = test["first"].to_numpy()
    seeds = test["seed"].to_numpy()
    pops = {"p10": p10, "all": np.ones(len(y), dtype=bool)}
    rank_items, alert_items = {}, {}
    for pop, pm in pops.items():
        for k in models:
            sc = score_t[k]
            m = pm & np.isfinite(sc)
            if not m.any() or (pop == "all" and k in ("sim_s0", "default_plus_sim")):
                continue
            idx = np.flatnonzero(m)
            rank_items[(pop, k)] = RankItem(idx, y, sc)
            if cut[k] is not None:
                alert = np.zeros(len(y), dtype=bool)
                alert[idx] = sc[idx] >= cut[k]
                alert_items[(pop, k)] = AlertItem(idx, y, alert, first)
    ap, auc = boot_rank(rank_items, rs)
    amet = {key: {m: np.empty(rs.reps) for m in ("precision", "recall", "f1", "legit_alerts_per_1000_legit", "first_fraud_recall")}
            for key in alert_items}
    for r in range(rs.reps):
        w = rs.weights(r)
        for key, it in alert_items.items():
            mm = AlertItem.metrics(it.sums(w))
            for m, v in mm.items():
                amet[key][m][r] = v
    for pop in pops:
        block = {}
        for k in models:
            if (pop, k) not in rank_items:
                continue
            it = rank_items[(pop, k)]
            pap, pauc = it.point()
            row = {"label": models[k], "rows": int(len(it.idx)), "fraud": int(it.y.sum()),
                   "pr_auc": _r(pap), "pr_auc_ci95": ci(ap[(pop, k)]), "roc_auc": _r(pauc), "roc_auc_ci95": ci(auc[(pop, k)])}
            if (pop, k) in alert_items:
                ai = alert_items[(pop, k)]
                tp, fp, fn, tn, ff, ffa = ai.sums(None)
                point = AlertItem.metrics((tp, fp, fn, tn, ff, ffa))
                row["confusion"] = {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}
                for m, v in point.items():
                    row[m] = _r(v)
                    row[m + "_ci95"] = ci(amet[(pop, k)][m])
            else:
                row["note"] = "no cut-off keeps the validation false-positive rate within the target"
            block[k] = row
        rep["pooled_test"][pop] = block
        # per seed point values
        per = {}
        for sd in TEST_SEEDS:
            per[str(sd)] = {}
            for k in models:
                if (pop, k) not in rank_items:
                    continue
                sc = score_t[k]
                m = pops[pop] & np.isfinite(sc) & (seeds == sd)
                ap_, auc_ = RankItem(np.flatnonzero(m), y, sc).point()
                e = {"pr_auc": _r(ap_), "roc_auc": _r(auc_)}
                if cut[k] is not None:
                    idx = np.flatnonzero(m)
                    a = np.zeros(len(y), dtype=bool)
                    a[idx] = sc[idx] >= cut[k]
                    mm = AlertItem.metrics(AlertItem(idx, y, a, first).sums(None))
                    e.update({"recall": _r(mm["recall"]), "precision": _r(mm["precision"]),
                              "legit_alerts_per_1000_legit": _r(mm["legit_alerts_per_1000_legit"])})
                per[str(sd)][k] = e
        rep["per_seed"][pop] = per
    # paired differences
    for pop in pops:
        rep["paired"][pop] = {}
        for a, b in (("default", "rf_nolstm"), ("default", "dnn_lstm"), ("default", "dnn_only"), ("default", "logreg"),
                     ("default", "rule"), ("default_plus_sim", "default")):
            if (pop, a) not in rank_items or (pop, b) not in rank_items:
                continue
            row = {}
            for name, va, vb, pa_, pb_ in (("pr_auc", ap[(pop, a)], ap[(pop, b)], rank_items[(pop, a)].point()[0], rank_items[(pop, b)].point()[0]),
                                           ("roc_auc", auc[(pop, a)], auc[(pop, b)], rank_items[(pop, a)].point()[1], rank_items[(pop, b)].point()[1])):
                lo, hi = np.percentile(va - vb, [2.5, 97.5])
                row[name] = {"difference": _r(pa_ - pb_), "ci95": [_r(lo), _r(hi)], "excludes_zero": bool(lo > 0 or hi < 0)}
            if (pop, a) in alert_items and (pop, b) in alert_items:
                for m in ("recall", "legit_alerts_per_1000_legit", "precision"):
                    diff = amet[(pop, a)][m] - amet[(pop, b)][m]
                    lo, hi = np.nanpercentile(diff, [2.5, 97.5])
                    pa_ = AlertItem.metrics(alert_items[(pop, a)].sums(None))[m]
                    pb_ = AlertItem.metrics(alert_items[(pop, b)].sums(None))[m]
                    row[m] = {"difference": _r(pa_ - pb_), "ci95": [_r(lo), _r(hi)], "excludes_zero": bool(lo > 0 or hi < 0)}
            rep["paired"][pop][f"{a}_minus_{b}"] = row
    return rep


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("--data", required=True)
    s.add_argument("--scores", default=str(RESULTS / "scores"))
    a = sub.add_parser("analyze")
    a.add_argument("--scores", default=str(RESULTS / "scores"))
    args = p.parse_args(argv)
    {"generate": cmd_generate, "score": cmd_score, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
