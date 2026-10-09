"""
Step 4D: choosing the downstream classifier behind the LSTM (pre-registered in
docs/step4d-downstream-selection-protocol.md, protocol "4D v1").

    previous 10 transactions -> LSTM (unchanged) -> risk_score
    9 behavioral features + risk_score -> downstream classifier -> Fraud Score

Candidates: the existing DNN (incumbent), logistic regression, random forest and
histogram gradient boosting (scikit-learn). XGBoost is not installed in the
project environment and is not evaluated.

Stages (each writes JSON under models/evaluation/downstream/):

    cd backend
    python -m app.evaluation.downstream data           # generate the development datasets 301-305 if missing
    python -m app.evaluation.downstream reproduce      # out-of-fold LSTM scores for seeds 11-15, verified bit for bit
    python -m app.evaluation.downstream tune           # each family's setting, on the validation period only
    python -m app.evaluation.downstream develop        # all families x seeds on development data 301-305 + selection record
    python -m app.evaluation.downstream confirm        # fresh hold-out 501-505 / 511-515, scored once (needs the record)

Nothing here reads or writes models/saved/, and the hold-out seeds used before
(101-105, 201-205, 401-405, 411-415) are never loaded.
"""

import argparse
import hashlib
import io
import json
import platform
import time
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd

from .. import config
from ..features.feature_engineering import FEATURE_COLUMNS
from ..features.ground_truth import assert_no_ground_truth
from ..models.downstream_classifier import ClassifierModel
from . import holdout as h
from . import multiseed as ev
from .datasets import load_evaluation_data, resolve_dataset
from .stacking import CLIP, fit_dnn, fit_lstm, predict_fixed_batch, stacked_risk_scores, N_FOLDS
from .windows import assert_past_only, build_windows

PROTOCOL_VERSION = "4D v1"
PROTOCOL_DOC = config.PROJECT_ROOT / "docs" / "step4d-downstream-selection-protocol.md"
TRAINING_SEEDS = (11, 12, 13, 14, 15)
DEPLOY_SEED = 14                                   # inherited from 4C-3E.6 stage B (fixed before this protocol)
DEVELOPMENT_SEEDS = (301, 302, 303, 304, 305)
FINAL_SEEDS = (501, 502, 503, 504, 505)
FINAL_NEW_CUSTOMER_SEEDS = (511, 512, 513, 514, 515)
SPENT_SEEDS = (42, 101, 102, 103, 104, 105, 201, 202, 203, 204, 205, 401, 402, 403, 404, 405,
               411, 412, 413, 414, 415)
FINAL_DATA_DIR = h.HOLDOUT_DATA_DIR / "final_4d"
OUTPUT_DIR = config.EVALUATION_DIR / "downstream"
CACHE_DIR = OUTPUT_DIR / "cache"                   # large intermediate arrays (not part of the reports)
RECORD_NAME = "selection_record.json"
INPUT_COLUMNS = list(FEATURE_COLUMNS) + ["risk_score"]
FAMILIES = ("dnn", "logistic_regression", "random_forest", "hist_gradient_boosting")
CHALLENGERS = FAMILIES[1:]
INCUMBENT = "dnn"
REPS = h.BOOTSTRAP_REPS
# selection-rule constants (protocol section 6)
RECALL_MARGIN = 0.05
LEGIT_ALERT_MARGIN_PER_1000 = 1.0
CRITICAL_RECALL_MARGIN = 0.05
MIN_RECALL = 0.40
MIN_SEEDS_HIGHER = 4
MIN_DATASETS_HIGHER = 4
TIE_MARGIN_PR_AUC = 0.005
DIGITS = 6


class DownstreamError(ValueError):
    pass


def _r(x, d=DIGITS):
    if x is None:
        return None
    x = float(x)
    return None if np.isnan(x) else round(x, d)


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256(path) -> str:
    return _sha256_bytes(Path(path).read_bytes())


def _write_json(obj, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default) + "\n")
    return path


def _json_default(x):
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x))


def assert_allowed_seed(seed: int, allowed) -> int:
    seed = int(seed)
    if seed in SPENT_SEEDS or seed not in allowed:
        raise DownstreamError(f"data seed {seed} may not be used here (allowed: {list(allowed)})")
    return seed


# ---- the hyperparameter grids (protocol section 3) ----------------------------------------------

def grid(family: str) -> list:
    if family == "logistic_regression":
        return [{"C": c, "class_weight": w} for c in (0.01, 0.1, 1.0, 10.0) for w in (None, "balanced")]
    if family == "random_forest":
        return [{"min_samples_leaf": leaf, "max_depth": depth, "class_weight": w}
                for leaf in (1, 5, 20) for depth in (None, 12) for w in (None, "balanced_subsample")]
    if family == "hist_gradient_boosting":
        return [{"learning_rate": lr, "max_leaf_nodes": n, "max_iter": it, "class_weight": w}
                for lr in (0.05, 0.1) for n in (15, 31) for it in (100, 300) for w in (None, "balanced")]
    if family == "dnn":
        return [{}]
    raise DownstreamError(f"unknown family {family!r}")


def make_estimator(family: str, params: dict, seed: int):
    if family == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(C=params["C"], class_weight=params["class_weight"], solver="lbfgs", max_iter=2000)
    if family == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=200, max_features="sqrt", min_samples_leaf=params["min_samples_leaf"],
                                      max_depth=params["max_depth"], class_weight=params["class_weight"],
                                      random_state=seed, n_jobs=-1)
    if family == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(learning_rate=params["learning_rate"], max_leaf_nodes=params["max_leaf_nodes"],
                                              max_iter=params["max_iter"], class_weight=params["class_weight"],
                                              early_stopping=False, random_state=seed)
    raise DownstreamError(f"no scikit-learn estimator for {family!r}")


# ---- the training data and the unchanged LSTM ----------------------------------------------------

def training_data():
    """(data, split labels) of the training dataset; the split must equal the saved one."""
    from ..training import candidates as cand
    spec = resolve_dataset("v2")
    data = load_evaluation_data(spec)
    labels, _, _, _ = cand.load_saved_split(spec, data)
    return data, labels


def saved_artifact_dir(seed: int, root=None) -> Path:
    from ..training import multiseed as training
    return training.seed_root(seed, root) / "dnn_lstm"


def reproduce_seed(seed: int, data=None, labels=None, root=None, log=print) -> dict:
    """Recomputes the out-of-fold LSTM scores of the training rows for training seed
    `seed` with the unchanged recipe, and accepts them only if (1) the final LSTM and
    (2) a DNN refit on them reproduce the saved v2_dnn_lstm@seed weights exactly."""
    from ..training.candidates import weights_sha256
    if data is None:
        data, labels = training_data()
    manifest = json.loads((saved_artifact_dir(seed, root) / "manifest.json").read_text())
    df = data.frame
    assert_no_ground_truth([c for c in df.columns if c != "is_fraud"], "columns of the training frame")
    X, y, target = build_windows(df)
    assert_past_only(df, target)
    split = np.asarray(labels)[target]
    tr, va = split == "train", split == "validation"
    ts = df["timestamp"].to_numpy()[target]
    groups = df["customer_id"].to_numpy()[target]
    F = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)[target]
    t0 = time.time()
    lstm_p, final_lstm, prov = stacked_risk_scores(X, y, ts, groups, tr, fit_fn=fit_lstm, n_folds=N_FOLDS, seed=seed)
    lstm_sha = weights_sha256(final_lstm.model)
    risk = np.round(lstm_p * 100, 2)
    XD = np.column_stack([F, risk]).astype(np.float32)
    dnn = fit_dnn(XD[tr], y[tr], ts[tr], seed=seed)
    dnn_sha = weights_sha256(dnn.model)
    ok = {"final_lstm_weights_match_saved": lstm_sha == manifest["model"]["lstm_weights_sha256"],
          "dnn_refit_weights_match_saved": dnn_sha == manifest["model"]["dnn_weights_sha256"]}
    log(f"seed {seed}: {ok} ({time.time() - t0:.0f}s)")
    if not all(ok.values()):
        raise DownstreamError(f"seed {seed}: the recomputed out-of-fold scores do not reproduce the saved artifact: {ok}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE_DIR / f"train_seed_{seed}.npz", XD=XD[tr], y=y[tr], ts=ts[tr].astype("datetime64[ns]").astype(np.int64),
                        transaction_id=df["transaction_id"].to_numpy()[target][tr].astype(str))
    return {"seed": seed, **ok, "lstm_weights_sha256": lstm_sha, "dnn_weights_sha256": dnn_sha,
            "training_rows": int(tr.sum()), "training_fraud": int(y[tr].sum()),
            "dnn_mean_matches_saved": bool(np.array_equal(dnn.mean, np.load(saved_artifact_dir(seed, root) / "dnn_feature_mean.npy"))),
            "seconds": round(time.time() - t0, 1)}


def load_training_matrix(seed: int):
    z = np.load(CACHE_DIR / f"train_seed_{seed}.npz")
    return z["XD"], z["y"]


def lstm_bundle(seed: int, root=None):
    """The saved seed-`seed` model set (LSTM + DNN + scalers), validated against its manifest."""
    from ..model_sets import load_model_set
    from ..training import multiseed as training
    return load_model_set("v2_dnn_lstm", candidates_root=training.seed_root(seed, root))


class _Capture:
    """Stands in for the classifier inside holdout.score_frame and records the exact
    scaled, clipped and imputed input matrix it is given (padding rows removed)."""

    def __init__(self, n_inputs):
        self.chunks, self.input_shape = [], (None, n_inputs)

    def predict_on_batch(self, X):
        self.chunks.append(np.array(X, dtype=np.float32))
        return np.zeros((len(X), 1), dtype=np.float32)


def scoring_inputs(frame: pd.DataFrame, bundle) -> np.ndarray:
    """The classifier input matrix the application would build for every row of `frame`
    (LSTM risk score, scaling with the DNN scaler, +-6 clip, cold-start imputation),
    obtained by running holdout.score_frame itself with a recording classifier."""
    cap = _Capture(len(bundle.dnn_input_columns))
    ms = SimpleNamespace(uses_lstm=True, dnn_input_columns=list(bundle.dnn_input_columns),
                         lstm_model=bundle.lstm_model, lstm_mean=bundle.lstm_mean, lstm_std=bundle.lstm_std,
                         dnn_mean=bundle.dnn_mean, dnn_std=bundle.dnn_std, dnn_model=cap)
    h.score_frame(frame, ms)
    Z = np.concatenate(cap.chunks)[: len(frame)] if cap.chunks else np.zeros((0, len(INPUT_COLUMNS)), np.float32)
    return Z


def cached_inputs(kind: str, seed_data: int, seed_training: int, frame: pd.DataFrame, bundle) -> np.ndarray:
    path = CACHE_DIR / f"inputs_{kind}_{seed_data}_t{seed_training}.npy"
    if path.exists():
        Z = np.load(path)
        if len(Z) == len(frame):
            return Z
    Z = scoring_inputs(frame, bundle)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(path, Z)
    return Z


def scaled_training(XD: np.ndarray, bundle) -> np.ndarray:
    """Training inputs of the non-DNN families: the DNN's scaler (fitted on these rows), clipped to +-6."""
    mean = XD.mean(axis=0)
    std = XD.std(axis=0)
    std[std == 0] = 1.0
    if not (np.array_equal(mean.astype(np.float32), bundle.dnn_mean.astype(np.float32))
            and np.array_equal(std.astype(np.float32), bundle.dnn_std.astype(np.float32))):
        raise DownstreamError("the training rows do not reproduce the saved DNN scaler")
    return np.clip((XD - bundle.dnn_mean) / bundle.dnn_std, -CLIP, CLIP).astype(np.float32)


def fit_family(family: str, params: dict, seed: int, Xs: np.ndarray, y: np.ndarray):
    est = make_estimator(family, params, seed)
    est.fit(Xs, y.astype(int))
    return ClassifierModel(est, family, Xs.shape[1])


def model_scores(model, Z: np.ndarray) -> np.ndarray:
    """Scores exactly as holdout.score_frame returns them: fixed-batch prediction, capped at 0.999."""
    return np.minimum(predict_fixed_batch(model, Z), np.float32(h.SCORE_CAP))


def key(family: str, seed: int) -> str:
    return f"{family}@{int(seed)}"


def policy_cutoffs_from_scores(y: np.ndarray, scores: np.ndarray) -> dict:
    """The 4C-3B rule on validation scores (multiseed.tie_safe_cutoff)."""
    out = {}
    for tier, target in ev.POLICY_FPR.items():
        cut = ev.tie_safe_cutoff(y, scores, target)
        if cut is None:
            raise DownstreamError(f"no cut-off keeps the validation false-positive rate within {target}")
        out[tier] = cut
    return out


# ---- stage: tune (validation period only) ---------------------------------------------------------

def tune(seeds=TRAINING_SEEDS, families=CHALLENGERS, log=print) -> dict:
    from sklearn.metrics import average_precision_score
    data, labels = training_data()
    mask = ev.validation_mask(data, labels)
    yv = data.frame["is_fraud"].to_numpy()[mask]
    results = {f: [] for f in families}
    dnn_validation = {}
    for s in seeds:
        bundle = lstm_bundle(s)
        Zv = cached_inputs("train", 42, s, data.frame, bundle)[mask]
        dnn_scores = model_scores(bundle.dnn_model, Zv)
        dnn_validation[str(s)] = {"pr_auc": _r(average_precision_score(yv, dnn_scores)),
                                  "cutoffs": policy_cutoffs_from_scores(yv, dnn_scores)}
        XD, y = load_training_matrix(s)
        Xs = scaled_training(XD, bundle)
        for f in families:
            for i, params in enumerate(grid(f)):
                t0 = time.time()
                model = fit_family(f, params, s, Xs, y)
                pr = average_precision_score(yv, model_scores(model, Zv))
                if len(results[f]) <= i:
                    results[f].append({"params": params, "validation_pr_auc": {}})
                results[f][i]["validation_pr_auc"][str(s)] = _r(pr)
                log(f"tune {f} {params} seed {s}: validation PR-AUC {pr:.4f} ({time.time() - t0:.1f}s)")
    chosen = {}
    for f in families:
        for entry in results[f]:
            v = list(entry["validation_pr_auc"].values())
            entry["mean_validation_pr_auc"] = _r(np.mean(v))
        best = max(range(len(results[f])), key=lambda i: (results[f][i]["mean_validation_pr_auc"], -i))
        chosen[f] = {"params": results[f][best]["params"], "index_in_grid": best,
                     "mean_validation_pr_auc": results[f][best]["mean_validation_pr_auc"]}
    out = {"protocol": PROTOCOL_VERSION, "stage": "tune (validation period of the training dataset only)",
           "validation_rows": int(mask.sum()), "validation_fraud": int(yv.sum()),
           "training_seeds": list(seeds), "dnn_validation": dnn_validation,
           "grid_results": results, "chosen": chosen,
           "rule": "highest mean validation PR-AUC over the training seeds; ties to the earlier setting"}
    _write_json(out, OUTPUT_DIR / "tuning.json")
    return out


def chosen_params() -> dict:
    t = json.loads((OUTPUT_DIR / "tuning.json").read_text())
    return {f: t["chosen"][f]["params"] for f in t["chosen"]}


# ---- calibration and score distributions ------------------------------------------------------------

def calibration_metrics(y: np.ndarray, p: np.ndarray, bins: int = 10) -> dict:
    """Brier score, log loss and expected calibration error (equal-width and
    equal-mass bins) of the score read as if it were a probability."""
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    brier = float(np.mean((p - y) ** 2))
    logloss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    prevalence = float(y.mean())
    brier_ref = prevalence * (1 - prevalence)

    def ece(edges):
        idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
        total, table = 0.0, []
        for b in range(len(edges) - 1):
            m = idx == b
            if not m.any():
                continue
            gap = abs(p[m].mean() - y[m].mean())
            total += m.mean() * gap
            table.append({"bin": [_r(edges[b], 4), _r(edges[b + 1], 4)], "rows": int(m.sum()),
                          "mean_score": _r(p[m].mean()), "observed_fraud_rate": _r(y[m].mean())})
        return total, table

    ece_w, table_w = ece(np.linspace(0, 1, bins + 1))
    q = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    ece_m, _ = ece(q) if len(q) > 2 else (float("nan"), [])
    return {"brier_score": _r(brier), "brier_score_of_predicting_the_prevalence": _r(brier_ref),
            "log_loss": _r(logloss), "ece_equal_width_10_bins": _r(ece_w), "ece_equal_mass_10_bins": _r(ece_m),
            "mean_score": _r(p.mean()), "observed_fraud_rate": _r(prevalence),
            "reliability_equal_width": table_w}


def score_distribution(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y).astype(bool)
    p = np.asarray(p, dtype=np.float64)
    qs = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
    out = {}
    for name, m in (("legitimate", ~y), ("fraud", y)):
        v = p[m]
        if len(v) == 0:                      # e.g. no fraud rows in a population
            out[name] = {"rows": 0}
            continue
        bands = {"low_below_25": float((v < 0.25).mean()), "medium_25_to_50": float(((v >= 0.25) & (v < 0.5)).mean()),
                 "high_50_to_80": float(((v >= 0.5) & (v < 0.8)).mean()), "critical_80_and_above": float((v >= 0.8).mean())}
        out[name] = {"rows": int(m.sum()), "quantiles": {f"{q:g}": _r(np.quantile(v, q), 4) for q in qs},
                     "share_at_or_above_0.99": _r((v >= 0.99).mean(), 4), "share_at_or_below_0.01": _r((v <= 0.01).mean(), 4),
                     "distinct_values": int(len(np.unique(v))),
                     "application_alert_bands_25_50_80": {k: _r(x, 4) for k, x in bands.items()},
                     "histogram_20_bins": np.histogram(v, bins=20, range=(0, 1))[0].tolist()}
    return out


def posthoc_calibration(y_val, s_val, y_dev, s_dev) -> dict:
    """Informational: sigmoid (Platt) and isotonic maps fitted on the validation period
    only, evaluated on development data. Not applied to the deployed model."""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    eps = 1e-6

    def logit(s):
        s = np.clip(np.asarray(s, dtype=np.float64), eps, 1 - eps)
        return np.log(s / (1 - s)).reshape(-1, 1)

    platt = LogisticRegression(C=1e6, max_iter=1000).fit(logit(s_val), y_val)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(np.asarray(s_val, dtype=np.float64), y_val)
    return {"fitted_on": "validation period of the training dataset", "evaluated_on": "development datasets",
            "sigmoid": {k: v for k, v in calibration_metrics(y_dev, platt.predict_proba(logit(s_dev))[:, 1]).items()
                        if k != "reliability_equal_width"},
            "isotonic": {k: v for k, v in calibration_metrics(y_dev, iso.predict(np.asarray(s_dev, dtype=np.float64))).items()
                         if k != "reliability_equal_width"}}


# ---- cost ---------------------------------------------------------------------------------------------

def model_size_bytes(model) -> int:
    if isinstance(model, ClassifierModel):
        buf = io.BytesIO()
        joblib.dump(model.estimator, buf)
        return len(buf.getvalue())
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "m.keras"
        model.save(p)
        return p.stat().st_size


def inference_cost(model, Z: np.ndarray, n_single: int = 300, warmup: int = 20) -> dict:
    """Classifier only (the LSTM is the same for every family): one-row calls as /predict
    makes them, and one 4,096-row batch."""
    rows = Z[:n_single + warmup]
    is_keras = not isinstance(model, ClassifierModel)
    call = (lambda x: model.predict(x, verbose=0)) if is_keras else (lambda x: model.predict(x))
    for i in range(warmup):
        call(rows[i:i + 1])
    times = []
    for i in range(warmup, warmup + n_single):
        t0 = time.perf_counter()
        call(rows[i:i + 1])
        times.append(time.perf_counter() - t0)
    batch = Z[:4096]
    t0 = time.perf_counter()
    predict_fixed_batch(model, batch)
    tb = time.perf_counter() - t0
    return {"single_row_ms_median": _r(1000 * np.median(times), 3), "single_row_ms_p95": _r(1000 * np.percentile(times, 95), 3),
            "batch_4096_rows_ms": _r(1000 * tb, 1), "size_bytes": model_size_bytes(model),
            "single_row_calls": n_single, "cpu": platform.processor() or platform.machine()}


# ---- statistics across training seeds -------------------------------------------------------------

SUMMARY_METRICS = ("pr_auc", "roc_auc", "recall", "precision", "f1", "legit_alerts_per_1000", "false_positive_rate",
                   "critical_recall", "critical_precision", "critical_legit_alerts_per_1000",
                   "first_fraud_recall", "episode_detection_rate")


def confusion(res: dict, name: str) -> dict:
    c = res["models"][name]["counts"]
    tp, fp, fn = c["true_positives"], c["false_positives"], c["false_negatives"]
    tn = res["legitimate_transactions"] - fp
    return {"true_negative": tn, "false_positive": fp, "false_negative": fn, "true_positive": tp,
            "accuracy": _r((tp + tn) / (tp + tn + fp + fn))}


def family_summary(res: dict, boot: dict, families, seeds, metrics=SUMMARY_METRICS, bootstrap_seed: int = h.BOOTSTRAP_SEED) -> dict:
    """Per family: each metric across training seeds (mean, sd, min, max, 95% intervals).
    Challenger - incumbent: difference of the means over training seeds, with a 95%
    interval including data variation (the group bootstrap, same resamples for every
    model) and training-seed variation (seeds redrawn per resample), as multiseed.py."""
    seeds = list(seeds)
    n = len(seeds)
    reps = len(next(iter(next(iter(boot.values())).values())))
    rng = np.random.default_rng(bootstrap_seed + 1)
    draws = {f: rng.integers(0, n, size=(reps, n)) for f in families}
    cols = np.arange(reps)
    out = {"families": {f: {} for f in families}, "versus_incumbent": {f: {} for f in families if f != INCUMBENT}}
    for m in metrics:
        point = {f: [res["models"][key(f, s)]["metrics"][m]["value"] for s in seeds] for f in families}
        stacked = {f: np.vstack([boot[key(f, s)][m] for s in seeds]) for f in families}
        for f in families:
            stats = ev.across_seeds(point[f])
            stats["per_training_seed"] = {str(s): v for s, v in zip(seeds, point[f])}
            stats["mean_ci95_data"] = h._ci(stacked[f].mean(axis=0))
            out["families"][f][m] = stats
        for f in families:
            if f == INCUMBENT or any(v is None for v in point[f] + point[INCUMBENT]):
                continue
            d = float(np.mean(point[f]) - np.mean(point[INCUMBENT]))
            combined = (stacked[f][draws[f].T, cols].mean(axis=0)
                        - stacked[INCUMBENT][draws[INCUMBENT].T, cols].mean(axis=0))
            ci = h._ci(combined)
            out["versus_incumbent"][f][m] = {
                "difference_of_means": _r(d), "ci95_data_and_training_seeds": ci,
                "ci95_data_only": h._ci(stacked[f].mean(axis=0) - stacked[INCUMBENT].mean(axis=0)),
                "ci95_training_seeds_only": ev._welch(point[f], point[INCUMBENT]),
                "same_seed_differences": {str(s): _r(a - b) for s, a, b in zip(seeds, point[f], point[INCUMBENT])},
                "seeds_higher": int(sum(a > b for a, b in zip(point[f], point[INCUMBENT]))),
            }
    return out


def per_dataset_pr_auc(table: pd.DataFrame, names) -> dict:
    out = {}
    for ds, rows in table.groupby("seed", sort=True):
        r = h.evaluate_population(rows, names, 0)
        out[str(ds)] = {m: r["models"][m]["metrics"]["pr_auc"]["value"] for m in names}
    return out


def eligibility(summary: dict, dataset_pr: dict, seeds, reproducible: dict) -> dict:
    """Protocol section 6, E1-E7, for every challenger."""
    out = {}
    for f in CHALLENGERS:
        v = summary["versus_incumbent"][f]
        mean_recall = summary["families"][f]["recall"]["mean"]
        ds_higher = sum(
            np.mean([dataset_pr[ds][key(f, s)] for s in seeds]) > np.mean([dataset_pr[ds][key(INCUMBENT, s)] for s in seeds])
            for ds in dataset_pr)
        c = {
            "E1_pr_auc_clearly_higher": {"ci95": v["pr_auc"]["ci95_data_and_training_seeds"],
                                         "holds": bool(v["pr_auc"]["ci95_data_and_training_seeds"][0] > 0)},
            "E2_recall_not_materially_worse": {"ci95": v["recall"]["ci95_data_and_training_seeds"],
                                               "holds": bool(v["recall"]["ci95_data_and_training_seeds"][0] > -RECALL_MARGIN)},
            "E3_alert_burden_not_materially_higher": {
                "ci95": v["legit_alerts_per_1000"]["ci95_data_and_training_seeds"],
                "holds": bool(v["legit_alerts_per_1000"]["ci95_data_and_training_seeds"][1] < LEGIT_ALERT_MARGIN_PER_1000)},
            "E4_critical_recall_not_materially_worse": {
                "ci95": v["critical_recall"]["ci95_data_and_training_seeds"],
                "holds": bool(v["critical_recall"]["ci95_data_and_training_seeds"][0] > -CRITICAL_RECALL_MARGIN)},
            "E5_mean_recall_at_least_0.40": {"value": mean_recall, "holds": bool(mean_recall is not None and mean_recall >= MIN_RECALL)},
            "E6_stable": {"training_seeds_with_higher_pr_auc": v["pr_auc"]["seeds_higher"],
                          "development_datasets_with_higher_mean_pr_auc": int(ds_higher),
                          "holds": bool(v["pr_auc"]["seeds_higher"] >= MIN_SEEDS_HIGHER and ds_higher >= MIN_DATASETS_HIGHER)},
            "E7_reproducible": {"holds": bool(reproducible.get(f, False))},
        }
        out[f] = {"conditions": c, "eligible": all(x["holds"] for x in c.values())}
    return out


def choose_winner(elig: dict, summary: dict, calibration_by_family: dict, cost_by_family: dict) -> dict:
    eligible = [f for f in CHALLENGERS if elig[f]["eligible"]]
    if not eligible:
        return {"winner": None, "eligible": [], "outcome": "no challenger is eligible: the DNN stays (nothing is replaced)"}
    pr = {f: summary["families"][f]["pr_auc"]["mean"] for f in eligible}
    ranked = sorted(eligible, key=lambda f: -pr[f])
    winner, note = ranked[0], "highest mean PR-AUC among eligible families"
    if len(ranked) > 1 and pr[ranked[0]] - pr[ranked[1]] < TIE_MARGIN_PR_AUC:
        a, b = ranked[0], ranked[1]
        ba, bb = calibration_by_family[a]["brier_score_mean"], calibration_by_family[b]["brier_score_mean"]
        if ba != bb:
            winner = a if ba < bb else b
            note = f"within {TIE_MARGIN_PR_AUC} PR-AUC of {b if winner == a else a}; lower Brier score"
        else:
            winner = min((a, b), key=lambda f: cost_by_family[f]["single_row_ms_median"])
            note = f"within {TIE_MARGIN_PR_AUC} PR-AUC; same Brier score; cheaper to run"
    return {"winner": winner, "eligible": eligible, "mean_pr_auc": pr, "note": note,
            "outcome": f"{winner} replaces the DNN, subject to confirmation on the fresh hold-out"}


# ---- stage: develop (development datasets 301-305) + selection record ----------------------------

def develop(seeds=TRAINING_SEEDS, dev_seeds=DEVELOPMENT_SEEDS, reps: int = REPS, log=print) -> dict:
    from . import selection as sel
    t_start = time.time()
    params = chosen_params()
    data, labels = training_data()
    mask = ev.validation_mask(data, labels)
    yv = data.frame["is_fraud"].to_numpy()[mask]
    dev = {ds: h.load_holdout(assert_allowed_seed(ds, DEVELOPMENT_SEEDS), sel.DEVELOPMENT_DATA_DIR) for ds in dev_seeds}
    scores = {ds: {} for ds in dev_seeds}
    val_scores, cutoffs, models = {}, {"policy_b": {}, "critical": {}}, {}
    Z14 = None
    for s in seeds:
        bundle = lstm_bundle(s)
        Zv = cached_inputs("train", 42, s, data.frame, bundle)[mask]
        XD, y = load_training_matrix(s)
        Xs = scaled_training(XD, bundle)
        fitted = {INCUMBENT: bundle.dnn_model}
        for f in CHALLENGERS:
            fitted[f] = fit_family(f, params[f], s, Xs, y)
        for f, model in fitted.items():
            k = key(f, s)
            sv = model_scores(model, Zv)
            val_scores[k] = sv
            for tier, cut in policy_cutoffs_from_scores(yv, sv).items():
                cutoffs[tier][k] = cut
        for ds in dev_seeds:
            Z = cached_inputs("dev", ds, s, dev[ds].frame, bundle)
            for f, model in fitted.items():
                scores[ds][key(f, s)] = model_scores(model, Z)
            if s == DEPLOY_SEED and ds == dev_seeds[0]:
                Z14 = Z
        if s == DEPLOY_SEED:
            models = fitted
        log(f"develop: seed {s} scored ({time.time() - t_start:.0f}s)")
    names = [key(f, s) for f in FAMILIES for s in seeds]
    table = pd.concat([h.build_table(dev[ds], scores[ds], cutoffs) for ds in dev_seeds], ignore_index=True)
    primary = h.population(table, "primary")
    log(f"develop: bootstrap over {len(primary)} rows, {len(names)} models, {reps} resamples")
    res, boot = h.evaluate_population(primary, names, reps, return_resamples=True, pairs=[])
    summary = family_summary(res, boot, FAMILIES, seeds)
    dataset_pr = per_dataset_pr_auc(primary, names)

    y_dev = primary["is_fraud"].to_numpy()
    calibration = {k: calibration_metrics(y_dev, primary[k].to_numpy()) for k in names}
    calib_family = {f: {"brier_score_mean": _r(np.mean([calibration[key(f, s)]["brier_score"] for s in seeds])),
                        "log_loss_mean": _r(np.mean([calibration[key(f, s)]["log_loss"] for s in seeds])),
                        "ece_equal_width_mean": _r(np.mean([calibration[key(f, s)]["ece_equal_width_10_bins"] for s in seeds])),
                        "ece_equal_mass_mean": _r(np.mean([calibration[key(f, s)]["ece_equal_mass_10_bins"] for s in seeds]))}
                    for f in FAMILIES}
    posthoc = {f: posthoc_calibration(yv, val_scores[key(f, DEPLOY_SEED)], y_dev, primary[key(f, DEPLOY_SEED)].to_numpy())
               for f in FAMILIES}
    distribution = {k: score_distribution(y_dev, primary[k].to_numpy()) for k in names}
    cost = {f: inference_cost(models[f], Z14) for f in FAMILIES}

    # E7: refit with the same seed and compare the scores bit for bit
    s0 = seeds[0]
    bundle = lstm_bundle(s0)
    XD, y = load_training_matrix(s0)
    Xs = scaled_training(XD, bundle)
    Zr = cached_inputs("dev", dev_seeds[0], s0, dev[dev_seeds[0]].frame, bundle)
    reproducible = {f: bool(np.array_equal(model_scores(fit_family(f, params[f], s0, Xs, y), Zr), scores[dev_seeds[0]][key(f, s0)]))
                    for f in CHALLENGERS}

    per_model = {k: {"metrics": {m: res["models"][k]["metrics"][m] for m in SUMMARY_METRICS},
                     "confusion_matrix_policy_b": confusion(res, k), "counts": res["models"][k]["counts"],
                     "cutoffs": {"policy_b": cutoffs["policy_b"][k], "critical": cutoffs["critical"][k]},
                     "calibration": calibration[k]}
                 for k in names}
    accuracy = {f: ev.across_seeds([per_model[key(f, s)]["confusion_matrix_policy_b"]["accuracy"] for s in seeds]) for f in FAMILIES}
    elig = eligibility(summary, dataset_pr, seeds, reproducible)
    decision = choose_winner(elig, summary, calib_family, cost)
    report = {
        "protocol": PROTOCOL_VERSION, "protocol_document_sha256": _sha256(PROTOCOL_DOC),
        "stage": "develop: all families x training seeds on the development datasets",
        "population": {k: res[k] for k in ("rows", "fraud_transactions", "legitimate_transactions", "fraud_episodes",
                                           "first_fraud_transactions", "customers", "groups", "datasets")},
        "development_datasets": {str(ds): {"sha256": dev[ds].sha256, "transactions": int(len(dev[ds].frame)),
                                           "generator_version": dev[ds].manifest.get("generator_version")} for ds in dev_seeds},
        "training_seeds": list(seeds), "chosen_params": params, "bootstrap_resamples": reps,
        "summary": summary, "accuracy_at_policy_b": accuracy, "per_dataset_pr_auc": dataset_pr,
        "per_model": per_model, "calibration_by_family": calib_family, "posthoc_calibration_seed_14": posthoc,
        "score_distribution": distribution, "inference_cost_seed_14": cost, "reproducible_refit": reproducible,
        "eligibility": elig, "decision": decision, "runtime_seconds": round(time.time() - t_start, 1),
        "versions": _versions(),
    }
    _write_json(report, OUTPUT_DIR / "development_report.json")
    record = {
        "protocol": PROTOCOL_VERSION, "protocol_document_sha256": report["protocol_document_sha256"],
        "written_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "decision": decision, "eligibility": {f: elig[f]["eligible"] for f in CHALLENGERS},
        "deploy": None if decision["winner"] is None else {
            "family": decision["winner"], "params": params[decision["winner"]], "training_seed": DEPLOY_SEED,
            "lstm": f"models/candidates_multiseed/v2/seed_{DEPLOY_SEED}/dnn_lstm (unchanged)",
            "cutoffs": {"policy_b": cutoffs["policy_b"][key(decision["winner"], DEPLOY_SEED)],
                        "critical": cutoffs["critical"][key(decision["winner"], DEPLOY_SEED)]},
            "incumbent_cutoffs": {"policy_b": cutoffs["policy_b"][key(INCUMBENT, DEPLOY_SEED)],
                                  "critical": cutoffs["critical"][key(INCUMBENT, DEPLOY_SEED)]}},
        "final_holdout_seeds": list(FINAL_SEEDS), "final_new_customer_seeds": list(FINAL_NEW_CUSTOMER_SEEDS),
        "final_holdout_existed_when_written": FINAL_DATA_DIR.exists(),
    }
    if FINAL_DATA_DIR.exists():
        raise DownstreamError("the fresh hold-out already exists; the selection record must be written before it is generated")
    _write_json(record, OUTPUT_DIR / RECORD_NAME)
    return report


def _versions() -> dict:
    import sklearn
    out = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
           "scikit_learn": sklearn.__version__}
    try:
        import tensorflow as tf
        out["tensorflow"] = tf.__version__
    except Exception:       # pragma: no cover
        pass
    return out


# ---- the deployable artifact ------------------------------------------------------------------------

ARTIFACT_ROOT = config.BACKEND_DIR / "models" / "candidates_downstream" / "v2"
CLASSIFIER_FILE = "classifier.joblib"
ARTIFACT_FILES = {
    "lstm_model": "lstm_risk_model.keras", "lstm_mean": "lstm_feature_mean.npy", "lstm_std": "lstm_feature_std.npy",
    "classifier": CLASSIFIER_FILE, "input_mean": "classifier_input_mean.npy", "input_std": "classifier_input_std.npy",
    "shap_background": "shap_background.npy",
}
SHAP_BACKGROUND_ROWS = 200


def artifact_dir(family: str, seed: int = DEPLOY_SEED, root=None) -> Path:
    return Path(root or ARTIFACT_ROOT) / f"seed_{int(seed)}" / f"lstm_{family}"


def save_artifact(family: str, params: dict, seed: int = DEPLOY_SEED, root=None, cutoffs=None, log=print) -> dict:
    """Fits the winning family for `seed` on the training rows (out-of-fold risk scores)
    and writes it next to a byte-identical copy of the unchanged seed-`seed` LSTM."""
    from ..training.candidates import weights_sha256
    src = saved_artifact_dir(seed)
    src_manifest = json.loads((src / "manifest.json").read_text())
    bundle = lstm_bundle(seed)
    XD, y = load_training_matrix(seed)
    Xs = scaled_training(XD, bundle)
    model = fit_family(family, params, seed, Xs, y)
    d = artifact_dir(family, seed, root)
    if config.MODELS_SAVED_DIR.resolve() in d.resolve().parents:
        raise DownstreamError("refusing to write inside models/saved/")
    d.mkdir(parents=True, exist_ok=True)
    files = {}
    for role in ("lstm_model", "lstm_mean", "lstm_std"):
        data = (src / ARTIFACT_FILES[role]).read_bytes()
        (d / ARTIFACT_FILES[role]).write_bytes(data)
        files[ARTIFACT_FILES[role]] = _sha256_bytes(data)
        if files[ARTIFACT_FILES[role]] != src_manifest["files"][ARTIFACT_FILES[role]]:
            raise DownstreamError(f"{role}: the copied LSTM file differs from the saved seed-{seed} artifact")
    for role, arr in (("input_mean", bundle.dnn_mean), ("input_std", bundle.dnn_std)):
        np.save(d / ARTIFACT_FILES[role], np.asarray(arr))
        files[ARTIFACT_FILES[role]] = _sha256(d / ARTIFACT_FILES[role])
    pick = np.sort(np.random.default_rng(seed).choice(len(Xs), size=min(SHAP_BACKGROUND_ROWS, len(Xs)), replace=False))
    np.save(d / ARTIFACT_FILES["shap_background"], Xs[pick])
    files[ARTIFACT_FILES["shap_background"]] = _sha256(d / ARTIFACT_FILES["shap_background"])
    joblib.dump(model.estimator, d / CLASSIFIER_FILE)
    files[CLASSIFIER_FILE] = _sha256(d / CLASSIFIER_FILE)
    manifest = {
        "model_set_kind": "lstm_classifier",
        "family": family, "classifier_class": type(model.estimator).__name__, "params": params,
        "seed": int(seed), "protocol": PROTOCOL_VERSION,
        "dataset": src_manifest["dataset"], "split": src_manifest["split"], "features": src_manifest["features"],
        "sequence_length": src_manifest["sequence_length"],
        "input_columns": INPUT_COLUMNS,
        "lstm": {"source": f"models/candidates_multiseed/v2/seed_{seed}/dnn_lstm (copied byte for byte, unchanged)",
                 "lstm_weights_sha256": weights_sha256(bundle.lstm_model),
                 "lstm_input_shape": [None, src_manifest["sequence_length"], len(FEATURE_COLUMNS)],
                 "risk_score": src_manifest["model"]["risk_score"]},
        "training": {"rows": int(len(y)), "fraud": int(y.sum()),
                     "inputs": "9 behavioral features + out-of-fold risk_score, scaled with the training mean/std "
                               "(identical to the seed DNN's scaler) and clipped to +-6, as when scoring"},
        "scoring": "scaled with classifier_input_mean/std, clipped to +-6, cold-start imputation as for the DNN; "
                   "output = predict_proba[:, 1], capped at 0.999; a model score (see calibration in the report)",
        "cutoffs_validation": cutoffs,
        "shap_background": {"file": ARTIFACT_FILES["shap_background"], "rows": int(len(pick)), "seed": int(seed)},
        "files": files, "versions": _versions(),
    }
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default) + "\n")
    log(f"artifact written: {d}")
    return manifest


# ---- stage: confirm (fresh hold-out, once) -----------------------------------------------------------

def load_record() -> dict:
    path = OUTPUT_DIR / RECORD_NAME
    if not path.exists():
        raise DownstreamError("no selection record: run the develop stage first")
    return json.loads(path.read_text())


def generate_final(log=print) -> None:
    load_record()                                    # refuses without a selection record
    for s in FINAL_SEEDS:
        assert_allowed_seed(s, FINAL_SEEDS)
        h.generate_dataset(s, root=FINAL_DATA_DIR)
        log(f"generated final hold-out seed {s}")
    for s in FINAL_NEW_CUSTOMER_SEEDS:
        assert_allowed_seed(s, FINAL_NEW_CUSTOMER_SEEDS)
        h.generate_dataset(s, root=FINAL_DATA_DIR / "new_customer", late_joiner_share=h.NEW_CUSTOMER_LATE_JOINER_SHARE,
                           new_customer_fraud_episodes=h.NEW_CUSTOMER_EPISODES_PER_SEED)
        log(f"generated final new-customer hold-out seed {s}")


def confirm(reps: int = REPS, log=print) -> dict:
    from ..model_sets import load_model_set
    record = load_record()
    deploy = record["deploy"]
    if deploy is None:
        raise DownstreamError("the selection record selected no challenger; nothing to confirm")
    t0 = time.time()
    family, params, seed = deploy["family"], deploy["params"], int(deploy["training_seed"])
    manifest = save_artifact(family, params, seed, cutoffs=deploy["cutoffs"])
    est = joblib.load(artifact_dir(family, seed) / CLASSIFIER_FILE)
    selected = ClassifierModel(est, family, len(INPUT_COLUMNS))
    bundle = lstm_bundle(seed)
    production = load_model_set("production")
    names = ["selected", "dnn_seed14", "production"]
    cutoffs = {"policy_b": {"selected": deploy["cutoffs"]["policy_b"], "dnn_seed14": deploy["incumbent_cutoffs"]["policy_b"],
                            "production": h.CUTOFFS["policy_b"]["production"]},
               "critical": {"selected": deploy["cutoffs"]["critical"], "dnn_seed14": deploy["incumbent_cutoffs"]["critical"],
                            "production": h.CUTOFFS["critical"]["production"]}}
    generate_final(log)

    def score(data):
        Z = scoring_inputs(data.frame, bundle)
        return {"selected": model_scores(selected, Z), "dnn_seed14": model_scores(bundle.dnn_model, Z),
                "production": h.score_frame(data.frame, production)}

    pairs = [("selected", "dnn_seed14"), ("selected", "production"), ("dnn_seed14", "production")]
    out = {"protocol": PROTOCOL_VERSION, "stage": "confirm: fresh hold-out, scored once", "record": record,
           "artifact_manifest": manifest, "cutoffs": cutoffs}
    for label, seeds_, root in (("final", FINAL_SEEDS, FINAL_DATA_DIR),
                                ("new_customer", FINAL_NEW_CUSTOMER_SEEDS, FINAL_DATA_DIR / "new_customer")):
        datasets = {s: h.load_holdout(s, root) for s in seeds_}
        table = pd.concat([h.build_table(datasets[s], score(datasets[s]), cutoffs) for s in seeds_], ignore_index=True)
        block = {"datasets": {str(s): h.dataset_summary(datasets[s]) for s in seeds_}}
        for pop in ("primary", "early_history"):
            rows = h.population(table, pop)
            res = h.evaluate_population(rows, names, reps, pairs=pairs)
            for n in names:
                res["models"][n]["confusion_matrix_policy_b"] = confusion(res, n)
            yv = rows["is_fraud"].to_numpy()
            res["calibration"] = {n: calibration_metrics(yv, rows[n].to_numpy()) for n in names}
            res["score_distribution"] = {n: score_distribution(yv, rows[n].to_numpy()) for n in names}
            res["fraud_types"] = h.fraud_type_breakdown(rows, names)
            block[pop] = res
        block["per_dataset_pr_auc"] = per_dataset_pr_auc(h.population(table, "primary"), names)
        out[label] = block
        log(f"confirm: {label} scored ({time.time() - t0:.0f}s)")
    p = out["final"]["primary"]
    diff = p["paired_differences"]["selected - dnn_seed14"]
    gates = {
        "C1_pr_auc_higher_than_seed14_dnn": {"difference": diff["pr_auc"]["difference"], "ci95": diff["pr_auc"]["ci95"],
                                              "holds": bool(diff["pr_auc"]["ci95"][0] > 0)},
        "C2_recall_at_least_0.40": {"value": p["models"]["selected"]["metrics"]["recall"]["value"],
                                    "holds": bool(p["models"]["selected"]["metrics"]["recall"]["value"] >= MIN_RECALL)},
        "C3_alert_burden_not_materially_higher": {"difference": diff["legit_alerts_per_1000"]["difference"],
                                                  "ci95": diff["legit_alerts_per_1000"]["ci95"],
                                                  "holds": bool(diff["legit_alerts_per_1000"]["ci95"][1] < LEGIT_ALERT_MARGIN_PER_1000)},
    }
    out["gates"] = gates
    out["confirmed"] = all(g["holds"] for g in gates.values())
    out["outcome"] = (f"{family} confirmed: it replaces the DNN behind the unchanged LSTM" if out["confirmed"]
                      else f"{family} NOT confirmed on the fresh hold-out: the DNN is not replaced")
    out["runtime_seconds"] = round(time.time() - t0, 1)
    out["versions"] = _versions()
    _write_json(out, OUTPUT_DIR / "final_holdout_report.json")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Step 4D: downstream classifier selection")
    ap.add_argument("stage", choices=["data", "reproduce", "tune", "develop", "confirm"])
    ap.add_argument("--reps", type=int, default=REPS)
    args = ap.parse_args(argv)
    if args.stage == "data":
        from . import selection as sel
        for s in DEVELOPMENT_SEEDS:
            print(h.generate_dataset(assert_allowed_seed(s, DEVELOPMENT_SEEDS), sel.DEVELOPMENT_DATA_DIR))
    elif args.stage == "reproduce":
        data, labels = training_data()
        out = {str(s): reproduce_seed(s, data, labels) for s in TRAINING_SEEDS}
        _write_json(out, OUTPUT_DIR / "reproduction.json")
    elif args.stage == "tune":
        tune()
    elif args.stage == "develop":
        r = develop(reps=args.reps)
        print(json.dumps(r["decision"], indent=2, default=_json_default))
    else:
        r = confirm(reps=args.reps)
        print(r["outcome"])


if __name__ == "__main__":
    main()
