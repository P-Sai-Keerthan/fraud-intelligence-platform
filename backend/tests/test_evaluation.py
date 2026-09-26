"""Corrected evaluation methodology (app/evaluation/).

Fast tests: the splits and windows run on the real dataset; model-fitting
logic runs with small stand-in models (logistic regression) instead of the
LSTM/DNN so it takes seconds. The saved report is checked against a fresh
recomputation of the split.
"""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from app.config import EVAL_REPORT_PATH, EVAL_TIME_SPLIT_PATH, EVAL_CUSTOMER_SPLIT_PATH, FEATURES_CSV
from app.evaluation import baselines
from app.evaluation.metrics import (
    classification_metrics, episode_metrics, evaluate_scores, select_threshold, threshold_for_fpr,
)
from app.evaluation.run import evaluate_split, first_fraud_flags
from app.evaluation.split import (
    SPLITS, build_customer_split, build_time_split, canonical_order, find_episodes,
)
from app.evaluation.stacking import chronological_holdout, stacked_risk_scores
from app.evaluation.windows import assert_past_only, build_windows, window_row_indices
from app.features.feature_engineering import FEATURE_COLUMNS, build_sequences


# ---- shared fixtures -------------------------------------------------------------

@pytest.fixture(scope="module")
def real():
    df = canonical_order(pd.read_csv(FEATURES_CSV))
    labels, episode_id, definition = build_time_split(df)
    return df, labels, episode_id, definition


@pytest.fixture(scope="module")
def real_windows(real):
    return build_windows(real[0])


class _Stub:
    def __init__(self, model, flatten):
        self.model, self.flatten, self.info = model, flatten, {"stub": True}

    def predict(self, X):
        return self.model.predict_proba(X.reshape(len(X), -1) if self.flatten else X)[:, 1]


class RecordingFitter:
    """Stand-in for fit_lstm / fit_dnn that remembers what each call trained on."""

    def __init__(self, flatten):
        self.flatten, self.calls = flatten, []

    def __call__(self, X, y, timestamps, seed=0):
        self.calls.append({"X": X, "y": y, "timestamps": timestamps, "seed": seed})
        Xf = X.reshape(len(X), -1) if self.flatten else X
        return _Stub(LogisticRegression(max_iter=2000).fit(Xf, y), self.flatten)


BURST_STARTS = [11, 14, 22, 30, 16, 33, 12, 24, 31]   # transaction index of each fraud burst


def _toy_frame(n_customers=12, n_tx=40, seed=0):
    """Small synthetic history, one transaction every 3 days from 2026-01-01.
    Every 3rd customer has one 4-transaction fraud burst; the bursts are
    spread so each period (before Mar 1 / Mar 1-25 / after) has fraud."""
    rng = np.random.RandomState(seed)
    rows = []
    start = pd.Timestamp("2026-01-01")
    for c in range(n_customers):
        b = BURST_STARTS[(c // 3) % len(BURST_STARTS)]
        burst = range(b, b + 4) if c % 3 == 0 else range(0)
        for i in range(n_tx):
            fraud = int(i in burst)
            feats = rng.normal(0, 1, len(FEATURE_COLUMNS)) + (4 if fraud else 0)
            rows.append({
                "customer_id": f"C{c:02d}", "transaction_id": f"T{c:02d}{i:03d}",
                "timestamp": start + pd.Timedelta(days=i * 3, hours=c), "is_fraud": fraud,
                **dict(zip(FEATURE_COLUMNS, feats)),
            })
    return canonical_order(pd.DataFrame(rows))


# ---- 1. chronological split / no future leakage ------------------------------------

def test_time_split_is_chronological(real):
    df, labels, _, d = real
    b1, b2 = pd.Timestamp(d["train_before"]), pd.Timestamp(d["validation_before"])
    assert b1 < b2
    assert df.loc[labels == "train", "timestamp"].max() < b1          # no future in training
    assert df.loc[labels == "validation", "timestamp"].max() < b2
    # test holds the latest transactions: everything at/after b2 is test
    assert (labels[df["timestamp"] >= b2] == "test").all()
    assert set(labels.unique()) == set(SPLITS)


def test_only_straddling_episodes_move_and_only_later(real):
    df, labels, episode_id, d = real
    b1, b2 = pd.Timestamp(d["train_before"]), pd.Timestamp(d["validation_before"])
    by_time = np.where(df["timestamp"] < b1, 0, np.where(df["timestamp"] < b2, 1, 2))
    rank = labels.map({s: i for i, s in enumerate(SPLITS)}).to_numpy()
    assert (rank >= by_time).all()                                       # rows only move later
    moved = rank != by_time
    moved_customers = set(df.loc[moved, "customer_id"])
    assert moved_customers == {m["customer_id"] for m in d["episodes_moved"]}
    assert int(moved.sum()) == sum(m["rows_moved"] for m in d["episodes_moved"])


def test_no_fraud_episode_crosses_a_split(real):
    _, labels, episode_id, d = real
    per_episode = labels[episode_id > 0].groupby(episode_id[episode_id > 0]).nunique()
    assert (per_episode == 1).all()
    assert d["episodes_total"] == 60
    assert sum(s["fraud_episodes"] for s in d["splits"].values()) == 60


def test_every_split_has_fraud_episodes(real):
    for name, s in real[3]["splits"].items():
        assert s["fraud_episodes"] >= 5, name


def test_saved_time_split_matches_recomputation(real):
    saved = json.loads(EVAL_TIME_SPLIT_PATH.read_text())
    assert saved == json.loads(json.dumps(real[3]))


def test_customer_split_keeps_customers_together(real):
    df, _, episode_id, _ = real
    labels, definition = build_customer_split(df, episode_id)
    assert (labels.groupby(df["customer_id"]).nunique() == 1).all()
    assert json.loads(EVAL_CUSTOMER_SPLIT_PATH.read_text()) == json.loads(json.dumps(definition))
    assert all(s["fraud_episodes"] > 0 for s in definition["splits"].values())


def test_episode_rule_links_interleaved_fraud():
    df = _toy_frame()
    # put a legit transaction inside customer C00's burst
    idx = df.index[(df.customer_id == "C00") & (df.is_fraud == 1)][1]
    df.loc[idx, "is_fraud"] = 0
    episode_id, table = find_episodes(df)
    assert len(table) == 4                     # C00, C03, C06, C09 -- still one episode each
    assert set(table["customer_id"]) == {"C00", "C03", "C06", "C09"}


# ---- 2. LSTM windows use only past transactions ------------------------------------

def test_windows_are_past_only_on_real_data(real, real_windows):
    df = real[0]
    X, y, target = real_windows
    assert_past_only(df, target)                                    # raises if not
    rows = window_row_indices(target)
    np.testing.assert_array_equal(X, df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)[rows])
    np.testing.assert_array_equal(y, df["is_fraud"].to_numpy()[target])


def test_window_never_contains_its_target(real_windows):
    _, _, target = real_windows
    assert not (window_row_indices(target) == target[:, None]).any()


def test_windows_match_production_sequence_builder(real, real_windows):
    X_prod, y_prod, _ = build_sequences(real[0])
    np.testing.assert_array_equal(real_windows[0], X_prod)
    np.testing.assert_array_equal(real_windows[1], y_prod)


def test_past_only_check_catches_future_rows():
    df = _toy_frame()
    X, y, target = build_windows(df)
    bad = df.copy()
    # swap one customer's 5th and 20th transaction timestamps -> a window now holds a later transaction
    rows = bad.index[bad.customer_id == "C01"]
    a, b = rows[5], rows[20]
    bad.loc[a, "timestamp"], bad.loc[b, "timestamp"] = bad.loc[b, "timestamp"], bad.loc[a, "timestamp"]
    with pytest.raises(AssertionError):
        assert_past_only(bad, target)


# ---- 3. out-of-fold LSTM scores -----------------------------------------------------

def test_out_of_fold_scores_never_come_from_a_model_that_saw_the_row():
    df = _toy_frame(n_customers=15)
    X, y, target = build_windows(df)
    groups = df["customer_id"].to_numpy()[target]
    ts = df["timestamp"].to_numpy()[target]
    train_mask = ts < pd.Timestamp("2026-03-20").to_datetime64()
    fitter = RecordingFitter(flatten=True)
    probs, final, prov = stacked_risk_scores(X, y, ts, groups, train_mask, fit_fn=fitter, n_folds=3)

    assert not np.isnan(probs).any()
    folds = prov["fold_of_row"]
    assert (folds[train_mask] >= 0).all() and (folds[~train_mask] == -1).all()
    for k, seen_customers in enumerate(prov["fold_train_groups"]):
        held = folds == k
        assert held.any()
        # the fold model never trained on the customers (hence rows/windows) it scores
        assert not set(groups[held]) & set(seen_customers)
    # 3 fold models + 1 final model; the final model trained on exactly the training rows
    assert len(fitter.calls) == 4
    np.testing.assert_array_equal(fitter.calls[-1]["y"], y[train_mask])
    np.testing.assert_array_equal(fitter.calls[-1]["timestamps"], ts[train_mask])
    # validation/test rows are scored by the final model
    np.testing.assert_allclose(probs[~train_mask], final.predict(X[~train_mask]))


def test_training_rows_do_not_get_in_sample_scores():
    df = _toy_frame(n_customers=15)
    X, y, target = build_windows(df)
    groups = df["customer_id"].to_numpy()[target]
    ts = df["timestamp"].to_numpy()[target]
    train_mask = np.ones(len(y), dtype=bool)
    fitter = RecordingFitter(flatten=True)
    probs, final, _ = stacked_risk_scores(X, y, ts, groups, train_mask, fit_fn=fitter, n_folds=3)
    in_sample = final.predict(X)
    assert not np.allclose(probs, in_sample)


def test_early_stopping_holdout_is_the_latest_rows():
    ts = np.array(pd.date_range("2026-01-01", periods=100, freq="D"))[::-1]
    fit_idx, es_idx = chronological_holdout(ts, 0.15)
    assert len(es_idx) == 15
    assert ts[es_idx].min() > ts[fit_idx].max()


# ---- 4. validation-only threshold selection -----------------------------------------

def test_threshold_is_midpoint_of_separable_validation():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    assert select_threshold(y, p) == pytest.approx(0.55)


def test_threshold_ignores_test_data():
    rng = np.random.RandomState(0)
    y_val = np.r_[np.zeros(200), np.ones(10)]
    p_val = np.r_[rng.uniform(0, 0.6, 200), rng.uniform(0.4, 1, 10)]
    y_te = np.r_[np.zeros(300), np.ones(15)]
    p_te = rng.uniform(0, 1, 315)
    a = evaluate_scores(y_val, p_val, y_te, p_te)
    b = evaluate_scores(y_val, p_val, 1 - y_te, rng.uniform(0, 1, 315))
    assert a["threshold"] == b["threshold"] == pytest.approx(select_threshold(y_val, p_val), abs=1e-6)
    assert a["threshold_source"].startswith("validation")
    assert a["recall_at_fpr"] and all(
        a["recall_at_fpr"][k]["threshold"] == b["recall_at_fpr"][k]["threshold"] for k in a["recall_at_fpr"]
    )


def test_fpr_operating_point_respects_target_on_validation():
    rng = np.random.RandomState(1)
    y = np.r_[np.zeros(5000), np.ones(50)]
    p = np.r_[rng.uniform(0, 1, 5000), rng.uniform(0.5, 1, 50)]
    for target in (0.001, 0.01):
        t = threshold_for_fpr(y, p, target)
        assert classification_metrics(y, p, t)["false_positive_rate"] <= target


def test_classification_metrics_fields():
    m = classification_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.9, 0.8, 0.2]), 0.5)
    assert m["confusion_matrix"] == {"true_negative": 1, "false_positive": 1, "false_negative": 1, "true_positive": 1}
    assert m["alerts_per_1000"] == 500.0
    assert m["precision"] == m["recall"] == 0.5


# ---- 5. baselines + full pipeline wiring (stand-in models) ---------------------------

def test_amount_hour_rule_scores():
    F = np.zeros((3, len(FEATURE_COLUMNS)))
    F[:, baselines.AMOUNT] = [400, 400, 90]
    F[:, baselines.HOUR] = [1, 0, 1]
    np.testing.assert_array_equal(baselines.amount_hour_rule_scores(F), [400, 0, 90])


def test_evaluate_split_trains_only_on_training_rows_and_reports_baselines():
    df = _toy_frame(n_customers=18)
    episode_id, _ = find_episodes(df)
    ts = df["timestamp"]
    labels = pd.Series(np.where(ts < "2026-03-01", "train", np.where(ts < "2026-03-25", "validation", "test")), index=df.index)
    # every toy burst sits inside one period and each period has fraud
    assert (labels[episode_id > 0].groupby(episode_id[episode_id > 0]).nunique() == 1).all()
    assert set(labels[df.is_fraud == 1]) == {"train", "validation", "test"}
    windows = build_windows(df)
    lstm_fit, dnn_fit = RecordingFitter(flatten=True), RecordingFitter(flatten=False)
    result, _ = evaluate_split(df, labels, episode_id, windows, fit_lstm_fn=lstm_fit, fit_dnn_fn=dnn_fit,
                               n_folds=3, log=lambda m: None)
    train_rows = result["evaluated_rows"]["train"]["windows"]
    # final LSTM, stacked DNN and DNN-without-risk all trained on the training rows only
    assert len(lstm_fit.calls[-1]["y"]) == train_rows
    assert [c["X"].shape[1] for c in dnn_fit.calls] == [len(FEATURE_COLUMNS) + 1, len(FEATURE_COLUMNS)]
    assert all(len(c["y"]) == train_rows for c in dnn_fit.calls)
    assert all(pd.Timestamp(c["timestamps"].max()) < pd.Timestamp("2026-03-01") for c in dnn_fit.calls)
    assert set(result["baselines"]) == {"amount_hour_rule", "logistic_regression", "dnn_without_risk_score"}
    n_test = result["evaluated_rows"]["test"]["windows"]
    for m in [result["lstm_risk_predictor"], result["dnn_fraud_classifier"], *result["baselines"].values()]:
        assert m["test_set_size"] == n_test
        assert m["threshold_source"].startswith("validation")
        assert {"pr_auc", "auc_roc", "precision", "recall", "f1_score", "confusion_matrix",
                "alerts_per_1000", "recall_at_fpr", "episodes"} <= set(m)


# ---- 6. first-fraud / episode metrics -----------------------------------------------

def test_episode_metrics_by_hand():
    t0 = pd.Timestamp("2026-06-01")
    frame = pd.DataFrame({
        "customer_id": ["A"] * 4 + ["B"] * 3 + ["C"],
        "timestamp": [t0 + pd.Timedelta(hours=h) for h in (0, 1, 5, 9, 0, 2, 4, 0)],
        "is_fraud":   [1, 1, 1, 0, 1, 1, 0, 0],
        "episode_id": [1, 1, 1, 0, 2, 2, 0, 0],
        "is_first_fraud": [True, False, False, False, True, False, False, False],
    })
    alert = [False, False, True, False, True, True, False, True]
    m = episode_metrics(frame, alert)
    assert m["fraud_episodes"] == 2 and m["first_fraud_transactions"] == 2
    assert m["first_fraud_recall"] == 0.5          # A's first missed, B's first caught
    assert m["overall_fraud_recall"] == 0.6        # 3 of 5 fraud transactions
    assert m["episodes_detected"] == 2 and m["episode_detection_rate"] == 1.0
    assert m["detection_delay_fraud_transactions"] == {"median": 1.0, "mean": 1.0, "max": 2}
    assert m["detection_delay_hours"] == {"median": 2.5, "max": 5.0}


def test_first_fraud_flags_mark_earliest_fraud_of_each_episode(real):
    df, _, episode_id, _ = real
    flags = first_fraud_flags(df, episode_id)
    assert flags.sum() == 60
    firsts = df[flags]
    earliest = df[episode_id > 0].groupby(episode_id[episode_id > 0])["timestamp"].min()
    assert sorted(firsts["timestamp"]) == sorted(earliest)


# ---- 7. the saved report ---------------------------------------------------------------

@pytest.fixture(scope="module")
def report():
    return json.loads(EVAL_REPORT_PATH.read_text())


def test_report_uses_the_saved_split(report, real):
    assert report["splits"]["time"] == json.loads(EVAL_TIME_SPLIT_PATH.read_text())
    rows = report["primary"]["evaluated_rows"]
    d = real[3]["splits"]
    # every fraud transaction has a window, so evaluated fraud == split fraud
    for name in SPLITS:
        assert rows[name]["fraud"] == d[name]["fraud_transactions"]


def test_report_has_all_models_and_early_fraud(report):
    p = report["primary"]
    assert set(p["baselines"]) == {"amount_hour_rule", "logistic_regression", "dnn_without_risk_score"}
    test_episodes = report["splits"]["time"]["splits"]["test"]["fraud_episodes"]
    for m in [p["lstm_risk_predictor"], p["dnn_fraud_classifier"], *p["baselines"].values()]:
        assert m["episodes"]["fraud_episodes"] == test_episodes
        assert m["episodes"]["first_fraud_transactions"] == test_episodes
    assert report["legacy_random_split"]["dnn_fraud_classifier"]["threshold"] == 0.5
    assert report["secondary_customer_grouped"] is not None
