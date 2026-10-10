"""v2 evaluation analysis (app/evaluation/analysis.py) and the per-window score
export of run.py. Runs the real evaluate_split on a small generated v2 sample
with fast stand-in models (logistic regression) instead of the LSTM/DNN, so it
takes seconds and trains nothing that is saved."""

import gzip
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from app import config
from app.evaluation import analysis
from app.evaluation.datasets import dataset_info, load_evaluation_data, resolve_dataset
from app.evaluation.run import _save_scores, evaluate_split
from app.evaluation.split import build_customer_split, build_time_split
from app.evaluation.windows import build_windows


class _Stub:
    def __init__(self, model, flatten):
        self.model, self.flatten, self.info = model, flatten, {"stub": True}

    def predict(self, X):
        return self.model.predict_proba(X.reshape(len(X), -1) if self.flatten else X)[:, 1]


def _fitter(flatten):
    def fit(X, y, timestamps, seed=0):
        Xf = X.reshape(len(X), -1) if flatten else X
        return _Stub(LogisticRegression(max_iter=3000).fit(Xf, y), flatten)
    return fit


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = tmp_path_factory.mktemp("v2_analysis_sample")
    subprocess.run([sys.executable, str(config.DATA_V2_DIR / "generate.py"), "--customers", "100", "--out", str(out)],
                   check=True, capture_output=True)
    data = load_evaluation_data(resolve_dataset("v2").with_data_dir(out))
    labels, episode_id, time_def = build_time_split(data.frame, data.grouping)
    _, cust_def = build_customer_split(data.frame, episode_id, data.grouping)
    result, models = evaluate_split(data.frame, labels, episode_id, build_windows(data.frame),
                                    fit_lstm_fn=_fitter(True), fit_dnn_fn=_fitter(False), n_folds=3, log=lambda m: None)
    report = {"primary": result, "splits": {"time": time_def, "customer": cust_def},
              "dataset": dataset_info(data), "secondary_customer_grouped": None}
    return data, report, models["scores"], time_def


@pytest.fixture(scope="module")
def result(run):
    data, report, scores, _ = run
    return analysis.analyse(data, report, scores, v1_report=None)


def test_scores_cover_every_window_and_match_the_report(run):
    data, report, scores, time_def = run
    assert list(scores.columns[:3]) == ["transaction_id", "split", "is_fraud"]
    assert sorted(scores.columns[3:]) == sorted(analysis.MAJOR)
    assert scores["transaction_id"].is_unique
    for name, rows in report["primary"]["evaluated_rows"].items():
        part = scores[scores["split"] == name]
        assert len(part) == rows["windows"] and int(part["is_fraud"].sum()) == rows["fraud"]


def test_thresholds_reproduce_the_report(run, result):
    _, report, _, _ = run
    for m, t in result["thresholds"].items():
        block = report["primary"][m] if m in report["primary"] else report["primary"]["baselines"][m]
        assert round(t, 6) == block["threshold"]


def test_first_fraud_counts_match_the_test_split(run, result):
    _, report, _, time_def = run
    n = time_def["splits"]["test"]["fraud_episodes"]
    for m in analysis.MAJOR:
        f = result["first_fraud"][m]
        assert f["episodes"] == n
        block = report["primary"][m] if m in report["primary"] else report["primary"]["baselines"][m]
        assert f["first_fraud_detected"] == round(block["episodes"]["first_fraud_recall"] * n)
        assert f["episodes_detected"] == f["first_fraud_detected"] + f["episodes_detected_only_after_onset"]


def test_ablation_partitions_fraud_and_false_positives(run, result):
    _, report, scores, _ = run
    a = result["lstm_ablation"]
    test_fraud = int(scores.loc[scores["split"] == "test", "is_fraud"].sum())
    assert a["fraud_caught_only_with_lstm"] + a["fraud_caught_only_without_lstm"] + a["fraud_caught_by_both"] \
        + a["fraud_missed_by_both"] == test_fraud
    with_fp = report["primary"]["dnn_fraud_classifier"]["confusion_matrix"]["false_positive"]
    assert a["false_positives_only_with_lstm"] + a["false_positives_both"] == with_fp


def test_legit_false_positive_table(run, result):
    _, report, _, _ = run
    rows = {r["context"]: r for r in result["legit_false_positives"]}
    all_legit = rows["(all legitimate)"]
    cm = report["primary"]["dnn_fraud_classifier"]["confusion_matrix"]
    assert all_legit["legitimate_rows"] == cm["true_negative"] + cm["false_positive"]
    assert all_legit["dnn_fraud_classifier.flagged"] == cm["false_positive"]


def test_uncertainty_intervals_contain_the_point_estimate(result):
    u = result["uncertainty"]
    for m, v in u["models"].items():
        lo, hi = v["recall_ci95"]
        assert lo <= v["recall"] <= hi, m
        assert v["first_fraud_recall_wilson95"][0] <= v["first_fraud_recall_wilson95"][1]
    assert analysis.wilson(0, 16) == [0.0, 0.1936]


def test_rings_and_sanity(run, result):
    data, _, _, _ = run
    r = result["rings"]
    assert r["rings_total"] == data.grouping.episodes.loc[data.grouping.episodes["fraud_ring_id"] > 0, "fraud_ring_id"].nunique()
    det = r["production_ring_detector"]
    assert det["flagged_devices"] == sum(det["by_kind"].values())
    s = result["sanity"]
    assert s["ground_truth_in_model_frame"] == [] and s["feature_count"] == 9
    assert s["dataset_sha256"] == s["report_dataset_sha256"]


def test_warning_period_groups(result):
    w = result["warning_period"]["lstm_all_splits_out_of_sample"]
    assert set(w) >= {"1_before_warning", "2_warning_period", "3_first_fraud", "4_later_fraud"}
    assert w["2_warning_period"]["rows"] > 0


def test_analysis_refuses_datasets_without_metadata():
    with pytest.raises(ValueError, match="metadata"):
        analysis.load_inputs("v1")


def test_score_file_is_byte_identical_on_rewrite(run, tmp_path):
    _, _, scores, _ = run
    _save_scores(scores, tmp_path / "a.csv.gz")
    _save_scores(scores, tmp_path / "b.csv.gz")
    assert (tmp_path / "a.csv.gz").read_bytes() == (tmp_path / "b.csv.gz").read_bytes()
    back = pd.read_csv(tmp_path / "a.csv.gz", float_precision="round_trip")
    np.testing.assert_array_equal(back["dnn_fraud_classifier"].to_numpy(), scores["dnn_fraud_classifier"].to_numpy())
    with gzip.open(tmp_path / "a.csv.gz", "rt") as fh:
        assert fh.readline().startswith("transaction_id,split,is_fraud")
