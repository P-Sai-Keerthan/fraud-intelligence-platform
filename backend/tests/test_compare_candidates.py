"""Candidate comparison (app/evaluation/compare_candidates.py) on a small
generated v2 sample, with candidates trained for one epoch by the real
candidate-training code. Writes only to temp directories."""

import json
import shutil
import subprocess
import sys

import numpy as np
import pytest

from app import config
from app.evaluation import compare_candidates as cc
from app.evaluation import stacking
from app.evaluation.datasets import load_evaluation_data, resolve_dataset
from app.evaluation.split import build_time_split, save_definition
from app.features.feature_engineering import FEATURE_COLUMNS
from app.training import candidates as cand

A, B = cc.A, cc.B


def _quick_lstm(X, y, timestamps, seed=0):
    from app.models.lstm_model import build_lstm_model
    stacking._set_seeds(seed)
    flat = X.reshape(-1, X.shape[2])
    mean, std = flat.mean(axis=0), flat.std(axis=0)
    std[std == 0] = 1.0
    model = build_lstm_model(X.shape[1], X.shape[2])
    model.fit((X - mean) / std, y, epochs=1, batch_size=256, class_weight=stacking._class_weight(y), verbose=0)
    return stacking.LSTMScorer(model, mean, std, {"quick": True})


def _quick_dnn(X, y, timestamps, seed=0):
    from app.models.dnn_model import build_dnn_model
    stacking._set_seeds(seed)
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std == 0] = 1.0
    model = build_dnn_model(X.shape[1])
    model.fit((X - mean) / std, y, epochs=1, batch_size=256, class_weight=stacking._class_weight(y), verbose=0)
    return stacking.DNNScorer(model, mean, std, {"quick": True})


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("cmp_data")
    eval_dir = tmp_path_factory.mktemp("cmp_eval")
    root = tmp_path_factory.mktemp("cmp_candidates")
    subprocess.run([sys.executable, str(config.DATA_V2_DIR / "generate.py"), "--customers", "80", "--out", str(data_dir)],
                   check=True, capture_output=True)
    spec = resolve_dataset("v2").with_data_dir(data_dir).with_output_dir(eval_dir)
    data = load_evaluation_data(spec)
    labels, _, definition = build_time_split(data.frame, data.grouping)
    save_definition(spec.time_split_path, definition)
    before = cand.production_checksums()
    cand.train_candidates(spec=spec, output_root=root, fit_lstm_fn=_quick_lstm, fit_dnn_fn=_quick_dnn,
                          n_folds=2, log=lambda m: None)
    result = cc.compare(spec=spec, candidates_root=root)
    cand.verify_production_unchanged(before, cand.production_checksums())
    return spec, root, result, definition


def test_identity_and_row_counts(setup):
    spec, root, r, definition = setup
    assert r["dataset"]["version"] == "v2"
    assert r["split"]["sha256"] == cand._sha256(spec.time_split_path)
    assert r["features"]["columns"] == list(FEATURE_COLUMNS)
    test = definition["splits"]["test"]
    assert r["rows"]["test"] == test["transactions"] and r["rows"]["test_fraud"] == test["fraud_transactions"]
    assert r["rows"]["test_fraud_episodes"] == test["fraud_episodes"]
    assert r["rows"]["validation"] == definition["splits"]["validation"]["transactions"]
    for m, d in cc.CANDIDATE_DIRS.items():
        man = json.loads((root / d / "manifest.json").read_text())
        assert r["candidates"][m]["dnn_weights_sha256"] == man["model"]["dnn_weights_sha256"]
        assert r["candidates"][m]["thresholds"] == man["thresholds"]
    assert r["candidates"][A]["dnn_input_columns"] == list(FEATURE_COLUMNS) + ["risk_score"]
    assert r["candidates"][B]["dnn_input_columns"] == list(FEATURE_COLUMNS)
    assert r["matches_4c2d_evaluation_scores"] == {"available": False}     # no evaluation scores in this sample


def test_overall_metrics_are_consistent(setup):
    _, _, r, _ = setup
    for m in (A, B):
        o = r["overall_test"][m]
        cm = o["confusion_matrix"]
        assert sum(cm.values()) == r["rows"]["test"]
        assert cm["true_positive"] + cm["false_negative"] == r["rows"]["test_fraud"]
        assert o["false_positives"] == cm["false_positive"] and o["false_negatives"] == cm["false_negative"]
        assert o["threshold"] == r["candidates"][m]["thresholds"]["f1_optimal"]
        assert r["operating_points"][m]["f1_optimal"]["test_false_positives"] == cm["false_positive"]


def test_operating_points_use_manifest_thresholds(setup):
    _, _, r, _ = setup
    for m in (A, B):
        pts = r["operating_points"][m]
        assert set(pts) == {"f1_optimal", "fpr_0.001", "fpr_0.01", "fpr_0.05"}
        for k, v in r["candidates"][m]["thresholds"]["fpr_operating_points"].items():
            assert pts[f"fpr_{k}"]["threshold"] == v
            assert 0 <= pts[f"fpr_{k}"]["validation_fpr"] <= 1       # the realised FPR is reported, not assumed
        # looser false-positive budgets mean lower thresholds and at least as many alerts
        t = [pts[k]["threshold"] for k in ("fpr_0.001", "fpr_0.01", "fpr_0.05")]
        a = [pts[k]["test_alerts_per_1000"] for k in ("fpr_0.001", "fpr_0.01", "fpr_0.05")]
        assert t == sorted(t, reverse=True) and a == sorted(a)


def test_alert_bands_cover_every_test_row(setup):
    _, _, r, _ = setup
    for m in (A, B):
        bands = r["alert_bands_production_25_50_80"][m]["bands"]
        assert sum(b["legitimate"] + b["fraud"] for b in bands.values()) == r["rows"]["test"]
        assert sum(b["fraud"] for b in bands.values()) == r["rows"]["test_fraud"]


def test_episode_and_false_positive_sections(setup):
    _, _, r, _ = setup
    for m in (A, B):
        e = r["episodes_test"][m]
        assert e["episodes"] == r["rows"]["test_fraud_episodes"]
        assert e["episodes_detected"] == e["first_fraud_detected"] + e["episodes_detected_only_after_onset"]
    sig = {row["signal"]: row for row in r["false_positives"]["by_signal"]}
    ctx = {row["context"]: row for row in r["false_positives"]["by_legit_context"]}
    for m in (A, B):
        fp = r["overall_test"][m]["false_positives"]
        assert sig["(all legitimate)"][f"{m}.false_positives_with_signal"] == fp
        assert ctx["(all legitimate)"][f"{m}.flagged"] == fp
        assert sum(row[f"{m}.false_positives"] for row in r["false_positives"]["by_customer_segment"]) == fp


def test_lstm_contribution_partitions(setup):
    _, _, r, _ = setup
    d = r["lstm_contribution"]["detection_overlap_test"]
    assert d["fraud_caught_only_with_lstm"] + d["fraud_caught_only_without_lstm"] + d["fraud_caught_by_both"] \
        + d["fraud_missed_by_both"] == r["rows"]["test_fraud"]
    assert d["false_positives_only_with_lstm"] + d["false_positives_both"] == r["overall_test"][A]["false_positives"]
    assert d["false_positives_only_without_lstm"] + d["false_positives_both"] == r["overall_test"][B]["false_positives"]
    sd = r["lstm_contribution"]["score_differences_test"]
    assert sd["all"]["rows"] == r["rows"]["test"] and len(sd["fraud_rows"]) == r["rows"]["test_fraud"]
    u = r["lstm_contribution"]["uncertainty_test"]
    assert set(u["models"]) == {A, B}
    lo, hi = u["pr_auc_difference_dnn_with_minus_without_lstm"]["ci95"]
    assert lo <= hi


def test_comparison_is_deterministic(setup):
    spec, root, r, _ = setup
    again = cc.compare(spec=spec, candidates_root=root)
    assert json.dumps(again, default=str) == json.dumps(r, default=str)


def test_tampered_threshold_is_refused(setup, tmp_path):
    spec, root, _, _ = setup
    bad = tmp_path / "bad"
    shutil.copytree(root, bad)
    man = json.loads((bad / "dnn_only" / "manifest.json").read_text())
    man["thresholds"]["f1_optimal"] = 0.123
    (bad / "dnn_only" / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(ValueError, match="do not reproduce"):
        cc.compare(spec=spec, candidates_root=bad)


def test_default_output_location():
    assert config.CANDIDATES_DIR / "v2" == config.BACKEND_DIR / "models" / "candidates" / "v2"
    assert config.MODELS_SAVED_DIR not in (config.CANDIDATES_DIR / "v2" / "comparison.json").parents
