"""Candidate training (app/training/candidates.py).

Runs the real training code path on a small generated v2 sample with the
production architectures trained for one epoch (fast stand-ins for fit_lstm
/ fit_dnn), writing only to temp directories. Nothing here touches
models/saved/ or models/candidates/.
"""

import json
import subprocess
import sys

import numpy as np
import pytest

from app import config
from app.evaluation import stacking
from app.evaluation.datasets import load_evaluation_data, resolve_dataset
from app.evaluation.split import build_time_split, save_definition
from app.evaluation.windows import build_windows
from app.features.feature_engineering import FEATURE_COLUMNS
from app.features.ground_truth import GROUND_TRUTH_COLUMNS
from app.training import candidates as cand


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
def sample_spec(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("cand_data")
    eval_dir = tmp_path_factory.mktemp("cand_eval")
    subprocess.run([sys.executable, str(config.DATA_V2_DIR / "generate.py"), "--customers", "60", "--out", str(data_dir)],
                   check=True, capture_output=True)
    spec = resolve_dataset("v2").with_data_dir(data_dir).with_output_dir(eval_dir)
    data = load_evaluation_data(spec)
    _, _, definition = build_time_split(data.frame, data.grouping)
    save_definition(spec.time_split_path, definition)       # stands in for the saved 4C-2d split
    return spec


@pytest.fixture(scope="module")
def trained(sample_spec, tmp_path_factory):
    out = tmp_path_factory.mktemp("cand_out")
    before = cand.production_checksums()
    result = cand.train_candidates(spec=sample_spec, output_root=out, fit_lstm_fn=_quick_lstm,
                                   fit_dnn_fn=_quick_dnn, n_folds=2, log=lambda m: None)
    cand.verify_production_unchanged(before, cand.production_checksums())
    return out, result


# ---- output paths and safety ----------------------------------------------------------------

def test_default_candidate_paths():
    assert cand.CANDIDATES == ("dnn_lstm", "dnn_only")
    assert config.CANDIDATES_DIR == config.BACKEND_DIR / "models" / "candidates"
    for name in cand.CANDIDATES:
        assert cand.candidate_dir(name) == (config.CANDIDATES_DIR / "v2" / name).resolve()
    with pytest.raises(ValueError, match="unknown candidate"):
        cand.candidate_dir("lstm_only")


@pytest.mark.parametrize("bad", [config.MODELS_SAVED_DIR, config.MODELS_SAVED_DIR / "v2", config.MODELS_SAVED_DIR / "x" / "y"])
def test_production_directory_can_never_be_an_output(bad):
    with pytest.raises(ValueError, match="production model directory"):
        cand.assert_safe_output(bad)
    with pytest.raises(ValueError, match="production model directory"):
        cand.train_candidates(output_root=bad, log=lambda m: None)      # refused before loading any data


def test_v2_is_required():
    with pytest.raises(ValueError, match="v2 only"):
        cand.train_candidates("v1", log=lambda m: None)
    with pytest.raises(ValueError, match="unknown dataset version"):
        cand.train_candidates("v3", log=lambda m: None)


def test_saved_split_is_required(sample_spec, tmp_path):
    no_split = sample_spec.with_output_dir(tmp_path / "empty")
    with pytest.raises(FileNotFoundError, match="saved split"):
        cand.train_candidates(spec=no_split, output_root=tmp_path / "out", log=lambda m: None)


def test_a_different_saved_split_is_refused(sample_spec, tmp_path):
    other = sample_spec.with_output_dir(tmp_path / "tampered")
    saved = json.loads(sample_spec.time_split_path.read_text())
    saved["train_before"] = "2026-01-01 00:00:00"
    other.time_split_path.parent.mkdir(parents=True)
    other.time_split_path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="does not match the saved split"):
        cand.train_candidates(spec=other, output_root=tmp_path / "out", log=lambda m: None)


def test_production_checksum_protection():
    before = cand.production_checksums()
    assert set(before) == {f.name for f in config.MODELS_SAVED_DIR.iterdir() if f.is_file()}
    assert len(before) == 7
    changed = dict(before, **{"dnn_fraud_model.keras": "0" * 64})
    with pytest.raises(RuntimeError, match="dnn_fraud_model.keras"):
        cand.verify_production_unchanged(before, changed)
    cand.verify_production_unchanged(before, dict(before))


# ---- what gets trained and saved -----------------------------------------------------------

def test_candidate_files(trained):
    out, _ = trained
    dnn_files = {"dnn_fraud_model.keras", "dnn_feature_mean.npy", "dnn_feature_std.npy", "shap_background.npy", "manifest.json"}
    lstm_files = {"lstm_risk_model.keras", "lstm_feature_mean.npy", "lstm_feature_std.npy"}
    assert {p.name for p in (out / "dnn_lstm").iterdir()} == dnn_files | lstm_files
    assert {p.name for p in (out / "dnn_only").iterdir()} == dnn_files


def test_manifest_contents(trained, sample_spec):
    out, result = trained
    for name in cand.CANDIDATES:
        m = json.loads((out / name / "manifest.json").read_text())
        assert m["candidate"] == name and m["seed"] == 42 and m["sequence_length"] == 10
        assert m["dataset"]["version"] == "v2"
        assert m["split"]["sha256"] == cand._sha256(sample_spec.time_split_path)
        assert m["split"]["test_rows_used"] == 0
        assert m["features"]["columns"] == list(FEATURE_COLUMNS) and m["features"]["count"] == 9
        assert m["clipping"]["dnn_training"].startswith("none") and "+-6" in m["clipping"]["dnn_scoring"]
        assert 0 < m["thresholds"]["f1_optimal"] < 1 and m["thresholds"]["source"].startswith("validation")
        assert set(m["thresholds"]["fpr_operating_points"]) == {"0.001", "0.01", "0.05"}
        assert m["model"]["dnn_training"]["class_weight"]["0"] == 1.0
        assert m["model"]["dnn_training"]["class_weight"]["1"] > 1
        for fname, sha in m["files"].items():
            assert cand._sha256(out / name / fname) == sha
        text = json.dumps(m)
        assert str(config.PROJECT_ROOT) not in text and "/tmp" not in text      # no local paths
    a = json.loads((out / "dnn_lstm" / "manifest.json").read_text())
    b = json.loads((out / "dnn_only" / "manifest.json").read_text())
    assert a["model"]["dnn_input_columns"] == list(FEATURE_COLUMNS) + ["risk_score"]
    assert b["model"]["dnn_input_columns"] == list(FEATURE_COLUMNS)
    assert "lstm_weights_sha256" in a["model"] and "lstm_weights_sha256" not in b["model"]


def test_metadata_never_reaches_a_model(trained):
    out, _ = trained
    for name in cand.CANDIDATES:
        cols = json.loads((out / name / "manifest.json").read_text())["model"]["dnn_input_columns"]
        assert not set(cols) & set(GROUND_TRUTH_COLUMNS) and "is_fraud" not in cols


def test_thresholds_come_from_validation_predictions_only(trained):
    _, result = trained
    from app.evaluation.metrics import select_threshold, threshold_for_fpr
    y = result["y"]
    for name, preds in result["predictions"].items():
        va = preds["split"] == "validation"
        th = result["manifests"][name]["thresholds"]
        assert th["f1_optimal"] == select_threshold(y[va], preds["dnn"][va])
        assert th["fpr_operating_points"]["0.01"] == threshold_for_fpr(y[va], preds["dnn"][va], 0.01)
        assert "test" in set(preds["split"])            # test rows exist but played no part


# ---- loading and scoring ---------------------------------------------------------------------

def test_candidates_load_and_score(trained, sample_spec):
    out, result = trained
    data = load_evaluation_data(sample_spec)
    X, y, target = build_windows(data.frame)
    F = data.frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)[target]

    a = cand.load_candidate("dnn_lstm", output_root=out)
    assert a.lstm is not None and a.lstm.input_shape == (None, 10, 9) and a.dnn.input_shape == (None, 10)
    va = result["predictions"]["dnn_lstm"]["split"] != "train"          # final-LSTM rows (not out-of-fold)
    # score the same rows in the same batch as training did, so the 2-decimal risk_score rounds identically
    lstm_p = a.risk_probability(X[va])
    np.testing.assert_array_equal(lstm_p, result["predictions"]["dnn_lstm"]["lstm"][va])
    scores = a.score(F[va], a.risk_score(X[va]))
    np.testing.assert_array_equal(scores, result["predictions"]["dnn_lstm"]["dnn"][va])
    assert a.threshold == result["manifests"]["dnn_lstm"]["thresholds"]["f1_optimal"]

    b = cand.load_candidate("dnn_only", output_root=out)
    assert b.lstm is None and b.dnn.input_shape == (None, 9)
    np.testing.assert_array_equal(b.score(F), result["predictions"]["dnn_only"]["dnn"])
    assert b.threshold == result["manifests"]["dnn_only"]["thresholds"]["f1_optimal"]


def test_loading_refuses_tampered_artifacts(trained, tmp_path):
    out, _ = trained
    import shutil
    bad = tmp_path / "dnn_only"
    shutil.copytree(out / "dnn_only", bad)
    np.save(bad / "dnn_feature_mean.npy", np.zeros(9))
    with pytest.raises(ValueError, match="manifest checksum"):
        cand.Candidate(bad)
