"""Controlled-promotion preparation (Step 4C-3F): the pinned model set
v2_dnn_lstm_seed14.

It is the exact artifact selected in 4C-3E.6 (v2_dnn_lstm, training seed 14).
These tests check that it loads only when asked for by name, only from the
selected files, that production stays the default and is untouched, that its
metadata says "EVALUATION / NOT DEPLOYED", and that switching back to
production is just MODEL_SET=production.

Tests that need the trained artifact are skipped when
models/candidates_multiseed/v2/seed_14/ is not present locally."""

import hashlib
import json
import shutil

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import config, inference_pipeline
from app import model_sets as ms
from app.inference_pipeline import FraudIntelligencePipeline
from app.main import app
from app.model_metadata import (FINAL_HOLDOUT_REPORT_PATH, SELECTION_RECORD_PATH, STATUS_EVALUATION,
                                STATUS_PRODUCTION, final_holdout_evaluation, model_metadata)

from conftest import NORMAL_TXN, SUSPICIOUS_TXN
from test_observability import _pdf_text, _row

NAME = ms.SEED14_MODEL_SET
SEED14_DIR = config.BACKEND_DIR / "models" / "candidates_multiseed" / "v2" / "seed_14" / "dnn_lstm"
HAVE_ARTIFACT = (SEED14_DIR / "manifest.json").exists()
artifact = pytest.mark.skipif(not HAVE_ARTIFACT, reason="the seed-14 artifact is not present locally")
HAVE_RECORDS = SELECTION_RECORD_PATH.exists() and FINAL_HOLDOUT_REPORT_PATH.exists()
records = pytest.mark.skipif(not HAVE_RECORDS, reason="selection record / final hold-out report not present locally")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _saved_checksums():
    return {p.name: _sha(p) for p in sorted(config.MODELS_SAVED_DIR.iterdir()) if p.is_file()}


@pytest.fixture(scope="module")
def seed14():
    if not HAVE_ARTIFACT:
        pytest.skip("the seed-14 artifact is not present locally")
    return FraudIntelligencePipeline(model_set=NAME)


@pytest.fixture
def use_pipeline(monkeypatch):
    def swap(p):
        monkeypatch.setattr(inference_pipeline, "_pipeline_instance", p)
    return swap


@pytest.fixture
def artifact_copy(tmp_path):
    """A private copy of seed_14/ to tamper with."""
    if not HAVE_ARTIFACT:
        pytest.skip("the seed-14 artifact is not present locally")
    root = tmp_path / "seed_14"
    shutil.copytree(SEED14_DIR, root / "dnn_lstm")
    return root


# ---- registry: explicit opt-in, production stays the default ------------------------------------------------

def test_registered_as_an_explicit_evaluation_model_set():
    spec = ms.MODEL_SETS[NAME]
    assert NAME == "v2_dnn_lstm_seed14" and spec.uses_lstm and spec.training_seed == 14
    assert spec.candidate == "dnn_lstm" and spec.dataset_version == "v2" and spec.artifact == "v2_dnn_lstm@14"
    assert ms.architecture_name(NAME) == "v2_dnn_lstm" and ms.architecture_name("production") == "production"
    assert ms.DEFAULT_MODEL_SET == "production"


def test_directory_is_the_multiseed_artifact_not_production():
    directory = ms.model_set_directory(NAME)
    assert directory == SEED14_DIR.resolve()
    assert config.MODELS_SAVED_DIR.resolve() not in directory.parents
    assert directory != (config.CANDIDATES_DIR / "v2" / "dnn_lstm").resolve()      # not the seed-42 candidate


def test_default_and_explicit_production_are_unaffected(monkeypatch):
    monkeypatch.delenv("MODEL_SET", raising=False)
    assert ms.resolve_model_set_name() == "production"
    monkeypatch.setenv("MODEL_SET", "production")
    assert ms.resolve_model_set_name() == "production"
    monkeypatch.setenv("MODEL_SET", NAME)
    assert ms.resolve_model_set_name() == NAME


@pytest.mark.parametrize("value", ["v2_dnn_lstm_seed_14", "v2_dnn_lstm_seed14 ", "V2_DNN_LSTM_SEED14", "seed14",
                                   "v2_dnn_lstm@14", "v2_dnn_lstm_seed15", "v2_dnn_only_seed14"])
def test_near_miss_names_fail_and_never_fall_back(monkeypatch, value):
    monkeypatch.setenv("MODEL_SET", value)
    with pytest.raises(ms.ModelSetError, match="invalid environment variable MODEL_SET"):
        ms.resolve_model_set_name()
    monkeypatch.setattr(inference_pipeline, "_pipeline_instance", None)
    with pytest.raises(ms.ModelSetError):
        with TestClient(app):
            pass
    assert inference_pipeline._pipeline_instance is None


def test_missing_artifact_stops_startup_instead_of_loading_production(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "BACKEND_DIR", tmp_path)                 # no models/candidates_multiseed here
    monkeypatch.setattr(inference_pipeline, "_pipeline_instance", None)
    monkeypatch.setenv("MODEL_SET", NAME)
    with pytest.raises(ms.ModelSetError, match="manifest .* not found"):
        with TestClient(app):
            pass
    assert inference_pipeline._pipeline_instance is None


# ---- the pinned values are the recorded ones --------------------------------------------------------------

@records
def test_pinned_values_equal_the_selection_record_and_stage_c_report():
    record = json.loads(SELECTION_RECORD_PATH.read_text())["selected_artifacts"]["v2_dnn_lstm@14"]
    assert record["architecture"] == "v2_dnn_lstm" and record["training_seed"] == 14
    assert dict(ms.SEED14_PINNED["weights"]) == record["weights_sha256"]
    assert dict(ms.SEED14_PINNED["files"]) == record["files_sha256"]
    assert dict(ms.SEED14_FROZEN_CUTOFFS) == record["cutoffs"]
    report = json.loads(FINAL_HOLDOUT_REPORT_PATH.read_text())
    frozen = report["fixed_inputs"]["frozen_cutoffs_score_0_1"]
    assert frozen["policy_b"]["v2_dnn_lstm@14"] == ms.SEED14_FROZEN_CUTOFFS["policy_b"]
    assert frozen["critical"]["v2_dnn_lstm@14"] == ms.SEED14_FROZEN_CUTOFFS["critical"]
    assert round(ms.SEED14_FROZEN_CUTOFFS["policy_b"], 4) == 0.7929
    assert round(ms.SEED14_FROZEN_CUTOFFS["critical"], 4) == 0.9431


@artifact
def test_artifact_on_disk_is_the_selected_one():
    manifest = json.loads((SEED14_DIR / "manifest.json").read_text())
    assert manifest["seed"] == 14 and manifest["candidate"] == "dnn_lstm" and manifest["dataset"]["version"] == "v2"
    assert manifest["features"]["count"] == 9 and manifest["sequence_length"] == 10
    assert manifest["model"]["dnn_input_shape"] == [None, 10] and manifest["model"]["lstm_input_shape"] == [None, 10, 9]
    assert manifest["files"] == dict(ms.SEED14_PINNED["files"])
    for fname, sha in ms.SEED14_PINNED["files"].items():
        assert _sha(SEED14_DIR / fname) == sha, fname


# ---- loading ----------------------------------------------------------------------------------------------

@artifact
def test_loads_the_exact_artifact_without_writing(seed14):
    before_artifact = {p.name: _sha(p) for p in sorted(SEED14_DIR.iterdir())}
    before_saved = _saved_checksums()
    loaded = ms.load_model_set(NAME)
    assert loaded.name == NAME and loaded.uses_lstm and loaded.directory == SEED14_DIR.resolve()
    assert loaded.manifest["seed"] == 14
    assert loaded.dnn_input_columns == ms.MODEL_SETS["v2_dnn_lstm"].dnn_input_columns
    assert tuple(loaded.dnn_model.input_shape[1:]) == (10,) and tuple(loaded.lstm_model.input_shape[1:]) == (10, 9)
    assert loaded.shap_background.shape == (200, 10)
    assert {p.name: _sha(p) for p in sorted(SEED14_DIR.iterdir())} == before_artifact
    assert _saved_checksums() == before_saved


@artifact
def test_a_changed_scaler_is_refused(artifact_copy):
    path = artifact_copy / "dnn_lstm" / "dnn_feature_mean.npy"
    np.save(path, np.load(path) + 1e-6)
    with pytest.raises(ms.ModelSetError, match="dnn_feature_mean.npy"):
        ms.load_model_set(NAME, candidates_root=artifact_copy)


@artifact
def test_a_changed_model_file_is_refused(artifact_copy):
    path = artifact_copy / "dnn_lstm" / "lstm_risk_model.keras"
    path.write_bytes(path.read_bytes() + b"\0")
    with pytest.raises(ms.ModelSetError, match="lstm_risk_model.keras is not the selected artifact"):
        ms.load_model_set(NAME, candidates_root=artifact_copy)


@artifact
@pytest.mark.parametrize("edit, message", [
    (lambda m: m.__setitem__("seed", 13), "training seed 13, expected 14"),
    (lambda m: m["model"].__setitem__("dnn_weights_sha256", "0" * 64), "dnn_weights_sha256 is not the selected"),
    (lambda m: m["model"].__setitem__("lstm_weights_sha256", "0" * 64), "lstm_weights_sha256 is not the selected"),
    (lambda m: m["files"].pop("shap_background.npy"), "manifest lists files"),
])
def test_a_changed_manifest_is_refused(artifact_copy, edit, message):
    path = artifact_copy / "dnn_lstm" / "manifest.json"
    manifest = json.loads(path.read_text())
    edit(manifest)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ms.ModelSetError, match=message):
        ms.load_model_set(NAME, candidates_root=artifact_copy)


def test_another_training_seed_is_refused():
    """Seed 13's files (or the seed-42 candidate) under this name are rejected."""
    for other in (config.BACKEND_DIR / "models" / "candidates_multiseed" / "v2" / "seed_13", config.CANDIDATES_DIR / "v2"):
        if not (other / "dnn_lstm" / "manifest.json").exists():
            continue
        with pytest.raises(ms.ModelSetError, match="training seed|not the selected artifact"):
            ms.load_model_set(NAME, candidates_root=other)


# ---- metadata ---------------------------------------------------------------------------------------------

def test_production_metadata_reports_production_status(pipeline):
    meta = pipeline.model_metadata
    assert meta["model_set"] == "production" and meta["status"] == STATUS_PRODUCTION
    assert meta["architecture"] == "DNN + LSTM" and meta["training_seed"] is None
    assert "selection" not in meta and "frozen_cutoffs" not in meta["thresholds"]


@artifact
def test_metadata_says_evaluation_not_deployed(client, seed14, use_pipeline):
    use_pipeline(seed14)
    meta = client.get("/model-info").json()
    assert meta["model_set"] == "v2_dnn_lstm_seed14"
    assert meta["architecture"] == "DNN + LSTM" and meta["uses_lstm"] is True
    assert meta["training_seed"] == 14
    assert meta["status"] == STATUS_EVALUATION == "EVALUATION / NOT DEPLOYED"
    assert meta["model_version"] == "v2_dnn_lstm_seed14-32fc53979e06"
    assert meta["directory"] == "backend/models/candidates_multiseed/v2/seed_14/dnn_lstm"
    assert meta["model"]["dnn_weights_sha256"] == ms.SEED14_PINNED["weights"]["dnn_weights_sha256"]
    assert meta["model"]["lstm_weights_sha256"] == ms.SEED14_PINNED["weights"]["lstm_weights_sha256"]
    assert meta["model"]["files_sha256"] == dict(ms.SEED14_PINNED["files"])
    assert meta["dataset"]["version"] == "v2" and meta["features"]["columns"] == list(seed14.model_set.dnn_input_columns[:9])
    assert meta["score_semantics"]["calibrated_probabilities"] is False
    assert "LSTM behavioral-risk component" in meta["score_semantics"]["risk_score"]
    frozen = meta["thresholds"]["frozen_cutoffs"]
    assert round(frozen["policy_b"], 4) == 0.7929 and round(frozen["critical"], 4) == 0.9431
    assert "NOT applied by /predict" in frozen["status"]
    assert meta["thresholds"]["alert_bands"] == seed14.model_metadata["thresholds"]["alert_bands"]   # legacy bands unchanged
    selection = meta["selection"]
    assert selection["artifact"] == "v2_dnn_lstm@14" and selection["presentation"] == "Validated candidate - not yet deployed"
    assert "25.6 legitimate alerts per 1,000" in selection["new_customer_limitation"]
    assert any("new-customer" in b for b in selection["promotion_blockers"])


@artifact
@records
def test_metrics_serve_the_stage_c_slice_and_no_production_numbers(client, seed14, use_pipeline):
    use_pipeline(seed14)
    body = client.get("/metrics").json()
    assert body["model_set"] == NAME
    assert body["lstm_risk_predictor"] is None and body["dnn_fraud_classifier"] is None
    assert body["candidate_evaluation"]["available"] is False
    ev = body["final_holdout_evaluation"]
    report = json.loads(FINAL_HOLDOUT_REPORT_PATH.read_text())
    gates = report["decision"]["gates"]["v2_dnn_lstm@14"]
    assert ev["available"] and ev["artifact"] == "v2_dnn_lstm@14" and ev["gates"]["result"] == gates
    primary = report["pooled"]["primary"]["models"]["v2_dnn_lstm@14"]
    assert ev["primary"]["metrics"]["recall"] == primary["metrics"]["recall"] == gates["recall"]
    assert ev["primary"]["metrics"]["legit_alerts_per_1000"] == primary["metrics"]["legit_alerts_per_1000"]
    assert ev["primary"]["counts"] == primary["counts"]
    new = report["new_customer"]["current_behaviour"]["models"]["v2_dnn_lstm@14"]["metrics"]["legit_alerts_per_1000"]
    assert ev["new_customer"]["metrics"]["legit_alerts_per_1000"] == new and round(new["value"], 1) == 25.6
    assert "25.6" in ev["new_customer"]["limitation"]
    assert ev["promotion"].startswith("none")
    assert client.get("/metrics/report").json()["final_holdout_evaluation"]["available"]


@artifact
def test_stage_c_slice_is_refused_for_other_model_sets(pipeline):
    assert final_holdout_evaluation(pipeline.model_set)["available"] is False


# ---- controlled scoring -----------------------------------------------------------------------------------

@artifact
def test_candidate_scores_with_shap_similarity_and_provenance(client, seed14, use_pipeline):
    use_pipeline(seed14)
    normal = client.post("/predict", json=dict(NORMAL_TXN))
    suspicious = client.post("/predict", json=dict(SUSPICIOUS_TXN))
    assert normal.status_code == 200 and suspicious.status_code == 200
    low, high = normal.json(), suspicious.json()
    for body in (low, high):
        assert 0 <= body["fraud_probability"] <= 99.9 and 0 <= body["risk_score"] <= 100
        assert 0 <= body["similarity_pct"] <= 100
        assert round(body["similarity_pct"] + body["deviation_pct"], 1) == 100.0
    assert high["fraud_probability"] > low["fraud_probability"]
    assert high["reasons"] and all(r["shap_value"] > 0 and r["display_name"] for r in high["reasons"])
    row = _row(high["transaction_id"])
    assert row.model_set == NAME and row.model_version == seed14.model_version


@artifact
def test_candidate_batch_scoring(client, seed14, use_pipeline):
    use_pipeline(seed14)
    csv = "customer_id,amount,merchant_category\nCUST_0001,1500,grocery\nCUST_0002,92000,electronics\n"
    r = client.post("/predict/batch", files={"file": ("t.csv", csv, "text/csv")})
    assert r.status_code == 200 and r.json()["count"] == 2
    assert all(_row(x["transaction_id"]).model_set == NAME for x in r.json()["results"])


@artifact
def test_candidate_pdf_names_the_candidate_as_not_deployed(client, seed14, use_pipeline):
    use_pipeline(seed14)
    pred = client.post("/predict", json=dict(SUSPICIOUS_TXN)).json()
    r = client.post("/report/pdf", json=pred)
    assert r.status_code == 200 and r.content[:5] == b"%PDF-"
    text = _pdf_text(r.content)
    assert "Model set v2_dnn_lstm_seed14" in text and "Training seed 14" in text
    assert "EVALUATION / NOT DEPLOYED" in text and "NOT deployed" in text
    assert "Fraud Score" in text and "Fraud Probability" not in text
    assert "not a calibrated probability" in text and "Trajectory" not in text


def test_production_pdf_says_fraud_score(client, pipeline):
    pred = client.post("/predict", json=dict(SUSPICIOUS_TXN)).json()
    text = _pdf_text(client.post("/report/pdf", json=pred).content)
    assert "Fraud Score" in text and "Fraud Probability" not in text
    assert "Model set production" in text and "Status PRODUCTION" in text


# ---- rollback ---------------------------------------------------------------------------------------------

@artifact
def test_rollback_production_seed14_production(monkeypatch, pipeline):
    """production -> seed 14 -> production: each start loads what MODEL_SET names,
    production scores exactly as before, and models/saved/ is byte-identical."""
    saved_before = _saved_checksums()
    production_version = pipeline.model_version
    txn = {k: v for k, v in SUSPICIOUS_TXN.items() if k != "timestamp"}
    from datetime import datetime
    when = datetime.fromisoformat(SUSPICIOUS_TXN["timestamp"])

    def score(p):
        snapshot = dict(p.customer_histories)
        try:
            np.random.seed(1234)
            return p.score_transaction(dict(txn, timestamp=when))
        finally:
            p.customer_histories.clear()
            p.customer_histories.update(snapshot)

    reference = score(pipeline)

    monkeypatch.setenv("MODEL_SET", "production")
    first = FraudIntelligencePipeline()
    assert first.model_set.name == "production" and first.model_version == production_version

    monkeypatch.setenv("MODEL_SET", NAME)
    candidate = FraudIntelligencePipeline()
    assert candidate.model_set.name == NAME and candidate.model_version == "v2_dnn_lstm_seed14-32fc53979e06"
    assert candidate.model_set.directory == SEED14_DIR.resolve()
    candidate_result = score(candidate)
    assert candidate_result["model_set"] == NAME

    monkeypatch.setenv("MODEL_SET", "production")
    back = FraudIntelligencePipeline()
    assert back.model_set.name == "production" and back.model_version == production_version
    assert back.model_set.directory == config.MODELS_SAVED_DIR
    again = score(back)
    assert again["model_set"] == "production"
    assert again["fraud_probability"] == reference["fraud_probability"]
    assert again["risk_score"] == reference["risk_score"] and again["alert_level"] == reference["alert_level"]

    monkeypatch.delenv("MODEL_SET")
    assert FraudIntelligencePipeline().model_set.name == "production"        # unset is production too
    assert _saved_checksums() == saved_before
