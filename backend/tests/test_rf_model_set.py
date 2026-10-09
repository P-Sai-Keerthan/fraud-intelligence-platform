"""Step 4D: the default model set v2_lstm_rf_seed14 (unchanged seed-14 LSTM -> random forest).

The shared test pipeline is pinned to the previous default ("production") by
conftest.py so that the existing suite keeps testing it; this module builds the
default pipeline itself and swaps it into the application for the API tests."""

import json
import shutil

import numpy as np
import pytest

from app import config
from app import inference_pipeline as ip
from app import model_sets as ms
from app.model_metadata import STATUS_LEGACY, STATUS_PRODUCTION, downstream_evaluation
from app.models.downstream_classifier import ClassifierModel
from conftest import BASE_TIME, NORMAL_TXN, SUSPICIOUS_TXN

NAME = ms.RF_MODEL_SET
RF_DIR = config.BACKEND_DIR / "models" / "candidates_downstream" / "v2" / "seed_14" / "lstm_random_forest"
SEED14_LSTM_DIR = config.BACKEND_DIR / "models" / "candidates_multiseed" / "v2" / "seed_14" / "dnn_lstm"
DOWNSTREAM = config.BACKEND_DIR / "models" / "evaluation" / "downstream"


@pytest.fixture(scope="module")
def rf_pipeline(monkeypatch_module):
    monkeypatch_module.delenv("MODEL_SET", raising=False)
    return ip.FraudIntelligencePipeline()          # unset -> the default


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture
def rf_client(client, rf_pipeline, monkeypatch):
    """The application's API with the default (random-forest) pipeline loaded."""
    snapshot = dict(rf_pipeline.customer_histories)
    monkeypatch.setattr(ip, "_pipeline_instance", rf_pipeline)
    yield client
    rf_pipeline.customer_histories.clear()
    rf_pipeline.customer_histories.update(snapshot)


# ---- configuration -------------------------------------------------------------------------------

def test_default_is_the_random_forest_and_production_stays_loadable(monkeypatch):
    assert ms.DEFAULT_MODEL_SET == NAME == "v2_lstm_rf_seed14"
    monkeypatch.delenv("MODEL_SET", raising=False)
    assert ms.resolve_model_set_name() == NAME
    monkeypatch.setenv("MODEL_SET", "production")
    assert ms.resolve_model_set_name() == "production"
    spec = ms.MODEL_SETS[NAME]
    assert spec.kind == "lstm_classifier" and spec.family == "random_forest" and spec.training_seed == 14
    assert spec.dnn_input_columns[-1] == "risk_score" and len(spec.dnn_input_columns) == 10


@pytest.mark.parametrize("value", ["", "rf", "random_forest", "v2_lstm_rf", "PRODUCTION"])
def test_invalid_model_set_values_are_refused(monkeypatch, value):
    monkeypatch.setenv("MODEL_SET", value)
    with pytest.raises(ms.ModelSetError):
        ms.resolve_model_set_name()


def test_pinned_values_match_the_artifact_manifest():
    man = json.loads((RF_DIR / "manifest.json").read_text())
    assert man["files"] == dict(ms.RF_PINNED["files"])
    assert man["lstm"]["lstm_weights_sha256"] == ms.RF_PINNED["weights"]["lstm_weights_sha256"]
    assert man["family"] == "random_forest" and man["seed"] == 14
    record = json.loads((DOWNSTREAM / "selection_record.json").read_text())
    assert record["deploy"]["family"] == "random_forest" and record["deploy"]["training_seed"] == 14
    assert record["deploy"]["params"] == man["params"]
    assert record["deploy"]["cutoffs"] == dict(ms.RF_FROZEN_CUTOFFS)


def test_the_lstm_is_the_unchanged_seed14_lstm():
    src = json.loads((SEED14_LSTM_DIR / "manifest.json").read_text())
    for f in ("lstm_risk_model.keras", "lstm_feature_mean.npy", "lstm_feature_std.npy"):
        assert (RF_DIR / f).read_bytes() == (SEED14_LSTM_DIR / f).read_bytes()
        assert ms.RF_PINNED["files"][f] == src["files"][f]
    assert src["model"]["lstm_weights_sha256"] == ms.RF_PINNED["weights"]["lstm_weights_sha256"]


def test_the_final_holdout_confirmed_exactly_this_classifier():
    final = json.loads((DOWNSTREAM / "final_holdout_report.json").read_text())
    assert final["confirmed"] is True
    assert final["artifact_manifest"]["files"]["classifier.joblib"] == ms.RF_PINNED["files"]["classifier.joblib"]


def _copy(tmp_path):
    d = tmp_path / "lstm_random_forest"
    shutil.copytree(RF_DIR, d)
    return tmp_path


@pytest.mark.parametrize("tamper", ["classifier", "manifest_family", "missing_file", "lstm_mean"])
def test_tampered_artifact_is_refused_before_loading(tmp_path, tamper):
    root = _copy(tmp_path)
    d = root / "lstm_random_forest"
    if tamper == "classifier":
        (d / "classifier.joblib").write_bytes((d / "classifier.joblib").read_bytes() + b"x")
    elif tamper == "manifest_family":
        man = json.loads((d / "manifest.json").read_text())
        man["family"] = "hist_gradient_boosting"
        (d / "manifest.json").write_text(json.dumps(man))
    elif tamper == "missing_file":
        (d / "shap_background.npy").unlink()
    else:
        np.save(d / "lstm_feature_mean.npy", np.zeros(9))
    with pytest.raises(ms.ModelSetError):
        ms.load_model_set(NAME, candidates_root=root)


# ---- loading, model-info, inference ------------------------------------------------------------------

def test_loads_the_random_forest_behind_the_lstm(rf_pipeline):
    m = rf_pipeline.model_set
    assert m.name == NAME and m.uses_lstm and m.classifier_family == "random_forest"
    assert isinstance(m.dnn_model, ClassifierModel) and type(m.dnn_model.estimator).__name__ == "RandomForestClassifier"
    assert tuple(m.lstm_model.input_shape[1:]) == (config.SEQUENCE_LENGTH, 9)
    assert rf_pipeline.explainer.method == "TreeExplainer"
    assert rf_pipeline.model_version == f"{NAME}-{ms.RF_PINNED['files']['classifier.joblib'][:12]}"


def test_model_info(rf_client):
    info = rf_client.get("/model-info").json()
    assert info["model_set"] == NAME and info["status"] == STATUS_PRODUCTION
    assert info["architecture"] == "LSTM + Random Forest" and info["uses_lstm"] is True
    assert info["downstream_classifier"]["family"] == "random_forest"
    assert info["training_seed"] == 14
    assert info["classifier_input_columns"][-1] == "risk_score"
    assert info["score_semantics"]["calibrated_probabilities"] is False
    assert "not a calibrated probability" in info["score_semantics"]["fraud_probability"]
    assert info["calibration"]["status"].startswith("not demonstrated")
    assert info["evaluation"]["confirmed_on_fresh_holdout"] is True
    assert "10 previous transactions" in info["lstm"]["window"]


def test_previous_default_is_labelled(pipeline):
    assert pipeline.model_set.name == "production"
    assert pipeline.model_metadata["status"] == STATUS_LEGACY


@pytest.mark.parametrize("txn", [NORMAL_TXN, SUSPICIOUS_TXN])
def test_predict_score_range_and_shap(rf_client, rf_pipeline, txn):
    r = rf_client.post("/predict", json=txn)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["fraud_probability"] <= 99.9 and 0 <= body["risk_score"] <= 100
    assert body["alert_level"] in ("Low Risk", "Medium Risk", "High Risk", "Critical Risk")
    for reason in body["reasons"]:
        assert reason["shap_value"] > 0 and reason["feature"] in rf_pipeline.model_set.dnn_input_columns


def test_normal_low_and_suspicious_high(rf_client):
    normal = rf_client.post("/predict", json=NORMAL_TXN).json()
    suspicious = rf_client.post("/predict", json=SUSPICIOUS_TXN).json()
    assert normal["fraud_probability"] < 25 and normal["alert_level"] == "Low Risk"
    assert suspicious["fraud_probability"] > normal["fraud_probability"] + 50
    assert suspicious["reasons"], "a high score must come with SHAP reasons"


def test_shap_explains_the_same_forest_that_scored(rf_pipeline):
    """Exact Tree SHAP: contributions + expected value == the forest's own output."""
    x = rf_pipeline.model_set.shap_background[:5]
    est = rf_pipeline.model_set.dnn_model.estimator
    sv = np.asarray(rf_pipeline.explainer.explainer.shap_values(x))
    sv = sv[..., 1] if sv.ndim == 3 else sv
    base = float(np.ravel(rf_pipeline.explainer.explainer.expected_value)[-1])
    assert np.allclose(sv.sum(axis=1) + base, est.predict_proba(x)[:, 1], atol=1e-6)


def test_scoring_is_deterministic(rf_pipeline):
    txn = {**SUSPICIOUS_TXN, "timestamp": BASE_TIME.replace(hour=2).isoformat()}
    snapshot = dict(rf_pipeline.customer_histories)
    a = rf_pipeline.score_transaction(txn)
    rf_pipeline.customer_histories.clear(); rf_pipeline.customer_histories.update(snapshot)
    b = rf_pipeline.score_transaction(txn)
    rf_pipeline.customer_histories.clear(); rf_pipeline.customer_histories.update(snapshot)
    assert a["fraud_probability"] == b["fraud_probability"] and a["reasons"] == b["reasons"]


def test_batch_scoring(rf_client):
    csv = ("customer_id,amount,merchant_category,device_id,location,failed_logins_24h\n"
           "CUST_0001,3800,fashion,DEV_0001_A,Pune,0\n"
           "CUST_0002,92000,electronics,DEV_UNKNOWN_5001,Lagos,5\n")
    r = rf_client.post("/predict/batch", files={"file": ("b.csv", csv, "text/csv")})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2 and len(body["results"]) == 2
    for row in body["results"]:
        assert 0 <= row["fraud_probability"] <= 99.9


def test_pdf_report(rf_client):
    pred = rf_client.post("/predict", json=SUSPICIOUS_TXN).json()
    r = rf_client.post("/report/pdf", json=pred)
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf"
    import io
    import re
    import pypdf
    text = re.sub(r"\s+", " ", " ".join(p.extract_text() or "" for p in pypdf.PdfReader(io.BytesIO(r.content)).pages))
    assert NAME in text and "random forest" in text and "Tree SHAP" in text
    assert "not a calibrated probability" in text and "LSTM risk score -> Random Forest" in text


def test_metrics_serve_the_model_selection_evidence(rf_client):
    m = rf_client.get("/metrics").json()
    assert m["model_set"] == NAME and m["dnn_fraud_classifier"] is None
    de = m["downstream_evaluation"]
    assert de["available"] is True and de["final_holdout"]["confirmed"] is True
    families = [r["family"] for r in de["development"]["comparison"]]
    assert families == ["dnn", "logistic_regression", "random_forest", "hist_gradient_boosting"]
    assert [r["family"] for r in de["development"]["comparison"] if r["selected"]] == ["random_forest"]
    assert de["development"]["decision"]["winner"] == "random_forest"


def test_downstream_evaluation_refuses_other_model_sets(pipeline):
    assert downstream_evaluation(pipeline.model_set)["available"] is False
