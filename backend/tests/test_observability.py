"""Model-set provenance, model metadata, /metrics labelling and model-aware
PDF wording (Step 4C-2f-2).

Candidate model sets are exercised with the untrained stand-in candidates
from test_model_sets (no local artifacts needed) and, where marked, with the
trained local candidates in models/candidates/v2/."""

import io
import json
import re
import shutil

import pytest
from sqlalchemy import create_engine, inspect, text

from app import config, inference_pipeline
from app.db import database as db_module
from app.db import models as db_models
from app.db.migrations import ensure_schema
from app.inference_pipeline import FraudIntelligencePipeline
from app.model_metadata import current_feature_implementation, feature_list_sha256
from app.report import build_pdf_report

from conftest import NORMAL_TXN, SUSPICIOUS_TXN
from test_model_sets import build_standin_candidates

REAL_ROOT = config.CANDIDATES_DIR / "v2"
HAVE_REAL = all((REAL_ROOT / c / "manifest.json").exists() for c in ("dnn_lstm", "dnn_only")) \
    and (REAL_ROOT / "comparison.json").exists()
real = pytest.mark.skipif(not HAVE_REAL, reason="trained v2 candidates / comparison.json not present locally")


@pytest.fixture(scope="module")
def standin_root(tmp_path_factory):
    return build_standin_candidates(tmp_path_factory.mktemp("obs_standin") / "v2")


@pytest.fixture(scope="module")
def standin(standin_root):
    return {n: FraudIntelligencePipeline(model_set=n, candidates_root=standin_root)
            for n in ("v2_dnn_lstm", "v2_dnn_only")}


@pytest.fixture(scope="module")
def real_pipelines():
    if not HAVE_REAL:
        pytest.skip("trained v2 candidates are not present locally")
    return {n: FraudIntelligencePipeline(model_set=n) for n in ("v2_dnn_lstm", "v2_dnn_only")}


@pytest.fixture
def use_pipeline(monkeypatch):
    def swap(p):
        monkeypatch.setattr(inference_pipeline, "_pipeline_instance", p)
    return swap


def _row(transaction_id):
    session = db_module.SessionLocal()
    try:
        return session.get(db_models.Transaction, transaction_id)
    finally:
        session.close()


def _pdf_text(content):
    pypdf = pytest.importorskip("pypdf")
    reader = pypdf.PdfReader(io.BytesIO(content))
    return re.sub(r"\s+", " ", " ".join(page.extract_text() or "" for page in reader.pages))


# ---- migration --------------------------------------------------------------------------------------

OLD_SCHEMA = """CREATE TABLE transactions (
    transaction_id VARCHAR PRIMARY KEY, customer_id VARCHAR NOT NULL, timestamp DATETIME, amount FLOAT NOT NULL,
    merchant_category VARCHAR, device_id VARCHAR, location VARCHAR, failed_logins_24h INTEGER, risk_score FLOAT,
    fraud_probability FLOAT, similarity_pct FLOAT, deviation_pct FLOAT, alert_level VARCHAR, is_fraud_actual BOOLEAN)"""


def test_migration_adds_provenance_columns_to_an_old_database(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'old.db'}")
    with engine.begin() as conn:
        conn.execute(text(OLD_SCHEMA))
        conn.execute(text("INSERT INTO transactions (transaction_id, customer_id, timestamp, amount, risk_score, "
                          "fraud_probability, alert_level) VALUES ('TXN_OLD', 'CUST_0001', '2026-07-10 13:30:00', "
                          "3800, 20.0, 1.5, 'Low Risk')"))
    assert ensure_schema(engine) == ["transactions.model_set", "transactions.model_version", "transactions.reasons_json"]
    cols = {c["name"] for c in inspect(engine).get_columns("transactions")}
    assert {"model_set", "model_version", "reasons_json"} <= cols
    with engine.connect() as conn:
        row = conn.execute(text("SELECT risk_score, model_set, model_version FROM transactions")).one()
    assert row == (20.0, None, None)                     # readable; provenance NOT back-filled
    assert ensure_schema(engine) == []                    # idempotent
    engine.dispose()


def test_migration_on_a_new_database_is_a_no_op(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'new.db'}")
    db_module.Base.metadata.create_all(bind=engine)
    assert ensure_schema(engine) == []
    engine.dispose()


def test_migration_without_the_table_is_a_no_op(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'empty.db'}")
    assert ensure_schema(engine) == []
    engine.dispose()


def test_application_database_has_provenance_columns():
    cols = {c["name"] for c in inspect(db_module.engine).get_columns("transactions")}
    assert {"model_set", "model_version"} <= cols


# ---- provenance -----------------------------------------------------------------------------------------

def test_predict_records_production_provenance(client, pipeline):
    body = client.post("/predict", json=dict(NORMAL_TXN)).json()
    assert "model_set" not in body                        # /predict response shape unchanged
    row = _row(body["transaction_id"])
    assert row.model_set == "production"
    assert row.model_version == pipeline.model_version
    assert row.model_version.startswith("production-")


def test_batch_records_provenance(client, pipeline):
    csv = "customer_id,amount,merchant_category\nCUST_0002,3500,fuel\nCUST_0003,90000,electronics\n"
    body = client.post("/predict/batch", files={"file": ("b.csv", csv.encode(), "text/csv")}).json()
    for r in body["results"]:
        row = _row(r["transaction_id"])
        assert (row.model_set, row.model_version) == ("production", pipeline.model_version)


@pytest.mark.parametrize("name", ["v2_dnn_lstm", "v2_dnn_only"])
def test_candidate_provenance(client, standin, use_pipeline, name):
    use_pipeline(standin[name])
    body = client.post("/predict", json=dict(SUSPICIOUS_TXN)).json()
    csv = "customer_id,amount,merchant_category\nCUST_0002,3500,fuel\n"
    batch = client.post("/predict/batch", files={"file": ("b.csv", csv.encode(), "text/csv")}).json()
    for tid in (body["transaction_id"], batch["results"][0]["transaction_id"]):
        row = _row(tid)
        assert row.model_set == name
        assert row.model_version == standin[name].model_version
        assert row.model_version.startswith(name + "-")


def test_restore_preserves_provenance_and_does_not_backfill(client, pipeline):
    body = client.post("/predict", json=dict(NORMAL_TXN)).json()
    session = db_module.SessionLocal()
    try:
        session.add(db_models.Transaction(
            transaction_id="TXN_LEGACY_ROW", customer_id="CUST_0001",
            timestamp=__import__("datetime").datetime(2026, 7, 9, 12, 0), amount=1200.0,
            merchant_category="grocery", device_id="DEV_0001_A", location="Pune", failed_logins_24h=0,
            risk_score=10.0, fraud_probability=1.0, alert_level="Low Risk"))       # no provenance
        session.commit()
    finally:
        session.close()
    from app.main import _restore_history_from_db
    snapshot = dict(pipeline.customer_histories)
    try:
        assert _restore_history_from_db(pipeline) == 2
        ids = set(pipeline.customer_histories["CUST_0001"]["transaction_id"])
        assert {"TXN_LEGACY_ROW", body["transaction_id"]} <= ids
    finally:
        pipeline.customer_histories.clear()
        pipeline.customer_histories.update(snapshot)
    assert _row("TXN_LEGACY_ROW").model_set is None                # not assigned
    assert _row(body["transaction_id"]).model_set == "production"  # kept


# ---- metadata ----------------------------------------------------------------------------------------------

def test_model_info_production(client, pipeline):
    info = client.get("/model-info").json()
    assert info == json.loads(json.dumps(pipeline.model_metadata))
    assert info["model_set"] == "production" and info["uses_lstm"] is True
    assert info["model_version"] == pipeline.model_version
    assert set(info["model"]["files_sha256"]) == {p.name for p in config.MODELS_SAVED_DIR.iterdir() if p.is_file()}
    assert info["dataset"]["version"] == "v1"
    assert info["score_semantics"]["calibrated_probabilities"] is False
    assert "not a calibrated probability" in info["score_semantics"]["fraud_probability"]
    assert "LSTM" in info["score_semantics"]["risk_score"]
    f = info["features"]
    assert f["training_feature_version"]["name"] == "legacy-v1"
    assert f["live_feature_version"] == current_feature_implementation()
    assert f["training_and_live_features_consistent"] is False
    assert f["list_sha256"] == feature_list_sha256()
    assert "candidate_thresholds" not in info["thresholds"]
    assert info["thresholds"]["alert_bands"]["Critical Risk"] == "fraud_probability >= 80"
    assert "not been approved" in info["thresholds"]["alert_bands_status"]


@pytest.mark.parametrize("name", ["v2_dnn_lstm", "v2_dnn_only"])
def test_model_info_candidates(client, standin, use_pipeline, name):
    use_pipeline(standin[name])
    info = client.get("/model-info").json()
    man = standin[name].model_set.manifest
    assert info["model_set"] == name and info["uses_lstm"] is (name == "v2_dnn_lstm")
    assert info["model"]["dnn_weights_sha256"] == man["model"]["dnn_weights_sha256"]
    assert len(info["model"]["manifest_sha256"]) == 64
    assert info["dataset"]["version"] == "v2" and info["dataset"]["sha256"] == man["dataset"]["sha256"]
    assert info["features"]["training_feature_version"]["name"] == "current"
    assert info["features"]["training_and_live_features_consistent"] is True
    ct = info["thresholds"]["candidate_thresholds"]
    assert ct["f1_optimal"] == man["thresholds"]["f1_optimal"]
    assert "NOT approved" in ct["status"] and "NOT applied" in ct["status"]
    if name == "v2_dnn_only":
        assert "LSTM" not in info["score_semantics"]["risk_score"].replace("No LSTM", "")
        assert "repeats the DNN fraud score" in info["score_semantics"]["risk_score"]


# ---- /metrics ------------------------------------------------------------------------------------------------

def test_metrics_production_is_labelled(client, pipeline):
    m = client.get("/metrics").json()
    assert m["model_set"] == "production" and m["model_version"] == pipeline.model_version
    assert m["evaluation"]["dataset_version"] == "v1"
    assert m["evaluation"]["applies_to_loaded_model_set"] is True
    assert "evaluation copies" in m["evaluation"]["evaluates"]
    assert m["alerting"]["alert_bands"]["High Risk"] == "50 <= fraud_probability < 80"
    assert m["lstm_risk_predictor"]["pr_auc"] is not None
    report = client.get("/metrics/report").json()
    assert report["model_set"] == "production" and "primary" in report


@pytest.mark.parametrize("name", ["v2_dnn_lstm", "v2_dnn_only"])
def test_metrics_candidate_never_serves_v1_numbers(client, standin, use_pipeline, name):
    use_pipeline(standin[name])
    m = client.get("/metrics").json()
    assert m["model_set"] == name
    assert m["lstm_risk_predictor"] is None and m["dnn_fraud_classifier"] is None
    assert "v1" in m["v1_metrics_withheld"]
    assert m["evaluation"]["dataset_version"] == "v2"
    # stand-ins have no comparison.json
    assert m["candidate_evaluation"] == {"available": False,
                                         "reason": m["candidate_evaluation"]["reason"]}
    assert m["evaluation"]["available"] is False
    report = client.get("/metrics/report").json()
    assert report["model_set"] == name and "primary" not in report and "v1_report_withheld" in report


@real
@pytest.mark.parametrize("name,key", [("v2_dnn_lstm", "candidate_a_dnn_lstm"), ("v2_dnn_only", "candidate_b_dnn_only")])
def test_metrics_candidate_evaluation_from_comparison(client, real_pipelines, use_pipeline, name, key):
    use_pipeline(real_pipelines[name])
    m = client.get("/metrics").json()
    comp = json.loads((REAL_ROOT / "comparison.json").read_text())
    ev = m["candidate_evaluation"]
    assert ev["available"] is True and m["evaluation"]["available"] is True
    assert ev["overall_test"] == comp["overall_test"][key]
    assert ev["operating_points"] == comp["operating_points"][key]
    assert ev["dataset"]["version"] == "v2"
    assert "not calibrated" in ev["scores_note"]
    assert m["lstm_risk_predictor"] is None and m["dnn_fraud_classifier"] is None


@real
def test_candidate_evaluation_refused_for_other_weights(real_pipelines, tmp_path):
    from app.model_metadata import candidate_evaluation
    shutil.copytree(REAL_ROOT, tmp_path / "v2")
    comp_path = tmp_path / "v2" / "comparison.json"
    comp = json.loads(comp_path.read_text())
    comp["candidates"]["candidate_b_dnn_only"]["dnn_weights_sha256"] = "0" * 64
    comp_path.write_text(json.dumps(comp))
    p = FraudIntelligencePipeline(model_set="v2_dnn_only", candidates_root=tmp_path / "v2")
    ev = candidate_evaluation(p.model_set)
    assert ev == {"available": False, "reason": "comparison.json was produced for different weights"}


# ---- PDF ----------------------------------------------------------------------------------------------------

def _pdf_for(client, txn=SUSPICIOUS_TXN):
    pred = client.post("/predict", json=dict(txn)).json()
    r = client.post("/report/pdf", json=pred)
    assert r.status_code == 200
    return pred, _pdf_text(r.content)


def test_pdf_production_wording(client, pipeline):
    _, text_ = _pdf_for(client)
    assert "Model set production" in text_ and pipeline.model_version in text_
    assert "Temporal behavioral risk" in text_ and "production LSTM" in text_
    assert "not a calibrated probability" in text_


@pytest.mark.parametrize("source", ["standin", "real"])
def test_pdf_v2_dnn_lstm_wording(client, standin, request, use_pipeline, source):
    pipes = standin if source == "standin" else request.getfixturevalue("real_pipelines")
    use_pipeline(pipes["v2_dnn_lstm"])
    _, text_ = _pdf_for(client)
    assert "Model set v2_dnn_lstm" in text_
    assert "LSTM behavioral-risk component" in text_
    assert "Trajectory" not in text_ and "production" not in text_
    assert "not a calibrated probability" in text_


@pytest.mark.parametrize("source", ["standin", "real"])
def test_pdf_v2_dnn_only_never_claims_lstm(client, standin, request, use_pipeline, source):
    pipes = standin if source == "standin" else request.getfixturevalue("real_pipelines")
    use_pipeline(pipes["v2_dnn_only"])
    pred, text_ = _pdf_for(client)
    assert pred["risk_score"] == pred["fraud_probability"]
    assert "Model set v2_dnn_only" in text_
    assert "Risk Score (compatibility)" in text_
    assert "repeats the DNN fraud score" in text_
    assert "LSTM" not in text_ and "trajectory" not in text_.lower()


def test_pdf_uses_the_recorded_model_set_not_the_loaded_one(client, standin, use_pipeline):
    pred = client.post("/predict", json=dict(SUSPICIOUS_TXN)).json()      # scored by production
    use_pipeline(standin["v2_dnn_only"])                                   # then the loaded set changes
    text_ = _pdf_text(client.post("/report/pdf", json=pred).content)
    assert "Model set production" in text_ and "Temporal behavioral risk" in text_
    assert "v2_dnn_only" not in text_


def test_pdf_without_recorded_provenance_is_neutral(client):
    pred = client.post("/predict", json=dict(NORMAL_TXN)).json()
    session = db_module.SessionLocal()
    try:
        row = session.get(db_models.Transaction, pred["transaction_id"])
        row.model_set, row.model_version = None, None                     # a row from before provenance
        session.commit()
    finally:
        session.close()
    text_ = _pdf_text(client.post("/report/pdf", json=pred).content)
    assert "Model set not recorded" in text_ and "scored before model-set provenance was recorded" in text_
    assert "Trajectory" not in text_ and "LSTM" not in text_


def test_pdf_for_unknown_transaction_is_404_not_a_report_built_from_client_data(client):
    # before the audit this produced a neutral report from whatever the client sent; reports are now only
    # available for transactions that were scored and stored (compatibility change, see FIX_AND_TEST_LOG.md)
    pred = client.post("/predict", json=dict(NORMAL_TXN)).json()
    pred["transaction_id"] = "TXN_NOT_IN_DB"
    r = client.post("/report/pdf", json=pred)
    assert r.status_code == 404


def test_build_pdf_report_default_is_neutral():
    pdf = build_pdf_report({"transaction_id": "T", "customer_id": "C", "amount": 1.0, "risk_score": 1.0,
                            "fraud_probability": 1.0, "alert_level": "Low Risk", "reasons": []})
    text_ = _pdf_text(pdf)
    assert "Model set not recorded" in text_ and "LSTM" not in text_
