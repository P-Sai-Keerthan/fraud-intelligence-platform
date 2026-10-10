"""MODEL_SET selection (app/model_sets.py) and the inference pipeline /
API with each model set.

Most tests use small, untrained stand-in candidates written to a temp
directory by the real candidate-saving code (training.candidates._save), so
they need no local artifacts. Tests marked `real_candidates` use the trained
v2 candidates in models/candidates/v2/ (kept local, not committed) and are
skipped when those are absent. Nothing here writes into models/saved/ or
models/candidates/; the production checksums are compared before and after.
"""

import hashlib
import json
import random
import shutil
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import config, inference_pipeline
from app import model_sets as ms
from app.evaluation.stacking import CLIP, DNNScorer, LSTMScorer
from app.features.feature_engineering import FEATURE_COLUMNS
from app.inference_pipeline import FraudIntelligencePipeline
from app.main import app
from app.models.dnn_model import DNN_INPUT_COLUMNS, build_dnn_model
from app.models.lstm_model import build_lstm_model
from app.models.shap_explainer import FraudExplainer
from app.training import candidates as cand

from conftest import NORMAL_TXN, SUSPICIOUS_TXN

REAL_ROOT = config.CANDIDATES_DIR / "v2"
HAVE_REAL = all((REAL_ROOT / c / "manifest.json").exists() for c in cand.CANDIDATES)
real_candidates = pytest.mark.skipif(not HAVE_REAL, reason="trained v2 candidates are not present locally")

PREDICT_KEYS = {"transaction_id", "customer_id", "timestamp", "amount", "merchant_category", "device_id",
                "location", "failed_logins_24h", "risk_score", "fraud_probability", "alert_level",
                "similarity_pct", "deviation_pct", "similarity_status", "history_transactions", "reasons"}
ALERT_LEVELS = {"Low Risk", "Medium Risk", "High Risk", "Critical Risk"}


def _tree_sha(root: Path) -> dict:
    return {f.relative_to(root).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(root.rglob("*")) if f.is_file()}


# ---- stand-in candidates ----------------------------------------------------------------------

def build_standin_candidates(root: Path) -> Path:
    """Both candidate directories, untrained models, saved by candidates._save
    so the manifests have exactly the production format."""
    import tensorflow as tf
    tf.keras.utils.set_random_seed(0)
    rng = np.random.default_rng(0)
    n, k = 240, len(FEATURE_COLUMNS)
    F = rng.normal(size=(n, k)).astype(np.float32)
    y = (np.arange(n) % 7 == 0).astype(np.float32)
    tr, va = np.arange(n) < 160, np.arange(n) >= 160
    common = {
        "dataset": {"version": "v2", "file": "transactions_with_features.csv", "sha256": "0" * 64,
                    "generator_version": "test"},
        "split": {"file": "split_time.json", "sha256": "0" * 64},
        "features": {"columns": list(FEATURE_COLUMNS), "count": k,
                     "sha256": ms.feature_list_sha256(FEATURE_COLUMNS)},
        "seed": 0,
        "sequence_length": config.SEQUENCE_LENGTH,
        "clipping": {"dnn_training": "none (scaled only)",
                     "dnn_scoring": f"scaled inputs clipped to +-{CLIP:g} (as in 4C-2d and production inference)",
                     "lstm": "scaled only, never clipped"},
    }
    lstm = LSTMScorer(build_lstm_model(config.SEQUENCE_LENGTH, k), np.zeros(k, np.float32),
                      np.ones(k, np.float32), {})
    lstm_p = rng.random(n).astype(np.float32)
    XD = np.column_stack([F, np.round(lstm_p * 100, 2)]).astype(np.float32)
    dnn_a = DNNScorer(build_dnn_model(k + 1), XD[tr].mean(0), XD[tr].std(0), {})
    cand._save("dnn_lstm", root / "dnn_lstm", common, dnn_a, XD[tr], y, tr, va, dnn_a.predict(XD), 0,
               input_columns=list(FEATURE_COLUMNS) + ["risk_score"], lstm=lstm, lstm_p=lstm_p, n_folds=2,
               log=lambda m: None)
    dnn_b = DNNScorer(build_dnn_model(k), F[tr].mean(0), F[tr].std(0), {})
    cand._save("dnn_only", root / "dnn_only", common, dnn_b, F[tr], y, tr, va, dnn_b.predict(F), 0,
               input_columns=list(FEATURE_COLUMNS), log=lambda m: None)
    return root


@pytest.fixture(scope="module", autouse=True)
def production_untouched():
    before = cand.production_checksums()
    yield
    cand.verify_production_unchanged(before, cand.production_checksums())


@pytest.fixture(scope="module")
def standin_root(tmp_path_factory):
    return build_standin_candidates(tmp_path_factory.mktemp("standin") / "v2")


@pytest.fixture
def standin_copy(standin_root, tmp_path):
    """A private copy a test may tamper with."""
    return Path(shutil.copytree(standin_root, tmp_path / "v2"))


@pytest.fixture(scope="module")
def pipelines(standin_root):
    return {name: FraudIntelligencePipeline(model_set=name, candidates_root=standin_root)
            for name in ("v2_dnn_lstm", "v2_dnn_only")}


class RecordingModel:
    """Stands in for the pipeline's DNN and records the inputs it is given."""

    def __init__(self, model):
        self.model, self.inputs = model, []

    def predict(self, x, **kw):
        self.inputs.append(np.array(x))
        return self.model.predict(x, **kw)


def _score_recording(p, txn):
    """(response, the DNN inputs recorded while scoring txn)."""
    real = p.dnn_model
    p.dnn_model = RecordingModel(real)
    try:
        out = p.score_transaction(txn)
        inputs = p.dnn_model.inputs
    finally:
        p.dnn_model = real
    return out, inputs


def _edit_manifest(root: Path, candidate: str, fn) -> None:
    path = root / candidate / "manifest.json"
    manifest = json.loads(path.read_text())
    fn(manifest)
    path.write_text(json.dumps(manifest))


# ---- MODEL_SET resolution -------------------------------------------------------------------

def test_default_is_the_4d_model_and_production_is_unchanged(monkeypatch, pipeline):
    monkeypatch.delenv("MODEL_SET", raising=False)
    assert ms.resolve_model_set_name() == "v2_lstm_rf_seed14"       # Step 4D default
    # the test suite's shared pipeline (MODEL_SET=production, conftest.py) loads models/saved/ exactly as before
    assert pipeline.model_set.name == "production" and pipeline.model_set.uses_lstm
    assert pipeline.model_set.directory == config.MODELS_SAVED_DIR
    assert pipeline.model_set.files == {
        "lstm_model": config.LSTM_MODEL_PATH, "dnn_model": config.DNN_MODEL_PATH,
        "lstm_mean": config.LSTM_FEATURE_MEAN_PATH, "lstm_std": config.LSTM_FEATURE_STD_PATH,
        "dnn_mean": config.DNN_FEATURE_MEAN_PATH, "dnn_std": config.DNN_FEATURE_STD_PATH,
        "shap_background": config.SHAP_BACKGROUND_PATH}
    assert pipeline.model_set.dnn_input_columns == DNN_INPUT_COLUMNS
    assert pipeline.model_set.manifest is None


def test_explicit_production(monkeypatch):
    monkeypatch.setenv("MODEL_SET", "production")
    assert ms.resolve_model_set_name() == "production"
    assert ms.model_set_directory("production") == config.MODELS_SAVED_DIR


def test_candidate_directories():
    assert ms.model_set_directory("v2_dnn_lstm") == (config.CANDIDATES_DIR / "v2" / "dnn_lstm").resolve()
    assert ms.model_set_directory("v2_dnn_only") == (config.CANDIDATES_DIR / "v2" / "dnn_only").resolve()


@pytest.mark.parametrize("value", ["", "v2", "PRODUCTION", " production", "v2_dnn_lstm ", "dnn_only", "v1"])
def test_invalid_model_set_fails(monkeypatch, value):
    monkeypatch.setenv("MODEL_SET", value)
    with pytest.raises(ms.ModelSetError, match="invalid environment variable MODEL_SET"):
        ms.resolve_model_set_name()
    with pytest.raises(ms.ModelSetError):
        FraudIntelligencePipeline()


def test_invalid_model_set_stops_application_startup(monkeypatch):
    monkeypatch.setattr(inference_pipeline, "_pipeline_instance", None)
    monkeypatch.setenv("MODEL_SET", "v2_dnn_lstmm")
    with pytest.raises(ms.ModelSetError, match="v2_dnn_lstmm"):
        with TestClient(app):
            pass
    assert inference_pipeline._pipeline_instance is None


def test_requested_candidate_never_falls_back_to_production(monkeypatch, tmp_path):
    """An explicitly requested candidate whose files are missing stops startup."""
    monkeypatch.setattr(config, "CANDIDATES_DIR", tmp_path)            # empty: no candidates here
    monkeypatch.setattr(inference_pipeline, "_pipeline_instance", None)
    monkeypatch.setenv("MODEL_SET", "v2_dnn_only")
    with pytest.raises(ms.ModelSetError, match="manifest .* not found"):
        with TestClient(app):
            pass
    assert inference_pipeline._pipeline_instance is None


# ---- loading each model set ---------------------------------------------------------------------

def test_explicit_v2_dnn_lstm_via_environment(monkeypatch, standin_root):
    monkeypatch.setenv("MODEL_SET", "v2_dnn_lstm")
    p = FraudIntelligencePipeline(candidates_root=standin_root)
    m = p.model_set
    assert m.name == "v2_dnn_lstm" and m.uses_lstm and m.directory == (standin_root / "dnn_lstm").resolve()
    assert m.dnn_input_columns == list(FEATURE_COLUMNS) + ["risk_score"]
    assert m.lstm_model.input_shape[1:] == (config.SEQUENCE_LENGTH, len(FEATURE_COLUMNS))
    assert m.dnn_model.input_shape[1:] == (len(FEATURE_COLUMNS) + 1,)
    assert all(Path(f).parent == m.directory for f in m.files.values())


def test_explicit_v2_dnn_only_via_environment(monkeypatch, standin_root):
    monkeypatch.setenv("MODEL_SET", "v2_dnn_only")
    p = FraudIntelligencePipeline(candidates_root=standin_root)
    m = p.model_set
    assert m.name == "v2_dnn_only" and not m.uses_lstm
    assert m.lstm_model is None and p.lstm_model is None and m.lstm_mean is None
    assert "lstm_model" not in m.files
    assert m.dnn_input_columns == list(FEATURE_COLUMNS)
    assert m.dnn_model.input_shape[1:] == (len(FEATURE_COLUMNS),)


def test_dnn_only_does_not_need_lstm_files(standin_copy):
    """The DNN-only directory has no LSTM files at all, and scoring never touches an LSTM."""
    assert not any("lstm" in f.name for f in (standin_copy / "dnn_only").iterdir())
    p = FraudIntelligencePipeline(model_set="v2_dnn_only", candidates_root=standin_copy)
    out = p.score_transaction(dict(SUSPICIOUS_TXN))
    assert out["risk_score"] == out["fraud_probability"]


def _latest(txn, when):
    """txn, timestamped after everything already in the stand-in pipelines' histories."""
    return {**txn, "timestamp": when}


def test_dnn_lstm_scoring_matches_manual_preprocessing(pipelines):
    """risk_score = LSTM probability x 100 (rounded to 2) on the 10 prior feature
    rows, scaled and not clipped; DNN input = 9 features + risk_score, scaled
    and clipped to +-6."""
    p = pipelines["v2_dnn_lstm"]
    prior = p.customer_histories["CUST_0005"][FEATURE_COLUMNS].to_numpy(np.float32)[-config.SEQUENCE_LENGTH:]
    assert len(prior) == config.SEQUENCE_LENGTH
    out, inputs = _score_recording(p, _latest(NORMAL_TXN | {"customer_id": "CUST_0005"}, "2027-01-05T10:00:00"))
    lstm_prob = float(p.lstm_model.predict(((prior - p.lstm_mean) / p.lstm_std)[None], verbose=0)[0][0])
    assert out["risk_score"] == round(lstm_prob * 100, 2)
    hist = p.customer_histories["CUST_0005"]
    assert hist["amount"].iloc[-1] == NORMAL_TXN["amount"]
    features = hist[FEATURE_COLUMNS].to_numpy(np.float32)[-1]
    expected = np.clip((np.append(features, out["risk_score"]).astype(np.float32) - p.dnn_mean) / p.dnn_std,
                       -CLIP, CLIP)
    assert len(inputs) == 1 and inputs[0].shape == (1, len(FEATURE_COLUMNS) + 1)
    np.testing.assert_array_equal(inputs[0][0], expected)


def test_dnn_only_scoring_matches_manual_preprocessing(pipelines):
    p = pipelines["v2_dnn_only"]
    out, inputs = _score_recording(p, _latest(SUSPICIOUS_TXN | {"customer_id": "CUST_0006"}, "2027-01-06T03:00:00"))
    hist = p.customer_histories["CUST_0006"]
    assert hist["amount"].iloc[-1] == SUSPICIOUS_TXN["amount"]
    features = hist[FEATURE_COLUMNS].to_numpy(np.float32)[-1]
    expected = np.clip((features - p.dnn_mean) / p.dnn_std, -CLIP, CLIP)
    assert len(inputs) == 1 and inputs[0].shape == (1, len(FEATURE_COLUMNS))
    np.testing.assert_array_equal(inputs[0][0], expected)
    prob = min(float(p.dnn_model.predict(expected[None], verbose=0)[0][0]), 0.999)
    assert out["fraud_probability"] == round(prob * 100, 2) == out["risk_score"]


# ---- validation failures -----------------------------------------------------------------------------

@pytest.mark.parametrize("name,candidate,fname", [
    ("v2_dnn_only", "dnn_only", "dnn_feature_std.npy"),
    ("v2_dnn_only", "dnn_only", "dnn_fraud_model.keras"),
    ("v2_dnn_only", "dnn_only", "shap_background.npy"),
    ("v2_dnn_lstm", "dnn_lstm", "lstm_risk_model.keras"),
    ("v2_dnn_lstm", "dnn_lstm", "lstm_feature_mean.npy"),
])
def test_missing_candidate_artifact(standin_copy, name, candidate, fname):
    (standin_copy / candidate / fname).unlink()
    with pytest.raises(ms.ModelSetError, match=f"missing artifact .*{fname}"):
        ms.load_model_set(name, candidates_root=standin_copy)


def test_missing_manifest(standin_copy):
    (standin_copy / "dnn_lstm" / "manifest.json").unlink()
    with pytest.raises(ms.ModelSetError, match="manifest .* not found"):
        ms.load_model_set("v2_dnn_lstm", candidates_root=standin_copy)


def test_file_not_listed_in_manifest(standin_copy):
    _edit_manifest(standin_copy, "dnn_lstm", lambda m: m["files"].pop("lstm_feature_std.npy"))
    with pytest.raises(ms.ModelSetError, match="does not list the required file lstm_feature_std.npy"):
        ms.load_model_set("v2_dnn_lstm", candidates_root=standin_copy)


@pytest.mark.parametrize("candidate,name,edit,message", [
    ("dnn_only", "v2_dnn_only", lambda m: m.__setitem__("candidate", "dnn_lstm"), "manifest is for candidate"),
    ("dnn_only", "v2_dnn_only", lambda m: m["dataset"].__setitem__("version", "v1"), "dataset version"),
    ("dnn_only", "v2_dnn_only", lambda m: m["dataset"].pop("sha256"), "no dataset sha256"),
    ("dnn_only", "v2_dnn_only", lambda m: m["model"].__setitem__("dnn_weights_sha256", "0" * 64),
     "DNN weights do not match"),
    ("dnn_lstm", "v2_dnn_lstm", lambda m: m["model"].__setitem__("lstm_weights_sha256", "0" * 64),
     "LSTM weights do not match"),
    ("dnn_only", "v2_dnn_only", lambda m: m["files"].__setitem__("dnn_feature_mean.npy", "0" * 64),
     "does not match its manifest checksum"),
    ("dnn_only", "v2_dnn_only", lambda m: m["features"].__setitem__("sha256", "0" * 64), "feature list hash"),
    ("dnn_only", "v2_dnn_only", lambda m: m["features"].__setitem__("count", 10), "feature count"),
    ("dnn_only", "v2_dnn_only", lambda m: m["features"]["columns"].append("amount"), "feature list mismatch"),
    ("dnn_only", "v2_dnn_only", lambda m: m.__setitem__("sequence_length", 5), "sequence length"),
    ("dnn_lstm", "v2_dnn_lstm", lambda m: m["clipping"].__setitem__("lstm", "clipped to +-6"), "clipping"),
    ("dnn_only", "v2_dnn_only", lambda m: m["model"]["dnn_input_columns"].append("risk_score"),
     "DNN input columns"),
    ("dnn_lstm", "v2_dnn_lstm", lambda m: m["model"]["dnn_input_columns"].remove("risk_score"),
     "DNN input columns"),
    ("dnn_only", "v2_dnn_only", lambda m: m["model"]["dnn_input_columns"].__setitem__(-1, "is_fraud"),
     "DNN input columns"),
])
def test_manifest_mismatch(standin_copy, candidate, name, edit, message):
    _edit_manifest(standin_copy, candidate, edit)
    with pytest.raises(ms.ModelSetError, match=message):
        ms.load_model_set(name, candidates_root=standin_copy)


def test_candidate_directory_swap_is_refused(standin_copy):
    """v2_dnn_lstm pointed at DNN-only files (or the reverse) fails validation."""
    swapped = standin_copy.parent / "swapped"
    shutil.copytree(standin_copy / "dnn_only", swapped / "dnn_lstm")
    shutil.copytree(standin_copy / "dnn_lstm", swapped / "dnn_only")
    for name in ("v2_dnn_lstm", "v2_dnn_only"):
        with pytest.raises(ms.ModelSetError, match="manifest is for candidate"):
            ms.load_model_set(name, candidates_root=swapped)


def test_feature_order_mismatch(standin_copy):
    def swap(m):
        cols = m["features"]["columns"]
        cols[0], cols[1] = cols[1], cols[0]
    _edit_manifest(standin_copy, "dnn_lstm", swap)
    with pytest.raises(ms.ModelSetError, match="feature order mismatch"):
        ms.load_model_set("v2_dnn_lstm", candidates_root=standin_copy)


def test_dnn_input_order_mismatch(standin_copy):
    def swap(m):
        cols = m["model"]["dnn_input_columns"]
        cols[2], cols[3] = cols[3], cols[2]
    _edit_manifest(standin_copy, "dnn_only", swap)
    with pytest.raises(ms.ModelSetError, match="DNN input columns"):
        ms.load_model_set("v2_dnn_only", candidates_root=standin_copy)


def test_manifest_input_shape_mismatch(standin_copy):
    _edit_manifest(standin_copy, "dnn_only", lambda m: m["model"].__setitem__("dnn_input_shape", [None, 10]))
    with pytest.raises(ms.ModelSetError, match="dnn_input_shape"):
        ms.load_model_set("v2_dnn_only", candidates_root=standin_copy)
    _edit_manifest(standin_copy, "dnn_lstm", lambda m: m["model"].__setitem__("lstm_input_shape", [None, 5, 9]))
    with pytest.raises(ms.ModelSetError, match="lstm_input_shape"):
        ms.load_model_set("v2_dnn_lstm", candidates_root=standin_copy)


def _replace_file(root, candidate, fname, write):
    path = root / candidate / fname
    write(path)
    _edit_manifest(root, candidate, lambda m: m["files"].__setitem__(fname, cand._sha256(path)))


def test_model_input_shape_mismatch(standin_copy):
    """A (correctly checksummed) DNN that takes the wrong number of inputs is refused."""
    wrong = build_dnn_model(len(FEATURE_COLUMNS) + 1)
    _replace_file(standin_copy, "dnn_only", "dnn_fraud_model.keras", wrong.save)
    _edit_manifest(standin_copy, "dnn_only",
                   lambda m: m["model"].__setitem__("dnn_weights_sha256", cand.weights_sha256(wrong)))
    with pytest.raises(ms.ModelSetError, match="DNN input shape"):
        ms.load_model_set("v2_dnn_only", candidates_root=standin_copy)


def test_lstm_input_shape_mismatch(standin_copy):
    wrong = build_lstm_model(5, len(FEATURE_COLUMNS))
    _replace_file(standin_copy, "dnn_lstm", "lstm_risk_model.keras", wrong.save)
    _edit_manifest(standin_copy, "dnn_lstm",
                   lambda m: m["model"].__setitem__("lstm_weights_sha256", cand.weights_sha256(wrong)))
    with pytest.raises(ms.ModelSetError, match="LSTM input shape"):
        ms.load_model_set("v2_dnn_lstm", candidates_root=standin_copy)


@pytest.mark.parametrize("fname,shape,message", [
    ("dnn_feature_mean.npy", (10,), "dnn mean has shape"),
    ("shap_background.npy", (50, 10), "SHAP background has shape"),
])
def test_scaler_and_background_shape_mismatch(standin_copy, fname, shape, message):
    _replace_file(standin_copy, "dnn_only", fname, lambda p: np.save(p, np.ones(shape, np.float32)))
    with pytest.raises(ms.ModelSetError, match=message):
        ms.load_model_set("v2_dnn_only", candidates_root=standin_copy)


# ---- API compatibility ---------------------------------------------------------------------------

@pytest.fixture
def use_pipeline(monkeypatch):
    def swap(p):
        monkeypatch.setattr(inference_pipeline, "_pipeline_instance", p)
    return swap


@pytest.mark.parametrize("name", ["v2_dnn_lstm", "v2_dnn_only"])
def test_predict_response_compatible(client, pipelines, use_pipeline, name):
    txns = (NORMAL_TXN, SUSPICIOUS_TXN)
    reference = [client.post("/predict", json=dict(t)).json() for t in txns]
    use_pipeline(pipelines[name])
    for txn, prod in zip(txns, reference):
        assert set(prod) == PREDICT_KEYS
        r = client.post("/predict", json=dict(txn))
        assert r.status_code == 200
        body = r.json()
        assert set(body) == PREDICT_KEYS
        # same field types as production (integers such as 0.0 serialize as floats either way)
        for key in ("risk_score", "fraud_probability", "similarity_pct", "deviation_pct", "amount"):
            assert isinstance(body[key], (int, float)) and isinstance(prod[key], (int, float))
        for key in ("transaction_id", "customer_id", "timestamp", "merchant_category", "device_id", "location",
                    "alert_level"):
            assert isinstance(body[key], str)
        assert isinstance(body["failed_logins_24h"], int)
        assert 0 <= body["risk_score"] <= 100 and 0 <= body["fraud_probability"] <= 99.9
        assert body["alert_level"] in ALERT_LEVELS
        assert isinstance(body["reasons"], list)
        for reason in body["reasons"]:
            assert set(reason) == {"feature", "display_name", "shap_value"}
            assert reason["feature"] in pipelines[name].model_set.dnn_input_columns


@pytest.mark.parametrize("name", ["v2_dnn_lstm", "v2_dnn_only"])
def test_persistence_history_batch_and_pdf(client, pipelines, use_pipeline, name):
    use_pipeline(pipelines[name])
    pred = client.post("/predict", json=dict(SUSPICIOUS_TXN)).json()
    # persisted and visible in the customer's timeline with the same values
    timeline = client.get("/customer/CUST_0001/history").json()["timeline"]
    saved = [t for t in timeline if t["transaction_id"] == pred["transaction_id"]]
    assert len(saved) == 1
    assert saved[0]["risk_score"] == pred["risk_score"]
    assert saved[0]["fraud_probability"] == pred["fraud_probability"]
    assert saved[0]["alert_level"] == pred["alert_level"]
    # PDF from exactly what /predict returned
    pdf = client.post("/report/pdf", json=pred)
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF")
    # batch scoring
    csv = ("customer_id,amount,merchant_category,device_id,location,failed_logins_24h\n"
           "CUST_0002,3500,fuel,DEV_0002_A,Chennai,0\nCUST_0003,90000,electronics,DEV_UNKNOWN_4242,Lagos,5\n")
    r = client.post("/predict/batch", files={"file": ("b.csv", csv.encode(), "text/csv")})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2 and set(body["summary"]) == ALERT_LEVELS
    for row in body["results"]:
        assert set(row) == {"transaction_id", "customer_id", "amount", "risk_score", "fraud_probability",
                            "alert_level"}
        assert isinstance(row["risk_score"], (int, float)) and row["alert_level"] in ALERT_LEVELS
        if name == "v2_dnn_only":
            assert row["risk_score"] == row["fraud_probability"]


# ---- SHAP -------------------------------------------------------------------------------------------

def test_shap_feature_names_follow_the_model_set(pipeline, pipelines):
    assert pipeline.explainer.feature_names == DNN_INPUT_COLUMNS                     # production unchanged
    assert pipelines["v2_dnn_lstm"].explainer.feature_names == list(FEATURE_COLUMNS) + ["risk_score"]
    assert pipelines["v2_dnn_only"].explainer.feature_names == list(FEATURE_COLUMNS)
    assert "risk_score" not in pipelines["v2_dnn_only"].explainer.feature_names


@pytest.mark.parametrize("name", ["v2_dnn_lstm", "v2_dnn_only"])
def test_shap_reasons_use_only_model_inputs(pipelines, name):
    p = pipelines[name]
    x = np.clip(p.model_set.shap_background[:1] + 3.0, -CLIP, CLIP)
    reasons = p.explainer.explain(x, top_k=len(p.model_set.dnn_input_columns))
    assert {r["feature"] for r in reasons} <= set(p.model_set.dnn_input_columns)


def test_shap_values_are_attributed_to_the_right_feature():
    """A linear model that only looks at one input: every reason must name that input."""
    import tensorflow as tf
    target = FEATURE_COLUMNS.index("failed_logins_24h")
    model = tf.keras.Sequential([tf.keras.Input(shape=(len(FEATURE_COLUMNS),)),
                                 tf.keras.layers.Dense(1, activation="sigmoid")])
    w = np.zeros((len(FEATURE_COLUMNS), 1), np.float32)
    w[target] = 1.0
    model.layers[-1].set_weights([w, np.zeros(1, np.float32)])
    explainer = FraudExplainer(model, np.zeros((20, len(FEATURE_COLUMNS)), np.float32),
                               feature_names=FEATURE_COLUMNS)
    x = np.full((1, len(FEATURE_COLUMNS)), 2.0, np.float32)
    reasons = explainer.explain(x, top_k=9)
    assert [r["feature"] for r in reasons] == ["failed_logins_24h"]
    assert reasons[0]["display_name"] == "Multiple Failed Logins"


def test_shap_refuses_misaligned_feature_names(pipelines):
    p = pipelines["v2_dnn_only"]
    with pytest.raises(ValueError, match="model takes 9 inputs but 10 feature names"):
        FraudExplainer(p.dnn_model, p.model_set.shap_background, feature_names=DNN_INPUT_COLUMNS)
    with pytest.raises(ValueError, match="background shape"):
        FraudExplainer(p.dnn_model, np.zeros((5, 10), np.float32), feature_names=FEATURE_COLUMNS)


# ---- production protection / switching ------------------------------------------------------------

def test_candidates_root_inside_production_is_refused():
    for root in (config.MODELS_SAVED_DIR, config.MODELS_SAVED_DIR / "v2"):
        with pytest.raises(ms.ModelSetError, match="inside the production model directory"):
            ms.model_set_directory("v2_dnn_only", candidates_root=root)
        with pytest.raises(ms.ModelSetError, match="inside the production model directory"):
            ms.load_model_set("v2_dnn_lstm", candidates_root=root)


def test_switching_model_sets_does_not_modify_artifacts(pipeline, standin_root):
    prod_before = _tree_sha(config.MODELS_SAVED_DIR)
    standin_before = _tree_sha(standin_root)
    real_before = _tree_sha(REAL_ROOT) if HAVE_REAL else None
    sequence = ["production", "v2_dnn_lstm", "v2_dnn_only", "production"]
    outputs = []
    for name in sequence:
        p = FraudIntelligencePipeline(model_set=name, candidates_root=standin_root)
        assert p.model_set.name == name
        random.seed(5)
        np.random.seed(5)
        out = p.score_transaction(dict(SUSPICIOUS_TXN))
        out.pop("transaction_id")
        outputs.append(out)
    # production before and after the candidates: identical results
    assert json.dumps(outputs[0], default=str) == json.dumps(outputs[-1], default=str)
    # ...and identical to the application's own production pipeline
    random.seed(5)
    np.random.seed(5)
    snapshot = dict(pipeline.customer_histories)
    ref = pipeline.score_transaction(dict(SUSPICIOUS_TXN))
    pipeline.customer_histories.clear()
    pipeline.customer_histories.update(snapshot)
    ref.pop("transaction_id")
    assert json.dumps(ref, default=str) == json.dumps(outputs[0], default=str)
    assert _tree_sha(config.MODELS_SAVED_DIR) == prod_before
    assert _tree_sha(standin_root) == standin_before
    if HAVE_REAL:
        assert _tree_sha(REAL_ROOT) == real_before


# ---- the trained local v2 candidates -------------------------------------------------------------

@real_candidates
@pytest.mark.parametrize("name", ["v2_dnn_lstm", "v2_dnn_only"])
def test_real_candidates_load_and_validate(monkeypatch, name):
    before = _tree_sha(REAL_ROOT)
    monkeypatch.setenv("MODEL_SET", name)
    m = ms.load_model_set()
    assert m.name == name and m.directory == (REAL_ROOT / name.split("_", 1)[1]).resolve()
    assert m.manifest["dataset"]["version"] == "v2"
    assert m.manifest["features"]["sha256"] == ms.feature_list_sha256(FEATURE_COLUMNS)
    assert (m.lstm_model is not None) == (name == "v2_dnn_lstm")
    assert _tree_sha(REAL_ROOT) == before


REAL_EVAL_SCORES = config.EVALUATION_V2_DIR / "scores_time_split.csv.gz"


@real_candidates
@pytest.mark.skipif(not REAL_EVAL_SCORES.exists() or not (config.DATA_V2_DIR / "transactions_with_features.csv").exists(),
                    reason="v2 evaluation scores / v2 data are not present locally")
def test_real_candidates_reproduce_the_4c2d_evaluation_scores():
    """Pipeline scoring of real v2 test transactions (history rebuilt from the
    v2 rows before each one) gives the evaluation's scores for the same
    models: DNN+LSTM = dnn_fraud_classifier / lstm_risk_predictor,
    DNN-only = dnn_without_risk_score."""
    import pandas as pd
    from app.evaluation.datasets import load_evaluation_data, resolve_dataset
    from app.features.feature_engineering import build_point_features
    from app.inference_pipeline import RAW_COLUMNS_FOR_FEATURES
    df = load_evaluation_data(resolve_dataset("v2")).frame
    scores = pd.read_csv(REAL_EVAL_SCORES, float_precision="round_trip").set_index("transaction_id")
    test = scores[scores["split"] == "test"]
    ids = list(test[test["is_fraud"] == 1].index[:4]) + list(test[test["is_fraud"] == 0].sample(6, random_state=1).index)
    for name, col in (("v2_dnn_lstm", "dnn_fraud_classifier"), ("v2_dnn_only", "dnn_without_risk_score")):
        p = FraudIntelligencePipeline(model_set=name)
        checked = 0
        for tid in ids:
            row = df[df["transaction_id"] == tid].iloc[0]
            cust = df[df["customer_id"] == row["customer_id"]]
            prior = cust[cust.index < row.name]
            if prior["timestamp"].duplicated().any() or (prior["timestamp"] == row["timestamp"]).any():
                continue            # production's feature builder orders by timestamp only
            raw = prior[RAW_COLUMNS_FOR_FEATURES].copy()
            raw["is_fraud"] = 0
            p.customer_histories[row["customer_id"]] = build_point_features(raw)
            txn = {k: row[k] for k in ("customer_id", "amount", "merchant_category", "device_id", "location",
                                       "failed_logins_24h")}
            out = p.score_transaction({**txn, "timestamp": row["timestamp"]})
            assert out["fraud_probability"] == round(min(scores.loc[tid, col], 0.999) * 100, 2)
            if name == "v2_dnn_lstm":
                assert out["risk_score"] == round(scores.loc[tid, "lstm_risk_predictor"] * 100, 2)
            checked += 1
        assert checked >= 8
