"""R-02: alert levels of the default random-forest model set come from its validated cut-offs.

Decision record: docs/R-02-alert-score-decision.md. The legacy 25/50/80 bands stay for every other model set.
Nothing here calibrates a score; the protected evaluation reports are only read."""

import json

import pytest

from app import config
from app import inference_pipeline as ip
from app import model_sets as ms
from app.alerting import ALL_LEVELS, LEGACY, LEGACY_POLICY, VALIDATED, policy_for, validated_policy
from app.models.dnn_model import alert_level_from_probability
from conftest import NORMAL_TXN, SUSPICIOUS_TXN

RF_DIR = config.BACKEND_DIR / "models" / "candidates_downstream" / "v2" / "seed_14" / "lstm_random_forest"
DOWNSTREAM = config.BACKEND_DIR / "models" / "evaluation" / "downstream"
CUT = ms.RF_FROZEN_CUTOFFS
POLICY = validated_policy(CUT)


# ---- the policy itself ------------------------------------------------------------------------------

def test_validated_tiers_follow_the_frozen_cutoffs():
    a, c = CUT["policy_b"], CUT["critical"]
    assert POLICY.level(0.0) == "Low Risk"
    assert POLICY.level(a - 1e-9) == "Low Risk"
    assert POLICY.level(a) == "Medium Risk"                 # the cut-off itself alerts (score >= cut-off)
    assert POLICY.level(0.0601) == "Medium Risk"            # the audit's "typical purchase" (6.01): above the alert cut-off
    assert POLICY.level(c - 1e-9) == "Medium Risk"
    assert POLICY.level(c) == "Critical Risk"
    assert POLICY.level(0.999) == "Critical Risk"


def test_high_risk_is_never_emitted_by_the_validated_policy():
    scores = [i / 1000 for i in range(0, 1000)]
    assert {POLICY.level(s) for s in scores} == {"Low Risk", "Medium Risk", "Critical Risk"}


def test_legacy_policy_is_the_old_function_exactly():
    scores = [i / 1000 for i in range(0, 1000)] + [0.2499, 0.25, 0.4999, 0.5, 0.7999, 0.8]
    assert all(LEGACY_POLICY.level(s) == alert_level_from_probability(s) for s in scores)


def test_only_the_default_random_forest_gets_validated_cutoffs():
    assert policy_for(ms.RF_MODEL_SET, CUT).kind == VALIDATED
    for name in ("production", "v2_dnn_lstm", "v2_dnn_only", ms.SEED14_MODEL_SET):
        assert policy_for(name, None) is LEGACY_POLICY and LEGACY_POLICY.kind == LEGACY
        assert policy_for(name, ms.SEED14_FROZEN_CUTOFFS) is LEGACY_POLICY          # frozen cut-offs alone do not switch a set


def test_band_texts_describe_the_actual_cutoffs():
    bands = POLICY.bands()
    assert set(bands) == set(ALL_LEVELS)
    assert bands["Low Risk"] == "fraud_probability < 4.39"
    assert bands["Medium Risk"] == "4.39 <= fraud_probability < 26.23"
    assert bands["Critical Risk"] == "fraud_probability >= 26.23"
    assert "not used" in bands["High Risk"]


# ---- the cut-offs are the recorded, protected ones -------------------------------------------------

def test_cutoffs_equal_the_manifest_and_the_selection_and_final_records():
    manifest = json.loads((RF_DIR / "manifest.json").read_text())
    assert manifest["cutoffs_validation"] == {"policy_b": CUT["policy_b"], "critical": CUT["critical"]}
    record = json.loads((DOWNSTREAM / "selection_record.json").read_text())
    assert record["deploy"]["cutoffs"] == {"policy_b": CUT["policy_b"], "critical": CUT["critical"]}
    final = json.loads((DOWNSTREAM / "final_holdout_report.json").read_text())
    assert final["cutoffs"]["policy_b"]["selected"] == CUT["policy_b"]
    assert final["cutoffs"]["critical"]["selected"] == CUT["critical"]


def test_evaluation_code_does_not_use_the_serving_policy():
    """The evaluation protocols keep their own frozen tables; serving policy changes cannot move them."""
    from app.evaluation import holdout
    assert holdout.CUTOFFS["policy_b"]["production"] == 0.9680148363113403
    import app.evaluation as pkg, pathlib
    for f in pathlib.Path(pkg.__file__).parent.glob("*.py"):
        assert "from ..alerting" not in f.read_text() and "from .. import alerting" not in f.read_text(), f.name


# ---- end to end: the API ----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mp_module():
    m = pytest.MonkeyPatch()
    yield m
    m.undo()


@pytest.fixture(scope="module")
def rf_pipeline(mp_module):
    mp_module.delenv("MODEL_SET", raising=False)
    return ip.FraudIntelligencePipeline()


@pytest.fixture
def rf_client(client, rf_pipeline, monkeypatch):
    snapshot = dict(rf_pipeline.customer_histories)
    monkeypatch.setattr(ip, "_pipeline_instance", rf_pipeline)
    yield client
    rf_pipeline.customer_histories.clear()
    rf_pipeline.customer_histories.update(snapshot)


def test_pipeline_uses_the_validated_policy(rf_pipeline):
    assert rf_pipeline.alert_policy.kind == VALIDATED


def test_predict_levels_match_the_policy(rf_client):
    for txn in (NORMAL_TXN, SUSPICIOUS_TXN):
        body = rf_client.post("/predict", json=txn).json()
        score = body["fraud_probability"]
        expected = POLICY.level(score / 100)
        margin = min(abs(score - CUT["policy_b"] * 100), abs(score - CUT["critical"] * 100))
        assert body["alert_level"] == expected or margin < 0.01       # the field is rounded to 2 decimals
        assert body["alert_level"] != "High Risk"


def test_a_score_between_the_cutoffs_is_not_low(rf_client, rf_pipeline, monkeypatch):
    """The R-02 defect: 4.39 <= score < 25 used to be 'Low Risk'. Force a model score of 0.06 and check the level."""
    class Fixed:
        family = "random_forest"
        def predict(self, x, verbose=0):
            import numpy as np
            return np.array([[0.06]], dtype=np.float32)
    monkeypatch.setattr(rf_pipeline, "dnn_model", Fixed())
    body = rf_client.post("/predict", json=NORMAL_TXN).json()
    assert body["fraud_probability"] == 6.0 and body["alert_level"] == "Medium Risk"


def test_model_info_publishes_the_policy_and_its_evidence(rf_client):
    info = rf_client.get("/model-info").json()
    t = info["thresholds"]
    assert t["alert_policy"]["kind"] == VALIDATED
    assert t["alert_policy"]["levels_used"] == ["Low Risk", "Medium Risk", "Critical Risk"]
    tiers = {x["level"]: x["min_score"] for x in t["alert_policy"]["tiers"]}
    assert tiers == {"Low Risk": 0.0, "Medium Risk": round(CUT["policy_b"] * 100, 4), "Critical Risk": round(CUT["critical"] * 100, 4)}
    assert t["alert_bands"]["Medium Risk"] == "4.39 <= fraud_probability < 26.23"
    assert "legacy" not in t["alert_bands_status"].split(";")[0]
    assert "applied by /predict" in t["frozen_cutoffs"]["status"]
    ev = t["alert_policy"]["evidence"]                               # read from the committed hold-out report, not typed
    final = json.loads((DOWNSTREAM / "final_holdout_report.json").read_text())
    m = final["final"]["primary"]["models"]["selected"]["metrics"]
    assert ev["medium_tier_and_above"]["precision"] == m["precision"]["value"]
    assert ev["critical_tier"]["recall"] == m["critical_recall"]["value"]
    assert info["score_semantics"]["calibrated_probabilities"] is False


def test_other_model_sets_still_report_the_legacy_bands(client, pipeline):
    t = client.get("/model-info").json()["thresholds"]                 # the shared test pipeline is "production"
    assert t["alert_policy"]["kind"] == LEGACY
    assert t["alert_bands"]["Critical Risk"] == "fraud_probability >= 80"
