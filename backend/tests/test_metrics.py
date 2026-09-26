"""GET /metrics and GET /metrics/report -- held-out model performance.

Served from the saved corrected-evaluation report
(models/evaluation/evaluation_report.json, written by
`python -m app.evaluation.run`); the endpoint no longer recomputes anything.
"""

import json

import pytest

from app.config import EVAL_REPORT_PATH

MODELS = ("lstm_risk_predictor", "dnn_fraud_classifier")


@pytest.fixture(scope="module")
def metrics(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    return r.json()


pytestmark = pytest.mark.slow


def test_metrics_has_both_models(metrics):
    assert set(metrics) == set(MODELS)


@pytest.mark.parametrize("model", MODELS)
def test_metric_values_in_range(metrics, model):
    m = metrics[model]
    for key in ("precision", "recall", "f1_score", "auc_roc"):
        assert m[key] is not None
        assert 0.0 <= m[key] <= 1.0, key
    # the decision threshold is chosen on the validation period, not fixed at 0.5
    assert 0.0 < m["threshold"] < 1.0
    assert m["threshold_source"].startswith("validation")
    assert 0 < m["fraud_rate_pct"] < 5  # heavily imbalanced, ~1% fraud


@pytest.mark.parametrize("model", MODELS)
def test_confusion_matrix_consistent(metrics, model):
    m = metrics[model]
    cm = m["confusion_matrix"]
    assert set(cm) == {"true_negative", "false_positive", "false_negative", "true_positive"}
    assert all(v >= 0 for v in cm.values())
    assert sum(cm.values()) == m["test_set_size"]

    tp, fp, fn = cm["true_positive"], cm["false_positive"], cm["false_negative"]
    if tp + fp:
        assert m["precision"] == pytest.approx(tp / (tp + fp), abs=1e-4)
    if tp + fn:
        assert m["recall"] == pytest.approx(tp / (tp + fn), abs=1e-4)
    positives = tp + fn
    assert m["fraud_rate_pct"] == pytest.approx(positives / m["test_set_size"] * 100, abs=1e-3)


@pytest.mark.parametrize("model", MODELS)
def test_models_beat_chance(metrics, model):
    # regression guard for the committed model artifacts
    assert metrics[model]["auc_roc"] > 0.8


def test_metrics_cached_and_stable(client, metrics):
    again = client.get("/metrics").json()
    assert again == metrics


def test_metrics_refresh_is_deterministic(client, metrics):
    refreshed = client.get("/metrics", params={"refresh": True}).json()
    assert refreshed == metrics


@pytest.mark.parametrize("model", MODELS)
def test_metrics_include_corrected_evaluation_fields(metrics, model):
    m = metrics[model]
    assert 0.0 <= m["pr_auc"] <= 1.0
    assert m["alerts_per_1000"] >= 0
    assert set(m["recall_at_fpr"]) == {"0.001", "0.01"}
    ep = m["episodes"]
    assert ep["first_fraud_transactions"] == ep["fraud_episodes"] > 0
    assert 0.0 <= ep["first_fraud_recall"] <= 1.0
    assert set(m["validation"]) >= {"pr_auc", "auc_roc", "precision", "recall", "f1_score"}


def test_metrics_match_saved_report(metrics):
    report = json.loads(EVAL_REPORT_PATH.read_text())
    for model in MODELS:
        assert metrics[model] == report["primary"][model]


def test_metrics_report_endpoint(client):
    r = client.get("/metrics/report")
    assert r.status_code == 200
    body = r.json()
    assert {"methodology", "splits", "primary", "secondary_customer_grouped", "legacy_random_split"} <= set(body)
    assert set(body["primary"]["baselines"]) == {"amount_hour_rule", "logistic_regression", "dnn_without_risk_score"}
    assert body["splits"]["time"]["method"].startswith("time-based")


def test_metrics_report_missing_returns_503(client, monkeypatch, tmp_path):
    import app.models.evaluate as evaluate
    monkeypatch.setattr(evaluate, "EVAL_REPORT_PATH", tmp_path / "missing.json")
    monkeypatch.setattr(evaluate, "_cache", None)
    for path in ("/metrics", "/metrics/report"):
        r = client.get(path)
        assert r.status_code == 503
        assert "python -m app.evaluation.run" in r.json()["detail"]
