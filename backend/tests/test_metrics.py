"""GET /metrics -- held-out model performance.

The first call rebuilds the test split and scores it through both models,
which takes a few seconds; the result is cached in-process afterwards.
"""

import pytest

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
    assert m["threshold"] == 0.5
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
