"""
Phase 3 (low-priority cleanup) regression tests: the refactors must not change behaviour.
"""
import warnings

import numpy as np

from app.models import evaluate
from app.models.shap_explainer import FraudExplainer, DNN_INPUT_COLUMNS

KERAS_MSG = "The structure of `inputs` doesn't match the expected structure."


def test_startup_uses_lifespan_and_initialises_the_pipeline_once(client, pipeline):
    from app.main import app
    from app.inference_pipeline import get_pipeline

    deprecated = getattr(app.router, "on_startup", []) + getattr(app.router, "on_shutdown", [])
    assert deprecated == []                       # no @app.on_event handlers left
    assert get_pipeline() is pipeline             # lifespan loaded the models once; nothing created a second pipeline
    assert client.get("/health").json() == {"status": "ok"}


class _WarningStub:
    """Stands in for shap's GradientExplainer: emits the cosmetic Keras warning AND an unrelated one."""

    def shap_values(self, x):
        warnings.warn(KERAS_MSG + "\nExpected: transaction_features", UserWarning)
        warnings.warn("some other, genuinely useful warning", UserWarning)
        return np.full((1, len(DNN_INPUT_COLUMNS), 1), 0.01)


def test_only_the_known_keras_warning_is_silenced_and_only_around_the_shap_call():
    explainer = FraudExplainer.__new__(FraudExplainer)
    explainer.explainer = _WarningStub()
    filters_before = list(warnings.filters)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        explainer.explain(np.zeros((1, len(DNN_INPUT_COLUMNS))))
        # the same message raised OUTSIDE the SHAP call is not hidden: the filter did not leak out of explain()
        warnings.warn(KERAS_MSG + " (raised elsewhere)", UserWarning)

    messages = [str(w.message) for w in caught]
    assert sum("some other, genuinely useful warning" in m for m in messages) == 1   # unrelated warnings still get through
    assert sum("raised elsewhere" in m for m in messages) == 1                        # not globally silenced
    assert sum("Expected: transaction_features" in m for m in messages) == 0         # the cosmetic one is gone
    assert list(warnings.filters) == filters_before                                    # global filter list restored


def test_real_shap_call_emits_no_keras_structure_warning(pipeline):
    x = np.clip(np.zeros((1, len(DNN_INPUT_COLUMNS)), dtype=np.float32), -6, 6)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipeline.explainer.explain(x)
    assert not [w for w in caught if KERAS_MSG in str(w.message)]


def test_evaluation_loads_each_model_once(monkeypatch):
    calls = []
    real_load = evaluate.keras.models.load_model

    def counting_load(path, *args, **kwargs):
        calls.append(str(path))
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(evaluate.keras.models, "load_model", counting_load)
    monkeypatch.setattr(evaluate, "_cache", None)
    metrics = evaluate.evaluate_all()
    assert len(calls) == 2 and sum("lstm" in c for c in calls) == 1     # the LSTM used to be loaded twice (3 loads)
    assert metrics["dnn_fraud_classifier"]["recall"] == 1.0              # and the numbers are the shipped ones
    assert metrics["lstm_risk_predictor"]["precision"] == 0.7042


def test_openapi_examples_survive_the_pydantic_example_migration(client):
    props = client.get("/openapi.json").json()["components"]["schemas"]["TransactionInput"]["properties"]
    assert props["customer_id"]["example"] == "CUST_0001"
    assert props["amount"]["example"] == 15000.0
    assert props["failed_logins_24h"]["example"] == 3
