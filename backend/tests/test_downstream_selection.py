"""Step 4D: the downstream-classifier selection code (app/evaluation/downstream.py).

These tests use small synthetic arrays; they do not train the LSTM and do not
read the v2 datasets (which are not part of the repository)."""

import json

import numpy as np
import pytest

from app.evaluation import downstream as d
from app.models.downstream_classifier import ClassifierModel, shap_explainer_for


def test_protocol_constants_match_the_document():
    text = d.PROTOCOL_DOC.read_text()
    assert d.PROTOCOL_VERSION in text
    for seeds in (d.FINAL_SEEDS, d.FINAL_NEW_CUSTOMER_SEEDS, d.DEVELOPMENT_SEEDS, d.TRAINING_SEEDS):
        assert f"{seeds[0]}–{seeds[-1]}" in text
    assert d.DEPLOY_SEED == 14
    assert d.FAMILIES[0] == d.INCUMBENT == "dnn"


def test_grids_are_the_preregistered_ones():
    assert len(d.grid("logistic_regression")) == 8
    assert len(d.grid("random_forest")) == 12
    assert len(d.grid("hist_gradient_boosting")) == 16
    assert d.grid("dnn") == [{}]
    with pytest.raises(d.DownstreamError):
        d.grid("xgboost")


@pytest.mark.parametrize("seed", [42, 101, 205, 401, 415])
def test_spent_seeds_are_refused(seed):
    with pytest.raises(d.DownstreamError):
        d.assert_allowed_seed(seed, d.FINAL_SEEDS + d.DEVELOPMENT_SEEDS + (seed,))


def test_allowed_seed_passes():
    assert d.assert_allowed_seed(501, d.FINAL_SEEDS) == 501
    with pytest.raises(d.DownstreamError):
        d.assert_allowed_seed(301, d.FINAL_SEEDS)


def _toy(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 10)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 9] + rng.normal(scale=0.5, size=n) > 1.2).astype(int)
    return X, y


@pytest.mark.parametrize("family", d.CHALLENGERS)
def test_classifier_interface_and_reproducibility(family):
    X, y = _toy()
    params = d.grid(family)[0]
    a = d.fit_family(family, params, 11, X, y)
    b = d.fit_family(family, params, 11, X, y)
    assert isinstance(a, ClassifierModel) and a.input_shape == (None, 10)
    pa = a.predict(X[:5], verbose=0)
    assert pa.shape == (5, 1) and ((pa >= 0) & (pa <= 1)).all()
    assert np.array_equal(d.model_scores(a, X), d.model_scores(b, X))       # E7 on a toy problem
    assert d.model_scores(a, X).max() <= np.float32(0.999)


@pytest.mark.parametrize("family", d.CHALLENGERS)
def test_shap_values_are_exact_for_each_family(family):
    X, y = _toy()
    m = d.fit_family(family, d.grid(family)[0], 11, X, y)
    explainer, space = shap_explainer_for(m, X[:50])
    x = X[:3]
    sv = np.asarray(explainer.shap_values(x))
    sv = sv[..., 1] if sv.ndim == 3 else sv
    base = explainer.expected_value
    base = float(np.ravel(base)[-1])
    p = m.estimator.predict_proba(x)[:, 1]
    target = p if space == "probability" else np.log(p / (1 - p))
    assert np.allclose(sv.sum(axis=1) + base, target, atol=1e-6)


def test_calibration_metrics_on_known_values():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.0, 0.0, 1.0, 1.0])
    c = d.calibration_metrics(y, p)
    assert c["brier_score"] == pytest.approx(0.0, abs=1e-9)
    assert c["ece_equal_width_10_bins"] == pytest.approx(0.0, abs=1e-6)
    c2 = d.calibration_metrics(np.array([0, 1]), np.array([0.5, 0.5]))
    assert c2["brier_score"] == pytest.approx(0.25)
    assert c2["log_loss"] == pytest.approx(np.log(2), rel=1e-6)


def test_score_distribution_bands():
    y = np.array([0, 0, 0, 1])
    p = np.array([0.1, 0.3, 0.6, 0.9])
    dist = d.score_distribution(y, p)
    bands = dist["legitimate"]["application_alert_bands_25_50_80"]
    assert bands["low_below_25"] == pytest.approx(1 / 3, abs=1e-4)
    assert dist["fraud"]["application_alert_bands_25_50_80"]["critical_80_and_above"] == 1.0


def _summary(pr, recall, alerts, crit, seeds_higher=5):
    def ci(lo, hi):
        return {"ci95_data_and_training_seeds": [lo, hi], "seeds_higher": seeds_higher}
    fam = {"versus_incumbent": {}, "families": {}}
    for f in d.CHALLENGERS:
        fam["versus_incumbent"][f] = {"pr_auc": ci(*pr), "recall": ci(*recall),
                                      "legit_alerts_per_1000": ci(*alerts), "critical_recall": ci(*crit)}
        fam["families"][f] = {"recall": {"mean": 0.5}, "pr_auc": {"mean": 0.6}}
    return fam


def _datasets(higher=True):
    out = {}
    for ds in d.DEVELOPMENT_SEEDS:
        row = {}
        for s in d.TRAINING_SEEDS:
            row[d.key("dnn", s)] = 0.5
            for f in d.CHALLENGERS:
                row[d.key(f, s)] = 0.6 if higher else 0.4
        out[str(ds)] = row
    return out


def test_eligibility_requires_a_clear_pr_auc_gain():
    ok = d.eligibility(_summary((0.01, 0.05), (-0.01, 0.02), (-0.5, 0.5), (-0.01, 0.02)), _datasets(),
                       d.TRAINING_SEEDS, {f: True for f in d.CHALLENGERS})
    assert all(ok[f]["eligible"] for f in d.CHALLENGERS)
    no = d.eligibility(_summary((-0.001, 0.05), (-0.01, 0.02), (-0.5, 0.5), (-0.01, 0.02)), _datasets(),
                       d.TRAINING_SEEDS, {f: True for f in d.CHALLENGERS})
    assert not any(no[f]["eligible"] for f in d.CHALLENGERS)
    assert not no["random_forest"]["conditions"]["E1_pr_auc_clearly_higher"]["holds"]


@pytest.mark.parametrize("bad", ["recall", "alerts", "critical", "seeds", "datasets", "reproducible"])
def test_eligibility_each_other_condition_can_fail(bad):
    s = _summary((0.01, 0.05), (-0.06, 0.02) if bad == "recall" else (-0.01, 0.02),
                 (-0.5, 1.5) if bad == "alerts" else (-0.5, 0.5),
                 (-0.06, 0.02) if bad == "critical" else (-0.01, 0.02),
                 seeds_higher=3 if bad == "seeds" else 5)
    e = d.eligibility(s, _datasets(higher=bad != "datasets"), d.TRAINING_SEEDS,
                      {f: bad != "reproducible" for f in d.CHALLENGERS})
    assert not any(e[f]["eligible"] for f in d.CHALLENGERS)


def test_no_eligible_family_keeps_the_dnn():
    elig = {f: {"eligible": False} for f in d.CHALLENGERS}
    out = d.choose_winner(elig, {}, {}, {})
    assert out["winner"] is None and "DNN stays" in out["outcome"]


def test_winner_is_highest_mean_pr_auc_and_tie_uses_brier():
    elig = {f: {"eligible": True} for f in d.CHALLENGERS}
    summary = {"families": {"logistic_regression": {"pr_auc": {"mean": 0.60}},
                            "random_forest": {"pr_auc": {"mean": 0.70}},
                            "hist_gradient_boosting": {"pr_auc": {"mean": 0.698}}}}
    calib = {"random_forest": {"brier_score_mean": 0.004}, "hist_gradient_boosting": {"brier_score_mean": 0.003},
             "logistic_regression": {"brier_score_mean": 0.002}}
    out = d.choose_winner(elig, summary, calib, {})
    assert out["winner"] == "hist_gradient_boosting"       # within 0.005 of RF and better calibrated
    summary["families"]["hist_gradient_boosting"]["pr_auc"]["mean"] = 0.65
    assert d.choose_winner(elig, summary, calib, {})["winner"] == "random_forest"


def test_confirm_and_generation_need_a_selection_record(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "OUTPUT_DIR", tmp_path)
    with pytest.raises(d.DownstreamError):
        d.generate_final()
    with pytest.raises(d.DownstreamError):
        d.confirm()
    (tmp_path / d.RECORD_NAME).write_text(json.dumps({"deploy": None}))
    with pytest.raises(d.DownstreamError):
        d.confirm()
