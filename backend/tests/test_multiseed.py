"""Multi-seed candidate training and its evaluation
(app/training/multiseed.py, app/evaluation/multiseed.py).

The real training code path runs on a small generated v2 sample with the
production architectures trained for one epoch, into temp directories.
Nothing here writes to models/saved/, models/candidates/,
models/candidates_multiseed/ or data/.
"""

import json
import subprocess
import sys

import numpy as np
import pytest

from app import config
from app.evaluation import holdout, stacking
from app.evaluation import multiseed as ev
from app.evaluation.datasets import load_evaluation_data, resolve_dataset
from app.evaluation.split import build_time_split, save_definition
from app.model_sets import DEFAULT_MODEL_SET, MODEL_SETS, load_model_set
from app.training import candidates as cand
from app.training import multiseed as ms

SEEDS = (3, 4)
REAL_V2 = config.DATA_V2_DIR / "transactions_with_features.csv"


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


QUICK = dict(fit_lstm_fn=_quick_lstm, fit_dnn_fn=_quick_dnn, n_folds=2)


@pytest.fixture(scope="module")
def sample_spec(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("ms_data")
    eval_dir = tmp_path_factory.mktemp("ms_eval")
    subprocess.run([sys.executable, str(config.DATA_V2_DIR / "generate.py"), "--customers", "60", "--out", str(data_dir)],
                   check=True, capture_output=True)
    spec = resolve_dataset("v2").with_data_dir(data_dir).with_output_dir(eval_dir)
    data = load_evaluation_data(spec)
    _, _, definition = build_time_split(data.frame, data.grouping)
    save_definition(spec.time_split_path, definition)
    return spec


@pytest.fixture(scope="module")
def trained(sample_spec, tmp_path_factory):
    root = tmp_path_factory.mktemp("ms_models")
    before = ms.protected_checksums()
    records = {s: ms.train_seed(s, root, spec=sample_spec, log=lambda m: None, **QUICK) for s in SEEDS}
    assert ms.protected_checksums() == before
    return root, records


@pytest.fixture(scope="module")
def holdout_roots(tmp_path_factory):
    plain, new = tmp_path_factory.mktemp("ms_holdout"), tmp_path_factory.mktemp("ms_new_customer")
    holdout.generate_dataset(7, plain, customers=60)
    holdout.generate_dataset(9, new, customers=60, late_joiner_share=0.3, new_customer_fraud_episodes=10)
    return plain, new


# ---- fixed inputs and safety --------------------------------------------------------------------

def test_fixed_inputs():
    assert ms.TRAINING_SEEDS == (11, 12, 13, 14, 15) and ms.REFERENCE_SEED == 42
    assert 42 not in ms.TRAINING_SEEDS
    assert not set(ms.TRAINING_SEEDS) & (set(holdout.HOLDOUT_SEEDS) | set(holdout.NEW_CUSTOMER_SEEDS))
    assert ms.MULTISEED_DIR == config.BACKEND_DIR / "models" / "candidates_multiseed"
    assert ms.seed_root(11) == ms.MULTISEED_DIR / "v2" / "seed_11"
    assert ev.POLICY_FPR == {"critical": 0.001, "policy_b": 0.01}
    assert ev.key("v2_dnn_lstm", 11) == "v2_dnn_lstm@11"
    assert DEFAULT_MODEL_SET == "v2_lstm_rf_seed14"      # Step 4D: the confirmed LSTM -> random forest (was "production")
    # the application names exactly one multi-seed directory: the artifact selected in 4C-3E.6
    # (seed 14, pinned by checksum, evaluation only -- 4C-3F). No other training seed is loadable by name.
    assert set(MODEL_SETS) == {"production", "v2_dnn_lstm", "v2_dnn_only", "v2_dnn_lstm_seed14", "v2_lstm_rf_seed14"}
    # (Step 4D added the LSTM -> random forest artifact, which lives under candidates_downstream/)
    assert [n for n, spec in MODEL_SETS.items() if spec.root is not None] == ["v2_dnn_lstm_seed14", "v2_lstm_rf_seed14"]
    assert MODEL_SETS["v2_lstm_rf_seed14"].root[0] == "candidates_downstream"
    assert MODEL_SETS["v2_dnn_lstm_seed14"].root == ("candidates_multiseed", "v2", "seed_14")


@pytest.mark.parametrize("bad", [config.MODELS_SAVED_DIR, config.MODELS_SAVED_DIR / "x", config.CANDIDATES_DIR,
                                 config.CANDIDATES_DIR / "v2", config.CANDIDATES_DIR / "v2" / "dnn_lstm"])
def test_production_and_seed_42_candidate_directories_are_never_an_output(bad):
    with pytest.raises(ms.MultiSeedError, match="refusing"):
        ms.assert_safe_root(bad)
    with pytest.raises(ms.MultiSeedError, match="refusing"):
        ms.train_seed(11, bad, log=lambda m: None)                      # refused before loading any data
    assert ms.assert_safe_root(ms.seed_root(11)) == ms.seed_root(11).resolve()


def test_seed_42_is_not_retrained_in_the_default_location():
    with pytest.raises(ms.MultiSeedError, match="not retrained"):
        ms.train_seed(42, log=lambda m: None)


def test_protected_checksums_cover_production_and_the_15_candidate_files():
    sums = ms.protected_checksums()
    assert len(sums["production"]) == 7 and len(sums["candidates"]) == 15
    assert sums == holdout.model_file_checksums()


def test_a_trained_seed_is_never_overwritten(trained, sample_spec):
    root, _ = trained
    files = ev.multiseed_file_checksums(SEEDS, root)
    with pytest.raises(ms.MultiSeedError, match="never overwritten"):
        ms.train_seed(SEEDS[0], root, spec=sample_spec, log=lambda m: None, **QUICK)
    assert ev.multiseed_file_checksums(SEEDS, root) == files


# ---- what is trained and recorded -----------------------------------------------------------------

def test_layout_and_run_record(trained, sample_spec):
    root, records = trained
    data_sha = load_evaluation_data(sample_spec).sha256
    for s in SEEDS:
        d = ms.seed_root(s, root)
        assert {p.name for p in d.iterdir()} == {"dnn_lstm", "dnn_only", "training_run.json"}
        assert len(list((d / "dnn_lstm").iterdir())) == 8 and len(list((d / "dnn_only").iterdir())) == 5
        rec = json.loads((d / "training_run.json").read_text())
        assert rec == json.loads(json.dumps(records[s]))
        assert rec["training_seed"] == s and rec["candidates"] == ["dnn_lstm", "dnn_only"]
        assert rec["dataset"]["sha256"] == data_sha and rec["dataset"]["version"] == "v2"
        assert rec["features"]["count"] == 9 and len(rec["features"]["sha256"]) == 64
        assert set(rec["weights_sha256"]["dnn_lstm"]) == {"dnn", "lstm"} and set(rec["weights_sha256"]["dnn_only"]) == {"dnn"}
        assert set(rec["scalers_sha256"]["dnn_lstm"]) == {"dnn_feature_mean.npy", "dnn_feature_std.npy",
                                                          "lstm_feature_mean.npy", "lstm_feature_std.npy"}
        assert rec["production_and_seed_42_candidate_files_unchanged"] is True
        assert rec["protected_files_sha256"] == ms.protected_checksums()
        for name in cand.CANDIDATES:
            manifest = json.loads((d / name / "manifest.json").read_text())
            assert manifest["seed"] == s
            for fname, sha in manifest["files"].items():
                assert rec["files_sha256"][name][fname] == sha
                if not fname.endswith(".keras"):
                    assert cand._sha256(d / name / fname) == sha


def test_only_the_training_seed_differs_between_runs(trained):
    root, records = trained
    a, b = (json.loads((ms.seed_root(s, root) / "dnn_lstm" / "manifest.json").read_text()) for s in SEEDS)
    for field in ("candidate", "dataset", "split", "features", "sequence_length", "clipping", "recipe"):
        assert a[field] == b[field], field
    for field in ("type", "architecture", "dnn_input_columns", "dnn_input_shape", "dnn_layers", "dnn_parameters",
                  "lstm_input_shape", "lstm_layers", "lstm_parameters"):
        assert a["model"][field] == b["model"][field], field
    assert a["seed"] != b["seed"]
    assert records[SEEDS[0]]["weights_sha256"] != records[SEEDS[1]]["weights_sha256"]
    # the scalers are fitted on the training rows, which do not depend on the seed
    for name in ("dnn_only",):
        assert records[SEEDS[0]]["scalers_sha256"][name] == records[SEEDS[1]]["scalers_sha256"][name]
    assert records[SEEDS[0]]["scalers_sha256"]["dnn_lstm"]["lstm_feature_mean.npy"] == \
        records[SEEDS[1]]["scalers_sha256"]["dnn_lstm"]["lstm_feature_mean.npy"]


def test_the_same_seed_gives_the_same_weights(trained, sample_spec, tmp_path):
    _, records = trained
    again = ms.train_seed(SEEDS[0], tmp_path, spec=sample_spec, log=lambda m: None, **QUICK)
    assert again["weights_sha256"] == records[SEEDS[0]]["weights_sha256"]
    assert again["scalers_sha256"] == records[SEEDS[0]]["scalers_sha256"]


def test_trained_models_load_as_model_sets(trained):
    root, records = trained
    models = ev.load_trained(SEEDS, root)
    assert list(models) == ["v2_dnn_lstm@3", "v2_dnn_only@3", "v2_dnn_lstm@4", "v2_dnn_only@4"]
    assert models["v2_dnn_lstm@3"].uses_lstm and not models["v2_dnn_only@3"].uses_lstm
    assert models["v2_dnn_lstm@4"].manifest["seed"] == 4
    assert models["v2_dnn_only@4"].directory == (ms.seed_root(4, root) / "dnn_only").resolve()


# ---- cut-offs -----------------------------------------------------------------------------------

def test_tie_safe_cutoff():
    y = np.array([0, 0, 0, 0, 1])
    s = np.array([0.1, 0.2, 0.3, 0.3, 0.9], dtype=np.float32)
    assert ev.tie_safe_cutoff(y, s, 0.0) == float(np.float32(0.9))        # no false positive allowed
    # one of four allowed, but two negatives tie at 0.3: admitting them would be two, so the cut-off stays above
    assert ev.tie_safe_cutoff(y, s, 0.25) == float(np.float32(0.9))
    assert ev.tie_safe_cutoff(y, s, 0.5) == float(np.float32(0.3))
    assert ev.tie_safe_cutoff(y, s, 1.0) == float(np.float32(0.1))
    # nothing qualifies when the top score is shared by more negatives than the budget allows
    assert ev.tie_safe_cutoff(np.array([0, 0, 1]), np.array([0.5, 0.5, 0.5]), 0.0) is None
    # against the definition, on data with heavy ties
    rng = np.random.default_rng(1)
    y = (rng.random(3000) < 0.03).astype(int)
    s = np.round(rng.random(3000) * 0.6 + 0.4 * y * rng.random(3000), 2).astype(np.float32)
    for target in (0.001, 0.01, 0.05):
        allowed = np.floor(target * (y == 0).sum())
        ok = [v for v in np.unique(s) if ((s >= v) & (y == 0)).sum() <= allowed]
        cut = ev.tie_safe_cutoff(y, s, target)
        assert cut == float(min(ok))
        assert ((s >= np.float32(cut)) & (y == 0)).sum() <= allowed


@pytest.mark.skipif(not REAL_V2.exists(), reason="data/v2 has not been generated on this machine")
@pytest.mark.parametrize("name", ["production", "v2_dnn_lstm", "v2_dnn_only"])
def test_the_rule_reproduces_the_fixed_seed_42_cutoffs(name):
    spec = resolve_dataset("v2")
    data = load_evaluation_data(spec)
    labels, _, _, _ = cand.load_saved_split(spec, data)
    info = ev.policy_cutoffs(load_model_set(name), data, ev.validation_mask(data, labels))
    assert info["policy_b"] == holdout.CUTOFFS["policy_b"][name]
    assert info["critical"] == holdout.CUTOFFS["critical"][name]
    assert info["policy_b_validation"]["false_positive_rate"] <= 0.01
    assert info["critical_validation"]["false_positive_rate"] <= 0.001


# ---- statistics ---------------------------------------------------------------------------------

def test_across_seeds():
    s = ev.across_seeds([0.40, 0.42, 0.44, 0.46, 0.48])
    assert (s["n"], s["mean"], s["median"], s["min"], s["max"]) == (5, 0.44, 0.44, 0.40, 0.48)
    assert s["sd"] == pytest.approx(np.std([0.40, 0.42, 0.44, 0.46, 0.48], ddof=1), abs=1e-6)
    half = 2.7764451 * s["sd"] / np.sqrt(5)                       # t, 4 degrees of freedom
    assert s["mean_ci95_training_seeds"] == pytest.approx([0.44 - half, 0.44 + half], abs=1e-5)
    assert ev.across_seeds([0.5])["sd"] is None and ev.across_seeds([])["n"] == 0


def _fake(lstm, only, reps=400, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    seeds = list(range(1, len(lstm) + 1))
    result, boot = {"models": {}}, {}
    for arch, values in (("v2_dnn_lstm", lstm), ("v2_dnn_only", only)):
        for s, v in zip(seeds, values):
            k = ev.key(arch, s)
            result["models"][k] = {"metrics": {m: {"value": v[m]} for m in v}}
            boot[k] = {m: v[m] + rng.normal(0, noise * (10 if m == "legit_alerts_per_1000" else 1), reps) for m in v}
    return result, boot, seeds


def test_architecture_summary_separates_the_two_sources_of_variation():
    lstm = [{"recall": r} for r in (0.40, 0.50, 0.45, 0.55, 0.35)]            # large spread between training seeds
    only = [{"recall": r} for r in (0.42, 0.42, 0.42, 0.42, 0.42)]
    result, boot, seeds = _fake(lstm, only, noise=0.002)
    out = ev.architecture_summary(result, boot, seeds, ["recall"])
    c = out["comparison"]["recall"]
    assert c["difference_of_means"] == pytest.approx(0.03, abs=1e-6)
    assert c["same_seed_differences"] == {"1": -0.02, "2": 0.08, "3": 0.03, "4": 0.13, "5": -0.07}
    assert c["seed_pairs_with_v2_dnn_lstm_higher"] == "15/25"
    data, both, seeds_only = c["ci95_data_only"], c["ci95_data_and_training_seeds"], c["ci95_training_seeds_only"]
    assert data[0] > 0 and data[1] - data[0] < 0.01                 # the data alone would call it separated
    assert both[0] < 0 < both[1] and both[1] - both[0] > 5 * (data[1] - data[0])
    assert seeds_only[0] < 0 < seeds_only[1]
    assert not c["excludes_zero"]
    a = out["architectures"]["v2_dnn_lstm"]["recall"]
    assert (a["mean"], a["min"], a["max"]) == (0.45, 0.35, 0.55)
    assert a["per_training_seed"] == {"1": 0.40, "2": 0.50, "3": 0.45, "4": 0.55, "5": 0.35}
    assert out["architectures"]["v2_dnn_only"]["recall"]["sd"] == 0.0
    assert ev.architecture_summary(result, boot, seeds, ["recall"]) == out            # deterministic


def _comparison(recall, alerts, critical):
    def entry(ci):
        return {"difference_of_means": (ci[0] + ci[1]) / 2, "ci95_data_and_training_seeds": ci,
                "ci95_training_seeds_only": ci, "excludes_zero": ci[0] > 0 or ci[1] < 0}
    return {"recall": entry(recall), "legit_alerts_per_1000": entry(alerts), "critical_recall": entry(critical)}


def test_decision_keeps_the_lead_only_when_all_three_conditions_hold():
    d = ev.decision(_comparison([-0.02, 0.05], [-0.8, 0.6], [-0.03, 0.03]))
    assert d["all_conditions_hold"] and d["provisional_lead"] == "v2_dnn_lstm" and d["primary_endpoints_separated"] == []
    assert "MODEL_SET stays production" in d["promotion"]
    for bad, failed in ((_comparison([-0.06, 0.05], [-0.8, 0.6], [-0.03, 0.03]), "recall_at_least_matches"),
                        (_comparison([-0.02, 0.05], [-0.8, 1.0], [-0.03, 0.03]), "alert_burden_at_least_matches"),
                        (_comparison([-0.02, 0.05], [-0.8, 0.6], [-0.05, 0.03]), "critical_tier_not_worse")):
        d = ev.decision(bad)
        assert not d["conditions"][failed]["holds"] and not d["all_conditions_hold"]
        assert d["provisional_lead"] == "v2_dnn_only" and "simplicity" in d["outcome"]


# ---- the whole evaluation ---------------------------------------------------------------------------

def test_report(trained, sample_spec, holdout_roots, tmp_path):
    root, _ = trained
    plain, new = holdout_roots
    before = ms.protected_checksums()
    files = ev.multiseed_file_checksums(SEEDS, root)
    kwargs = dict(training_seeds=SEEDS, root=root, holdout_seeds=[7], holdout_root=plain, new_customer_seeds=[9],
                  new_customer_root=new, reps=25, spec=sample_spec)
    rep = ev.evaluate(**kwargs)
    again = ev.evaluate(**kwargs, cache_dir=tmp_path / "cache")
    cached = ev.evaluate(**kwargs, cache_dir=tmp_path / "cache")
    a = holdout.write_report(rep, tmp_path / "a", ev.REPORT_NAME).read_bytes()
    assert a == holdout.write_report(again, tmp_path / "b", ev.REPORT_NAME).read_bytes()
    assert a == holdout.write_report(cached, tmp_path / "c", ev.REPORT_NAME).read_bytes()
    assert ms.protected_checksums() == before == rep["protected_model_files_sha256"]
    assert ev.multiseed_file_checksums(SEEDS, root) == files == rep["multiseed_files_sha256"]

    trained_keys = [ev.key(a_, s) for s in SEEDS for a_ in ev.ARCHITECTURES]
    names = list(ev.REFERENCE) + trained_keys
    prim = rep["holdout"]["primary"]
    assert list(prim["per_model"]) == names and list(rep["cutoffs"]) == names
    assert rep["fixed_inputs"]["training_seeds"] == list(SEEDS)
    assert set(rep["training_runs"]) == {"3", "4"} and "protected_files_sha256" not in rep["training_runs"]["3"]
    # every trained model gets its own cut-offs by the validation rule; Critical is at or above Policy B
    data = load_evaluation_data(sample_spec)
    labels, _, _, _ = cand.load_saved_split(sample_spec, data)
    mask = ev.validation_mask(data, labels)
    models = ev.load_trained(SEEDS, root)
    for k in trained_keys:
        info = ev.policy_cutoffs(models[k], data, mask)
        assert rep["cutoffs"][k] == json.loads(json.dumps(info))
        assert info["critical"] >= info["policy_b"]
        assert info["policy_b_validation"]["false_positive_rate"] <= 0.01
        counts = prim["per_model"][k]["counts"]
        assert counts["true_positives"] + counts["false_negatives"] == prim["population"]["fraud_transactions"]
        assert counts["critical_true_positives"] <= counts["true_positives"]
    for a_ in ev.ARCHITECTURES:
        stats = prim["architectures"][a_]["recall"]
        values = [prim["per_model"][ev.key(a_, s)]["metrics"]["recall"]["value"] for s in SEEDS]
        assert stats["per_training_seed"] == {str(s): v for s, v in zip(SEEDS, values)}
        assert stats["mean"] == pytest.approx(np.mean(values), abs=1e-6) and stats["n"] == 2
        assert set(rep["alert_budget"]["architectures"][a_]["holdout_pooled_per_training_seed"]) == {"3", "4"}
        assert rep["alert_budget"]["architectures"][a_]["cells_training_seed_x_data_seed"]["n"] == 4
    c = prim["comparison"]["recall"]
    lstm, only = (np.mean([prim["per_model"][ev.key(a_, s)]["metrics"]["recall"]["value"] for s in SEEDS]) for a_ in ev.ARCHITECTURES)
    assert c["difference_of_means"] == pytest.approx(lstm - only, abs=1e-6)
    assert set(c) >= {"ci95_data_only", "ci95_training_seeds_only", "ci95_data_and_training_seeds", "same_seed_differences"}
    assert "v2_dnn_lstm@3 - v2_dnn_only@3" in prim["paired_differences"]
    assert "v2_dnn_only@4 - production" in prim["paired_differences"]
    assert set(rep["decision"]["conditions"]) == {"recall_at_least_matches", "alert_burden_at_least_matches", "critical_tier_not_worse"}
    assert rep["decision"]["provisional_lead"] in ev.ARCHITECTURES
    assert set(rep["holdout"]["per_data_seed"]) == {"7"}
    for tier in ("policy_b", "critical"):
        ring = rep["holdout"]["rings"][tier]
        assert list(ring["per_model"]) == names and ring["rings"] >= 1
    nc = rep["new_customer"]
    assert set(nc["variants"]) == {"current_policy_b", "critical_only", "policy_b_flags_neutral", "critical_only_flags_neutral"}
    assert nc["population"]["first_fraud_transactions"] == 10
    for k in names:
        cur, crit = (nc["variants"][v]["per_model"][k]["counts"] for v in ("current_policy_b", "critical_only"))
        assert crit["true_positives"] <= cur["true_positives"] and crit["false_positives"] <= cur["false_positives"]
    assert set(nc["unchanged_customers_full_history"]["per_data_seed"]) == {"9"}
    assert rep["seed_42_and_multi_seed"]["v2_dnn_lstm"]["recall"]["seed_42"] == prim["per_model"]["v2_dnn_lstm"]["metrics"]["recall"]["value"]
    assert rep["state"] == {"model_set_default": DEFAULT_MODEL_SET, "promotion": "none", "production_models_trained": "none",
                            "seed_42_candidates_retrained": "none", "thresholds_changed": "none"}


def test_reference_models_keep_the_fixed_cutoffs(trained, sample_spec, holdout_roots):
    """The seed-42 candidates and production are always alerted at holdout.CUTOFFS,
    whatever dataset supplies the trained models' cut-offs."""
    root, _ = trained
    plain, new = holdout_roots
    rep = ev.evaluate(training_seeds=SEEDS[:1], root=root, holdout_seeds=[7], holdout_root=plain, new_customer_seeds=[9],
                      new_customer_root=new, reps=0, spec=sample_spec)
    reference = holdout.evaluate([7], plain, reps=5)
    for name in ev.REFERENCE:
        assert rep["holdout"]["primary"]["per_model"][name]["counts"] == reference["pooled"]["primary"]["models"][name]["counts"]
