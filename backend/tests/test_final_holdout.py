"""Stage C of the selection protocol (app/evaluation/final_holdout.py).

The decision logic is tested on hand-built numbers. One end-to-end test runs
the whole stage on small generated samples with quickly trained models, in
temp directories. Nothing here generates or scores the real final hold-out
(seeds 401-405, 411-415) or writes into the repository.
"""

import json
import subprocess
import sys

import pytest

from app import config
from app.evaluation import final_holdout as fh
from app.evaluation import holdout, stacking
from app.evaluation import multiseed as ev
from app.evaluation import selection as sel
from app.evaluation.datasets import load_evaluation_data, resolve_dataset
from app.evaluation.split import build_time_split, save_definition
from app.model_sets import DEFAULT_MODEL_SET
from app.training import multiseed as ms

LSTM, ONLY = "v2_dnn_lstm@14", "v2_dnn_only@14"


# ---- the decision ---------------------------------------------------------------------------------

def _model(recall, rci, alerts, aci):
    return {"metrics": {"recall": {"value": recall, "ci95": rci}, "legit_alerts_per_1000": {"value": alerts, "ci95": aci}}}


def _diff(ci):
    return {"difference": round((ci[0] + ci[1]) / 2, 6), "ci95": ci, "excludes_zero": ci[0] > 0 or ci[1] < 0}


def _primary(lstm=(0.55, [0.51, 0.59], 8.9, [8.3, 9.5]), only=(0.43, [0.39, 0.47], 9.2, [8.6, 9.8]), production=0.30,
             recall=(0.08, 0.16), alerts=(-0.9, 0.3), critical=(-0.02, 0.06), first=(-0.14, -0.04), artifacts=(LSTM, ONLY)):
    models = {"production": _model(production, [0.27, 0.33], 10.3, [10.0, 10.6])}
    for name, m in zip((LSTM, ONLY), (lstm, only)):
        if name in artifacts:
            models[name] = _model(*m)
    pairs = {}
    if LSTM in artifacts and ONLY in artifacts:
        pairs[f"{LSTM} - {ONLY}"] = {"recall": _diff(list(recall)), "legit_alerts_per_1000": _diff(list(alerts)),
                                     "critical_recall": _diff(list(critical)), "first_fraud_recall": _diff(list(first))}
    return {"models": models, "paired_differences": pairs}


def test_gates_are_the_approved_ones():
    assert (holdout.ALERT_BUDGET_PER_1000, holdout.MIN_RECALL) == (10.0, 0.40)
    assert (holdout.RECALL_MARGIN, holdout.LEGIT_ALERT_MARGIN_PER_1000, holdout.CRITICAL_RECALL_MARGIN) == (0.05, 1.0, 0.05)
    assert DEFAULT_MODEL_SET == "production"


def test_both_eligible_and_tie_break_holds():
    d = fh.stage_c_decision(_primary(), [LSTM, ONLY])
    assert d["eligible_artifacts"] == [LSTM, ONLY]
    assert d["tie_break"]["applies"] and d["tie_break"]["all_conditions_hold"]
    assert d["eligible_for_a_controlled_promotion_decision"] == LSTM
    assert d["outcome"] == f"{LSTM} is eligible for a controlled-promotion decision"
    assert "MODEL_SET stays production" in d["promotion"]
    assert d["first_fraud_detection"]["excludes_zero"] and "excludes zero" in d["first_fraud_detection"]["statement"]
    g = d["gates"][LSTM]
    assert g["alert_budget_upper_bound_at_most_10"] and g["recall_at_least_0_40"] and g["recall_lower_bound_above_production"]


@pytest.mark.parametrize("override, failed", [
    (dict(recall=(-0.06, 0.02)), "recall_at_least_matches"),
    (dict(recall=(-0.05, 0.02)), "recall_at_least_matches"),
    (dict(alerts=(-0.2, 1.0)), "alert_burden_at_least_matches"),
    (dict(critical=(-0.07, 0.01)), "critical_tier_not_worse"),
])
def test_a_failed_tie_break_condition_gives_the_simpler_model(override, failed):
    d = fh.stage_c_decision(_primary(**override), [LSTM, ONLY])
    assert not d["tie_break"]["conditions"][failed]["holds"] and not d["tie_break"]["all_conditions_hold"]
    assert d["eligible_for_a_controlled_promotion_decision"] == ONLY and "simpler model" in d["reason"]


def test_the_alert_gate_is_the_upper_bound_at_10_not_a_band():
    ok = fh.stage_c_decision(_primary(lstm=(0.55, [0.51, 0.59], 9.4, [8.8, 10.0])), [LSTM, ONLY])
    assert ok["gates"][LSTM]["alert_budget_upper_bound_at_most_10"]
    # a point value under 10 (and far under 11) still fails when the upper bound is over 10
    over = fh.stage_c_decision(_primary(lstm=(0.55, [0.51, 0.59], 9.5, [8.9, 10.01])), [LSTM, ONLY])
    assert not over["gates"][LSTM]["alert_budget_upper_bound_at_most_10"] and not over["gates"][LSTM]["eligible"]
    assert over["eligible_artifacts"] == [ONLY] and over["eligible_for_a_controlled_promotion_decision"] == ONLY
    assert "only artifact" in over["reason"] and not over["tie_break"]["applies"]


def test_recall_gates():
    low = fh.stage_c_decision(_primary(only=(0.399, [0.36, 0.44], 9.2, [8.6, 9.8])), [LSTM, ONLY])
    assert not low["gates"][ONLY]["recall_at_least_0_40"] and low["eligible_artifacts"] == [LSTM]
    edge = fh.stage_c_decision(_primary(only=(0.40, [0.36, 0.44], 9.2, [8.6, 9.8])), [LSTM, ONLY])
    assert edge["gates"][ONLY]["recall_at_least_0_40"]
    # the lower bound must be strictly above production's point estimate
    near = fh.stage_c_decision(_primary(production=0.39), [LSTM, ONLY])
    assert not near["gates"][ONLY]["recall_lower_bound_above_production"] and near["gates"][LSTM]["recall_lower_bound_above_production"]
    assert near["eligible_for_a_controlled_promotion_decision"] == LSTM


def test_no_eligible_artifact_and_single_artifact():
    none = fh.stage_c_decision(_primary(lstm=(0.55, [0.51, 0.59], 10.4, [9.8, 11.0]), only=(0.43, [0.39, 0.47], 10.2, [9.6, 10.8])), [LSTM, ONLY])
    assert none["eligible_artifacts"] == [] and none["eligible_for_a_controlled_promotion_decision"] is None
    assert none["outcome"] == "no artifact is eligible; production stays"
    assert none["tie_break"] is not None and not none["tie_break"]["applies"]        # still reported
    alone = fh.stage_c_decision(_primary(artifacts=(ONLY,)), [ONLY])
    assert alone["tie_break"] is None and alone["first_fraud_detection"] is None
    assert alone["eligible_for_a_controlled_promotion_decision"] == ONLY


def test_pairs_and_frozen_cutoffs():
    assert fh.comparison_pairs(["production", LSTM, ONLY]) == [(LSTM, ONLY), (LSTM, "production"), (ONLY, "production")]
    assert fh.comparison_pairs(["production", ONLY, LSTM])[0] == (LSTM, ONLY)
    assert fh.comparison_pairs(["production", ONLY]) == [(ONLY, "production")]
    record = {"stage_b_outcome": {"artifacts": [LSTM, ONLY]},
              "selected_artifacts": {LSTM: {"cutoffs": {"policy_b": 0.79, "critical": 0.94}},
                                     ONLY: {"cutoffs": {"policy_b": 0.84, "critical": 0.97}}}}
    assert fh.frozen_cutoffs(record) == {
        "policy_b": {"production": holdout.CUTOFFS["policy_b"]["production"], LSTM: 0.79, ONLY: 0.84},
        "critical": {"production": holdout.CUTOFFS["critical"]["production"], LSTM: 0.94, ONLY: 0.97}}


def test_stage_c_needs_a_selection_record(tmp_path):
    with pytest.raises(fh.StageCError, match="must be recorded before Stage C"):
        fh.load_record(tmp_path)
    with pytest.raises(fh.StageCError, match="must be recorded before Stage C"):
        fh.run(output_dir=tmp_path, final_root=tmp_path / "final", new_customer_root=tmp_path / "nc")
    assert not (tmp_path / "final").exists()                    # nothing was generated
    (tmp_path / sel.RECORD_NAME).write_text(json.dumps({"protocol": "other", "stage_b_outcome": {"artifacts": [LSTM]}}))
    with pytest.raises(fh.StageCError, match="protocol"):
        fh.load_record(tmp_path)
    (tmp_path / sel.RECORD_NAME).write_text(json.dumps({"protocol": sel.PROTOCOL_VERSION, "stage_b_outcome": {"artifacts": []}}))
    with pytest.raises(fh.StageCError, match="nothing to evaluate"):
        fh.load_record(tmp_path)


def test_report_is_written_once_and_never_into_a_model_directory(tmp_path):
    path = fh.write_report({"a": 1}, tmp_path)
    assert path.name == "final_holdout_report.json"
    with pytest.raises(fh.StageCError, match="scored and reported once"):
        fh.write_report({"a": 2}, tmp_path)
    assert json.loads(path.read_text()) == {"a": 1}
    with pytest.raises(fh.StageCError, match="scored and reported once"):
        fh.run(output_dir=tmp_path)
    with pytest.raises(holdout.HoldoutError, match="refusing"):
        fh.write_report({}, config.CANDIDATES_DIR)
    assert fh.FINAL_NEW_CUSTOMER_DIR == config.DATA_DIR / "v2_holdout" / "final" / "new_customer"


# ---- end to end on small samples -------------------------------------------------------------------

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


def test_end_to_end_on_samples(tmp_path_factory):
    data_dir, eval_dir = tmp_path_factory.mktemp("fh_data"), tmp_path_factory.mktemp("fh_eval")
    subprocess.run([sys.executable, str(config.DATA_V2_DIR / "generate.py"), "--customers", "60", "--out", str(data_dir)],
                   check=True, capture_output=True)
    spec = resolve_dataset("v2").with_data_dir(data_dir).with_output_dir(eval_dir)
    data = load_evaluation_data(spec)
    _, _, definition = build_time_split(data.frame, data.grouping)
    save_definition(spec.time_split_path, definition)
    root, dev_root, out = tmp_path_factory.mktemp("fh_models"), tmp_path_factory.mktemp("fh_dev"), tmp_path_factory.mktemp("fh_out")
    final_root, nc_root = tmp_path_factory.mktemp("fh_final"), tmp_path_factory.mktemp("fh_final_nc")
    seeds = (3, 4, 5)
    for s in seeds:
        ms.train_seed(s, root, spec=spec, log=lambda m: None, fit_lstm_fn=_quick_lstm, fit_dnn_fn=_quick_dnn, n_folds=2)
    holdout.generate_dataset(8, dev_root, customers=60)
    dev = sel.development_metrics(training_seeds=seeds, root=root, development_seeds=(8,), development_root=dev_root, spec=spec)
    # quickly trained models need not pass the screens; give the record passing metrics so two artifacts are selected
    for a in sel.ELIGIBLE_ARCHITECTURES:
        for i, s in enumerate(seeds):
            dev["metrics"][a][s] = {"recall": 0.45 + 0.01 * i, "legit_alerts_per_1000": 9.0 + 0.1 * i, "critical_recall": 0.2,
                                    "first_fraud_recall": 0.5, "episode_detection_rate": 0.6}
    record = sel.build_record(dev, {"lead": "v2_dnn_lstm"}, training_seeds=seeds, root=root)
    assert record["stage_b_outcome"]["artifacts"] == ["v2_dnn_lstm@4", "v2_dnn_only@4"]
    record_path = sel.write_record(record, out, final_root=final_root)
    record_bytes = record_path.read_bytes()

    before = ms.protected_checksums()
    files = ev.multiseed_file_checksums(seeds, root)
    kwargs = dict(output_dir=out, root=root, final_seeds=(21,), new_customer_seeds=(22,), final_root=final_root,
                  new_customer_root=nc_root, reps=30, customers=60, new_customer_episodes=10)
    report = fh.run(**kwargs)
    assert json.loads((out / fh.REPORT_NAME).read_text()) == json.loads(json.dumps(report))
    assert sorted(p.name for p in (out / fh.MANIFEST_DIR).iterdir()) == ["seed_21_manifest.json", "seed_22_manifest.json"]
    assert record_path.read_bytes() == record_bytes
    assert ms.protected_checksums() == before and ev.multiseed_file_checksums(seeds, root) == files
    with pytest.raises(fh.StageCError, match="scored and reported once"):
        fh.run(**kwargs)

    names = ["production", "v2_dnn_lstm@4", "v2_dnn_only@4"]
    assert report["fixed_inputs"]["models_scored"] == names                       # nothing else is scored
    assert report["fixed_inputs"]["frozen_cutoffs_score_0_1"] == fh.frozen_cutoffs(record)
    assert report["selection_record_sha256"] == holdout._sha256(record_path)
    for seed, key, r in (("21", "final_holdout", final_root), ("22", "final_new_customer", nc_root)):
        info = report["datasets"][key][seed]
        manifest = json.loads((holdout.seed_dir(int(seed), r) / "manifest.json").read_text())
        assert info["files_sha256"] == {k: v["sha256"] for k, v in manifest["files"].items()}
        assert info["manifest_sha256"] == holdout._sha256(out / fh.MANIFEST_DIR / f"seed_{seed}_manifest.json")
        assert info["command"] == manifest["command"]
    assert report["datasets"]["final_holdout"]["21"]["generator_version"] == "2.0.1"
    assert report["datasets"]["final_new_customer"]["22"]["generator_version"] == "2.1.0"
    assert report["new_customer"]["new_customer_fraud_episodes"] == 10
    prim = report["pooled"]["primary"]
    assert list(prim["models"]) == names
    assert list(prim["paired_differences"]) == ["v2_dnn_lstm@4 - v2_dnn_only@4", "v2_dnn_lstm@4 - production", "v2_dnn_only@4 - production"]
    assert report["decision"] == json.loads(json.dumps(fh.stage_c_decision(prim, names[1:])))
    assert set(report["per_dataset_primary"]) == {"21"} and set(report["datasets_over_alert_budget"]) == set(names)
    assert list(report["rings"]["policy_b"]["summary"]) == names
    assert list(report["new_customer"]["current_behaviour"]["models"]) == names
    assert report["size_requirements"]["fraud_episodes_primary"]["met"] is False        # a 60-customer sample is too small
    assert report["state"] == {"model_set_default": "production", "promotion": "none", "models_trained": "none",
                               "cutoffs_recalibrated": "none", "seed_reselected": "none"}
    # the evaluation itself is deterministic
    again = fh.evaluate(record, root, (21,), final_root, (22,), nc_root, 30, record_sha256=holdout._sha256(record_path))
    assert json.dumps(again) == json.dumps(report)
    # a changed artifact is refused
    tampered = json.loads(json.dumps(record))
    tampered["selected_artifacts"]["v2_dnn_only@4"]["weights_sha256"]["dnn_weights_sha256"] = "0" * 64
    with pytest.raises(fh.StageCError, match="differs from the selection record"):
        fh.load_selected(tampered, root)
