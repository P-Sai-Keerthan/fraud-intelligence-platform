"""The pre-registered model-selection protocol (app/evaluation/selection.py,
docs/step4c3e-model-selection-protocol.md).

The rule is tested on hand-built numbers. One end-to-end test runs it on a
small generated sample with quickly trained models, in temp directories.
Nothing here scores the real trained seeds, generates a development or final
hold-out dataset in data/, or writes a selection record into the repository.
One test reads the stage B record, when it exists, and checks it against the rule.
"""

import itertools
import json
import subprocess
import sys

import numpy as np
import pytest

from app import config
from app.evaluation import holdout, stacking
from app.evaluation import multiseed as ev
from app.evaluation import selection as sel
from app.evaluation.datasets import load_evaluation_data, resolve_dataset
from app.evaluation.split import build_time_split, save_definition
from app.model_sets import DEFAULT_MODEL_SET
from app.training import multiseed as ms

PROTOCOL_DOC = config.PROJECT_ROOT / "docs" / "step4c3e-model-selection-protocol.md"
MULTISEED_REPORT = holdout.OUTPUT_DIR / ev.REPORT_NAME
RECORD = holdout.OUTPUT_DIR / sel.RECORD_NAME


def _m(recall, alerts, critical, first, episodes):
    return {"recall": recall, "legit_alerts_per_1000": alerts, "critical_recall": critical,
            "first_fraud_recall": first, "episode_detection_rate": episodes}


BASE = {11: _m(0.50, 9.0, 0.20, 0.55, 0.68), 12: _m(0.60, 9.5, 0.22, 0.50, 0.70), 13: _m(0.52, 9.1, 0.21, 0.56, 0.69),
        14: _m(0.45, 8.5, 0.15, 0.60, 0.60), 15: _m(0.53, 9.2, 0.20, 0.54, 0.70)}


def _reference(metrics):
    """The rule, written out independently of the implementation."""
    seeds = sorted(metrics)
    col = {m: [metrics[s][m] for s in seeds] for m in sel.SELECTION_METRICS}
    med = {m: float(np.median(v)) for m, v in col.items()}
    sd = {m: float(np.std(v, ddof=1)) for m, v in col.items()}
    ok = [s for s in seeds if metrics[s]["recall"] >= 0.40 and metrics[s]["legit_alerts_per_1000"] <= 11.0
          and round(metrics[s]["critical_recall"], 6) >= round(med["critical_recall"] - 0.05, 6)]
    if len(ok) < len(seeds) // 2 + 1:
        return None
    dist = {s: round(sum(abs(metrics[s][m] - med[m]) / sd[m] if sd[m] > 0 else 0.0 for m in sel.SELECTION_METRICS), 6) for s in ok}
    return min(ok, key=lambda s: (dist[s], round(abs(metrics[s]["recall"] - med["recall"]), 6), s))


# ---- what the protocol fixes ---------------------------------------------------------------------

def test_fixed_inputs():
    assert sel.ELIGIBLE_ARCHITECTURES == ("v2_dnn_lstm", "v2_dnn_only")
    assert sel.ELIGIBLE_SEEDS == (11, 12, 13, 14, 15)
    assert sel.DEVELOPMENT_SEEDS == (301, 302, 303, 304, 305)
    assert sel.FINAL_HOLDOUT_SEEDS == (401, 402, 403, 404, 405)
    assert sel.FINAL_NEW_CUSTOMER_SEEDS == (411, 412, 413, 414, 415)
    assert sel.SPENT_SEEDS == (42, 101, 102, 103, 104, 105, 201, 202, 203, 204, 205)
    groups = [sel.ELIGIBLE_SEEDS, sel.DEVELOPMENT_SEEDS, sel.FINAL_HOLDOUT_SEEDS, sel.FINAL_NEW_CUSTOMER_SEEDS, sel.SPENT_SEEDS]
    for a, b in itertools.combinations(groups, 2):
        assert not set(a) & set(b)
    assert sel.SELECTION_METRICS == ("recall", "legit_alerts_per_1000", "critical_recall", "first_fraud_recall",
                                     "episode_detection_rate")
    assert sel.majority(5) == 3 and sel.majority(3) == 2 and sel.majority(4) == 3
    assert sel.DEVELOPMENT_DATA_DIR == config.DATA_DIR / "v2_holdout" / "development"
    assert sel.FINAL_DATA_DIR == config.DATA_DIR / "v2_holdout" / "final"
    assert DEFAULT_MODEL_SET == "v2_lstm_rf_seed14"      # Step 4D: the confirmed LSTM -> random forest (was "production")


def test_every_number_is_an_already_approved_one():
    assert sel.MIN_RECALL == holdout.MIN_RECALL == 0.40
    assert sel.ALERT_SCREEN_PER_1000 == holdout.ALERT_BUDGET_PER_1000 + holdout.LEGIT_ALERT_MARGIN_PER_1000 == 11.0
    assert sel.CRITICAL_RECALL_MARGIN == holdout.CRITICAL_RECALL_MARGIN == 0.05
    # the architecture rule and its margins are the existing ones, untouched
    assert (holdout.RECALL_MARGIN, holdout.LEGIT_ALERT_MARGIN_PER_1000, holdout.CRITICAL_RECALL_MARGIN) == (0.05, 1.0, 0.05)
    assert holdout.ALERT_BUDGET_PER_1000 == 10.0


@pytest.mark.parametrize("seed, message", [
    (42, "already been used"), (101, "already been used"), (105, "already been used"), (201, "already been used"),
    (205, "already been used"), (401, "final hold-out"), (405, "final hold-out"), (411, "final hold-out"),
    (11, "training seed"), (306, "not a development seed"), (7, "not a development seed")])
def test_only_development_seeds_may_be_used_for_selection(seed, message):
    with pytest.raises(sel.SelectionError, match=message):
        sel.assert_development_seed(seed)
    if seed in sel.SPENT_SEEDS or seed in sel.FINAL_HOLDOUT_SEEDS + sel.FINAL_NEW_CUSTOMER_SEEDS + sel.ELIGIBLE_SEEDS:
        with pytest.raises(sel.SelectionError):                 # no override can admit them
            sel.assert_development_seed(seed, allowed=(seed,))


def test_development_seeds_are_accepted():
    assert [sel.assert_development_seed(s) for s in sel.DEVELOPMENT_SEEDS] == list(sel.DEVELOPMENT_SEEDS)


# ---- the artifact rule ---------------------------------------------------------------------------

def test_the_most_typical_seed_is_selected_not_the_best():
    out = sel.select_seed(BASE)
    assert out["seeds"] == [11, 12, 13, 14, 15]
    assert out["median"] == {"recall": 0.52, "legit_alerts_per_1000": 9.1, "critical_recall": 0.20,
                             "first_fraud_recall": 0.55, "episode_detection_rate": 0.69}
    assert out["eligible_seeds"] == [11, 12, 13, 14, 15] and out["eligible_needed"] == 3 and out["architecture_passes"]
    assert out["selected_seed"] == 13 == _reference(BASE)
    assert out["highest_recall_seed"] == 12 and not out["selected_is_highest_recall_seed"]
    d = {int(s): e["distance"] for s, e in out["per_seed"].items()}
    assert d[13] == min(d.values()) and d[12] > d[13] and d[14] > d[13]
    assert out["per_seed"]["13"]["distance_parts"]["recall"] == 0.0


def test_raising_one_seeds_recall_never_gets_it_selected_for_that():
    """Making a run better on the existing data moves it away from the typical run."""
    better = {s: dict(v) for s, v in BASE.items()}
    better[12]["recall"] = 0.95
    out = sel.select_seed(better)
    assert out["highest_recall_seed"] == 12 and out["selected_seed"] != 12
    assert out["per_seed"]["12"]["distance"] > sel.select_seed(BASE)["per_seed"]["12"]["distance"]


def test_screens():
    m = {s: dict(v) for s, v in BASE.items()}
    m[13]["legit_alerts_per_1000"] = 11.01              # over target + margin
    m[11]["recall"] = 0.399                             # under the approved minimum
    out = sel.select_seed(m)
    assert not out["per_seed"]["13"]["screens"]["alert_burden_within_screen"] and not out["per_seed"]["13"]["eligible"]
    assert not out["per_seed"]["11"]["screens"]["recall_at_least_minimum"]
    assert out["eligible_seeds"] == [12, 14, 15] and out["architecture_passes"]
    assert out["selected_seed"] == _reference(m) and out["selected_seed"] in (12, 14, 15)
    # boundaries are inside
    edge = {s: dict(v) for s, v in BASE.items()}
    edge[11]["recall"], edge[11]["legit_alerts_per_1000"] = 0.40, 11.0
    assert sel.select_seed(edge)["per_seed"]["11"]["eligible"]


def test_critical_tier_outlier_is_not_eligible():
    m = {s: dict(v) for s, v in BASE.items()}
    m[14]["critical_recall"] = 0.149                    # median 0.20, margin 0.05
    out = sel.select_seed(m)
    assert not out["per_seed"]["14"]["screens"]["critical_recall_not_an_outlier"]
    assert out["per_seed"]["12"]["screens"]["critical_recall_not_an_outlier"]
    m[14]["critical_recall"] = 0.15
    assert sel.select_seed(m)["per_seed"]["14"]["eligible"]


def test_an_architecture_needs_a_majority_of_eligible_seeds():
    m = {s: dict(v) for s, v in BASE.items()}
    for s in (11, 13, 15):
        m[s]["recall"] = 0.35
    out = sel.select_seed(m)
    assert out["eligible_seeds"] == [12, 14] and not out["architecture_passes"] and out["selected_seed"] is None
    assert _reference(m) is None
    m[11]["recall"] = 0.41
    out = sel.select_seed(m)
    assert out["architecture_passes"] and out["selected_seed"] == _reference(m)


def test_ties_and_degenerate_inputs():
    same = {s: _m(0.5, 9.0, 0.2, 0.5, 0.7) for s in (11, 12, 13, 14, 15)}
    out = sel.select_seed(same)
    assert out["selected_seed"] == 11 and all(e["distance"] == 0.0 for e in out["per_seed"].values())
    twins = {s: dict(v) for s, v in BASE.items()}
    twins[11] = dict(twins[13])                          # two identical most-typical runs: the lower seed
    out = sel.select_seed(twins)
    assert out["per_seed"]["11"]["distance"] == out["per_seed"]["13"]["distance"] == min(e["distance"] for e in out["per_seed"].values())
    assert out["selected_seed"] == 11
    with pytest.raises(sel.SelectionError, match="at least three"):
        sel.select_seed({11: BASE[11], 12: BASE[12]})
    broken = {s: dict(v) for s, v in BASE.items()}
    broken[12]["recall"] = float("nan")
    with pytest.raises(sel.SelectionError, match="missing"):
        sel.select_seed(broken)


def test_the_rule_is_deterministic_and_ignores_input_order():
    rng = np.random.default_rng(0)
    for _ in range(200):
        m = {s: _m(rng.uniform(0.3, 0.7), rng.uniform(7, 13), rng.uniform(0.05, 0.3), rng.uniform(0.3, 0.8), rng.uniform(0.4, 0.9))
             for s in (11, 12, 13, 14, 15)}
        out = sel.select_seed(m)
        shuffled = {s: m[s] for s in (14, 11, 15, 13, 12)}
        strings = {str(s): v for s, v in m.items()}
        assert sel.select_seed(shuffled) == out == sel.select_seed(strings)
        assert out["selected_seed"] == _reference(m)
        if out["selected_seed"] is not None:
            assert out["selected_seed"] in out["eligible_seeds"]


# ---- what goes forward ---------------------------------------------------------------------------

def _stage(lstm_ok=True, only_ok=True):
    def one(ok, seed):
        return {"architecture_passes": ok, "selected_seed": seed if ok else None}
    return {"v2_dnn_lstm": one(lstm_ok, 13), "v2_dnn_only": one(only_ok, 14)}


def test_stage_outcome():
    both = sel.stage_outcome(_stage(), "v2_dnn_lstm")
    assert both["primary_candidate"] == "v2_dnn_lstm@13" and both["challenger"] == "v2_dnn_only@14"
    assert both["artifacts"] == ["v2_dnn_lstm@13", "v2_dnn_only@14"]
    lead_fails = sel.stage_outcome(_stage(lstm_ok=False), "v2_dnn_lstm")
    assert lead_fails["primary_candidate"] == "v2_dnn_only@14" and lead_fails["challenger"] is None
    assert "v2_dnn_lstm did not pass" in lead_fails["outcome"]
    challenger_fails = sel.stage_outcome(_stage(only_ok=False), "v2_dnn_lstm")
    assert challenger_fails["artifacts"] == ["v2_dnn_lstm@13"] and "v2_dnn_only did not pass" in challenger_fails["outcome"]
    none = sel.stage_outcome(_stage(False, False), "v2_dnn_lstm")
    assert none["artifacts"] == [] and none["primary_candidate"] is None and "production stays" in none["outcome"]
    # if stage A had chosen the simpler architecture, it leads
    assert sel.stage_outcome(_stage(), "v2_dnn_only")["primary_candidate"] == "v2_dnn_only@14"
    with pytest.raises(sel.SelectionError):
        sel.stage_outcome(_stage(), "production")


@pytest.mark.skipif(not MULTISEED_REPORT.exists(), reason="the multi-seed report has not been generated on this machine")
def test_stage_a_is_the_existing_rule_on_the_multi_seed_comparison():
    report = json.loads(MULTISEED_REPORT.read_text())
    lead = sel.architecture_lead(report)
    assert lead["lead"] == report["decision"]["provisional_lead"] == "v2_dnn_lstm" and lead["all_conditions_hold"]
    assert set(lead["conditions"]) == {"recall_at_least_matches", "alert_burden_at_least_matches", "critical_tier_not_worse"}
    # it reads the architecture-level comparison only: no per-seed hold-out result is needed
    stripped = {"holdout": {"primary": {"comparison": report["holdout"]["primary"]["comparison"]}}}
    assert sel.architecture_lead(stripped) == lead


# ---- the record is written once, before any final hold-out exists ------------------------------------

def test_record_is_written_once_and_never_into_a_model_directory(tmp_path):
    record = {"protocol": sel.PROTOCOL_VERSION}
    path = sel.write_record(record, tmp_path / "out", final_root=tmp_path / "final")
    assert path.name == "selection_record.json" and json.loads(path.read_text()) == record
    with pytest.raises(sel.SelectionError, match="made once"):
        sel.write_record(record, tmp_path / "out", final_root=tmp_path / "final")
    for bad in (config.MODELS_SAVED_DIR, config.CANDIDATES_DIR / "v2"):
        with pytest.raises(holdout.HoldoutError, match="refusing"):
            sel.write_record(record, bad, final_root=tmp_path / "final")


def test_selection_is_refused_once_a_final_holdout_exists(tmp_path):
    final = tmp_path / "final"
    assert not sel.final_holdout_exists(final)
    final.mkdir()
    assert not sel.final_holdout_exists(final)                  # an empty directory is not a hold-out
    (final / "seed_401").mkdir()
    assert sel.final_holdout_exists(final)
    with pytest.raises(sel.SelectionError, match="before it is generated"):
        sel.write_record({}, tmp_path / "out", final_root=final)
    assert not (tmp_path / "out" / sel.RECORD_NAME).exists()


def test_a_final_holdout_exists_only_after_the_selection_was_recorded():
    """Stage C generates the final hold-out; it may exist only once the Stage B record does."""
    assert not sel.final_holdout_exists() or RECORD.exists()


@pytest.mark.skipif(not RECORD.exists(), reason="stage B has not been run on this machine")
def test_the_recorded_selection_is_the_rule_applied_to_the_recorded_metrics():
    """The stage B record (written once, after the protocol was approved) holds exactly
    what the rule gives for the development metrics it records, and the selected
    artifacts are the files on disk."""
    record = json.loads(RECORD.read_text())
    assert record["protocol"] == sel.PROTOCOL_VERSION
    fixed = record["fixed_inputs"]
    assert fixed["eligible_training_seeds"] == list(sel.ELIGIBLE_SEEDS)
    assert fixed["development_seeds"] == [str(s) for s in sel.DEVELOPMENT_SEEDS]
    assert fixed["screens"] == {"minimum_recall": 0.40, "alert_screen_per_1000": 11.0,
                                "critical_recall_margin_below_median": 0.05, "eligible_seeds_needed": 3}
    assert not set(int(s) for s in record["development"]["datasets"]) & set(sel.SPENT_SEEDS + sel.FINAL_HOLDOUT_SEEDS + sel.FINAL_NEW_CUSTOMER_SEEDS)
    assert all(d["generator_version"] == "2.0.1" for d in record["development"]["datasets"].values())
    for a in sel.ELIGIBLE_ARCHITECTURES:
        recorded = record["stage_b_selection"][a]
        metrics = {s: e["metrics"] for s, e in recorded["per_seed"].items()}
        assert json.loads(json.dumps(sel.select_seed(metrics))) == recorded
    lead = record["stage_a_architecture"]["lead"]
    assert sel.stage_outcome(record["stage_b_selection"], lead) == record["stage_b_outcome"]
    assert list(record["selected_artifacts"]) == record["stage_b_outcome"]["artifacts"]
    for name, art in record["selected_artifacts"].items():
        arch, seed = name.split("@")
        assert arch in sel.ELIGIBLE_ARCHITECTURES and int(seed) in sel.ELIGIBLE_SEEDS
        assert int(seed) == record["stage_b_selection"][arch]["selected_seed"]
        directory = ms.seed_root(int(seed)) / ev.CANDIDATE_DIR[arch]
        manifest = json.loads((directory / "manifest.json").read_text())
        assert art["files_sha256"] == manifest["files"]
        assert art["weights_sha256"]["dnn_weights_sha256"] == manifest["model"]["dnn_weights_sha256"]
        for fname, sha in art["files_sha256"].items():
            if not fname.endswith(".keras"):
                assert holdout._sha256(directory / fname) == sha
        assert art["cutoffs"]["critical"] >= art["cutoffs"]["policy_b"]
    assert record["stage_c_final_holdout"] == "not generated, not scored"
    # the stored 4C-3E.6 record was written when "production" was the default (before Step 4D)
    assert record["state"] == {"model_set_default": "production", "promotion": "none", "models_trained": "none"}


def test_protocol_document_states_the_fixed_inputs():
    text = PROTOCOL_DOC.read_text(encoding="utf-8")
    for needle in ("NO PROMOTION", "301–305", "401–405", "411–415", "11, 12, 13, 14, 15", "11.0", "0.40",
                   "at least 3 of the 5", "most typical", "101–105", "201–205"):
        assert needle in text, needle


# ---- end to end on a small sample ------------------------------------------------------------------

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


def test_end_to_end_on_a_sample(tmp_path_factory):
    data_dir, eval_dir = tmp_path_factory.mktemp("sel_data"), tmp_path_factory.mktemp("sel_eval")
    subprocess.run([sys.executable, str(config.DATA_V2_DIR / "generate.py"), "--customers", "60", "--out", str(data_dir)],
                   check=True, capture_output=True)
    spec = resolve_dataset("v2").with_data_dir(data_dir).with_output_dir(eval_dir)
    data = load_evaluation_data(spec)
    _, _, definition = build_time_split(data.frame, data.grouping)
    save_definition(spec.time_split_path, definition)
    root, dev_root = tmp_path_factory.mktemp("sel_models"), tmp_path_factory.mktemp("sel_dev")
    seeds, dev_seeds = (3, 4, 5), (8,)
    before = ms.protected_checksums()
    for s in seeds:
        ms.train_seed(s, root, spec=spec, log=lambda m: None, fit_lstm_fn=_quick_lstm, fit_dnn_fn=_quick_dnn, n_folds=2)
    holdout.generate_dataset(8, dev_root, customers=60)
    files = ev.multiseed_file_checksums(seeds, root)

    kwargs = dict(training_seeds=seeds, root=root, development_seeds=dev_seeds, development_root=dev_root, spec=spec)
    dev = sel.development_metrics(**kwargs)
    assert sel.development_metrics(**kwargs) == dev
    assert set(dev["metrics"]) == set(sel.ELIGIBLE_ARCHITECTURES) and dev["population"]["datasets"] == 1
    for a in sel.ELIGIBLE_ARCHITECTURES:
        assert set(dev["metrics"][a]) == set(seeds)
        assert all(set(v) == set(sel.SELECTION_METRICS) for v in dev["metrics"][a].values())
    lead = {"lead": "v2_dnn_lstm"}
    record = sel.build_record(dev, lead, training_seeds=seeds, root=root)
    assert json.dumps(record) == json.dumps(sel.build_record(dev, lead, training_seeds=seeds, root=root))
    for a in sel.ELIGIBLE_ARCHITECTURES:
        assert record["stage_b_selection"][a] == json.loads(json.dumps(sel.select_seed(dev["metrics"][a])))
    for name, art in record["selected_artifacts"].items():
        arch, seed = name.split("@")
        manifest = json.loads((ms.seed_root(int(seed), root) / ev.CANDIDATE_DIR[arch] / "manifest.json").read_text())
        assert art["files_sha256"] == manifest["files"] and art["weights_sha256"]["dnn_weights_sha256"] == manifest["model"]["dnn_weights_sha256"]
        assert art["cutoffs"]["policy_b"] == dev["cutoffs"][name]["policy_b"]
    assert list(record["selected_artifacts"]) == record["stage_b_outcome"]["artifacts"]
    assert record["stage_c_final_holdout"] == "not generated, not scored"
    # the stored 4C-3E.6 record was written when "production" was the default (before Step 4D)
    assert record["state"] == {"model_set_default": DEFAULT_MODEL_SET, "promotion": "none", "models_trained": "none"}
    # the existing hold-out and the final hold-out can never be selection data
    for bad in ((101,), (201,), (401,), (42,)):
        with pytest.raises(sel.SelectionError):
            sel.development_metrics(**{**kwargs, "development_seeds": bad})
    assert ms.protected_checksums() == before and ev.multiseed_file_checksums(seeds, root) == files
