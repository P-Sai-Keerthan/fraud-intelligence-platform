"""Hold-out evaluation harness (app/evaluation/holdout.py, app/evaluation/ring_metrics.py).

Uses a small generated dataset (60 customers, seed 7) in a temp directory and
the saved model sets, read-only. Nothing here writes to models/saved/,
models/candidates/ or data/.
"""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from app import config
from app.evaluation import holdout, ring_metrics
from app.features.feature_engineering import FEATURE_COLUMNS, build_point_features
from app.inference_pipeline import RAW_COLUMNS_FOR_FEATURES, FraudIntelligencePipeline
from app.model_sets import DEFAULT_MODEL_SET, load_model_set

SAMPLE_SEED = 7
MODELS = list(holdout.MODEL_SETS)


@pytest.fixture(scope="module")
def sample_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("holdout_sample")
    holdout.generate_dataset(SAMPLE_SEED, root, customers=60)
    return root


@pytest.fixture(scope="module")
def sample(sample_root):
    return holdout.load_holdout(SAMPLE_SEED, sample_root)


@pytest.fixture(scope="module")
def model_sets():
    return {name: load_model_set(name) for name in MODELS}


@pytest.fixture(scope="module")
def scored(sample, model_sets):
    return holdout.score_dataset(sample, model_sets)


# ---- fixed inputs -------------------------------------------------------------------------------

def test_fixed_inputs_are_the_approved_ones():
    assert holdout.HOLDOUT_SEEDS == (101, 102, 103, 104, 105)
    assert holdout.TRAINING_SEED == 42 and 42 not in holdout.HOLDOUT_SEEDS
    assert (holdout.RECALL_MARGIN, holdout.LEGIT_ALERT_MARGIN_PER_1000, holdout.CRITICAL_RECALL_MARGIN) == (0.05, 1.0, 0.05)
    assert holdout.MIN_PRIOR == 10 and holdout.SECONDARY_START == "2026-06-01"
    assert holdout.MIN_RECALL == 0.40 and holdout.ALERT_BUDGET_PER_1000 == 10.0
    assert holdout.BOOTSTRAP_REPS == 2000
    assert DEFAULT_MODEL_SET == "production"


def test_cutoffs_are_the_4c3b_values_and_cannot_be_changed():
    assert dict(holdout.CUTOFFS["policy_b"]) == {"production": 0.9680148363113403, "v2_dnn_lstm": 0.76771479845047,
                                                 "v2_dnn_only": 0.7744449377059937}
    assert dict(holdout.CUTOFFS["critical"]) == {"production": 0.9950721263885498, "v2_dnn_lstm": 0.9680655598640442,
                                                 "v2_dnn_only": 0.9119433164596558}
    with pytest.raises(TypeError):
        holdout.CUTOFFS["policy_b"]["v2_dnn_lstm"] = 0.5
    with pytest.raises(TypeError):
        holdout.CUTOFFS["policy_b"] = {}


def test_alerts_use_only_the_fixed_cutoffs(scored):
    before = {tier: dict(v) for tier, v in holdout.CUTOFFS.items()}
    for m in MODELS:
        s = scored[m].to_numpy()
        assert np.array_equal(scored[f"alert_{m}"].to_numpy(), s >= np.float32(holdout.CUTOFFS["policy_b"][m]))
        assert np.array_equal(scored[f"critical_{m}"].to_numpy(), s >= np.float32(holdout.CUTOFFS["critical"][m]))
        assert not (scored[f"critical_{m}"] & ~scored[f"alert_{m}"]).any()      # Critical is inside Policy B
    assert {tier: dict(v) for tier, v in holdout.CUTOFFS.items()} == before


# ---- the training data is refused ------------------------------------------------------------------

def _fake(seed, sha="0" * 64, recorded=None):
    spec = SimpleNamespace(features_csv=SimpleNamespace(name="transactions_with_features.csv"))
    files = {"transactions_with_features.csv": {"sha256": sha if recorded is None else recorded}}
    return SimpleNamespace(manifest={"seed": seed, "files": files}, sha256=sha, spec=spec)


def test_seed_42_is_refused(tmp_path):
    with pytest.raises(holdout.HoldoutError, match="training dataset"):
        holdout.check_holdout(_fake(42))
    with pytest.raises(holdout.HoldoutError, match="training dataset"):
        holdout.generate_dataset(42, tmp_path)
    assert not list(tmp_path.iterdir())


def test_the_training_file_is_refused_whatever_its_seed_says():
    shas = holdout.training_dataset_sha256()
    assert shas, "the candidate manifests record the training dataset"
    with pytest.raises(holdout.HoldoutError, match="trained on"):
        holdout.check_holdout(_fake(101, sha=sorted(shas)[0]))


def test_wrong_seed_missing_seed_and_modified_file_are_refused(sample):
    holdout.check_holdout(sample, expected_seed=SAMPLE_SEED)
    with pytest.raises(holdout.HoldoutError, match="expected a dataset generated with seed 101"):
        holdout.check_holdout(sample, expected_seed=101)
    with pytest.raises(holdout.HoldoutError, match="no generator seed"):
        holdout.check_holdout(SimpleNamespace(manifest={}, sha256="x", spec=None))
    with pytest.raises(holdout.HoldoutError, match="does not match the checksum"):
        holdout.check_holdout(_fake(101, sha="a" * 64, recorded="b" * 64))


# ---- scoring ---------------------------------------------------------------------------------

def test_populations(scored):
    prim, sec, early = (holdout.population(scored, n) for n in ("primary", "secondary", "early_history"))
    assert (prim["n_prior"] >= 10).all() and (early["n_prior"] < 10).all()
    assert len(prim) + len(early) == len(scored)
    assert (sec["n_prior"] >= 10).all() and (sec["timestamp"] >= pd.Timestamp("2026-06-01")).all()
    assert len(sec) == ((scored["n_prior"] >= 10) & (scored["timestamp"] >= pd.Timestamp("2026-06-01"))).sum()
    first_rows = scored.groupby("customer_id").head(10)
    assert (first_rows["n_prior"] < 10).all() and len(first_rows) == len(early)
    with pytest.raises(ValueError):
        holdout.population(scored, "other")


@pytest.mark.parametrize("name", MODELS)
def test_scoring_matches_the_live_pipeline(name, sample, scored):
    """The batch replication and FraudIntelligencePipeline.score_transaction agree on
    sampled rows, with and without 10 earlier transactions. The pipeline reports
    the score x100 rounded to 2 decimals, so agreement is to that rounding."""
    frame = sample.frame
    n_prior = scored["n_prior"].to_numpy()
    rng = np.random.default_rng(0)
    rows = np.concatenate([rng.choice(np.flatnonzero(n_prior == k), 2, replace=False) for k in (0, 1, 5, 9, 10, 11, 40)]
                          + [np.flatnonzero((frame["is_fraud"] == 1) & (n_prior >= 10))[:4]])
    pipeline = FraudIntelligencePipeline(model_set=name)
    for r in rows:
        row = frame.iloc[r]
        customer = frame[frame["customer_id"] == row["customer_id"]]
        prior = customer[customer.index < r]
        assert len(prior) == n_prior[r]
        raw = prior[RAW_COLUMNS_FOR_FEATURES].copy()
        raw["is_fraud"] = 0
        pipeline.customer_histories[row["customer_id"]] = build_point_features(raw) if len(raw) else raw.iloc[:0]
        out = pipeline.score_transaction({k: row[k] for k in ("customer_id", "amount", "merchant_category", "device_id",
                                                               "location", "failed_logins_24h", "timestamp")})
        assert abs(float(scored[name].iloc[r]) * 100 - out["fraud_probability"]) <= 0.005 + 1e-6, (name, int(r), int(n_prior[r]))


def test_scores_are_capped_and_do_not_depend_on_the_other_rows(sample, model_sets, scored):
    for m in MODELS:
        assert float(scored[m].max()) <= np.float32(holdout.SCORE_CAP) and float(scored[m].min()) >= 0.0
    # scoring a subset of customers reproduces their scores bit for bit
    keep = sample.frame["customer_id"].isin(sorted(sample.frame["customer_id"].unique())[:7]).to_numpy()
    sub = sample.frame[keep].reset_index(drop=True)
    for m in MODELS:
        assert np.array_equal(holdout.score_frame(sub, model_sets[m]), scored[m].to_numpy()[keep])


def test_no_ground_truth_reaches_the_model_inputs(sample, model_sets):
    from app.features.ground_truth import GROUND_TRUTH_COLUMNS
    assert not set(sample.frame.columns) & set(GROUND_TRUTH_COLUMNS)
    for ms in model_sets.values():
        assert not set(ms.dnn_input_columns) & set(GROUND_TRUTH_COLUMNS)
        assert list(ms.dnn_input_columns)[:len(FEATURE_COLUMNS)] == list(FEATURE_COLUMNS)


# ---- weighted ranking metrics ---------------------------------------------------------------------

@pytest.mark.parametrize("ties", [False, True])
def test_weighted_rank_metrics_match_scikit_learn(ties):
    rng = np.random.default_rng(3)
    n = 4000
    y = (rng.random(n) < 0.05).astype(float)
    s = rng.random(n).astype(np.float32) + 0.3 * y.astype(np.float32)
    if ties:
        s = np.minimum(np.round(s, 1), np.float32(0.9))          # heavy ties, and a cap as in the pipeline
    w = rng.integers(0, 4, n).astype(float)                       # bootstrap-like counts, zeros included
    order, ends = holdout.rank_prepare(s)
    for weights in (np.ones(n), w):
        ap, auc = holdout.weighted_rank_metrics((weights * y)[order], (weights * (1 - y))[order], ends)
        assert ap == pytest.approx(average_precision_score(y, s, sample_weight=weights), abs=1e-12)
        assert auc == pytest.approx(roc_auc_score(y, s, sample_weight=weights), abs=1e-12)


def test_weighted_rank_metrics_without_positives_or_negatives():
    order, ends = holdout.rank_prepare(np.array([0.1, 0.2, 0.3]))
    assert all(np.isnan(v) for v in holdout.weighted_rank_metrics(np.zeros(3), np.ones(3), ends))
    assert all(np.isnan(v) for v in holdout.weighted_rank_metrics(np.ones(3), np.zeros(3), ends))


# ---- metrics and the group bootstrap -----------------------------------------------------------------

def _hand_table():
    """Two datasets. Every group has two customers: one whose fraud is caught and
    one whose fraud is missed, so recall is exactly 0.5 in every whole group."""
    rows = []
    for seed in (1, 2):
        for g in range(6):
            for half, caught in ((0, True), (1, False)):
                cust = f"C{seed}{g}{half}"
                for i in range(4):
                    fraud = i == 3
                    score = 0.99 if (fraud and caught) else (0.90 if (not fraud and i == 0 and g % 3 == 0) else 0.01 + 0.001 * i)
                    rows.append({"seed": seed, "customer_id": cust, "transaction_id": f"{cust}-{i}",
                                 "timestamp": pd.Timestamp("2026-06-10") + pd.Timedelta(hours=len(rows)),
                                 "device_id": f"D{cust}", "is_fraud": int(fraud), "n_prior": 10 + i,
                                 "group": f"{seed}:G{g}", "episode": f"{seed}:{g * 2 + half + 1}" if fraud else f"{seed}:0",
                                 "fraud_type": "x" if fraud else "none", "fraud_episode_id": g * 2 + half + 1 if fraud else 0,
                                 "fraud_stage": "first" if fraud else "none", "fraud_ring_id": 0, "legit_context": "none"})
                    for m in MODELS:
                        rows[-1][m] = np.float32(score)
    t = pd.DataFrame(rows)
    for m in MODELS:
        t[f"alert_{m}"] = t[m] >= 0.5
        t[f"critical_{m}"] = t[m] >= 0.95
    return t


def test_point_metrics_on_a_hand_built_table():
    t = _hand_table()
    res = holdout.evaluate_population(t, MODELS, reps=200, seed=1)
    assert (res["rows"], res["fraud_transactions"], res["fraud_episodes"], res["groups"], res["datasets"]) == (96, 24, 24, 12, 2)
    assert res["customers"] == 24
    m = res["models"]["v2_dnn_lstm"]
    # 12 frauds caught of 24; 8 legitimate rows alerted (groups 0 and 3, both customers, both datasets)
    assert m["counts"] == {"true_positives": 12, "false_positives": 8, "false_negatives": 12, "critical_true_positives": 12,
                           "critical_false_positives": 0, "first_frauds_detected": 12, "episodes_detected": 12}
    v = {k: e["value"] for k, e in m["metrics"].items()}
    assert v["recall"] == 0.5 and v["precision"] == 0.6 and v["critical_precision"] == 1.0
    assert v["f1"] == pytest.approx(2 * 12 / (2 * 12 + 8 + 12))
    assert v["legit_alerts_per_1000"] == pytest.approx(1000 * 8 / 96)
    assert v["false_positive_rate"] == pytest.approx(8 / 72)
    assert v["first_fraud_recall"] == 0.5 and v["episode_detection_rate"] == 0.5
    y, s = t["is_fraud"], t["v2_dnn_lstm"]
    assert v["pr_auc"] == pytest.approx(average_precision_score(y, s), abs=1e-6)
    assert v["roc_auc"] == pytest.approx(roc_auc_score(y, s), abs=1e-6)


def test_bootstrap_is_deterministic_and_keeps_groups_together():
    t = _hand_table()
    a = holdout.evaluate_population(t, MODELS, reps=300, seed=5)
    b = holdout.evaluate_population(t.sample(frac=1.0, random_state=0), MODELS, reps=300, seed=5)
    assert json.dumps(a["models"], sort_keys=True) == json.dumps(holdout.evaluate_population(t, MODELS, reps=300, seed=5)["models"], sort_keys=True)
    assert a["models"]["production"]["metrics"]["recall"] == b["models"]["production"]["metrics"]["recall"]
    # every whole group has recall exactly 0.5, so a group-level resample cannot move it;
    # resampling single customers would
    for m in MODELS:
        assert a["models"][m]["metrics"]["recall"]["ci95"] == [0.5, 0.5]
        assert a["models"][m]["metrics"]["episode_detection_rate"]["ci95"] == [0.5, 0.5]
        lo, hi = a["models"][m]["metrics"]["legit_alerts_per_1000"]["ci95"]
        assert lo < 1000 * 8 / 96 < hi                           # false positives differ between groups
    by_customer = holdout.evaluate_population(t.assign(group=t["seed"].astype(str) + ":" + t["customer_id"]), MODELS, reps=300, seed=5)
    lo, hi = by_customer["models"]["production"]["metrics"]["recall"]["ci95"]
    assert lo < 0.5 < hi
    # identical scores for every model set: every paired difference is exactly zero
    for pair in a["paired_differences"].values():
        assert pair["recall"] == {"difference": 0.0, "ci95": [0.0, 0.0], "excludes_zero": False}


def test_bootstrap_resamples_within_each_dataset():
    """With all fraud in one dataset, a stratified resample always contains that dataset's
    groups, so the number of fraud episodes in every resample stays 12 of 12 groups drawn."""
    t = _hand_table()
    t.loc[t["seed"] == 2, ["is_fraud"]] = 0
    t.loc[t["seed"] == 2, ["fraud_stage"]] = "none"
    res = holdout.evaluate_population(t, MODELS, reps=200, seed=2)
    assert res["fraud_episodes"] == 12
    assert res["models"]["production"]["metrics"]["recall"]["ci95"] == [0.5, 0.5]     # never an empty (NaN-only) resample


# ---- rings -------------------------------------------------------------------------------------

def _ring_table():
    ts = pd.Timestamp("2026-03-01")
    rows = [
        # ring 1 (seed 1): victims A (2 frauds), B (1), C (1)
        (1, "A", 1, "first", 1, 0, True, True), (1, "A", 1, "subsequent", 1, 5, False, True),
        (1, "B", 2, "first", 1, 2, True, False), (1, "C", 3, "first", 1, 3, False, False),
        # ring 1 of another dataset (seed 2): a different ring; nothing caught by "a"
        (2, "D", 1, "first", 1, 0, False, False), (2, "E", 2, "first", 1, 1, False, True),
        # non-ring fraud
        (1, "F", 9, "first", 0, 0, True, True), (1, "G", 10, "first", 0, 0, False, True),
        # legitimate
        (1, "A", 0, "none", 0, 9, True, False),
    ]
    return pd.DataFrame([{"seed": s, "customer_id": c, "transaction_id": f"T{i}", "timestamp": ts + pd.Timedelta(hours=h),
                          "is_fraud": int(e > 0), "fraud_episode_id": e, "fraud_stage": st, "fraud_ring_id": r,
                          "alert_a": a, "alert_b": b} for i, (s, c, e, st, r, h, a, b) in enumerate(rows)])


def test_ring_metrics_on_a_hand_built_example():
    rep = ring_metrics.ring_report(_ring_table(), ["a", "b"], reps=200, seed=1)
    assert (rep["rings"], rep["victim_episodes"], rep["ring_fraud_transactions"]) == (2, 5, 6)
    r1, r2 = rep["per_ring"]
    assert (r1["seed"], r1["ring"], r1["victims"], r1["fraud_transactions"], r1["first_fraud_transactions"]) == (1, 1, 3, 4, 3)
    assert r1["a"] == {"fraud_caught": 2, "victims_alerted": 2, "first_frauds_caught": 2, "hours_from_first_fraud_to_first_alert": 0.0}
    assert r1["b"] == {"fraud_caught": 2, "victims_alerted": 1, "first_frauds_caught": 1, "hours_from_first_fraud_to_first_alert": 0.0}
    assert r2["a"] == {"fraud_caught": 0, "victims_alerted": 0, "first_frauds_caught": 0, "hours_from_first_fraud_to_first_alert": None}
    assert r2["b"]["hours_from_first_fraud_to_first_alert"] == 1.0
    a, b = rep["summary"]["a"], rep["summary"]["b"]
    assert a["ring_level"]["rings_with_at_least_one_victim_alerted"]["k"] == 1
    assert a["ring_level"]["rings_with_at_least_two_victims_alerted"]["k"] == 1
    assert b["ring_level"]["rings_with_at_least_one_victim_alerted"]["k"] == 2
    assert b["ring_level"]["rings_with_at_least_two_victims_alerted"]["k"] == 0
    assert b["ring_level"]["median_hours_from_first_fraud_to_first_alert"] == 0.5
    assert (a["victim_level"]["victim_episodes_detected"]["k"], a["victim_level"]["victim_episodes_detected"]["n"]) == (2, 5)
    assert (b["victim_level"]["victim_first_frauds_detected"]["k"], b["victim_level"]["victim_first_frauds_detected"]["n"]) == (2, 5)
    assert (a["transaction_level"]["ring_fraud_caught"]["k"], a["transaction_level"]["ring_fraud_caught"]["n"]) == (2, 6)
    assert (a["transaction_level"]["non_ring_fraud_caught"]["k"], b["transaction_level"]["non_ring_fraud_caught"]["k"]) == (1, 2)
    assert a["transaction_level"]["ring_fraud_caught"]["wilson95"] is not None
    d = rep["paired_differences"]["a - b"]
    assert d["victim_episode_detection_rate"]["difference"] == 0.0
    assert d["ring_fraud_transaction_recall"]["difference"] == pytest.approx(-1 / 6, abs=1e-4)
    assert ring_metrics.ring_report(_ring_table(), ["a", "b"], reps=200, seed=1) == rep


def test_shared_device_rule_counts(sample):
    res = ring_metrics.shared_device_rule(sample.frame, sample.metadata, sample.grouping.episodes)
    assert sum(res["by_kind"].values()) == res["devices_flagged"]
    assert res["rings"] == sample.grouping.episodes.query("fraud_ring_id > 0")["fraud_ring_id"].nunique() >= 2
    assert 0 <= res["rings_with_a_flagged_ring_device"] <= res["rings"]


# ---- the decision rule ---------------------------------------------------------------------------

def _primary(lstm_minus_only, production_recall=0.30, lstm=(0.45, [0.41, 0.50], 7.0, [6.0, 8.0]),
             only=(0.44, [0.40, 0.49], 8.0, [7.0, 9.0])):
    def model(recall, rci, alerts, aci):
        return {"metrics": {"recall": {"value": recall, "ci95": rci}, "legit_alerts_per_1000": {"value": alerts, "ci95": aci}}}

    def diff(ci):
        return {"difference": (ci[0] + ci[1]) / 2, "ci95": ci, "excludes_zero": ci[0] > 0 or ci[1] < 0}
    zero = {k: diff([-0.01, 0.01]) for k in holdout.PRIMARY_ENDPOINTS}
    return {"models": {"production": model(production_recall, [0.25, 0.35], 9.0, [8.0, 10.0]),
                       "v2_dnn_lstm": model(*lstm), "v2_dnn_only": model(*only)},
            "paired_differences": {"v2_dnn_lstm - v2_dnn_only": {k: diff(v) for k, v in lstm_minus_only.items()},
                                   "v2_dnn_lstm - production": zero, "v2_dnn_only - production": zero}}


def test_decision_rule_lead_stays_when_all_three_conditions_hold():
    d = holdout.decision_rule(_primary({"recall": [-0.02, 0.04], "legit_alerts_per_1000": [-1.5, -0.2],
                                        "critical_recall": [-0.03, 0.03]}))
    tb = d["candidate_tie_break"]
    assert tb["all_conditions_hold"] and tb["provisional_lead"] == "v2_dnn_lstm"
    assert tb["primary_endpoints_separated"] == ["legit_alerts_per_1000"]
    assert [c["endpoint"] for c in d["primary_endpoint_intervals_excluding_zero"]] == ["legit_alerts_per_1000"]
    assert "MODEL_SET stays production" in d["promotion"]


@pytest.mark.parametrize("override, failed", [
    ({"recall": [-0.06, 0.02]}, "recall_at_least_matches"),
    ({"recall": [-0.05, 0.02]}, "recall_at_least_matches"),                       # the margin itself is not inside
    ({"legit_alerts_per_1000": [-0.5, 1.2]}, "alert_burden_at_least_matches"),
    ({"legit_alerts_per_1000": [-0.5, 1.0]}, "alert_burden_at_least_matches"),
    ({"critical_recall": [-0.08, 0.0]}, "critical_tier_not_worse"),
])
def test_decision_rule_any_failed_condition_gives_the_simpler_model(override, failed):
    base = {"recall": [-0.02, 0.04], "legit_alerts_per_1000": [-0.5, 0.5], "critical_recall": [-0.03, 0.03]}
    tb = holdout.decision_rule(_primary({**base, **override}))["candidate_tie_break"]
    assert not tb["conditions"][failed]["holds"] and not tb["all_conditions_hold"]
    assert tb["provisional_lead"] == "v2_dnn_only" and "simplicity" in tb["outcome"]


def test_decision_rule_against_production():
    base = {"recall": [-0.02, 0.04], "legit_alerts_per_1000": [-0.5, 0.5], "critical_recall": [-0.03, 0.03]}
    d = holdout.decision_rule(_primary(base))["against_production"]
    assert d["candidates"]["v2_dnn_lstm"]["eligible_on_this_evidence"] and d["candidates"]["v2_dnn_only"]["eligible_on_this_evidence"]
    # over budget (upper bound), recall below 0.40, or interval reaching production's point estimate
    over = holdout.decision_rule(_primary(base, lstm=(0.45, [0.41, 0.50], 9.5, [8.5, 10.4])))["against_production"]["candidates"]["v2_dnn_lstm"]
    assert not over["within_alert_budget"] and not over["eligible_on_this_evidence"]
    low = holdout.decision_rule(_primary(base, lstm=(0.39, [0.35, 0.43], 7.0, [6.0, 8.0])))["against_production"]["candidates"]["v2_dnn_lstm"]
    assert not low["recall_at_least_minimum"] and not low["eligible_on_this_evidence"]
    near = holdout.decision_rule(_primary(base, production_recall=0.41))["against_production"]
    assert not near["candidates"]["v2_dnn_lstm"]["recall_lower_bound_above_production"]
    assert not near["any_candidate_eligible"]


# ---- the whole run ---------------------------------------------------------------------------------

def test_report_is_reproducible_and_model_files_are_untouched(sample_root, model_sets, scored, tmp_path):
    before = holdout.model_file_checksums()
    assert before["production"] and before["candidates"]
    first = holdout.evaluate([SAMPLE_SEED], sample_root, reps=40, model_sets=model_sets)
    second = holdout.evaluate([SAMPLE_SEED], sample_root, reps=40, model_sets=model_sets)
    a = holdout.write_report(first, tmp_path / "a").read_bytes()
    b = holdout.write_report(second, tmp_path / "b").read_bytes()
    assert a == b and b"\r" not in a
    assert holdout.model_file_checksums() == before == first["model_files_sha256"]
    assert first["state"] == {"model_set_default": "production", "promotion": "none", "models_trained": "none",
                              "thresholds_changed": "none"}
    assert first["fixed_inputs"]["cutoffs_score_0_1"] == {t: dict(v) for t, v in holdout.CUTOFFS.items()}
    prim = first["pooled"]["primary"]
    assert prim["rows"] == int((scored["n_prior"] >= 10).sum())
    assert prim["fraud_transactions"] == int(((scored["n_prior"] >= 10) & (scored["is_fraud"] == 1)).sum())
    assert set(first["decision_rule"]["candidate_tie_break"]["conditions"]) == {
        "recall_at_least_matches", "alert_burden_at_least_matches", "critical_tier_not_worse"}
    assert first["gates"]["independent_data_seeds"]["met"] is False           # one sample dataset is not five seeds
    for m in MODELS:
        for metric in holdout.METRICS:
            assert set(prim["models"][m]["metrics"][metric]) == {"value", "ci95"}


def test_output_into_a_model_directory_is_refused(tmp_path):
    for bad in (config.MODELS_SAVED_DIR, config.CANDIDATES_DIR, config.CANDIDATES_DIR / "v2" / "dnn_lstm" / "x"):
        with pytest.raises(holdout.HoldoutError, match="refusing"):
            holdout.write_report({}, bad)
        with pytest.raises(holdout.HoldoutError, match="refusing"):
            holdout.assert_safe_output(bad)
    assert holdout.assert_safe_output(tmp_path) == tmp_path.resolve()
    assert holdout.seed_dir(101) == config.DATA_DIR / "v2_holdout" / "seed_101"
    assert holdout.OUTPUT_DIR == config.EVALUATION_DIR / "v2_holdout"


# ---- new-customer fraud (generator extension, separate datasets) ---------------------------------------

NEW_SEED = 9


@pytest.fixture(scope="module")
def new_customer_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("new_customer_sample")
    holdout.generate_dataset(NEW_SEED, root, customers=60, late_joiner_share=0.3, new_customer_fraud_episodes=10)
    return root


def test_new_customer_fixed_inputs():
    assert holdout.NEW_CUSTOMER_SEEDS == (201, 202, 203, 204, 205)
    assert not set(holdout.NEW_CUSTOMER_SEEDS) & set(holdout.HOLDOUT_SEEDS) and 42 not in holdout.NEW_CUSTOMER_SEEDS
    assert holdout.NEW_CUSTOMER_DATA_DIR == config.DATA_DIR / "v2_holdout" / "new_customer"
    assert holdout.NEUTRAL_FLAGS == ("hour_is_unusual", "category_is_unusual")
    assert len(holdout.NEW_CUSTOMER_SEEDS) * holdout.NEW_CUSTOMER_EPISODES_PER_SEED >= holdout.MIN_EPISODES


def test_neutral_flags_change_only_rows_with_fewer_than_ten_earlier_transactions(sample, model_sets, scored):
    full = scored["n_prior"].to_numpy() >= 10
    for m in MODELS:
        neutral = holdout.score_frame(sample.frame, model_sets[m], neutral=holdout.NEUTRAL_FLAGS)
        assert np.array_equal(neutral[full], scored[m].to_numpy()[full])
        assert not np.array_equal(neutral[~full], scored[m].to_numpy()[~full])


def test_new_customer_report(new_customer_root, model_sets, tmp_path):
    before = holdout.model_file_checksums()
    rep = holdout.evaluate_new_customer([NEW_SEED], new_customer_root, reps=30, model_sets=model_sets)
    again = holdout.evaluate_new_customer([NEW_SEED], new_customer_root, reps=30, model_sets=model_sets)
    a = holdout.write_report(rep, tmp_path / "a", holdout.NEW_CUSTOMER_REPORT_NAME).read_bytes()
    assert a == holdout.write_report(again, tmp_path / "b", holdout.NEW_CUSTOMER_REPORT_NAME).read_bytes()
    assert (tmp_path / "a" / "new_customer_report.json").exists()
    assert holdout.model_file_checksums() == before

    data = holdout.load_holdout(NEW_SEED, new_customer_root)
    assert data.manifest["generator_version"] == "2.1.0"
    assert "prior_transactions_at_first_fraud" not in data.frame.columns
    n_prior = holdout.prior_counts(data.frame)
    early_fraud = int(((n_prior < 10) & (data.frame["is_fraud"] == 1)).sum())
    info = rep["datasets"][str(NEW_SEED)]
    assert info["new_customer_fraud_episodes"] == 10 and info["late_joiners"] == 18
    assert rep["gates"]["new_customer_fraud_episodes"] == {"value": 10, "required": 100, "met": False}
    pop = rep["population"]
    assert pop["fraud_transactions"] == early_fraud >= 10 and pop["first_fraud_transactions"] == 10
    assert pop["rows"] == int((n_prior < 10).sum())

    assert set(rep["variants"]) == {"current_policy_b", "critical_only", "critical_only_flags_neutral", "policy_b_flags_neutral"}
    for m in MODELS:
        cur, crit = (rep["variants"][v]["models"][m]["counts"] for v in ("current_policy_b", "critical_only"))
        # the Critical cut-off is above the Policy B cut-off, so it alerts on a subset
        assert crit["true_positives"] <= cur["true_positives"] and crit["false_positives"] <= cur["false_positives"]
        assert cur["true_positives"] + cur["false_negatives"] == early_fraud
        d = rep["variant_differences_per_model_set"]["critical_only - current_policy_b"][m]
        assert d["recall"]["difference"] <= 0 and d["legit_alerts_per_1000"]["difference"] <= 0
        for variant in rep["variants"]:
            buckets = rep["by_earlier_transactions"][variant]
            parts = [buckets[b] for b in ("0", "1", "2", "3-9")]
            assert sum(p["fraud_transactions"] for p in parts) == buckets["all"]["fraud_transactions"] == early_fraud
            assert sum(p[m]["legitimate_alerts"] for p in parts) == buckets["all"][m]["legitimate_alerts"] \
                == rep["variants"][variant]["models"][m]["counts"]["false_positives"]
            assert sum(p[m]["fraud_caught"] for p in parts) == rep["variants"][variant]["models"][m]["counts"]["true_positives"]
    full = rep["full_history_policy_b"]
    assert set(full["unchanged_customers_per_seed"]) == {str(NEW_SEED)}
    assert full["unchanged_customers_pooled"]["rows"] + full["late_joiners_pooled"]["rows"] == int((n_prior >= 10).sum())
    assert full["unchanged_customers_pooled"] == full["unchanged_customers_per_seed"][str(NEW_SEED)]
    assert rep["state"]["model_set_default"] == "production" and rep["state"]["cold_start_behaviour_changed"].startswith("none")
    assert "not that it works on real new-customer fraud" in rep["limit"]
