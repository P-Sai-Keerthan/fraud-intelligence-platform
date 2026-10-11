# Required model and code adaptations (nothing implemented yet)

Rule: **no weights, scalers, thresholds or manifests of the platform are reused.** A new, separate package (proposed `experiments/independent_ieee_cis/ieeecis/`) trains every model from scratch on IEEE-CIS. Code is reused only by copying small, tested functions, each listed below with the reason, so the platform and its protected evidence stay untouched.

## 1. What cannot be reused as is, and why

| Existing component [CODE] | Coupling that blocks reuse | Adaptation |
|---|---|---|
| `features/feature_engineering.py` (`FEATURE_COLUMNS`, `build_point_features`) | fixed nine columns; reads `is_new_device`, `is_new_location`, `failed_logins_24h` from the generator; `HOME_LOCATIONS` (ten Indian cities); per-row Python loop (about 441 s per 416 k rows) | new vectorised builder for the mapped schema (`FEATURE_MAPPING.md`); keep the *definitions* of features 1, 8, 9 and the 5 % rarity rules as documented functions; log-amount variant added; no `failed_logins_24h` |
| `evaluation/datasets.py` (`DatasetSpec`, `load_evaluation_data`) | assumes the v2 file layout, generator metadata columns, `groups.py` episodes and rings | new loader for the two tables with a stage-0 audit; entity and component grouping from the chosen proxy entity; no generator metadata (episodes, warning periods, rings do not exist) |
| `evaluation/split.py` | episode-aware time and customer splits | new time-segment splitter with embargo (the cut-points are protocol constants) |
| `evaluation/windows.py` (`build_windows`) | built on the nine features and `customer_id` | reuse the *logic* (window of the K earlier rows, current excluded, next-row target) in a new function over the new step features; keep `assert_past_only` as a test |
| `evaluation/stacking.py` (`fit_lstm`, `stacked_risk_scores`, 3 customer-grouped folds, chronological early stopping) | Keras, nine features | reuse the recipe for M2: out-of-fold LSTM scores by entity-grouped folds, early stopping on the chronologically last 15 % of train; new inputs |
| `models/lstm_model.py` | architecture `Masking → LSTM64 → LSTM32 → Dense16 → Dense1`, input 10 × 9 | same architecture family, input dimension from the new step features; weights trained from scratch |
| `models/shap_explainer.py`, `FEATURE_DISPLAY_NAMES` | display names of the nine features | new display-name table for the mapped features; masked Vesta features have no display names and are not offered |
| `model_sets.py`, `model_metadata.py`, `inference_pipeline.py`, `alerting.py` | SHA-pinned platform artifacts, nine-feature shapes, synthetic provenance | **not touched**; the experiment writes its own manifests |
| `evaluation/metrics.py` | `episode_metrics` calls first-fraud recall an "early-warning rate" (Experiment 2 finding) | new metric code with wording-lint test; do not reuse the name |
| Alert levels (R-02) | serving policy | out of scope; the experiment reports operating points at validation-fitted FPR targets, not the platform's tiers |

## 2. New components required

1. **Stage-0 auditor** — label-blind schema report (`AUDIT0.json`) with checks listed in the protocol.
2. **Entity builder** — E1–E3 variants; consistency diagnostics; connected components when entities overlap.
3. **Vectorised past-only feature builder** — grouped cumulative statistics by entity ordered by `TransactionDT` (strict inequality), tested against a slow reference implementation on a sample (IL1, IL5).
4. **Sequence builder** — lazy windows (K = 10 for M2; K = 32 optional for M3), padding masks, no stored N×K×D tensor (IL20).
5. **Learners** — scikit-learn random forest and logistic regression (installed); **LightGBM** (not installed in the project environment; verified absent here) and, for M3, a recurrent classifier: the project's TensorFlow/Keras 2.21 can be used (installed, CPU) — PyTorch is optional (not installed; measured 1.6–2.8× faster on GRU in Experiment 2's micro-benchmarks).
6. **Randomised cut-off** (Experiment 2 v1 §7, test L16) and **paired entity-cluster bootstrap**.
7. **Ledger and freeze tooling** — tags, hashes, one-time final access, refusal without both tags.
8. **Wording lint** — forbidden phrases (§1 of the protocol).

## 3. Dependencies to add (environment, not the platform's `requirements.txt`)

`lightgbm`, `pyarrow` (columnar cache of the joined table; optional), `pandas`/`numpy`/`scikit-learn`/`tensorflow` already pinned in the project; pin exact versions in `experiments/independent_ieee_cis/requirements.txt` at Freeze-1. A separate virtual environment is used so the platform's pins are unaffected.

## 4. Sequential-model design choices to settle before Freeze-1

* M2 keeps the platform's target (is the *next* row fraudulent) because that is the deployed design; M3 uses the *current* row's label. Both are reported; neither is tuned on final data.
* Window length K = 10 (platform) with a sensitivity K = 32 only if stage 0 shows long histories.
* Unknown-device and missing-identity states are explicit inputs (never zero-filled as "seen").
* Class imbalance (about 3.5 % positives): the platform's LSTM used class weights `n_neg/n_pos`; here class weight and negative-sampling ratio are search-space entries for every arm.
