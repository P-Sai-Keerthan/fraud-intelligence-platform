# Step 4C-2e-a: production retraining design (audit)

This is a read-only audit of the code at `ae2f2f5`. No model was trained or replaced, and no application code, frontend or data was changed. Where the audit states a number, it was computed read-only from the committed files and the local v2 evaluation artefacts.

Evidence it builds on: `docs/step4c2-v2-evaluation.md`. On the v2 test split (14 episodes), PR-AUC is DNN + LSTM 0.104, DNN only 0.128, LSTM 0.135 and logistic regression 0.138, with overlapping intervals. The LSTM contribution is not consistently shown. First-fraud recall is 1/14 for DNN + LSTM and 5/14 for DNN only.

## 1. Current production training architecture

| Item | Current production (`app/models/lstm_model.py`, `dnn_model.py`, `features/feature_engineering.py`) |
|---|---|
| Training data | `config.FEATURES_CSV` = `data/transactions_with_features.csv` (v1), fixed path |
| Features | 9 `FEATURE_COLUMNS`: `amount_zscore, hour_is_unusual, is_new_device, is_new_location, is_foreign_location, failed_logins_24h, category_is_unusual, txn_velocity_1h, amount_pct_of_avg`. `build_point_features` computes them from each customer's earlier history only. |
| LSTM | Input (10, 9) → Masking(0.0) → LSTM(64, return_sequences) → Dropout 0.3 → LSTM(32) → Dropout 0.2 → Dense(16, relu) → Dense(1, sigmoid). Adam 1e-3, binary cross-entropy. |
| LSTM data | `build_sequences`: one window per transaction that has ≥ 10 earlier transactions. Target = `is_fraud` of the next transaction. Saved to `data/lstm_X.npy`, `lstm_y.npy`, `lstm_meta.csv`. |
| DNN | Input (10) = 9 features + `risk_score` (LSTM probability × 100) → Dense 64 relu → BatchNorm → Dropout 0.3 → Dense 32 relu → Dropout 0.2 → Dense 16 relu → Dense(1, sigmoid). Adam 1e-3, binary cross-entropy. |
| DNN data | Every sequence is scored by the **final, fully trained** LSTM (in-sample risk scores), then joined to its transaction. |
| Split | `train_test_split(test_size=0.2, stratify=y, random_state=42)` over rows, separately in each script. Early stopping uses Keras `validation_split=0.15`. |
| Class weights | {0: 1, 1: n_neg / n_pos}, computed on the training rows |
| Epochs / batch / early stop | LSTM 30 / 64, DNN 40 / 128; `val_auc`, patience 5, best weights restored |
| Scaling | Mean and standard deviation from the training rows; standard deviation 0 is replaced by 1. LSTM: per feature over all window rows. DNN: per column, `risk_score` included. |
| Clipping | **None during training.** Inference clips the normalized DNN input to ±6. |
| Seeds / determinism | Only the split's `random_state`. TensorFlow is not seeded, and `np.random.choice` for the SHAP background is unseeded, so retraining is not reproducible. |
| Serialization | `models/saved/`: `lstm_risk_model.keras`, `lstm_feature_{mean,std}.npy`, `dnn_fraud_model.keras`, `dnn_feature_{mean,std}.npy`, and `shap_background.npy` (200 unclipped, normalized training rows of shape (200, 10)) |
| Alert levels | `alert_level_from_probability`: Low < 25%, Medium < 50%, High < 80%, otherwise Critical |

**How inference works (`inference_pipeline.py`):**

- It loads the files above plus `FEATURES_CSV` as the seed history. Everything is fixed through `config` paths.
- On every call it recomputes all features from the customer's full history plus the new row.
- `is_new_device` and `is_new_location` mean "never seen before in this customer's history".
- `failed_logins_24h` is taken as given in the request.
- The LSTM input is the previous 10 feature rows. With fewer rows the first row is repeated as padding; with no history the input is all zeros.
- `risk_score = LSTM probability × 100`.
- The DNN input is `[9 features, risk_score]`, normalized, then clipped to ±6.
- The output is capped at 0.999. Alert level comes from the fixed 25/50/80 bands.
- SHAP explains the clipped DNN input.
- `similarity` compares the new row with the mean and standard deviation of the customer's earlier feature rows.

**`/predict` and SHAP dependencies:**

- The response always contains `risk_score` (0–100), `fraud_probability` (0–100), `alert_level`, similarity and `reasons`. The frontend shows `risk_score` in `App.jsx` (gauge), `FraudEvolutionTimeline.jsx` and `BatchScoring.jsx`. It is also stored in the database and printed in the PDF.
- `shap_explainer.FraudExplainer` names its values by zipping them with the hard-coded 10-name `DNN_INPUT_COLUMNS`. The names must match the DNN's input order, or reasons get the wrong labels.

**Assumptions about paths and columns:**

- Every path is a module constant in `config.py`. There is no model-set or dataset selection for production.
- The seed loader reads the whole features file. A v2 file would bring its 9 metadata columns into memory, where they are unused but present.
- Scoring uses the raw columns in `RAW_COLUMNS_FOR_FEATURES` plus `FEATURE_COLUMNS`.

## 2. Differences from the corrected evaluation

| Aspect | Current production training | Corrected evaluation (`app/evaluation/`, used for v2) | Retraining issue? |
|---|---|---|---|
| Split | Random stratified 80/20 over rows (interleaved in time, same customers in both parts) | Time-based: train / validation / test, with v2 episode groups (rings and warning periods kept together); secondary customer-group split | Yes: candidates must use the saved time split |
| LSTM → DNN handoff | In-sample: the final LSTM scores its own training windows | Out-of-fold: 3 customer-grouped folds for training rows; the final LSTM scores validation and test | Yes |
| Early stopping | Keras `validation_split=0.15` (last 15% of a random array) | Chronologically last 15% of the model's own training rows | Yes |
| Threshold | None at training. Inference uses fixed 25/50/80 bands on an uncalibrated score. | F1-maximizing threshold and recall-at-FPR points chosen on validation | Yes (§8) |
| Class weights | n_neg/n_pos on training rows | Same formula | No |
| Scaling | Mean/std from training rows | Same, from the training split only | No |
| DNN clipping | Not in training; ±6 at inference | Same: not in training; ±6 when scoring (`DNNScorer.predict`) | No difference. Both train on unclipped inputs and score clipped ones; on v2, 1,821 of 58,840 training rows (3.1%) have a normalized value beyond ±6. Corrected in 4C-2e-b: an earlier version of this row said the evaluation also clips in training. |
| LSTM windows | `build_sequences`, customers with more than 10 rows | Same windows (`build_windows` equals `build_sequences`, tested), with a past-only check | No |
| Cold start | Training never contains padded windows. Inference pads (repeat first row, or zeros). | Same: evaluation only scores full windows | Yes, known mismatch (§6) |
| Masking | `Masking(mask_value=0.0)` is ineffective: normalized padding is never exactly 0 (zero rows normalize to about −0.01, −0.15, −0.10, …) | Same architecture | Documented, not fixed |
| Seeds | Not reproducible | Seed 42 + TensorFlow op determinism; two runs byte-identical in weights | Yes |
| Saved format | `.keras` + `.npy` in `models/saved/`, fixed names | Same file names, in `models/evaluation/<v>/time_split/`, no SHAP background | Candidates need the SHAP background and a manifest |
| Dataset | v1 only, fixed path | v1 default, v2 selectable (`datasets.resolve_dataset`) | Yes |
| Evaluated rows | Random 20% of windows | Every windowed row of the test period (v2: 21,142) | — |

## 3. Candidate retraining options

| Option | What is trained | What it would let us conclude | What it would not |
|---|---|---|---|
| **A**: DNN + LSTM on v2, corrected recipe | LSTM + stacked DNN (out-of-fold `risk_score`) | Whether the proposed architecture, trained correctly on v2, behaves acceptably in the live pipeline (loading, SHAP, alerts). Its offline test metrics are already known from 4C-2d (PR-AUC 0.104). | Whether the LSTM helps. Without the ablation there is no comparison, and the 4C-2d ablation showed no consistent benefit. |
| **B**: DNN only on v2 | DNN on the 9 features | A simpler model without the unproven LSTM input, and fewer moving parts at inference | Anything about the architecture as proposed. It also forces a decision on `risk_score` in the API and UI (§9) before any comparison. |
| **C**: both, as candidates, then select | A and B from the same split, seed and recipe | A like-for-like comparison of the two candidates and the current production model on the same fixed splits, plus integration checks for both. The production choice is then made from evidence. | A definitive ranking: with 14 test episodes, the intervals overlap (4C-2d), so selection must also weigh simplicity, false-positive behaviour and explainability. |

The recommended experiment structure is **Option C**. It is the only option that keeps the production decision open and answers the ablation question inside the same pipeline code that `/predict` will use. It is not a claim that either candidate is better.

## 4. Recommended experiment structure

1. **Train both candidates:**
   - Use the saved v2 time split (`models/evaluation/v2/split_time.json`): train rows for fitting, validation rows for early stopping and thresholds, test rows untouched.
   - Use the exact evaluation recipe: `evaluation/stacking.fit_lstm`, `fit_dnn` and `stacked_risk_scores`, seed 42, op determinism.
   - Candidate A should reproduce the 4C-2d evaluation weights exactly. That serves as a determinism check.
2. **Save each candidate** to its own directory (§10), with its thresholds, a seeded SHAP background (clipped, normalized training rows) and a manifest.
3. **Do not refit on train + validation** in this experiment. Refitting removes the untouched test check; it can be decided later for the model finally chosen.
4. **Compare** the current production models, candidate B and candidate A on the same fixed splits (§10, "Required comparison").
5. **Only after review:** wire the chosen model into `/predict` behind a configuration switch, run the full test plan, and switch.

## 5. 90-day home-device implementation plan

**Current rule:** `get_customer_profile` returns the most frequent `device_id` and `location` over the customer's whole in-memory history, with ties broken alphabetically. It is used only by:

- `GET /customer/{id}/profile`, which gives the dashboard its form defaults;
- batch scoring, where it fills a missing `device_id` or `location`.

**The home device is not a model input.** The features use `is_new_device` ("never seen in this customer's history"), not "differs from the home device". So the 90-day rule changes the defaults and the profile endpoint, not any score, unless a caller relies on the defaulted device.

**Measured on the data:**

- **v1:** the 90-day rule and the whole-history rule give the same home device for all 500 customers.
- **v2:** they differ for 32 customers, all of whom upgraded their phone and whose upgraded device is now the most frequent in the last 90 days.
- Every customer in both datasets has at least 162 days of history and 31 or more transactions in their last 90 days. So the fallback only matters for new customers and customers created through the API.

**Rule to implement (next stage):**

- **Anchor:** the customer's newest transaction timestamp in history, not the wall clock. The seed data ends on 2026-07-06, so a wall-clock window would be empty today.
- **Home device** = the most frequent `device_id` among transactions in `(anchor − 90 days, anchor]`, ties broken alphabetically (unchanged).
- **Fallback:** if that window has fewer than a minimum number of transactions (proposed: 5), or the history spans less than 90 days, use the most frequent device over the whole history. That is the current rule, which already handles short histories.
- **No history:** return `None`, as now: `/profile` gives 404 and batch scoring reports a row error.
- **Location:** apply the same rule to `home_location`, for consistency (to be confirmed).
- **Code:** update `inference_pipeline.get_customer_profile` and the `schemas.CustomerProfile` description, and document it in the README.
  - Add tests for an upgrade within the window, a window with fewer transactions than the minimum, a history shorter than 90 days, and ties.
  - `test_profile.py` and the frontend's placeholder text need no change for v1 data (identical results).

## 6. Cold-start findings

- **Sequence length:** `SEQUENCE_LENGTH = 10`.
- **Training** (`build_sequences` and evaluation `build_windows`): only customers with more than 10 transactions contribute, and each window is 10 real earlier rows. **Padding never appears in training.**
- **Live inference:**
  - With 1–9 earlier rows, the earliest row is repeated at the front to make up 10.
  - With no earlier rows, the input is all zeros.
  - Both are normalized with the training mean and standard deviation, so the zeros become non-zero and the `Masking(0.0)` layer never masks anything.
  - These inputs are outside the training distribution.
- **Who is affected:** in the seed data every customer has at least 62 transactions (v1; 83 in v2), so seeded customers never hit this. Any new customer ID sent to `/predict` or batch scoring does. `_get_history` creates an empty history, and there is no minimum-history check.
- **Features for new customers:** `build_point_features` also uses special values for them (z-score denominator from a 30% floor, `hour_is_unusual = 0` and `category_is_unusual = 1` on the first row).
- **Conclusion:** this is a **production-retraining issue**. Retraining alone does not fix it. Options for a later stage:
  - (a) add padded windows to training;
  - (b) fall back to a DNN-only score (or a fixed `risk_score`) below 10 earlier transactions, and flag low confidence in the response;
  - (c) require a minimum history.

  No redesign is proposed in this stage.

## 7. Feature compatibility

- **Names and order:** v2 uses exactly the 9 production features, in the same order.
  - The generator builds `transactions_with_features.csv` with the unchanged `build_point_features`, which never sees the metadata.
  - The sanity audit recomputed all 101,297 rows with a maximum difference of 0.
- **Guards:**
  - `feature_engineering.py` refuses ground-truth columns in `FEATURE_COLUMNS` at import.
  - `evaluate_split` refuses a frame that contains them.
  - The evaluation's model frame holds only identifiers, `is_fraud` and the 9 features.
  - `DNN_INPUT_COLUMNS` holds the 9 features plus `risk_score`.
- **Preprocessing and scaling:** mean and standard deviation from the training split; standard deviation 0 is replaced by 1. The LSTM uses per-feature statistics over window rows; the DNN uses per-column statistics.
- **Clipping:** the DNN input is clipped to ±6 when scoring. The evaluation recipe, like current production, trains on unclipped inputs (corrected in 4C-2e-b; an earlier version said the evaluation also clips in training). The candidates follow the evaluation recipe exactly, as decided in 4C-2e-b, so candidate A reproduces the evaluated model. Clipping in training as well would be a separate, not-yet-evaluated model change.
- **Missing values:** there are none in either dataset (the sanity audit counted 0 NaN/inf). The feature code has no imputation: new customers get the defaults in §6, and `failed_logins_24h` defaults to 0 in the API. A missing numeric value would pass through as NaN, which is not guarded. That is out of scope, but noted.
- **One difference in meaning:** in v2, `failed_logins_24h` is a true 24-hour count of login-failure events. At inference, the caller supplies it, as in v1. The name and meaning are unchanged; how accurate it is depends on the caller.

## 8. Threshold strategy

- **The scores are not probabilities:** the models are trained with heavy class weights (about 200:1 on v2), so the scores are not calibrated. The 25/50/80 bands have no statistical basis.
- **Measured on the v2 test period** with the 4C-2d evaluation DNN + LSTM (DNN-only figures in brackets):

| Band | Legitimate rows | Legitimate rows per 1,000 transactions | Fraud rows |
|---|---|---|---|
| Medium (25–50) | 1,817 (2,501) | 86 (118) | 7 (10) |
| High (50–80) | 1,223 (1,530) | 58 (72) | 23 (22) |
| Critical (≥ 80) | 112 (144) | 5.3 (6.8) | 22 (22) |

  The High band alone would add about 58 legitimate alerts per 1,000 transactions (72 for DNN only). The F1-optimal thresholds chosen on validation were 0.959 and 0.770.

**Recommendation for retraining (to be implemented later):**

- **Store thresholds with each candidate:** in its manifest, chosen on the v2 validation split only:
  - the F1-optimal threshold;
  - operating points for fixed alert budgets or false-positive rates (e.g. about 0.1%, 1% and 5% of legitimate traffic);
  - optionally, a precision target.
- **Derive the alert bands from those points,** for example Critical ≈ the 0.1% FPR point, High ≈ 1%, Medium ≈ 5%, instead of fixed 25/50/80.
- **Calibration** (Platt or isotonic on validation) would make the 0–100% display meaningful. There are only 119 validation fraud rows, though, so it would be unstable; report it as an option, not a requirement.
- **API shape:** keep `alert_level` values and response fields unchanged. Only the cut-points move, and they come from the model manifest instead of constants in `dnn_model.py`.
- **Test data:** never use it to choose or adjust thresholds.

## 9. Explainability considerations

- **SHAP:** `GradientExplainer(dnn_model, background)` works for any Keras DNN. What depends on the candidate:
  - **Feature names:** `FraudExplainer.explain` zips SHAP values with the hard-coded 10-name `DNN_INPUT_COLUMNS`. A DNN-only candidate has 9 inputs, so the names must come from the candidate's manifest. Otherwise every reason after position 9 is mislabeled (`zip` truncates silently).
  - **Background:** it must match the candidate's input width and normalization, ideally from clipped training rows with a fixed seed. The current background is unclipped and unseeded.
  - **Format:** `.keras` stays; no change to the loading code is needed for the model files themselves.
  - **`/predict` response:** `reasons` keeps the same shape. With DNN only, "Elevated Behavioral Risk Score" can no longer appear.
- **`risk_score` in the API and UI** (candidate B, or a stage without the LSTM): `PredictionResponse.risk_score`, the database column, the PDF, the gauge, the timeline and the batch table all expect it. Options:
  - (i) keep computing the LSTM score for display only, clearly labelled as not used in the fraud score;
  - (ii) return null and adapt the schema and frontend.

  This is a product decision to take before any switch to candidate B.
- **Known issue:** GradientExplainer samples background rows and interpolation points at random, so identical calls can return different top-4 reasons (seen in Step 1: 7 different sets in 8 runs). Tests seed the RNG. Retraining does not change this. Fixing it (a fixed seed per call, or more samples) is optional and is not required for compatibility.

## 10. Safe model replacement plan

- **Existing conventions:** `models/saved/` holds production and `models/evaluation/<version>/` holds evaluation copies. Following those, candidates go to **`backend/models/candidates/v2/<candidate>/`** (`dnn_lstm/`, `dnn_only/`).
- **Each candidate directory** uses the production file names, so it can be loaded as a drop-in set:
  - `lstm_risk_model.keras`, `lstm_feature_{mean,std}.npy` (A only);
  - `dnn_fraud_model.keras`, `dnn_feature_{mean,std}.npy`, `shap_background.npy`;
  - `manifest.json` with:
    - the input columns, feature-list hash, dataset version and SHA-256, and the split file and its hash;
    - seeds, training row counts, library versions and the recipe reference;
    - the validation thresholds and alert bands;
    - SHA-256 of every model file.
- **Protecting production:**
  - Nothing writes to `models/saved/` during the experiment.
  - A test asserts that the training command refuses an output path under `models/saved/`.
  - Before the experiment, record the SHA-256 of every file in `models/saved/`, and check it again afterwards.
- **Switching (later, after review):**
  - Add `config.MODEL_SET` (environment variable, default `saved`) so the pipeline loads from a named directory.
  - Validate the manifest at startup: input width against columns, and the feature hash against `FEATURE_COLUMNS`.
  - Roll back by setting `MODEL_SET=saved`.
  - Copy files into `models/saved/` only as a separate, reviewed, committed step, with the old files archived (e.g. `models/archive/v1-<date>/`).
- **Git:** candidate files stay local and uncommitted until chosen (like the v2 evaluation artefacts).

## 11. Required comparison (after retraining)

**Models:** (1) current production (`models/saved`), (2) candidate B (DNN only, v2), (3) candidate A (DNN + LSTM, v2).

**Data and splits:**

- Use the **saved** v2 split files (`models/evaluation/v2/split_time.json`, `split_customer.json`); no new splits.
- Score the same test rows (21,142 transactions, 61 fraud, 14 episodes) with the same windows and features.
- Production is scored through the production preprocessing (its scalers, the ±6 clip, and `risk_score` from the production LSTM). v2 is entirely out-of-sample for production, which was trained on v1.

**Thresholds:**

- Report each model at its own threshold chosen on v2 validation.
- For production, also report the fixed 25/50/80 bands as deployed.

**Metrics** (the same as 4C-2d, from `app/evaluation/metrics.py` and `analysis.py`):

- PR-AUC, ROC-AUC, precision, recall, F1, false-positive rate, alerts per 1,000;
- recall at about 0.1% and 1% false-positive rate;
- first-fraud recall and detection delay;
- per fraud type;
- false positives on legitimate unusual behaviour;
- warning-period scores;
- customer-cluster bootstrap intervals, including paired differences A − B and B − production.

**Secondary evidence:**

- The same comparison on the customer-grouped split.
- The v1 time split, as a regression check that the candidates do not break on v1-like data (not for selection).

**Integration checks:** the same five fixed transactions (normal and suspicious, known and new customer) scored through the full pipeline for each model set, covering latency, alert levels, reasons and similarity.

## 12. Testing plan (before any production switch)

| Area | Tests |
|---|---|
| Model loading | Each candidate directory loads. Manifest input width equals the model input shape. The feature hash matches `FEATURE_COLUMNS`. A missing or corrupt file gives a clear error. `MODEL_SET` defaults to `saved`. |
| Feature compatibility | The candidate is trained on exactly the 9 features (+ `risk_score` for A). No ground-truth column in any training matrix (existing guards plus a manifest check). The same scaler at training and inference; ±6 clip when scoring, as evaluated. |
| Inference | The pipeline gives the same score for a fixed transaction as calling the model directly on the same vector. The output stays within [0, 0.999]. Results are deterministic with SHAP seeded. |
| Customer history | Seed loading drops metadata columns. `restore_scored_history` behaves as before. History ordering is unchanged. |
| Home device | The 90-day rule, fallback, ties and no-history cases (§5). v1 profiles are unchanged. |
| Cold start | New customer with 0, 1 and 9 earlier transactions: no crash, the documented behaviour, and the response flags whatever policy is chosen. |
| Threshold behaviour | Bands come from the manifest. Values at band edges map correctly. `alert_level` values are unchanged. |
| SHAP | Reasons are named from the candidate's columns (9 or 10). The top-k positive-only rule holds. There is no `risk_score` reason for DNN only. The background shape matches. |
| Batch scoring | Same validation, summary counts and defaults (with the 90-day profile). |
| Persistence | Scored rows persist and restore under a candidate model set. The history endpoint still works. |
| API endpoints | `/predict`, `/predict/batch`, `/customer/*`, `/fraud-rings`, `/metrics`, `/metrics/report`, `/report/pdf` all still return their current response shapes. |
| Existing suite | All 234 current tests stay green. Tests that encode v1 data facts are updated deliberately at switch time, not before: `conftest` `NORMAL_TXN`/`SUSPICIOUS_TXN` expectations, `test_profile` (DEV_nnnn_A), `test_predict` (Low / Critical), `test_fraud_rings`, `test_batch`. |
| Protection | `models/saved/` hashes are unchanged after the candidate training run. The training command refuses `models/saved/` as its output. |

## 13. Risks and limitations

- **Weak evidence:** with 14 test episodes, no test rings and one test warning period, the evidence cannot separate the candidates. Selection will partly be a judgement call on simplicity, false-positive behaviour and explainability.
- **Synthetic data:** v2 difficulty comes from generator settings. A model tuned or selected on v2 is selected for this generator, not for real traffic.
- **Thresholds:** validation has only 119 fraud rows, so thresholds and any calibration are noisy. Alert bands derived from them can move a lot between reruns with different data.
- **Cold-start mismatch:** it remains (§6), and Masking is ineffective.
- **Switching data changes the app:** moving the live seed history to v2 would change every customer's history, the ring detector output (mostly legitimate shared devices, §13 of the v2 report) and several v1-specific tests. That is a separate decision from switching models.
- **DNN only means a product decision:** the `risk_score` field would need one (§9).
- **SHAP:** remains stochastic.
- **Pre-existing performance limit:** inference recomputes each customer's full history on every call (O(n)). Unchanged by retraining.

## 14. Exact next implementation steps

**Step 4C-2e-b: train candidates (no application changes)**

- **New:** `backend/app/training/__init__.py` and `backend/app/training/candidates.py`.
  - Reuses `evaluation.datasets`, `split` (loading the saved split JSON) and `stacking`.
  - Trains A and B, and saves each to `models/candidates/v2/<name>/` with a manifest, validation thresholds and a seeded SHAP background.
  - Refuses `models/saved/` as its output.
  - Command: `python -m app.training.candidates --dataset v2`.
- **`backend/app/config.py`:** add `CANDIDATES_DIR` (paths only).
- **New tests:** `backend/tests/test_candidates.py` (tiny generated sample and stub fitters), covering the manifest, file set, determinism, the refusal of `models/saved`, and threshold fields.
- **Check:** candidate A's weights equal the 4C-2d evaluation weights.

**Step 4C-2e-c: comparison (no application changes)**

- **New:** `backend/app/evaluation/compare_models.py`. It scores production, B and A on the saved v2 splits and reuses `analysis.py` functions. Output goes to `models/evaluation/v2/comparison.json`.
- **New:** `docs/step4c2e-candidate-comparison.md`.
- **Tests:** a comparison test on a small sample.

**Step 4C-2e-d: inference readiness behind a switch (default unchanged)**

- **`backend/app/config.py`:** `MODEL_SET` selection.
- **`backend/app/inference_pipeline.py`:**
  - load the model set from its manifest;
  - support a DNN-only model (input columns from the manifest);
  - `risk_score` policy;
  - thresholds from the manifest;
  - drop metadata columns on seed load;
  - the 90-day home device (§5);
  - the cold-start policy, once decided.
- **`backend/app/models/shap_explainer.py`:** feature names passed in, not hard-coded.
- **`backend/app/models/dnn_model.py`:** alert bands as a parameter (the constants stay the default).
- **`backend/app/schemas.py`:** only if `risk_score` becomes optional.
- **Tests:** new tests per §12, and `test_profile.py` extended.

**Step 4C-2e-e: switch decision (separate approval)**

- Choose the model and data (v1 seed or v2 seed).
- Update the v1-specific tests and README/docs.
- Adjust the frontend only if `risk_score` changes.
- Archive `models/saved/` before replacing it.

**Not touched in any of these steps unless separately approved:** the v1 data and generator, the evaluation methodology, and the frontend (until the `risk_score` decision).
