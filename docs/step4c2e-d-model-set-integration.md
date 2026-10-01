# Step 4C-2e-d: MODEL_SET integration

This step lets the application load a different set of models without changing
the API. It does not choose a production winner, change the production models,
change the frontend or change the `/predict` response.

## Selecting a model set

The model set comes from the `MODEL_SET` environment variable. It is read once,
when the inference pipeline is created at application startup.

| `MODEL_SET` | Directory | Models |
|---|---|---|
| unset (default) or `production` | `backend/models/saved/` | the existing production LSTM and DNN, unchanged |
| `v2_dnn_lstm` | `backend/models/candidates/v2/dnn_lstm/` | candidate A from 4C-2e-b: the LSTM risk score feeds the DNN |
| `v2_dnn_only` | `backend/models/candidates/v2/dnn_only/` | candidate B from 4C-2e-b: the DNN scores the 9 features, with no LSTM |

Any other value stops startup with `ModelSetError`, and the message lists the
valid values. This includes an empty string and differences in case or
whitespace. A candidate that you ask for explicitly never falls back to
production. If its directory, manifest or any of its files is missing or fails
validation, startup stops.

For example:

```
cd backend
MODEL_SET=v2_dnn_only uvicorn app.main:app --port 8000
```

The startup log shows which model set was loaded:

```
[pipeline] model set 'v2_dnn_only' from .../models/candidates/v2/dnn_only; DNN only (no LSTM); DNN inputs [...]
```

## Code

* **`app/model_sets.py`** (new) does all model-set resolution, loading and
  validation.
* **`app/inference_pipeline.py`** gets its models from `load_model_set()`.
  For a DNN-only set it skips the LSTM step. Everything else is unchanged: the
  feature recomputation, the sequence construction, scaling, clipping, the
  0.999 cap, the alert bands, the SHAP display threshold and the similarity
  calculation.
* **`app/models/shap_explainer.py`**: `FraudExplainer` takes the model set's
  input column names. If the names do not match the model's input width or the
  background width, it raises `ValueError`.

## Production

* With `MODEL_SET` unset or set to `production`, the pipeline loads exactly the
  same seven files from `models/saved/` as before (`config.*_PATH`) and scores
  in the same way.
* The only additions are read-only shape checks:
  * the DNN takes 10 inputs;
  * the LSTM takes 10×9 windows;
  * the scaler and SHAP background shapes match those widths.
* Nothing in this step writes to `models/saved/`. A candidate directory inside
  `models/saved/` is refused.
* The old and new pipelines were compared on 61 transactions (60 random plus
  one new customer), including SHAP reasons with seeded RNG. The complete
  responses were byte-identical.

## Candidate validation

Before a candidate model set is used, `manifest.json` is checked:

| Check | What must hold |
|---|---|
| Candidate name | `dnn_lstm` or `dnn_only`, matching the model set |
| Dataset metadata | version `v2`, with a dataset SHA-256 present |
| Feature list | exactly the application's `FEATURE_COLUMNS`: the same names, order and count (9), and the same list hash. A reordered list is reported as a feature order mismatch. |
| DNN inputs | the 9 features plus `risk_score` for `v2_dnn_lstm`; the 9 features for `v2_dnn_only`. No ground-truth columns. |
| Input shapes | manifest `dnn_input_shape` `[None, n]`; `lstm_input_shape` `[None, 10, 9]` (`v2_dnn_lstm` only); the loaded models' real input shapes |
| Sequence length | 10 |
| Clipping metadata | the DNN is trained unclipped and its scaled inputs are clipped to ±6 when scoring; LSTM inputs are never clipped |
| Files | every required file is listed and exists, and each `.npy` matches its SHA-256 |
| Weights | `dnn_weights_sha256` and `lstm_weights_sha256` match. The `.keras` archives embed their save time, so the weights are the reproducible identity. |
| Shapes | scaler shapes and the SHAP background width equal the input width. The DNN std values are positive. |
| `v2_dnn_only` | no LSTM is described or loaded, and no LSTM files are needed |

## Scoring with the candidates

**`v2_dnn_lstm`**
* The LSTM sees the 10 prior feature rows. When there are fewer, it uses the
  production padding.
* Its inputs are scaled with the candidate's LSTM scaler and are not clipped.
* `risk_score = round(prob × 100, 2)`.
* The DNN input is the 9 features plus `risk_score`, scaled with the
  candidate's DNN scaler and clipped to ±6. This is exactly the 4C-2d /
  4C-2e-b recipe.

**`v2_dnn_only`**
* The DNN input is the 9 features, scaled with the candidate's scaler and
  clipped to ±6.
* No LSTM is loaded or called.

**End-to-end check against the evaluation.** Real v2 test transactions were
scored through the pipeline, with each customer's history rebuilt from the v2
rows before the transaction.
* `fraud_probability` equals the saved 4C-2d evaluation score for the same
  model, to the displayed 2 decimals: `dnn_fraud_classifier` for
  `v2_dnn_lstm` and `dnn_without_risk_score` for `v2_dnn_only`.
* For `v2_dnn_lstm`, `risk_score` also equals `lstm_risk_predictor`.
* 40 of 40 transactions matched for each candidate during development. The
  test suite rechecks 10 of them.

## API compatibility

The `/predict` response has the same 14 fields with the same types for every
model set, and `/predict/batch` has the same 6 fields per row.

The following were checked for every model set:
* rows are persisted in the database;
* the customer history timeline;
* PDF generation from the `/predict` response;
* batch scoring.

The frontend receives the same response shape, so it needs no changes.

### `risk_score` for `v2_dnn_only`

`v2_dnn_only` has no LSTM, so there is no behavioral risk score. To keep the
field and its type (float, 0–100), `risk_score` carries the DNN's own score. It
is therefore equal to `fraud_probability`. This is a compatibility choice and
not a second signal.

**Known limitation.** The PDF report and the frontend still describe
`risk_score` as LSTM or trajectory risk:
* PDF row: "Trajectory risk from prior activity…";
* PDF footer: "…from the LSTM risk predictor…".

Those descriptions are wrong while `v2_dnn_only` is loaded. They were left
unchanged in this step (no redesign) and are a decision for 4C-2e-e.

### Alert levels

Every model set uses the existing bands (Low < 25, Medium < 50, High < 80,
otherwise Critical) on the DNN output. The validation thresholds in the
candidate manifests are not used for alerting. Choosing operating points is
deferred to 4C-2e-e. 4C-2e-c showed that the fixed bands give very different
alert volumes on v2.

### Other things that do not follow the model set

* **Seed customer history.** Every model set scores against the same live seed
  history (`data/transactions_with_features.csv`, v1 customers), because the
  live dataset is not switched. The candidates were trained on v2, so
  absolute scores for v1 customers are not comparable between model sets.
* **`/metrics` and `/metrics/report`.** They still serve the saved v1
  evaluation report whichever model set is loaded.

## SHAP

* Feature names come from the loaded model set:
  * production: the unchanged `DNN_INPUT_COLUMNS` (9 features plus
    `risk_score`);
  * `v2_dnn_lstm`: the 9 features plus `risk_score`;
  * `v2_dnn_only`: exactly the 9 model inputs. `risk_score` never appears as
    a reason.
* The background is the candidate's own `shap_background.npy`: 200 training
  rows, scaled with the candidate's scaler and clipped.
* A mismatch between the names and the model's input width or the background
  width raises `ValueError`, rather than attributing a reason to the wrong
  feature. The loader also refuses such a set at startup.
* SHAP GradientExplainer is stochastic. Without seeding, the order of the top
  reasons can differ between identical calls. This is existing behavior.

## Tests

The new file `backend/tests/test_model_sets.py` has 60 tests.

Most of them use small untrained stand-in candidates. These are written to a
temp directory by the real `training.candidates._save`, so their manifests
have exactly the real format.

Four tests use the trained local candidates and are skipped when those are
absent:
* loading each of the two candidates (2 tests);
* reproducing the evaluation scores;
* a check inside the model-switching test.

The tests cover:
* the default production model set;
* explicit production, `v2_dnn_lstm` and `v2_dnn_only`;
* invalid values, including at application startup;
* no fallback for a missing requested candidate;
* missing artifacts and a missing manifest;
* manifest mismatches: 14 cases;
* feature order and DNN input order mismatches;
* manifest shape mismatches and real model, scaler and background shape
  mismatches;
* swapped candidate directories;
* exact preprocessing for both candidates;
* API response compatibility, persistence, history, PDF and batch scoring;
* SHAP name alignment and attribution to the right feature;
* refusal of a candidate root inside `models/saved/`;
* switching model sets with no change to any artifact checksum;
* reproduction of the 4C-2d evaluation scores.
