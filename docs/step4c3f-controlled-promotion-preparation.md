# Step 4C-3F — Controlled promotion preparation

Status: **done; nothing promoted.** `MODEL_SET` still defaults to `production`,
`backend/models/saved/` is untouched, no model was trained, no cut-off was
recalibrated, no seed was reselected.

This step makes the exact artifact selected in 4C-3E.6 loadable **by name, for
controlled testing only**.

## 1. The artifact

| | |
|---|---|
| Selected artifact | `v2_dnn_lstm@14` (architecture `v2_dnn_lstm`, training seed 14) |
| Path | `backend/models/candidates_multiseed/v2/seed_14/dnn_lstm/` |
| Model-set identifier | `v2_dnn_lstm_seed14` |
| Model version | `v2_dnn_lstm_seed14-32fc53979e06` |
| Architecture | DNN + LSTM (LSTM risk score -> DNN), production layer definitions |
| Dataset | synthetic v2, generator 2.0.1, sha256 `a09d0611…f7b1ef08` |
| Features | 9, order as `FEATURE_COLUMNS`, list sha256 `78f61bd7…1052ed3e` |
| DNN input | `[None, 10]` (9 features + `risk_score`) |
| LSTM input | `[None, 10, 9]`, sequence length 10 |
| Clipping | DNN: scaled inputs clipped to +-6 when scoring; LSTM: never clipped |
| Status | EVALUATION / NOT DEPLOYED |

SHA-256 (pinned in `app/model_sets.py`, equal to `selection_record.json`):

| File | SHA-256 |
|---|---|
| `dnn_fraud_model.keras` | `c033fa8d4ed9888b135a308959ab7596c5db8824665121876cde0f2a28655351` |
| `lstm_risk_model.keras` | `5a327ccba85f86e90a927cf5c0a4cd2f1cc6d040f8ded2e47734705b0c1a7600` |
| `dnn_feature_mean.npy` | `2071d1baa335af2c06d26ec63cc6623e0e46e6626ee349f79face3bc0f01c681` |
| `dnn_feature_std.npy` | `b472068a64760b13ab4f959b81309cf5df7abd1c56afbb8372d878aa515eadd8` |
| `lstm_feature_mean.npy` | `fbfc1a0afd79ad236b27932198a6a6a6c1d6ac4b2e61b335db1269b9be91e90c` |
| `lstm_feature_std.npy` | `e3e7765d84d6b6d2f3ebc665c0798ef36d015cbf604acc53f00d8cd8aaff5a35` |
| `shap_background.npy` | `ef63280f701e04b5a25dfc47e41457782c1e3152474104560d6c71b3d9761a59` |
| DNN weights | `32fc53979e06db730742423133e54bee2305f89a439cbd634ae03b76f8c94f0f` |
| LSTM weights | `e4851c4c20e864b14f1f54a7051b0ad5dad5d0e4920270737ded6bad6db1a077` |

Frozen cut-offs (DNN output, 0-1; frozen at Stage B, used unchanged in Stage C):

| | Value | As a fraud score |
|---|---|---|
| Policy B (alert) | 0.7928694486618042 (0.7929) | 79.29 |
| Critical | 0.9430845379829407 (0.9431) | 94.31 |

## 2. How it is wired

`app/model_sets.py` has one new entry, `v2_dnn_lstm_seed14`, in the existing
`MODEL_SETS` registry. It uses the existing candidate loader and adds a
pinned check (`validate_pinned`):

* the manifest's training seed must be 14;
* the manifest's DNN and LSTM weight hashes must be the selected ones;
* the SHA-256 of **every** file, the `.keras` archives included, must equal the
  pinned value;
* the loader then also checks, as for every candidate, that the loaded weights
  hash to the manifest's values, and the feature list/order/count, input
  shapes, sequence length and clipping metadata.

Anything else is refused with `ModelSetError`. Unknown `MODEL_SET` values
(including near misses such as `v2_dnn_lstm_seed15` or an empty string) stop
application start-up; there is no fallback to production. Loading only reads
files.

`GET /model-info` reports `status`, `architecture` and `training_seed` for every
model set (production: `PRODUCTION`, `DNN + LSTM`, no seed). For seed 14 it also
reports the pinned file hashes, a `selection` block (artifact, protocol,
selection-record hash, limitation, blockers) and
`thresholds.frozen_cutoffs`. `GET /metrics` and `/metrics/report` add
`final_holdout_evaluation` for candidate model sets: the loaded artifact's slice
of `final_holdout_report.json`, copied, only when that report scored exactly
the loaded weights. Production responses keep their existing keys.

## 3. Score wording

The API field is still called `fraud_probability` (API contract unchanged).
Everything shown to a person says **Fraud Score**: the UI already did, and the
PDF report now does too ("Fraud Score … / 100" instead of "Fraud Probability
… %"). The score is not a calibrated probability.

## 4. What the live candidate does and does not do

In controlled mode `/predict` scores with the seed-14 weights and returns SHAP
reasons, behavioural similarity and PDF reports like production.

**The frozen cut-offs are not applied.** `/predict` keeps the legacy fixed bands
(25 / 50 / 80) for every model set, because changing live alert thresholds was
out of scope for this step. So the Stage C figures (9.11 legitimate alerts per
1,000, recall 0.558) describe alerts at fraud score >= 79.29, not the alert
levels shown on screen: "Critical Risk" on screen is fraud score >= 80, and
"High Risk" (50-80) and "Medium Risk" (25-50) are below the evaluated cut-off.

There is no simultaneous shadow scoring: one process loads one model set. A
controlled switch (or two processes on different ports with different
databases) is what this step provides.

## 5. New customers — promotion blocker

On the Stage C new-customer population (fewer than 10 earlier transactions,
current cold-start behaviour) this artifact raised **25.56 legitimate alerts per
1,000** [23.45, 27.92] at the frozen Policy B cut-off, with recall 0.313
[0.244, 0.391]. For customers with full history it raised 9.11 [8.41, 9.83].

* Policy B performance is **not** claimed for cold-start customers.
* No new-customer policy was approved before the data was generated, so none
  was evaluated.
* A new-customer policy remains an explicit promotion blocker.

## 6. Remaining blockers

1. No approved new-customer (cold-start) policy.
2. The frozen cut-offs are not applied by `/predict`; applying them is a
   threshold change that needs its own approval and tests.
3. All evidence is on synthetic data.
4. Promotion itself is the owner's decision; none has been made.

## 7. Running it

Commands are for Windows PowerShell, from the `backend` folder with the
project's Python environment active.

Production (the default, and the recommended configuration for the review):

```
Remove-Item Env:MODEL_SET -ErrorAction SilentlyContinue
Remove-Item Env:DATABASE_URL -ErrorAction SilentlyContinue
uvicorn app.main:app --port 8000
```

Seed 14, controlled mode, with its own database so that its scans are not
replayed into production's customer histories:

```
$env:MODEL_SET = "v2_dnn_lstm_seed14"
$env:DATABASE_URL = "sqlite:///./fraud_platform_seed14.db"
uvicorn app.main:app --port 8000
```

Rollback: stop the server, run the two `Remove-Item` lines, start it again.
(In `cmd.exe`: `set MODEL_SET=v2_dnn_lstm_seed14` to select, `set MODEL_SET=`
to clear.) Present seed 14 as "Validated candidate — not yet deployed".

## 8. Verification

`tests/test_seed14_model_set.py` (30 tests): registry and default, near-miss
names, missing artifact, pinned values equal the selection record and the
Stage C report, tampered scaler / model file / manifest / other seed refused,
metadata, Stage C slice, scoring with SHAP and similarity, batch, PDF,
production -> seed 14 -> production with identical production scores and
unchanged `models/saved/` checksums.
