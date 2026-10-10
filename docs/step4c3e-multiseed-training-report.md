# Step 4C-3E: multi-seed candidate training results

Written at commit `582fe60` (nothing committed). This is the second evidence
stage after `docs/step4c3e-evidence-report.md`.

**NO PROMOTION.** `MODEL_SET` still defaults to `production`. Production was
not trained or touched. The seed-42 candidates were not retrained or
replaced. No threshold or alert band was changed. Nothing was deployed.

Scores are model scores. They are not calibrated probabilities.

## 1. Summary

**Question.** Is the `v2_dnn_lstm` advantage over `v2_dnn_only` a property of
the architecture, or of the single seed-42 training run?

| Finding | Result |
|---|---|
| Decision under the tie-break rule | All three conditions hold. **`v2_dnn_lstm` remains the provisional lead.** |
| Architecture effect on Policy B recall | `v2_dnn_lstm` is higher by 0.092 on average over training seeds, [0.029, 0.157] with data and training-seed variation included. It is higher in 23 of 25 seed pairs. |
| Is it robust to training randomness? | Yes for recall and ROC-AUC. It is not uniform: with training seed 11, `v2_dnn_only` has the higher recall. |
| Where `v2_dnn_only` is better | First-fraud detection: 0.638 against 0.528 on average. The interval excludes zero on the primary population. |
| Training-seed variability | Large. Recall ranges over 0.45–0.57 (`v2_dnn_lstm`) and 0.36–0.53 (`v2_dnn_only`) across training seeds. The spread between runs is about two thirds of the gap between architectures. |
| The saved seed-42 candidates | They are among the weakest of the six runs of each architecture. Several weaknesses reported earlier (fewer episodes than production, weak Critical-tier ring detection) are specific to seed 42. |
| Policy B alert budget | **Still unstable.** It moves with the training seed and with the data seed. 18 of 50 cells are over 10 per 1,000 for each architecture. |
| New customers | Behaviour on early transactions depends heavily on the training run: 15 to 82 legitimate alerts per 1,000. |

## 2. Design

| Item | Value |
|---|---|
| Architectures | `v2_dnn_lstm`, `v2_dnn_only`; unchanged |
| Training seeds | **11, 12, 13, 14, 15** (fixed before training; 42 is the saved candidates' seed and is not among them) |
| What the seed controls | Weight initialisation, shuffling, dropout, and the customer folds of the out-of-fold LSTM scores |
| Everything else | Identical to the saved candidates: same function (`train_candidates`), same seed-42 v2 dataset, same saved time split, same features, scaling, clipping, hyperparameters, early stopping and validation rows. Nothing was tuned. |
| Training dataset SHA-256 | `a09d06114ce7…` (the same file the saved candidates were trained on; checked for every model) |
| Hold-out data | Seeds 101–105; new-customer seeds 201–205. The same files as in the first report. |
| Populations | Primary: at least 10 earlier transactions. Secondary: the same from 2026-06-01. |
| Bootstrap | 2,000 resamples, seed 42, customer groups resampled within each dataset |

**Cut-offs.** A cut-off belongs to one trained model. The seed-42 numbers
cannot be applied to another model's scores. Each new model gets its
cut-offs by **the same rule** that produced the fixed seed-42 table in 4C-3B:
on the validation period of the training dataset, the lowest tie-safe score
with a false-positive rate within 1% (Policy B) or 0.1% (Critical). Applied
to the saved models, this rule reproduces the fixed cut-offs exactly (a test
checks all six values). The saved models keep their fixed cut-offs. Nothing
is tuned on hold-out data.

**Two sources of variation, kept apart**

| Source | How it is measured |
|---|---|
| Data: which customers are in the hold-out | Group bootstrap, as in the first report |
| Training seed: which run produced the model | Spread across the 5 seeds of an architecture |

The architecture comparison is the difference between the two architectures'
means over training seeds. Three intervals are reported. The **decision
interval** includes both sources: every resample redraws the customer groups
and redraws the 5 training seeds of each architecture. It was fixed in the
code before the models were scored. With 5 seeds per architecture, any
interval over training seeds is rough.

## 3. Training runs and artifacts

| Training seed | v2_dnn_lstm: epochs (LSTM / DNN) | v2_dnn_lstm weights SHA-256 (LSTM / DNN, first 10) | v2_dnn_only: epochs | v2_dnn_only weights SHA-256 (first 10) | Seconds |
|---|---|---|---|---|---|
| 11 | 7 / 9 | `f370caba89` / `a9236eabe4` | 11 | `03401a6626` | 347 |
| 12 | 10 / 13 | `36a1614ec0` / `11b4328d13` | 6 | `3aa9002eab` | 411 |
| 13 | 7 / 8 | `151ec43d7b` / `b39332c0c8` | 9 | `390515d0e1` | 363 |
| 14 | 8 / 13 | `e4851c4c20` / `32fc53979e` | 8 | `8b79fbf09a` | 366 |
| 15 | 7 / 10 | `e466abd517` / `6a427aa893` | 17 | `70ff615f4e` | 357 |

* Location: `backend/models/candidates_multiseed/v2/seed_<seed>/{dnn_lstm,dnn_only}/`
  with a `training_run.json` per seed. 70 files, 3.4 MB.
* Each manifest records the training seed, dataset checksum, split checksum,
  feature list and its hash, layers, hyperparameters, epochs run, weight
  checksums and scaler checksums.
* The trainer refuses to write into `models/saved/` or `models/candidates/`,
  refuses to overwrite a trained seed, and compares the production and
  seed-42 candidate checksums before and after each run.
* The application cannot load these directories by a model-set name.
* **Reproducibility checks**
  * Seed 11 was trained a second time into a scratch directory. The weights
    are identical.
  * Seed 42 was trained again into a scratch directory with the same code.
    `v2_dnn_only` has identical weights to the saved candidate. `v2_dnn_lstm`
    differs in the last bits of its weights (the saved one was trained on a
    different machine) and gives the same hold-out counts (1,085 fraud
    caught, 4,321 false positives). So the saved candidates are ordinary
    runs of this recipe.

**Cut-offs by the validation rule**

| Model | Policy B cut-off | Critical cut-off | Validation recall at Policy B |
|---|---|---|---|
| production | 0.9680 | 0.9951 | 0.193 |
| v2_dnn_lstm (seed 42, saved) | 0.7677 | 0.9681 | 0.336 |
| v2_dnn_only (seed 42, saved) | 0.7744 | 0.9119 | 0.344 |
| v2_dnn_lstm @ seed 11 | 0.8426 | 0.9380 | 0.361 |
| v2_dnn_only @ seed 11 | 0.6740 | 0.9618 | 0.471 |
| v2_dnn_lstm @ seed 12 | 0.8230 | 0.9713 | 0.437 |
| v2_dnn_only @ seed 12 | 0.8637 | 0.9790 | 0.286 |
| v2_dnn_lstm @ seed 13 | 0.8329 | 0.9244 | 0.311 |
| v2_dnn_only @ seed 13 | 0.7483 | 0.8761 | 0.387 |
| v2_dnn_lstm @ seed 14 | 0.7929 | 0.9431 | 0.429 |
| v2_dnn_only @ seed 14 | 0.8425 | 0.9654 | 0.403 |
| v2_dnn_lstm @ seed 15 | 0.8225 | 0.9752 | 0.395 |
| v2_dnn_only @ seed 15 | 0.7726 | 0.9322 | 0.395 |

## 4. Seed-42 result (the saved candidates, from the first report)

| Metric, primary population | production | v2_dnn_lstm | v2_dnn_only |
|---|---|---|---|
| Recall (Policy B) | 0.296 | 0.453 | 0.420 |
| Legit alerts per 1,000 | 10.31 | 8.91 | 9.03 |
| Critical-tier recall | 0.160 | 0.146 | 0.138 |
| First-fraud detection | 0.437 | 0.408 | 0.421 |
| Episodes with any alert | 0.607 | 0.545 | 0.553 |
| ROC-AUC | 0.732 | 0.929 | 0.903 |

`v2_dnn_lstm − v2_dnn_only` on seed 42: recall +0.033 [+0.014, +0.051] (data
variation only).

## 5. Multi-seed result: every training seed

Primary population, hold-out seeds 101–105 pooled. Intervals are over the
data (the model is fixed).

**v2_dnn_lstm**

| Training seed | Recall (Policy B) | Legit alerts / 1,000 | Critical recall | First-fraud detection | Episodes with any alert (of 550) | Fraud caught (of 2,396) | False positives | ROC-AUC |
|---|---|---|---|---|---|---|---|---|
| 42 (saved) | 0.453 [0.409, 0.494] | 8.91 [8.29, 9.53] | 0.146 [0.118, 0.176] | 0.408 [0.366, 0.452] | 300 (0.545) | 1,085 | 4,321 | 0.929 |
| 11 | 0.496 [0.452, 0.538] | 8.34 [7.79, 8.91] | 0.240 [0.210, 0.271] | 0.543 [0.495, 0.588] | 358 (0.651) | 1,189 | 4,045 | 0.941 |
| 12 | 0.563 [0.522, 0.603] | 10.01 [9.38, 10.66] | 0.212 [0.180, 0.244] | 0.545 [0.500, 0.588] | 371 (0.675) | 1,350 | 4,853 | 0.938 |
| 13 | 0.452 [0.415, 0.492] | 9.10 [8.68, 9.54] | 0.186 [0.162, 0.212] | 0.435 [0.394, 0.479] | 358 (0.651) | 1,084 | 4,414 | 0.944 |
| 14 | 0.573 [0.529, 0.615] | 8.25 [7.67, 8.89] | 0.225 [0.194, 0.257] | 0.610 [0.563, 0.653] | 392 (0.713) | 1,373 | 4,003 | 0.941 |
| 15 | 0.568 [0.526, 0.608] | 8.24 [7.60, 8.86] | 0.169 [0.141, 0.199] | 0.506 [0.462, 0.551] | 380 (0.691) | 1,361 | 3,995 | 0.946 |

**v2_dnn_only**

| Training seed | Recall (Policy B) | Legit alerts / 1,000 | Critical recall | First-fraud detection | Episodes with any alert (of 550) | Fraud caught (of 2,396) | False positives | ROC-AUC |
|---|---|---|---|---|---|---|---|---|
| 42 (saved) | 0.420 [0.380, 0.460] | 9.03 [8.47, 9.62] | 0.138 [0.114, 0.166] | 0.421 [0.376, 0.468] | 304 (0.553) | 1,006 | 4,382 | 0.903 |
| 11 | 0.529 [0.489, 0.566] | 9.82 [9.24, 10.40] | 0.168 [0.139, 0.200] | 0.729 [0.687, 0.766] | 433 (0.787) | 1,267 | 4,761 | 0.912 |
| 12 | 0.361 [0.319, 0.402] | 8.72 [8.16, 9.33] | 0.093 [0.077, 0.112] | 0.515 [0.474, 0.559] | 298 (0.542) | 866 | 4,231 | 0.900 |
| 13 | 0.420 [0.380, 0.461] | 8.75 [8.21, 9.31] | 0.209 [0.182, 0.238] | 0.668 [0.624, 0.710] | 379 (0.689) | 1,007 | 4,242 | 0.907 |
| 14 | 0.431 [0.391, 0.472] | 8.87 [8.33, 9.48] | 0.201 [0.173, 0.229] | 0.681 [0.638, 0.722] | 394 (0.716) | 1,033 | 4,303 | 0.905 |
| 15 | 0.449 [0.409, 0.491] | 8.87 [8.30, 9.47] | 0.222 [0.190, 0.254] | 0.597 [0.554, 0.640] | 363 (0.660) | 1,076 | 4,303 | 0.905 |

**Each model against production** (bold: the data interval excludes zero)

| Model | Recall − production | Legit alerts per 1,000 − production | First-fraud − production | Episodes − production |
|---|---|---|---|---|
| v2_dnn_lstm (seed 42, saved) | **+0.157 [+0.117, +0.197]** | **-1.41 [-1.97, -0.81]** | -0.029 [-0.078, +0.019] | **-0.062 [-0.110, -0.014]** |
| v2_dnn_only (seed 42, saved) | **+0.124 [+0.085, +0.160]** | **-1.28 [-1.80, -0.73]** | -0.016 [-0.067, +0.031] | **-0.055 [-0.102, -0.007]** |
| v2_dnn_lstm @ seed 11 | **+0.200 [+0.166, +0.236]** | **-1.98 [-2.51, -1.40]** | **+0.106 [+0.065, +0.146]** | **+0.044 [+0.002, +0.084]** |
| v2_dnn_only @ seed 11 | **+0.233 [+0.196, +0.267]** | -0.50 [-1.04, +0.08] | **+0.291 [+0.247, +0.337]** | **+0.180 [+0.136, +0.225]** |
| v2_dnn_lstm @ seed 12 | **+0.268 [+0.232, +0.304]** | -0.31 [-0.90, +0.31] | **+0.107 [+0.063, +0.153]** | **+0.067 [+0.022, +0.112]** |
| v2_dnn_only @ seed 12 | **+0.066 [+0.027, +0.103]** | **-1.59 [-2.14, -1.00]** | **+0.078 [+0.024, +0.129]** | **-0.065 [-0.114, -0.017]** |
| v2_dnn_lstm @ seed 13 | **+0.157 [+0.126, +0.188]** | **-1.21 [-1.61, -0.80]** | -0.002 [-0.050, +0.049] | **+0.044 [+0.002, +0.088]** |
| v2_dnn_only @ seed 13 | **+0.124 [+0.093, +0.154]** | **-1.57 [-2.03, -1.08]** | **+0.231 [+0.187, +0.275]** | **+0.082 [+0.041, +0.121]** |
| v2_dnn_lstm @ seed 14 | **+0.277 [+0.242, +0.312]** | **-2.06 [-2.63, -1.43]** | **+0.173 [+0.129, +0.219]** | **+0.105 [+0.065, +0.147]** |
| v2_dnn_only @ seed 14 | **+0.135 [+0.105, +0.165]** | **-1.44 [-1.94, -0.89]** | **+0.244 [+0.200, +0.290]** | **+0.109 [+0.069, +0.151]** |
| v2_dnn_lstm @ seed 15 | **+0.272 [+0.239, +0.305]** | **-2.08 [-2.67, -1.47]** | **+0.069 [+0.027, +0.111]** | **+0.084 [+0.042, +0.126]** |
| v2_dnn_only @ seed 15 | **+0.153 [+0.118, +0.187]** | **-1.44 [-1.97, -0.88]** | **+0.160 [+0.116, +0.208]** | **+0.053 [+0.008, +0.096]** |

* All 10 new models catch more fraud transactions than production at
  Policy B.
* 9 of 10 detect more first frauds than production and 9 of 10 reach more
  episodes. The first report's caveat that the candidates detect fewer
  episodes than production holds for the saved seed-42 models; it does not
  hold for the architectures in general.

**Recall by data seed** (Policy B)

| Data seed | lstm @11 | lstm @12 | lstm @13 | lstm @14 | lstm @15 | only @11 | only @12 | only @13 | only @14 | only @15 |
|---|---|---|---|---|---|---|---|---|---|---|
| 101 | 0.479 | 0.539 | 0.451 | 0.543 | 0.562 | 0.506 | 0.342 | 0.395 | 0.414 | 0.422 |
| 102 | 0.485 | 0.548 | 0.429 | 0.537 | 0.537 | 0.519 | 0.365 | 0.411 | 0.425 | 0.442 |
| 103 | 0.567 | 0.644 | 0.504 | 0.672 | 0.650 | 0.601 | 0.423 | 0.498 | 0.504 | 0.511 |
| 104 | 0.470 | 0.555 | 0.429 | 0.565 | 0.549 | 0.494 | 0.308 | 0.383 | 0.397 | 0.415 |
| 105 | 0.483 | 0.534 | 0.451 | 0.551 | 0.545 | 0.528 | 0.374 | 0.419 | 0.419 | 0.459 |

## 6. Aggregate by architecture

Mean, median, standard deviation and range are over the 5 training seeds.
The interval of the mean is a t-interval over training seeds (4 degrees of
freedom); it does not include data variation.

| Metric | Architecture | Mean | Median | SD | Range | 95% interval of the mean (training seeds) | Seed 42 (saved) |
|---|---|---|---|---|---|---|---|
| Recall (Policy B) | v2_dnn_lstm | 0.531 | 0.563 | 0.054 | 0.452–0.573 | [0.464, 0.597] | 0.453 |
| Recall (Policy B) | v2_dnn_only | 0.438 | 0.431 | 0.060 | 0.361–0.529 | [0.363, 0.513] | 0.420 |
| Legit alerts per 1,000 | v2_dnn_lstm | 8.79 | 8.34 | 0.77 | 8.24–10.01 | [7.83, 9.74] | 8.91 |
| Legit alerts per 1,000 | v2_dnn_only | 9.01 | 8.87 | 0.46 | 8.72–9.82 | [8.44, 9.57] | 9.03 |
| Critical-tier recall | v2_dnn_lstm | 0.206 | 0.212 | 0.029 | 0.169–0.240 | [0.170, 0.242] | 0.146 |
| Critical-tier recall | v2_dnn_only | 0.179 | 0.201 | 0.052 | 0.093–0.222 | [0.115, 0.243] | 0.138 |
| First-fraud detection | v2_dnn_lstm | 0.528 | 0.543 | 0.064 | 0.435–0.610 | [0.449, 0.607] | 0.408 |
| First-fraud detection | v2_dnn_only | 0.638 | 0.668 | 0.083 | 0.515–0.729 | [0.535, 0.742] | 0.421 |
| Episodes with any alert | v2_dnn_lstm | 0.676 | 0.675 | 0.027 | 0.651–0.713 | [0.643, 0.709] | 0.545 |
| Episodes with any alert | v2_dnn_only | 0.679 | 0.689 | 0.090 | 0.542–0.787 | [0.567, 0.791] | 0.553 |
| Precision (Policy B) | v2_dnn_lstm | 0.230 | 0.227 | 0.025 | 0.197–0.255 | [0.199, 0.261] | 0.201 |
| Precision (Policy B) | v2_dnn_only | 0.193 | 0.194 | 0.015 | 0.170–0.210 | [0.175, 0.212] | 0.187 |
| Critical legit alerts per 1,000 | v2_dnn_lstm | 0.77 | 0.65 | 0.22 | 0.60–1.11 | [0.50, 1.03] | 0.52 |
| Critical legit alerts per 1,000 | v2_dnn_only | 0.56 | 0.52 | 0.14 | 0.40–0.74 | [0.38, 0.74] | 0.76 |
| PR-AUC | v2_dnn_lstm | 0.317 | 0.319 | 0.049 | 0.250–0.388 | [0.256, 0.378] | 0.252 |
| PR-AUC | v2_dnn_only | 0.271 | 0.297 | 0.050 | 0.182–0.300 | [0.208, 0.333] | 0.213 |
| ROC-AUC | v2_dnn_lstm | 0.942 | 0.941 | 0.003 | 0.938–0.946 | [0.938, 0.945] | 0.929 |
| ROC-AUC | v2_dnn_only | 0.906 | 0.905 | 0.004 | 0.900–0.912 | [0.900, 0.911] | 0.903 |

## 7. Architecture effect: v2_dnn_lstm against v2_dnn_only

Difference of the means over training seeds, primary population. Bold: the
decision interval excludes zero.

| Metric | Difference of means | Interval: data only | Interval: training seeds only | Interval: data and training seeds | Seed pairs with v2_dnn_lstm higher | Seed-42 difference |
|---|---|---|---|---|---|---|
| Recall (Policy B) | +0.092 | [+0.074, +0.111] | [+0.009, +0.176] | **[+0.029, +0.157]** | 23/25 | +0.033 |
| Legit alerts per 1,000 | -0.22 | [-0.42, -0.03] | [-1.18, +0.74] | [-0.93, +0.53] | 9/25 | -0.13 |
| Critical-tier recall | +0.027 | [+0.019, +0.036] | [-0.037, +0.091] | [-0.015, +0.077] | 18/25 | +0.008 |
| First-fraud detection | -0.110 | [-0.127, -0.094] | [-0.220, -0.001] | **[-0.194, -0.021]** | 4/25 | -0.013 |
| Episodes with any alert | -0.003 | [-0.019, +0.014] | [-0.113, +0.107] | [-0.072, +0.079] | 10/25 | -0.007 |
| Precision (Policy B) | +0.037 | [+0.030, +0.045] | [+0.006, +0.068] | **[+0.014, +0.063]** | 23/25 | +0.014 |
| Critical legit alerts per 1,000 | +0.21 | [+0.16, +0.25] | [-0.07, +0.48] | **[+0.00, +0.42]** | 19/25 | -0.24 |
| PR-AUC | +0.046 | [+0.034, +0.058] | [-0.026, +0.119] | [-0.006, +0.108] | 21/25 | +0.039 |
| ROC-AUC | +0.036 | [+0.031, +0.042] | [+0.031, +0.042] | **[+0.029, +0.044]** | 25/25 | +0.026 |

**Same training seed, model against model** (data interval; the pairing by
seed number is only a convenient way to list them)

| Training seed | Recall | Legit alerts per 1,000 | Critical recall | First-fraud detection |
|---|---|---|---|---|
| 11 | **-0.033 [-0.054, -0.011]** | **-1.48 [-1.81, -1.15]** | **+0.071 [+0.051, +0.093]** | **-0.186 [-0.222, -0.152]** |
| 12 | **+0.202 [+0.172, +0.230]** | **+1.28 [+0.94, +1.62]** | **+0.118 [+0.096, +0.143]** | +0.029 [-0.002, +0.061] |
| 13 | **+0.032 [+0.012, +0.055]** | +0.35 [-0.02, +0.70] | **-0.023 [-0.038, -0.008]** | **-0.233 [-0.270, -0.193]** |
| 14 | **+0.142 [+0.117, +0.168]** | **-0.62 [-0.90, -0.33]** | **+0.024 [+0.006, +0.042]** | **-0.071 [-0.094, -0.050]** |
| 15 | **+0.119 [+0.096, +0.143]** | **-0.64 [-0.92, -0.36]** | **-0.053 [-0.075, -0.034]** | **-0.091 [-0.122, -0.059]** |

**Reading**

* **Recall at Policy B.** `v2_dnn_lstm` is higher by 0.092. Every interval
  excludes zero, including the one over training seeds alone. This is larger
  than the seed-42 difference of 0.033.
* **Not uniform.** With seed 11, `v2_dnn_only` has the higher recall, and the
  best `v2_dnn_only` run (0.529) is above two of the five `v2_dnn_lstm` runs.
* **Alert burden.** No difference: −0.22 per 1,000, interval [−0.93, +0.53].
* **Critical tier.** No difference: +0.027, interval [−0.015, +0.077].
* **First-fraud detection.** `v2_dnn_only` is higher by 0.110, and the
  decision interval excludes zero. `v2_dnn_lstm` catches more fraud
  transactions; `v2_dnn_only` more often catches the first one.
* **Episodes with any alert.** No difference (0.676 and 0.679).
* **ROC-AUC.** `v2_dnn_lstm` is higher in all 25 seed pairs (+0.036), with
  very little training-seed spread.

**Secondary population (from 2026-06-01)**

| Metric | Difference of means | Interval: data only | Interval: training seeds only | Interval: data and training seeds | Seed pairs with v2_dnn_lstm higher | Seed-42 difference |
|---|---|---|---|---|---|---|
| Recall (Policy B) | +0.133 | [+0.097, +0.172] | [+0.024, +0.242] | **[+0.046, +0.228]** | 23/25 | +0.083 |
| Legit alerts per 1,000 | -0.30 | [-0.69, +0.11] | [-1.03, +0.43] | [-1.04, +0.41] | 9/25 | -0.04 |
| Critical-tier recall | +0.024 | [+0.008, +0.040] | [-0.040, +0.089] | [-0.024, +0.079] | 14/25 | -0.016 |
| First-fraud detection | -0.109 | [-0.150, -0.069] | [-0.261, +0.043] | [-0.235, +0.019] | 5/25 | +0.008 |
| Episodes with any alert | +0.038 | [-0.008, +0.083] | [-0.100, +0.175] | [-0.066, +0.151] | 13/25 | +0.008 |
| Precision (Policy B) | +0.056 | [+0.038, +0.076] | [+0.003, +0.109] | **[+0.014, +0.104]** | 22/25 | +0.032 |
| Critical legit alerts per 1,000 | +0.16 | [+0.09, +0.24] | [-0.13, +0.45] | [-0.06, +0.41] | 18/25 | -0.24 |
| PR-AUC | +0.071 | [+0.046, +0.095] | [-0.009, +0.150] | **[+0.012, +0.145]** | 21/25 | +0.043 |
| ROC-AUC | +0.038 | [+0.028, +0.048] | [+0.030, +0.045] | **[+0.026, +0.050]** | 25/25 | +0.025 |

The direction is the same. First-fraud detection is not separated here.

## 8. Training-seed variability

| Metric, primary | v2_dnn_lstm: SD (range) | v2_dnn_only: SD (range) | Gap between architectures |
|---|---|---|---|
| Recall (Policy B) | 0.054 (0.452–0.573) | 0.060 (0.361–0.529) | 0.092 |
| Legit alerts per 1,000 | 0.77 (8.24–10.01) | 0.46 (8.72–9.82) | 0.22 |
| Critical-tier recall | 0.029 (0.169–0.240) | 0.052 (0.093–0.222) | 0.027 |
| First-fraud detection | 0.064 (0.435–0.610) | 0.083 (0.515–0.729) | 0.110 |
| Episodes with any alert | 0.027 (0.651–0.713) | 0.090 (0.542–0.787) | 0.003 |
| ROC-AUC | 0.003 (0.938–0.946) | 0.004 (0.900–0.912) | 0.036 |

* **A single run is not a reliable picture of an architecture.** For recall,
  the spread between runs of one architecture is about two thirds of the gap
  between the architectures.
* **The data interval of one model understates the uncertainty.** A typical
  data interval for recall is ±0.04. Two runs of the same architecture differ
  by up to 0.12 (`v2_dnn_lstm`) and 0.17 (`v2_dnn_only`).
* `v2_dnn_only` varies more between runs than `v2_dnn_lstm` on recall,
  Critical-tier recall and episode detection.
* ROC-AUC is the only stable measure. Results at a cut-off move much more
  than the ranking quality does.

**Where the saved seed-42 candidates sit**

| Metric | Architecture | Seed 42 (saved) | Multi-seed mean | Multi-seed range | Seed 42 inside the range |
|---|---|---|---|---|---|
| Recall (Policy B) | v2_dnn_lstm | 0.453 | 0.531 | 0.452–0.573 | yes |
| Recall (Policy B) | v2_dnn_only | 0.420 | 0.438 | 0.361–0.529 | yes |
| Legit alerts per 1,000 | v2_dnn_lstm | 8.91 | 8.79 | 8.24–10.01 | yes |
| Legit alerts per 1,000 | v2_dnn_only | 9.03 | 9.01 | 8.72–9.82 | yes |
| Critical-tier recall | v2_dnn_lstm | 0.146 | 0.206 | 0.169–0.240 | **no** |
| Critical-tier recall | v2_dnn_only | 0.138 | 0.179 | 0.093–0.222 | yes |
| First-fraud detection | v2_dnn_lstm | 0.408 | 0.528 | 0.435–0.610 | **no** |
| First-fraud detection | v2_dnn_only | 0.421 | 0.638 | 0.515–0.729 | **no** |
| Episodes with any alert | v2_dnn_lstm | 0.545 | 0.676 | 0.651–0.713 | **no** |
| Episodes with any alert | v2_dnn_only | 0.553 | 0.679 | 0.542–0.787 | yes |
| ROC-AUC | v2_dnn_lstm | 0.929 | 0.942 | 0.938–0.946 | **no** |
| ROC-AUC | v2_dnn_only | 0.903 | 0.906 | 0.900–0.912 | yes |

* For `v2_dnn_lstm`, seed 42 is at the very bottom of the multi-seed range
  on recall and below it on Critical-tier recall, first-fraud detection,
  episode detection and ROC-AUC. For `v2_dnn_only` it is below the range on
  first-fraud detection.
* With five new runs this may be chance. No cause was found: the seed-42
  retraining above reproduces the saved models.
* **Consequence.** Conclusions drawn from seed 42 alone were pessimistic
  about both architectures and understated the gap between them.

## 9. Ring fraud

20 rings, 80 victim episodes, 176 ring fraud transactions (primary
population).

| Model (Policy B) | Victim episodes detected (of 80) | Victims' first fraud detected (of 80) | Ring fraud transactions caught (of 176) | Rings with at least one / two victims alerted (of 20) |
|---|---|---|---|---|
| production | 63 | 56 | 122 | 20 / 17 |
| v2_dnn_lstm (seed 42, saved) | 60 | 43 | 121 | 20 / 18 |
| v2_dnn_only (seed 42, saved) | 48 | 42 | 96 | 18 / 15 |
| v2_dnn_lstm @ seed 11 | 70 | 63 | 142 | 20 / 19 |
| v2_dnn_only @ seed 11 | 78 | 77 | 151 | 20 / 20 |
| v2_dnn_lstm @ seed 12 | 68 | 64 | 144 | 20 / 20 |
| v2_dnn_only @ seed 12 | 65 | 59 | 103 | 20 / 20 |
| v2_dnn_lstm @ seed 13 | 63 | 49 | 109 | 20 / 20 |
| v2_dnn_only @ seed 13 | 75 | 74 | 143 | 20 / 20 |
| v2_dnn_lstm @ seed 14 | 76 | 72 | 167 | 20 / 20 |
| v2_dnn_only @ seed 14 | 77 | 76 | 143 | 20 / 20 |
| v2_dnn_lstm @ seed 15 | 69 | 56 | 150 | 20 / 19 |
| v2_dnn_only @ seed 15 | 69 | 66 | 126 | 20 / 20 |

| Across training seeds (Policy B) | Architecture | Mean | SD | Range |
|---|---|---|---|---|
| Victim episodes detected | v2_dnn_lstm | 0.865 | 0.058 | 0.787–0.950 |
| Victim episodes detected | v2_dnn_only | 0.910 | 0.070 | 0.812–0.975 |
| Ring fraud transactions caught | v2_dnn_lstm | 0.809 | 0.120 | 0.619–0.949 |
| Ring fraud transactions caught | v2_dnn_only | 0.757 | 0.109 | 0.585–0.858 |

| Model (Critical) | Victim episodes detected (of 80) | Victims' first fraud detected (of 80) | Ring fraud transactions caught (of 176) | Rings with at least one / two victims alerted (of 20) |
|---|---|---|---|---|
| production | 43 | 34 | 77 | 18 / 14 |
| v2_dnn_lstm (seed 42, saved) | 18 | 7 | 22 | 9 / 5 |
| v2_dnn_only (seed 42, saved) | 23 | 16 | 31 | 10 / 8 |
| v2_dnn_lstm @ seed 11 | 47 | 36 | 76 | 19 / 16 |
| v2_dnn_only @ seed 11 | 25 | 20 | 33 | 9 / 8 |
| v2_dnn_lstm @ seed 12 | 36 | 23 | 51 | 16 / 10 |
| v2_dnn_only @ seed 12 | 26 | 21 | 31 | 9 / 9 |
| v2_dnn_lstm @ seed 13 | 52 | 37 | 67 | 20 / 18 |
| v2_dnn_only @ seed 13 | 61 | 53 | 91 | 20 / 20 |
| v2_dnn_lstm @ seed 14 | 42 | 27 | 76 | 17 / 13 |
| v2_dnn_only @ seed 14 | 54 | 49 | 92 | 20 / 17 |
| v2_dnn_lstm @ seed 15 | 30 | 10 | 39 | 15 / 10 |
| v2_dnn_only @ seed 15 | 43 | 39 | 66 | 17 / 15 |

| Across training seeds (Critical) | Architecture | Mean | SD | Range |
|---|---|---|---|---|
| Victim episodes detected | v2_dnn_lstm | 0.517 | 0.109 | 0.375–0.650 |
| Victim episodes detected | v2_dnn_only | 0.522 | 0.203 | 0.312–0.762 |
| Ring fraud transactions caught | v2_dnn_lstm | 0.351 | 0.093 | 0.222–0.432 |
| Ring fraud transactions caught | v2_dnn_only | 0.356 | 0.169 | 0.176–0.523 |

* **Policy B.** No clear difference between the architectures. Every new
  model alerts on all 20 rings.
* **Critical cut-off.** Results vary widely between training seeds (25 to 61
  victims of 80 for `v2_dnn_only`; 30 to 52 for `v2_dnn_lstm`). The earlier
  finding that production is far better here was specific to the seed-42
  candidates.
* 20 rings is a small sample. No ring superiority is claimed.

## 10. New-customer results (seeds 201–205, fewer than 10 earlier transactions)

25,000 transactions; 439 fraud transactions; 151 first frauds. No cold-start
behaviour was changed.

| Model: Current behaviour (Policy B cut-off) | Fraud caught (of 439) | First frauds caught (of 151) | Legitimate alerts | Legit alerts per 1,000 | Recall |
|---|---|---|---|---|---|
| production | 125 | 72 | 1,906 | 76.2 | 0.285 |
| v2_dnn_lstm (seed 42, saved) | 184 | 68 | 415 | 16.6 | 0.419 |
| v2_dnn_only (seed 42, saved) | 195 | 73 | 570 | 22.8 | 0.444 |
| v2_dnn_lstm @ seed 11 | 132 | 80 | 632 | 25.3 | 0.301 |
| v2_dnn_only @ seed 11 | 235 | 102 | 2,038 | 81.5 | 0.535 |
| v2_dnn_lstm @ seed 12 | 202 | 80 | 757 | 30.3 | 0.460 |
| v2_dnn_only @ seed 12 | 177 | 74 | 647 | 25.9 | 0.403 |
| v2_dnn_lstm @ seed 13 | 130 | 72 | 375 | 15.0 | 0.296 |
| v2_dnn_only @ seed 13 | 194 | 88 | 1,695 | 67.8 | 0.442 |
| v2_dnn_lstm @ seed 14 | 179 | 78 | 638 | 25.5 | 0.408 |
| v2_dnn_only @ seed 14 | 198 | 93 | 1,251 | 50.0 | 0.451 |
| v2_dnn_lstm @ seed 15 | 174 | 75 | 442 | 17.7 | 0.396 |
| v2_dnn_only @ seed 15 | 210 | 85 | 630 | 25.2 | 0.478 |

| Current behaviour (Policy B cut-off): across training seeds | Architecture | Mean | SD | Range |
|---|---|---|---|---|
| Recall | v2_dnn_lstm | 0.372 | 0.072 | 0.296–0.460 |
| Recall | v2_dnn_only | 0.462 | 0.049 | 0.403–0.535 |
| First-fraud detection | v2_dnn_lstm | 0.510 | 0.023 | 0.477–0.530 |
| First-fraud detection | v2_dnn_only | 0.585 | 0.068 | 0.490–0.675 |
| Legit alerts per 1,000 | v2_dnn_lstm | 22.8 | 6.3 | 15.0–30.3 |
| Legit alerts per 1,000 | v2_dnn_only | 50.1 | 25.0 | 25.2–81.5 |

| Current behaviour (Policy B cut-off): v2_dnn_lstm − v2_dnn_only | Difference of means | Interval: training seeds only | Interval: data and training seeds |
|---|---|---|---|
| Recall | -0.090 | [-0.181, +0.002] | **[-0.163, -0.021]** |
| First-fraud detection | -0.075 | [-0.159, +0.008] | **[-0.145, -0.006]** |
| Legit alerts per 1,000 | -27.3 | [-58.0, +3.4] | **[-47.7, -7.9]** |

| Model: Proposed policy (Critical cut-off only) | Fraud caught (of 439) | First frauds caught (of 151) | Legitimate alerts | Legit alerts per 1,000 | Recall |
|---|---|---|---|---|---|
| production | 51 | 45 | 509 | 20.4 | 0.116 |
| v2_dnn_lstm (seed 42, saved) | 12 | 5 | 0 | 0.0 | 0.027 |
| v2_dnn_only (seed 42, saved) | 67 | 34 | 28 | 1.1 | 0.153 |
| v2_dnn_lstm @ seed 11 | 79 | 58 | 119 | 4.8 | 0.180 |
| v2_dnn_only @ seed 11 | 85 | 27 | 15 | 0.6 | 0.194 |
| v2_dnn_lstm @ seed 12 | 31 | 26 | 8 | 0.3 | 0.071 |
| v2_dnn_only @ seed 12 | 43 | 33 | 12 | 0.5 | 0.098 |
| v2_dnn_lstm @ seed 13 | 52 | 44 | 23 | 0.9 | 0.118 |
| v2_dnn_only @ seed 13 | 113 | 69 | 111 | 4.4 | 0.257 |
| v2_dnn_lstm @ seed 14 | 40 | 30 | 9 | 0.4 | 0.091 |
| v2_dnn_only @ seed 14 | 94 | 58 | 61 | 2.4 | 0.214 |
| v2_dnn_lstm @ seed 15 | 4 | 4 | 0 | 0.0 | 0.009 |
| v2_dnn_only @ seed 15 | 121 | 55 | 40 | 1.6 | 0.276 |

| Proposed policy (Critical cut-off only): across training seeds | Architecture | Mean | SD | Range |
|---|---|---|---|---|
| Recall | v2_dnn_lstm | 0.094 | 0.063 | 0.009–0.180 |
| Recall | v2_dnn_only | 0.208 | 0.070 | 0.098–0.276 |
| First-fraud detection | v2_dnn_lstm | 0.215 | 0.134 | 0.026–0.384 |
| First-fraud detection | v2_dnn_only | 0.321 | 0.117 | 0.179–0.457 |
| Legit alerts per 1,000 | v2_dnn_lstm | 1.3 | 2.0 | 0.0–4.8 |
| Legit alerts per 1,000 | v2_dnn_only | 1.9 | 1.6 | 0.5–4.4 |

| Proposed policy (Critical cut-off only): v2_dnn_lstm − v2_dnn_only | Difference of means | Interval: training seeds only | Interval: data and training seeds |
|---|---|---|---|
| Recall | -0.114 | [-0.211, -0.017] | **[-0.200, -0.031]** |
| First-fraud detection | -0.106 | [-0.290, +0.078] | [-0.255, +0.032] |
| Legit alerts per 1,000 | -0.6 | [-3.3, +2.0] | [-2.6, +1.5] |

| Model: Measurement: Policy B, flags neutral | Fraud caught (of 439) | First frauds caught (of 151) | Legitimate alerts | Legit alerts per 1,000 | Recall |
|---|---|---|---|---|---|
| production | 14 | 14 | 2 | 0.1 | 0.032 |
| v2_dnn_lstm (seed 42, saved) | 164 | 59 | 139 | 5.6 | 0.374 |
| v2_dnn_only (seed 42, saved) | 177 | 66 | 166 | 6.6 | 0.403 |
| v2_dnn_lstm @ seed 11 | 115 | 74 | 260 | 10.4 | 0.262 |
| v2_dnn_only @ seed 11 | 214 | 93 | 1,353 | 54.1 | 0.487 |
| v2_dnn_lstm @ seed 12 | 176 | 73 | 247 | 9.9 | 0.401 |
| v2_dnn_only @ seed 12 | 178 | 77 | 432 | 17.3 | 0.405 |
| v2_dnn_lstm @ seed 13 | 88 | 59 | 62 | 2.5 | 0.200 |
| v2_dnn_only @ seed 13 | 170 | 83 | 425 | 17.0 | 0.387 |
| v2_dnn_lstm @ seed 14 | 169 | 76 | 339 | 13.6 | 0.385 |
| v2_dnn_only @ seed 14 | 175 | 85 | 628 | 25.1 | 0.399 |
| v2_dnn_lstm @ seed 15 | 158 | 74 | 298 | 11.9 | 0.360 |
| v2_dnn_only @ seed 15 | 188 | 76 | 289 | 11.6 | 0.428 |

| Measurement: Policy B, flags neutral: across training seeds | Architecture | Mean | SD | Range |
|---|---|---|---|---|
| Recall | v2_dnn_lstm | 0.322 | 0.087 | 0.200–0.401 |
| Recall | v2_dnn_only | 0.421 | 0.040 | 0.387–0.487 |
| First-fraud detection | v2_dnn_lstm | 0.472 | 0.046 | 0.391–0.503 |
| First-fraud detection | v2_dnn_only | 0.548 | 0.045 | 0.503–0.616 |
| Legit alerts per 1,000 | v2_dnn_lstm | 9.6 | 4.3 | 2.5–13.6 |
| Legit alerts per 1,000 | v2_dnn_only | 25.0 | 17.0 | 11.6–54.1 |

| Measurement: Policy B, flags neutral: v2_dnn_lstm − v2_dnn_only | Difference of means | Interval: training seeds only | Interval: data and training seeds |
|---|---|---|---|
| Recall | -0.100 | [-0.206, +0.006] | **[-0.183, -0.024]** |
| First-fraud detection | -0.077 | [-0.143, -0.010] | **[-0.144, -0.019]** |
| Legit alerts per 1,000 | -15.4 | [-36.2, +5.4] | **[-30.0, -4.2]** |

**Reading**

* **Current behaviour.** `v2_dnn_only` catches more early fraud (recall 0.462
  against 0.372) and raises about twice as many legitimate alerts (50 against
  23 per 1,000 on average). Both are far over the budget of 10.
* **The legitimate alert rate on early transactions is not a stable property
  of a model.** It runs from 15 to 30 per 1,000 for `v2_dnn_lstm` and from 25
  to 82 for `v2_dnn_only`.
* **Proposed Critical-only policy.** Legitimate alerts fall to about 1–2 per
  1,000. Detection falls to 0.09 (`v2_dnn_lstm`, range 0.01–0.18) and 0.21
  (`v2_dnn_only`). For one `v2_dnn_lstm` run (seed 15) it is close to no
  detection: 4 of 439.
* **Policy B with the two flags neutral.** The first report measured 5.6 and
  6.6 legitimate alerts per 1,000 on seed 42 and called it inside the budget.
  **That does not hold across training seeds:** 2.5 to 13.6 for
  `v2_dnn_lstm` and 11.6 to 54.1 for `v2_dnn_only`.
* The new-customer fraud is generator-written. These results describe those
  patterns only.

## 11. Is the Policy B alert budget stable across training seeds?

**No.** Budget: 10 legitimate alerts per 1,000 transactions (primary
population).

| Training seed | v2_dnn_lstm: seeds 101–105 pooled | v2_dnn_lstm: seeds 201–205 (unchanged customers) | v2_dnn_only: seeds 101–105 pooled | v2_dnn_only: seeds 201–205 (unchanged customers) |
|---|---|---|---|---|
| 42 (saved) | 8.91 [8.29, 9.53] | 10.42 | 9.03 [8.47, 9.62] | 10.23 |
| 11 | 8.34 [7.79, 8.91] | 9.65 | 9.82 [9.24, 10.40] | 10.93 |
| 12 | 10.01 [9.38, 10.66] | 11.57 | 8.72 [8.16, 9.33] | 10.17 |
| 13 | 9.10 [8.68, 9.54] | 9.98 | 8.75 [8.21, 9.31] | 10.19 |
| 14 | 8.25 [7.67, 8.89] | 9.49 | 8.87 [8.33, 9.48] | 10.30 |
| 15 | 8.24 [7.60, 8.86] | 9.95 | 8.87 [8.30, 9.47] | 10.30 |

| Measure | v2_dnn_lstm | v2_dnn_only |
|---|---|---|
| Mean across training seeds, seeds 101–105 | 8.79 | 9.01 |
| Range across training seeds, seeds 101–105 | 8.24–10.01 | 8.72–9.82 |
| SD between training seeds | 0.77 | 0.46 |
| SD between data seeds (ten datasets) | 0.97 | 0.92 |
| Training seeds over 10 on seeds 101–105 (point) | 1 of 5 | 0 of 5 |
| Training seeds whose upper bound is over 10 | 1 of 5 | 1 of 5 |
| Mean across training seeds, seeds 201–205 | 10.13 | 10.38 |
| Range across training seeds, seeds 201–205 | 9.49–11.57 | 10.17–10.93 |
| Training seed × data seed cells over 10 | 18 of 50 | 18 of 50 |
| Range over all cells | 7.52–13.15 | 8.37–12.00 |
| Rate on the validation period (where the cut-off is set) | 9.87 for every model | 9.87 for every model |

**Every training seed on every data seed** (legitimate alerts per 1,000;
seeds 201–205 are the customers the generator extension does not touch)

| Data seed | lstm @11 | lstm @12 | lstm @13 | lstm @14 | lstm @15 | only @11 | only @12 | only @13 | only @14 | only @15 |
|---|---|---|---|---|---|---|---|---|---|---|
| 101 | 8.5 | 10.2 | 9.3 | 8.6 | 8.3 | 9.9 | 8.5 | 8.9 | 9.1 | 9.0 |
| 102 | 8.5 | 10.5 | 8.7 | 9.1 | 8.8 | 10.1 | 9.0 | 8.9 | 9.3 | 9.1 |
| 103 | 8.7 | 10.0 | 9.2 | 7.5 | 7.9 | 9.3 | 8.7 | 8.4 | 8.8 | 8.6 |
| 104 | 7.6 | 9.2 | 9.1 | 7.5 | 7.9 | 9.6 | 8.5 | 8.7 | 8.4 | 8.7 |
| 105 | 8.3 | 10.1 | 9.2 | 8.5 | 8.3 | 10.2 | 8.9 | 8.8 | 8.7 | 9.0 |
| 201 | 10.8 | 12.6 | 10.3 | 10.7 | 10.8 | 11.4 | 11.1 | 10.7 | 10.8 | 11.0 |
| 202 | 10.3 | 13.1 | 10.5 | 10.6 | 11.2 | 12.0 | 10.9 | 11.3 | 11.5 | 11.3 |
| 203 | 9.2 | 10.9 | 10.4 | 9.0 | 9.4 | 10.7 | 9.5 | 9.9 | 10.0 | 10.0 |
| 204 | 8.5 | 10.1 | 9.3 | 8.1 | 8.9 | 10.1 | 9.4 | 9.5 | 9.2 | 9.1 |
| 205 | 9.4 | 11.1 | 9.4 | 9.1 | 9.4 | 10.5 | 9.9 | 9.5 | 10.0 | 10.1 |

* Every cut-off gives exactly 9.87 per 1,000 on the validation period where
  it is set. On new data the same cut-offs give 7.5 to 13.2.
* **Training seed.** `v2_dnn_lstm` with seed 12 is over the budget on the
  pre-registered hold-out (10.01); the other four are at 8.2–9.1. One
  `v2_dnn_only` run has an upper bound over 10.
* **Data seed.** The variation between datasets is larger than between
  training runs (SD about 0.9–1.0 against 0.5–0.8). Data seeds 201 and 202
  are over the budget for all 10 models.
* **Conclusion.** A cut-off measured once on one validation period does not
  hold the budget to within 1 per 1,000. This is true for both
  architectures, so it does not help choose between them. It remains a
  promotion blocker for the threshold policy.

## 12. Latency

Live pipeline, one transaction at a time, 200 scorings after 20 warm-up, one
multi-seed model per architecture (seed 11). Latency depends on the
architecture, not on the trained weights. Measured on the machine that ran
this evaluation.

| Model | Median | 95th percentile |
|---|---|---|
| v2_dnn_lstm @ seed 11 | 340 ms | 440 ms |
| v2_dnn_only @ seed 11 | 264 ms | 373 ms |

These match the first report (329 / 448 ms and 263 / 369 ms).

## 13. Decision

The tie-break rule, unchanged, applied to the difference of the
architectures' means over training seeds on the primary population.

| Condition | Test | Difference | Decision interval (data and training seeds) | Training seeds only | Holds |
|---|---|---|---|---|---|
| Recall at least matches | Lower bound above −0.05 | +0.092 | [+0.029, +0.157] | [+0.009, +0.176] | Yes |
| Alert burden at least matches | Upper bound below +1.0 per 1,000 | −0.22 | [−0.93, +0.53] | [−1.18, +0.74] | Yes |
| Critical tier not worse | Lower bound above −0.05 | +0.027 | [−0.015, +0.077] | [−0.037, +0.091] | Yes |

* All three conditions hold, on both intervals.
* **`v2_dnn_lstm` remains the provisional lead.** `v2_dnn_only` is not
  selected on simplicity.
* `MODEL_SET` was not changed.

**What the rule does not weigh** (for the owner)

* `v2_dnn_only` detects the first fraud more often (+0.11).
* `v2_dnn_only` is about 75 ms faster per transaction and has no LSTM.
* On early transactions `v2_dnn_lstm` raises about half as many legitimate
  alerts and catches less fraud.
* The lead describes the architecture on average. It does not say that a
  particular trained `v2_dnn_lstm` model is better than a particular
  `v2_dnn_only` model.

## 14. Remaining promotion blockers

| Blocker | State after this step |
|---|---|
| Which trained model would be promoted | **New, open.** The saved seed-42 candidates are the only ones the application can load, and they are among the weakest runs. Choosing a better seed by its hold-out result would be selection on the hold-out and would need a fresh hold-out or a rule fixed in advance (for example, choose on validation only). |
| Threshold policy and alert budget | Open. The budget is unstable across training seeds and data seeds (§11). The owner must approve a policy that accounts for this. |
| Thresholds recorded per model set | Open. Depends on the two rows above. |
| New-customer policy | Open. Early-transaction behaviour varies strongly between training runs (§10). |
| Live data version (v1 or v2) | Open. Not addressed here. |
| Rollback triggers | Open. Not addressed here. |
| "At least 5 seeds", training seeds | Evidence now exists: 5 training seeds per architecture, on 5 + 5 data seeds. |
| Candidate-versus-candidate analysis | Done with training-seed variation included (§7, §13). |
| Final regression at the promotion commit | Blocked: no promotion commit exists. |

## 15. Limits

* Five training seeds per architecture. Intervals over training seeds are
  rough, and the decision interval resamples only five runs.
* All data is synthetic and from one generator. The hold-out differs from the
  training data in its random seed, not in its design.
* All models were trained on one dataset (data seed 42). Variation from the
  training data itself is not measured.
* The new models were trained and scored in the cloud workspace (Linux,
  TensorFlow 2.21.0). Retraining on another machine gives slightly different
  weights. The saved artifacts are the reference.
* The live application still scores v1 customers. Nothing here measures any
  candidate on v1.

## 16. State and reproduction

| Item | State |
|---|---|
| `MODEL_SET` default | `production` |
| Promotion or deployment | none |
| `backend/models/saved/` | unchanged (7 checksums verified) |
| `backend/models/candidates/` (seed 42) | unchanged (15 checksums verified) |
| Cut-offs of the saved models; alert bands 25 / 50 / 80 | unchanged |
| HEAD | `582fe60`; nothing committed or pushed |

**Files added**

* `backend/app/training/multiseed.py`
* `backend/app/evaluation/multiseed.py`
* `backend/tests/test_multiseed.py` (22 tests)
* `backend/models/candidates_multiseed/v2/seed_{11..15}/` (70 files)
* `backend/models/evaluation/v2_holdout/multiseed_report.json`
* `docs/step4c3e-multiseed-training-report.md`

**Files changed**

* `backend/app/evaluation/holdout.py`: optional arguments so cut-offs and
  model pairs can be passed in. Defaults are unchanged, and the saved models'
  results in the multi-seed report equal the hold-out report exactly.

**Tests:** the 22 new tests pass; the full backend suite passes (492 tests) in the cloud workspace (Python 3.13, TensorFlow 2.21.0, Keras 3.15.1). They have not been run on your machine yet.

**To reproduce** (from `backend`, with the project's virtual environment)

```
python -m app.training.multiseed          # only seeds not yet trained; about 6 minutes each
python -m app.evaluation.multiseed        # about 25 minutes
python -m app.evaluation.multiseed latency
```

* The evaluation needs the hold-out datasets (`python -m
  app.evaluation.holdout --generate` and `... new-customer --generate`).
* The trained models are already in place, so the first command trains
  nothing unless a seed directory is removed.

**Not started:** Step 4C-3F, controlled promotion.
