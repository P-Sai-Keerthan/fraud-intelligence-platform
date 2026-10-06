# Step 4C-3E: evidence closure results

Written at commit `582fe60` (nothing committed). This report gives the
results of the approved evidence-closure plan
(`docs/step4c3e-evidence-closure-plan.md`).

**NO PROMOTION.** `MODEL_SET` still defaults to `production`. No model was
trained, retrained, promoted or deployed. No threshold or alert band was
changed. Production and candidate model files are unchanged (checksums
verified at every stage).

Scores are model scores. They are not calibrated probabilities.

## 1. Summary

| Question | Result |
|---|---|
| Hold-out size | 5 fresh datasets (seeds 101–105), 510,030 transactions, 550 fraud episodes, 20 rings. All out of sample for the saved candidates. |
| Candidates against production, Policy B | Both candidates catch more fraud transactions with fewer legitimate alerts. The paired intervals exclude zero. |
| `v2_dnn_lstm` against `v2_dnn_only` | All three pre-registered tie-break conditions hold. `v2_dnn_lstm` **remains the provisional lead**. Its recall is higher by 0.033 [0.014, 0.051]. |
| Pre-registered rule against production | Both candidates meet it on the primary population. |
| What the primary endpoints do not show | Neither candidate detects more fraud **episodes** or more **first frauds** than production. Both detect fewer episodes. Neither is better in the Critical tier. |
| Ring fraud | Measured on 20 rings. At Policy B `v2_dnn_lstm` is level with production; `v2_dnn_only` is lower. At the Critical cut-off production is clearly higher than both. |
| New-customer fraud | Generator extension added, off by default, default output byte-identical. 151 new-customer episodes evaluated. The proposed Critical-only policy removes almost all detection for `v2_dnn_lstm`. |
| Alert budget | Met on seeds 101–105. **Not robust:** on the five further seeds (201–205) the candidates are at about 10.2–10.4 per 1,000, above the budget of 10. |
| Multi-seed retraining | **Still necessary** before a final choice between the candidates and before the "5 seeds" gate can be called closed for training. See §12. |

## 2. What was run

| Stage | What | Result |
|---|---|---|
| 1 | Hold-out harness and tests | `backend/app/evaluation/holdout.py`, `ring_metrics.py`, `tests/test_holdout.py` |
| 2 | Generate and score seeds 101–105 | `data/v2_holdout/seed_<seed>/` (untracked, ignored by git) |
| 3 | Hold-out comparison report | `backend/models/evaluation/v2_holdout/holdout_report.json` |
| 4 | Review of the evidence gates | §11 |
| 5 | Generator extension for new-customer fraud | `data/v2/synth_v2/` (2.1.0, off by default) |
| 6 | New-customer datasets (seeds 201–205) and analysis | `data/v2_holdout/new_customer/seed_<seed>/` (untracked); `new_customer_report.json` |
| 7 | Multi-seed candidate retraining | **Not run.** Not approved. §12 says whether it is still needed. |

**Fixed before any model was scored on this data**

* Seeds 101–105. Cut-offs from the seed-42 validation period (4C-3B), not
  re-tuned: Critical 0.9951 / 0.9681 / 0.9119 and Policy B 0.9680 / 0.7677 /
  0.7744 for production / v2_dnn_lstm / v2_dnn_only.
* Primary population: transactions with at least 10 earlier transactions.
  Secondary: the same from 2026-06-01.
* Margins: recall 0.05; legitimate alerts 1.0 per 1,000; Critical-tier recall
  0.05. Budget: 10 legitimate alerts per 1,000. Minimum recall: 0.40.
* Uncertainty: 2,000 bootstrap resamples (seed 42). Whole customer groups
  (a ring's victims, a household) are resampled within each dataset. All
  model sets are scored on the same resample, so differences are paired.

**How each transaction was scored.** Exactly as the live pipeline does it:
the LSTM on the 10 earlier transactions when they exist, the cold-start
handling when they do not. A test checks the batch scoring against
`FraudIntelligencePipeline.score_transaction` on sampled rows for all three
model sets.

## 3. Hold-out datasets

| Seed | Transactions | Fraud transactions | Fraud episodes | Rings | Ring victim episodes | Customer groups | Fraud with fewer than 10 earlier | SHA-256 (first 12) |
|---|---|---|---|---|---|---|---|---|
| 101 | 101,033 | 486 | 110 | 4 | 16 | 443 | 0 | `996ae5cf2441` |
| 102 | 104,638 | 484 | 110 | 4 | 16 | 442 | 2 | `ea4dbb4ff12b` |
| 103 | 100,485 | 466 | 110 | 4 | 16 | 442 | 0 | `3321d13e3b7c` |
| 104 | 101,636 | 494 | 110 | 4 | 16 | 441 | 0 | `b43c57b05dd2` |
| 105 | 102,238 | 468 | 110 | 4 | 16 | 443 | 0 | `2b75986489d8` |

* Total: 510,030 transactions, 2,398 fraud transactions, 550 fraud episodes,
  20 rings, 80 ring victim episodes, 2,211 customer groups.
* The raw transactions are byte-identical to the feasibility data counted in
  the plan.
* Primary population: 485,030 transactions, 2,396 fraud transactions, 550
  episodes, 549 first frauds (one episode's first fraud has fewer than 10
  earlier transactions).
* Secondary population: 106,948 transactions, 496 fraud transactions, 122
  episodes. One episode spans the boundary, so the count is 122, not 121.
* Every dataset matches its manifest checksum. The harness refuses seed 42
  and the training file.

## 4. Model metrics: primary population, five seeds pooled

Values are point estimates with 95% bootstrap intervals.

| Metric | production | v2_dnn_lstm | v2_dnn_only |
|---|---|---|---|
| Recall (Policy B) | 0.296 [0.267, 0.328] | 0.453 [0.409, 0.494] | 0.420 [0.380, 0.460] |
| Legitimate alerts per 1,000 (Policy B) | 10.31 [10.00, 10.62] | 8.91 [8.29, 9.53] | 9.03 [8.47, 9.62] |
| Critical-tier recall | 0.160 [0.138, 0.183] | 0.146 [0.118, 0.176] | 0.138 [0.114, 0.166] |
| Precision (Policy B) | 0.124 [0.108, 0.139] | 0.201 [0.177, 0.225] | 0.187 [0.164, 0.210] |
| F1 (Policy B) | 0.175 [0.155, 0.192] | 0.278 [0.249, 0.306] | 0.258 [0.231, 0.286] |
| False-positive rate (Policy B) | 0.0104 [0.0101, 0.0107] | 0.0090 [0.0083, 0.0096] | 0.0091 [0.0085, 0.0097] |
| Critical-tier precision | 0.462 [0.414, 0.507] | 0.579 [0.497, 0.660] | 0.471 [0.403, 0.541] |
| Critical-tier legitimate alerts per 1,000 | 0.92 [0.83, 1.01] | 0.52 [0.40, 0.66] | 0.76 [0.63, 0.91] |
| First-fraud detection (Policy B) | 0.437 [0.389, 0.484] | 0.408 [0.366, 0.452] | 0.421 [0.376, 0.468] |
| Episodes with any fraud alerted (Policy B) | 0.607 [0.563, 0.650] | 0.545 [0.502, 0.588] | 0.553 [0.506, 0.599] |
| PR-AUC | 0.172 [0.147, 0.200] | 0.252 [0.209, 0.297] | 0.213 [0.177, 0.255] |
| ROC-AUC | 0.732 [0.706, 0.758] | 0.929 [0.916, 0.941] | 0.903 [0.892, 0.913] |
| Fraud transactions caught (of 2,396) | 709 | 1085 | 1006 |
| False positives (of 482,634 legitimate) | 5003 | 4321 | 4382 |
| Critical tier: fraud caught / false positives | 383 / 446 | 350 / 254 | 330 / 370 |
| First frauds detected (of 549), Wilson 95% | 240 [0.3962, 0.4789] | 224 [0.3677, 0.4496] | 231 [0.3802, 0.4625] |
| Episodes detected (of 550) | 334 | 300 | 304 |

**Paired differences.** Bold means the 95% interval excludes zero.

| Metric | v2_dnn_lstm − v2_dnn_only | v2_dnn_lstm − production | v2_dnn_only − production |
|---|---|---|---|
| Recall (Policy B) | **+0.033 [+0.014, +0.051]** | **+0.157 [+0.117, +0.197]** | **+0.124 [+0.085, +0.160]** |
| Legitimate alerts per 1,000 (Policy B) | -0.13 [-0.41, +0.14] | **-1.41 [-1.97, -0.81]** | **-1.28 [-1.80, -0.73]** |
| Critical-tier recall | +0.008 [-0.013, +0.029] | -0.014 [-0.048, +0.020] | -0.022 [-0.050, +0.008] |
| Precision (Policy B) | **+0.014 [+0.006, +0.022]** | **+0.077 [+0.058, +0.096]** | **+0.063 [+0.045, +0.079]** |
| F1 (Policy B) | **+0.020 [+0.009, +0.030]** | **+0.103 [+0.079, +0.128]** | **+0.084 [+0.061, +0.105]** |
| False-positive rate (Policy B) | -0.0001 [-0.0004, +0.0001] | **-0.0014 [-0.0020, -0.0008]** | **-0.0013 [-0.0018, -0.0007]** |
| Critical-tier precision | **+0.108 [+0.058, +0.162]** | **+0.117 [+0.035, +0.198]** | +0.009 [-0.062, +0.079] |
| Critical-tier legitimate alerts per 1,000 | **-0.24 [-0.33, -0.15]** | **-0.40 [-0.53, -0.25]** | **-0.16 [-0.29, -0.00]** |
| First-fraud detection (Policy B) | -0.013 [-0.044, +0.019] | -0.029 [-0.078, +0.019] | -0.016 [-0.067, +0.031] |
| Episodes with any fraud alerted (Policy B) | -0.007 [-0.045, +0.029] | **-0.062 [-0.110, -0.014]** | **-0.055 [-0.102, -0.007]** |
| PR-AUC | **+0.039 [+0.023, +0.055]** | **+0.080 [+0.036, +0.123]** | **+0.041 [+0.004, +0.077]** |
| ROC-AUC | **+0.026 [+0.018, +0.034]** | **+0.197 [+0.172, +0.222]** | **+0.171 [+0.147, +0.195]** |

**Reading**

* **Recall at Policy B.** Both candidates are above production: +0.157 and
  +0.124. `v2_dnn_lstm` is above `v2_dnn_only` by +0.033.
* **Legitimate alerts at Policy B.** Both candidates are below production by
  about 1.3–1.4 per 1,000. The two candidates are not separated.
* **Production is slightly over the budget at its own cut-off:** 10.31
  [10.00, 10.62] per 1,000.
* **Critical tier.** No model set is separated from another on Critical-tier
  recall. `v2_dnn_lstm` has the fewest Critical false positives.
* **First fraud and episodes.** This is the weak side of both candidates.
  Production detects the first fraud of 240 episodes; the candidates 224 and
  231 (not separated). Production has at least one alert in 334 of 550
  episodes; the candidates in 300 and 304. That difference is separated, in
  production's favour.
* **Ranking metrics.** ROC-AUC 0.93 and 0.90 against 0.73 for production.
  PR-AUC 0.25 and 0.21 against 0.17.

**Why transaction recall and episode detection disagree.** The candidates
catch many more transactions inside long episodes (card testing, account
takeover, stolen card). Production reaches more episodes but catches fewer
transactions in each. §6 shows this by fraud type.

## 5. Secondary population: 2026-06-01 onward

| Metric | production | v2_dnn_lstm | v2_dnn_only |
|---|---|---|---|
| Recall (Policy B) | 0.319 [0.252, 0.395] | 0.454 [0.359, 0.543] | 0.371 [0.287, 0.455] |
| Legitimate alerts per 1,000 (Policy B) | 9.47 [8.82, 10.12] | 8.02 [6.80, 9.29] | 8.06 [6.94, 9.29] |
| Critical-tier recall | 0.163 [0.118, 0.213] | 0.117 [0.061, 0.180] | 0.133 [0.077, 0.198] |
| Precision (Policy B) | 0.135 [0.100, 0.173] | 0.208 [0.154, 0.262] | 0.176 [0.127, 0.228] |
| F1 (Policy B) | 0.190 [0.147, 0.236] | 0.285 [0.220, 0.350] | 0.239 [0.178, 0.297] |
| False-positive rate (Policy B) | 0.0095 [0.0089, 0.0102] | 0.0081 [0.0068, 0.0093] | 0.0081 [0.0070, 0.0093] |
| Critical-tier precision | 0.551 [0.440, 0.653] | 0.513 [0.310, 0.707] | 0.449 [0.288, 0.617] |
| Critical-tier legitimate alerts per 1,000 | 0.62 [0.46, 0.79] | 0.51 [0.25, 0.83] | 0.76 [0.48, 1.07] |
| First-fraud detection (Policy B) | 0.463 [0.364, 0.563] | 0.380 [0.287, 0.474] | 0.372 [0.277, 0.467] |
| Episodes with any fraud alerted (Policy B) | 0.648 [0.553, 0.740] | 0.516 [0.420, 0.609] | 0.508 [0.406, 0.612] |
| PR-AUC | 0.204 [0.147, 0.268] | 0.235 [0.149, 0.331] | 0.191 [0.113, 0.278] |
| ROC-AUC | 0.755 [0.701, 0.807] | 0.948 [0.927, 0.965] | 0.922 [0.905, 0.937] |
| Fraud transactions caught (of 496) | 158 | 225 | 184 |
| False positives (of 106,452 legitimate) | 1013 | 858 | 862 |
| Critical tier: fraud caught / false positives | 81 / 66 | 58 / 55 | 66 / 81 |
| First frauds detected (of 121), Wilson 95% | 56 [0.3765, 0.5514] | 46 [0.2986, 0.4691] | 45 [0.291, 0.4607] |
| Episodes detected (of 122) | 79 | 63 | 62 |

| Metric | v2_dnn_lstm − v2_dnn_only | v2_dnn_lstm − production | v2_dnn_only − production |
|---|---|---|---|
| Recall (Policy B) | **+0.083 [+0.039, +0.132]** | **+0.135 [+0.031, +0.235]** | +0.052 [-0.049, +0.146] |
| Legitimate alerts per 1,000 (Policy B) | -0.04 [-0.57, +0.49] | **-1.45 [-2.60, -0.17]** | **-1.41 [-2.52, -0.23]** |
| Critical-tier recall | -0.016 [-0.056, +0.017] | -0.046 [-0.115, +0.023] | -0.030 [-0.101, +0.043] |
| Precision (Policy B) | **+0.032 [+0.013, +0.052]** | **+0.073 [+0.027, +0.119]** | +0.041 [-0.001, +0.085] |
| F1 (Policy B) | **+0.046 [+0.021, +0.075]** | **+0.095 [+0.034, +0.156]** | +0.049 [-0.010, +0.105] |
| False-positive rate (Policy B) | -0.0000 [-0.0006, +0.0005] | **-0.0015 [-0.0026, -0.0002]** | **-0.0014 [-0.0025, -0.0002]** |
| Critical-tier precision | +0.064 [-0.057, +0.180] | -0.038 [-0.220, +0.152] | -0.102 [-0.262, +0.056] |
| Critical-tier legitimate alerts per 1,000 | **-0.24 [-0.45, -0.04]** | -0.10 [-0.35, +0.20] | +0.14 [-0.13, +0.45] |
| First-fraud detection (Policy B) | +0.008 [-0.061, +0.079] | -0.083 [-0.206, +0.039] | -0.091 [-0.224, +0.033] |
| Episodes with any fraud alerted (Policy B) | +0.008 [-0.075, +0.089] | **-0.131 [-0.237, -0.017]** | **-0.139 [-0.268, -0.008]** |
| PR-AUC | **+0.043 [+0.010, +0.079]** | +0.031 [-0.056, +0.118] | -0.013 [-0.096, +0.072] |
| ROC-AUC | **+0.025 [+0.010, +0.041]** | **+0.193 [+0.139, +0.251]** | **+0.167 [+0.117, +0.222]** |

* The direction is the same as in the primary population, with wider
  intervals (122 episodes).
* `v2_dnn_only` is **not** separated from production on recall here.
  `v2_dnn_lstm` is.
* Both candidates again detect fewer episodes than production.

## 6. Per seed and per fraud type

**Per seed, primary population**

| Seed | Model set | Recall | Legit alerts / 1,000 | Critical recall | Precision | First-fraud detection | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|---|---|---|
| 101 | production | 0.309 [0.245, 0.382] | 10.20 [9.55, 10.93] | 0.177 [0.129, 0.235] | 0.133 | 0.418 | 0.196 | 0.717 |
| 101 | v2_dnn_lstm | 0.428 [0.331, 0.530] | 9.08 [7.72, 10.62] | 0.134 [0.077, 0.194] | 0.193 | 0.427 | 0.224 | 0.937 |
| 101 | v2_dnn_only | 0.395 [0.309, 0.483] | 9.05 [7.88, 10.42] | 0.105 [0.063, 0.151] | 0.181 | 0.436 | 0.179 | 0.897 |
| 102 | production | 0.301 [0.227, 0.371] | 10.56 [9.86, 11.30] | 0.166 [0.113, 0.216] | 0.121 | 0.394 | 0.176 | 0.736 |
| 102 | v2_dnn_lstm | 0.442 [0.339, 0.540] | 9.11 [7.64, 10.65] | 0.156 [0.087, 0.232] | 0.190 | 0.367 | 0.252 | 0.913 |
| 102 | v2_dnn_only | 0.423 [0.325, 0.518] | 9.31 [7.94, 10.76] | 0.131 [0.075, 0.197] | 0.180 | 0.367 | 0.214 | 0.906 |
| 103 | production | 0.309 [0.247, 0.378] | 10.12 [9.39, 10.89] | 0.176 [0.130, 0.233] | 0.130 | 0.455 | 0.185 | 0.735 |
| 103 | v2_dnn_lstm | 0.532 [0.440, 0.625] | 8.99 [7.86, 10.24] | 0.142 [0.086, 0.210] | 0.224 | 0.455 | 0.327 | 0.945 |
| 103 | v2_dnn_only | 0.479 [0.388, 0.570] | 8.79 [7.76, 9.86] | 0.142 [0.090, 0.204] | 0.210 | 0.436 | 0.249 | 0.919 |
| 104 | production | 0.261 [0.205, 0.323] | 10.37 [9.68, 11.13] | 0.138 [0.098, 0.183] | 0.114 | 0.455 | 0.146 | 0.716 |
| 104 | v2_dnn_lstm | 0.419 [0.331, 0.508] | 8.32 [7.05, 9.69] | 0.119 [0.065, 0.180] | 0.205 | 0.373 | 0.203 | 0.913 |
| 104 | v2_dnn_only | 0.389 [0.306, 0.474] | 8.96 [7.81, 10.22] | 0.123 [0.079, 0.178] | 0.181 | 0.436 | 0.174 | 0.886 |
| 105 | production | 0.301 [0.230, 0.373] | 10.31 [9.59, 11.03] | 0.143 [0.094, 0.198] | 0.123 | 0.464 | 0.162 | 0.756 |
| 105 | v2_dnn_lstm | 0.447 [0.348, 0.541] | 9.04 [7.78, 10.35] | 0.182 [0.104, 0.264] | 0.192 | 0.418 | 0.277 | 0.936 |
| 105 | v2_dnn_only | 0.417 [0.322, 0.515] | 9.05 [7.89, 10.35] | 0.190 [0.122, 0.267] | 0.181 | 0.427 | 0.262 | 0.908 |

* Recall at Policy B is higher for both candidates than for production in
  all five seeds, and higher for `v2_dnn_lstm` than for `v2_dnn_only` in all
  five.
* Range across seeds, recall: production 0.26–0.31; `v2_dnn_lstm` 0.42–0.53;
  `v2_dnn_only` 0.39–0.48.
* Range across seeds, legitimate alerts per 1,000: production 10.1–10.6;
  `v2_dnn_lstm` 8.3–9.1; `v2_dnn_only` 8.8–9.3.

**Per fraud type, primary population, Policy B.** Transaction recall, with
episodes detected in brackets. Descriptive; no intervals.

| Fraud type | Episodes | Fraud transactions | production | v2_dnn_lstm | v2_dnn_only |
|---|---|---|---|---|---|
| account_takeover | 110 | 570 | 0.43 (92 ep.) | 0.55 (84 ep.) | 0.53 (89 ep.) |
| card_testing_cashout | 85 | 605 | 0.13 (36 ep.) | 0.58 (68 ep.) | 0.51 (69 ep.) |
| device_takeover | 55 | 125 | 0.43 (32 ep.) | 0.16 (12 ep.) | 0.27 (24 ep.) |
| high_value_single | 55 | 78 | 0.74 (44 ep.) | 0.46 (28 ep.) | 0.54 (34 ep.) |
| normal_looking | 55 | 137 | 0.01 (2 ep.) | 0.01 (1 ep.) | 0.00 (0 ep.) |
| ring | 80 | 176 | 0.69 (63 ep.) | 0.69 (60 ep.) | 0.55 (48 ep.) |
| stolen_card_online | 110 | 705 | 0.21 (65 ep.) | 0.34 (47 ep.) | 0.31 (40 ep.) |

* The candidates gain most on `card_testing_cashout` (0.58 and 0.51 against
  0.13).
* The candidates are **worse** than production on `device_takeover` (0.16 and
  0.27 against 0.43) and on `high_value_single` (0.46 and 0.54 against 0.74).
* No model set detects `normal_looking` fraud.

## 7. Ring fraud

20 rings, 80 victim episodes, 176 ring fraud transactions, all in the primary
population. Counts carry Wilson 95% intervals.

| Level | Measure (Policy B) | production | v2_dnn_lstm | v2_dnn_only |
|---|---|---|---|---|
| Ring | Rings with at least one victim alerted | 20/20 [0.8389, 1.0] | 20/20 [0.8389, 1.0] | 18/20 [0.699, 0.9721] |
| Ring | Rings with at least two victims alerted | 17/20 [0.6396, 0.9476] | 18/20 [0.699, 0.9721] | 15/20 [0.5313, 0.8881] |
| Ring | Median hours from the ring's first fraud to the first alert | 0.0 | 0.33 | 0.0 |
| Victim | Victim episodes detected | 63/80 [0.6858, 0.8629] | 60/80 [0.6452, 0.8319] | 48/80 [0.4905, 0.7004] |
| Victim | Victims' first fraud detected | 56/80 [0.5923, 0.7894] | 43/80 [0.429, 0.6425] | 42/80 [0.417, 0.6308] |
| Transaction | Ring fraud transactions caught | 122/176 [0.6215, 0.7566] | 121/176 [0.6156, 0.7514] | 96/176 [0.4717, 0.6173] |
| Transaction | Non-ring fraud transactions caught | 587/2220 [0.2465, 0.2832] | 964/2220 [0.4137, 0.4549] | 910/2220 [0.3896, 0.4305] |

| Ring-level paired difference (Policy B) | Victim episodes detected | Ring fraud transactions caught |
|---|---|---|
| v2_dnn_lstm − v2_dnn_only | **+0.150 [+0.024, +0.284]** | **+0.142 [+0.062, +0.217]** |
| v2_dnn_lstm − production | -0.037 [-0.169, +0.111] | -0.006 [-0.126, +0.103] |
| v2_dnn_only − production | **-0.188 [-0.338, -0.051]** | **-0.148 [-0.286, -0.025]** |

| Level | Measure (Critical) | production | v2_dnn_lstm | v2_dnn_only |
|---|---|---|---|---|
| Ring | Rings with at least one victim alerted | 18/20 [0.699, 0.9721] | 9/20 [0.2582, 0.6579] | 10/20 [0.2993, 0.7007] |
| Ring | Rings with at least two victims alerted | 14/20 [0.481, 0.8545] | 5/20 [0.1119, 0.4687] | 8/20 [0.2188, 0.6134] |
| Ring | Median hours from the ring's first fraud to the first alert | 0.99 | 4.4 | 2.73 |
| Victim | Victim episodes detected | 43/80 [0.429, 0.6425] | 18/80 [0.1473, 0.3279] | 23/80 [0.1999, 0.3946] |
| Victim | Victims' first fraud detected | 34/80 [0.3226, 0.5343] | 7/80 [0.043, 0.1698] | 16/80 [0.127, 0.3005] |
| Transaction | Ring fraud transactions caught | 77/176 [0.3663, 0.5114] | 22/176 [0.084, 0.182] | 31/176 [0.127, 0.2392] |
| Transaction | Non-ring fraud transactions caught | 306/2220 [0.1241, 0.1528] | 328/2220 [0.1336, 0.1631] | 299/2220 [0.1211, 0.1495] |

| Ring-level paired difference (Critical) | Victim episodes detected | Ring fraud transactions caught |
|---|---|---|
| v2_dnn_lstm − v2_dnn_only | -0.062 [-0.145, +0.013] | **-0.051 [-0.096, -0.012]** |
| v2_dnn_lstm − production | **-0.312 [-0.403, -0.210]** | **-0.312 [-0.412, -0.212]** |
| v2_dnn_only − production | **-0.250 [-0.346, -0.149]** | **-0.261 [-0.361, -0.155]** |

* **Policy B.** Every model set alerts on nearly every ring. `v2_dnn_lstm`
  and production are not separated. `v2_dnn_only` detects fewer ring victims
  than either; both paired intervals exclude zero.
* **Critical cut-off.** Production detects ring fraud far more often than
  either candidate (77 of 176 transactions against 22 and 31). The intervals
  exclude zero.
* **No claim of ring superiority for any candidate over production is
  supported.** Between the candidates, `v2_dnn_lstm` is better at Policy B and
  `v2_dnn_only` is slightly better at the Critical cut-off.
* 20 rings is a small sample. The ring-level intervals are wide.

**Shared-device rule** (the same for every model set; full history of each
dataset)

| Seed | Devices flagged | Ring devices | Other fraud devices | Legitimate-only devices | Rings found (of 4) |
|---|---|---|---|---|---|
| 101 | 52 | 5 | 0 | 47 | 4 |
| 102 | 53 | 5 | 1 | 47 | 4 |
| 103 | 56 | 5 | 1 | 50 | 3 |
| 104 | 42 | 3 | 1 | 38 | 3 |
| 105 | 53 | 4 | 1 | 48 | 4 |

* The rule finds 18 of the 20 rings. About 9 in 10 flagged devices are
  legitimate shared devices (households, borrowed phones).

## 8. Pre-registered decision rule

Applied to the primary population, five seeds pooled.

**Between the candidates** (`v2_dnn_lstm − v2_dnn_only`)

| Condition | Test | Interval | Holds |
|---|---|---|---|
| Recall at least matches | Lower bound above −0.05 | [+0.014, +0.051] | Yes |
| Alert burden at least matches | Upper bound below +1.0 per 1,000 | [−0.41, +0.14] | Yes |
| Critical tier not worse | Lower bound above −0.05 | [−0.013, +0.029] | Yes |

* All three hold. **`v2_dnn_lstm` remains the provisional lead.**
  `v2_dnn_only` remains the required challenger.
* One primary endpoint is separated between the candidates: recall, +0.033
  [+0.014, +0.051] in favour of `v2_dnn_lstm`. Under the claims rule this is
  the only place where "better" may be used between the candidates. The
  difference is small.

**Against production**

| Check | v2_dnn_lstm | v2_dnn_only |
|---|---|---|
| Legitimate alerts per 1,000, upper bound at or below 10 | 8.91 [8.29, 9.53]: yes | 9.03 [8.47, 9.62]: yes |
| Recall at least 0.40 | 0.453: yes | 0.420: yes |
| Recall lower bound above production's 0.296 | 0.409: yes | 0.380: yes |
| Eligible on this evidence | Yes | Yes |

**Primary-endpoint intervals that exclude zero**

| Pair | Endpoint | Difference |
|---|---|---|
| v2_dnn_lstm − v2_dnn_only | Recall | +0.033 [+0.014, +0.051] |
| v2_dnn_lstm − production | Recall | +0.157 [+0.117, +0.197] |
| v2_dnn_lstm − production | Legitimate alerts per 1,000 | −1.41 [−1.97, −0.81] |
| v2_dnn_only − production | Recall | +0.124 [+0.085, +0.160] |
| v2_dnn_only − production | Legitimate alerts per 1,000 | −1.28 [−1.80, −0.73] |

* There are nine primary comparisons. About one in twenty such intervals
  excludes zero by chance. The four against production are large and repeat
  in every seed. The candidate-against-candidate one is small.
* **Eligible does not mean approved.** The rule's endpoints are
  transaction-level. §4 shows the candidates are not better at episode level
  or in the Critical tier. The owner should weigh that before any promotion
  decision.

## 9. New-customer fraud

### Generator extension (version 2.1.0, off by default)

* Two settings, both 0 by default: `late_joiner_share` and
  `new_customer_fraud_episodes` (CLI: `--late-joiner-share`,
  `--new-customer-fraud-episodes`).
* A late joiner is a customer with no other planned fraud whose activity
  starts on a later day (day 14 to 140). A new-customer episode starts after
  0 to 9 of a late joiner's own transactions. The existing non-ring fraud
  types are reused.
* **Default output is byte-identical to version 2.0.1.** With both settings
  at 0 the generator makes the same draws, writes the same files, and the
  manifest still says 2.0.1. Tests compare against checksums recorded before
  the change, and the tracked seed-42 files regenerate identically.
* With the extension on, every other customer and every existing episode is
  exactly as without it (tested).
* The count of earlier transactions at first fraud is recorded in
  `episodes.csv` only. No column was added to the transaction or feature
  files, so nothing new can reach a model.

### Datasets (seeds 201–205, 20% late joiners, 30 new-customer episodes each)

| Seed | Transactions | Fraud episodes | New-customer fraud episodes | Late joiners | Fraud transactions with fewer than 10 earlier | SHA-256 (first 12) |
|---|---|---|---|---|---|---|
| 201 | 94,710 | 140 | 30 | 100 | 86 | `fb8a6412d2af` |
| 202 | 92,474 | 140 | 30 | 100 | 81 | `61da8ecf49a3` |
| 203 | 92,265 | 140 | 30 | 100 | 94 | `665dbe27fe5b` |
| 204 | 94,532 | 140 | 31 | 100 | 100 | `d403d01f5cfd` |
| 205 | 94,431 | 140 | 30 | 100 | 78 | `82554a8848fb` |
{'rows': 25000, 'fraud_transactions': 439, 'legitimate_transactions': 24561, 'fraud_episodes': 151, 'first_fraud_transactions': 151, 'groups': 2220, 'datasets': 5}

* 151 new-customer fraud episodes in total (required: 100). One extra arises
  in seed 204 by chance.
* Population evaluated: 25,000 transactions with fewer than 10 earlier ones;
  24,561 legitimate, 439 fraud, 151 first frauds.

### Alert rules compared

Recall and first-fraud detection are within this population. The
false-positive rate is legitimate alerts divided by legitimate transactions.

| Alert rule | Model set | Fraud caught (of 439) | First frauds caught (of 151) | Recall | First-fraud detection | Legitimate alerts | False-positive rate | Legit alerts / 1,000 | Precision |
|---|---|---|---|---|---|---|---|---|---|
| Current behaviour (Policy B cut-off) | production | 125 | 72 | 0.285 [0.240, 0.333] | 0.477 [0.395, 0.560] | 1,906 | 0.0776 [0.0745, 0.0808] | 76.24 [73.17, 79.39] | 0.062 |
| Current behaviour (Policy B cut-off) | v2_dnn_lstm | 184 | 68 | 0.419 [0.335, 0.501] | 0.450 [0.372, 0.530] | 415 | 0.0169 [0.0148, 0.0192] | 16.60 [14.53, 18.82] | 0.307 |
| Current behaviour (Policy B cut-off) | v2_dnn_only | 195 | 73 | 0.444 [0.366, 0.524] | 0.483 [0.401, 0.561] | 570 | 0.0232 [0.0208, 0.0257] | 22.80 [20.38, 25.25] | 0.255 |
| Proposed policy (Critical cut-off only) | production | 51 | 45 | 0.116 [0.088, 0.147] | 0.298 [0.222, 0.366] | 509 | 0.0207 [0.0190, 0.0224] | 20.36 [18.66, 22.02] | 0.091 |
| Proposed policy (Critical cut-off only) | v2_dnn_lstm | 12 | 5 | 0.027 [0.005, 0.063] | 0.033 [0.007, 0.066] | 0 | 0.0000 [0.0000, 0.0000] | 0.00 [0.00, 0.00] | 1.000 |
| Proposed policy (Critical cut-off only) | v2_dnn_only | 67 | 34 | 0.153 [0.104, 0.206] | 0.225 [0.161, 0.291] | 28 | 0.0011 [0.0006, 0.0018] | 1.12 [0.56, 1.79] | 0.705 |
| Measurement: Critical cut-off, flags neutral | production | 7 | 7 | 0.016 [0.005, 0.029] | 0.046 [0.014, 0.081] | 0 | 0.0000 [0.0000, 0.0000] | 0.00 [0.00, 0.00] | 1.000 |
| Measurement: Critical cut-off, flags neutral | v2_dnn_lstm | 9 | 5 | 0.021 [0.003, 0.050] | 0.033 [0.007, 0.066] | 0 | 0.0000 [0.0000, 0.0000] | 0.00 [0.00, 0.00] | 1.000 |
| Measurement: Critical cut-off, flags neutral | v2_dnn_only | 57 | 32 | 0.130 [0.086, 0.178] | 0.212 [0.147, 0.274] | 28 | 0.0011 [0.0005, 0.0019] | 1.12 [0.52, 1.85] | 0.671 |
| Measurement: Policy B cut-off, flags neutral | production | 14 | 14 | 0.032 [0.017, 0.049] | 0.093 [0.050, 0.140] | 2 | 0.0001 [0.0000, 0.0002] | 0.08 [0.00, 0.20] | 0.875 |
| Measurement: Policy B cut-off, flags neutral | v2_dnn_lstm | 164 | 59 | 0.374 [0.290, 0.452] | 0.391 [0.315, 0.464] | 139 | 0.0057 [0.0042, 0.0073] | 5.56 [4.17, 7.13] | 0.541 |
| Measurement: Policy B cut-off, flags neutral | v2_dnn_only | 177 | 66 | 0.403 [0.318, 0.489] | 0.437 [0.358, 0.517] | 166 | 0.0068 [0.0053, 0.0084] | 6.64 [5.19, 8.26] | 0.516 |

**Effect of changing the rule, per model set** (bold: interval excludes zero)

| Change of rule | Model set | Recall | First-fraud detection | Legit alerts / 1,000 |
|---|---|---|---|---|
| critical_only − current_policy_b | production | **-0.169 [-0.208, -0.132]** | **-0.179 [-0.244, -0.117]** | **-55.88 [-58.55, -53.14]** |
| critical_only − current_policy_b | v2_dnn_lstm | **-0.392 [-0.469, -0.312]** | **-0.417 [-0.494, -0.342]** | **-16.60 [-18.82, -14.53]** |
| critical_only − current_policy_b | v2_dnn_only | **-0.292 [-0.360, -0.223]** | **-0.258 [-0.327, -0.190]** | **-21.68 [-23.93, -19.51]** |
| critical_only_flags_neutral − critical_only | production | **-0.100 [-0.130, -0.074]** | **-0.252 [-0.318, -0.183]** | **-20.36 [-22.02, -18.66]** |
| critical_only_flags_neutral − critical_only | v2_dnn_lstm | -0.007 [-0.015, +0.000] | +0.000 [+0.000, +0.000] | +0.00 [+0.00, +0.00] |
| critical_only_flags_neutral − critical_only | v2_dnn_only | **-0.023 [-0.044, -0.005]** | -0.013 [-0.048, +0.018] | +0.00 [-0.28, +0.32] |
| policy_b_flags_neutral − current_policy_b | production | **-0.253 [-0.300, -0.209]** | **-0.384 [-0.467, -0.305]** | **-76.16 [-79.33, -73.10]** |
| policy_b_flags_neutral − current_policy_b | v2_dnn_lstm | **-0.046 [-0.079, -0.013]** | **-0.060 [-0.113, -0.013]** | **-11.04 [-12.59, -9.59]** |
| policy_b_flags_neutral − current_policy_b | v2_dnn_only | **-0.041 [-0.073, -0.010]** | **-0.046 [-0.090, -0.006]** | **-16.16 [-17.96, -14.53]** |

**By number of earlier transactions, current behaviour (Policy B)**

| Earlier transactions | Legitimate | Fraud | First frauds | production: legit alerts (per 1,000) / fraud caught / first frauds | v2_dnn_lstm | v2_dnn_only |
|---|---|---|---|---|---|---|
| 0 | 2,482 | 18 | 18 | 0 (0.0) / 0 / 0 | 1 (0.4) / 4 / 4 | 1 (0.4) / 4 / 4 |
| 1 | 2,481 | 19 | 14 | 655 (264.01) / 11 / 10 | 103 (41.52) / 12 / 10 | 131 (52.8) / 11 / 9 |
| 2 | 2,474 | 26 | 9 | 412 (166.53) / 13 / 6 | 60 (24.25) / 15 / 7 | 73 (29.51) / 15 / 6 |
| 3-9 | 17,124 | 376 | 110 | 839 (49.0) / 101 / 56 | 251 (14.66) / 153 / 47 | 365 (21.32) / 165 / 54 |
| all | 24,561 | 439 | 151 | 1906 (77.6) / 125 / 72 | 415 (16.9) / 184 / 68 | 570 (23.21) / 195 / 73 |

**By number of earlier transactions, proposed policy (Critical only)**

| Earlier transactions | Legitimate | Fraud | First frauds | production: legit alerts (per 1,000) / fraud caught / first frauds | v2_dnn_lstm | v2_dnn_only |
|---|---|---|---|---|---|---|
| 0 | 2,482 | 18 | 18 | 0 (0.0) / 0 / 0 | 0 (0.0) / 0 / 0 | 0 (0.0) / 1 / 1 |
| 1 | 2,481 | 19 | 14 | 242 (97.54) / 9 / 8 | 0 (0.0) / 0 / 0 | 5 (2.02) / 5 / 5 |
| 2 | 2,474 | 26 | 9 | 109 (44.06) / 5 / 3 | 0 (0.0) / 1 / 1 | 4 (1.62) / 4 / 2 |
| 3-9 | 17,124 | 376 | 110 | 158 (9.23) / 37 / 34 | 0 (0.0) / 11 / 4 | 19 (1.11) / 57 / 26 |
| all | 24,561 | 439 | 151 | 509 (20.72) / 51 / 45 | 0 (0.0) / 12 / 5 | 28 (1.14) / 67 / 34 |

**Reading**

* **Current behaviour.** The candidates catch 42% and 44% of early fraud but
  flag 16.6 and 22.8 legitimate early transactions per 1,000, above the
  budget of 10. Production flags 76 per 1,000.
* **Proposed Critical-only policy.** It brings legitimate alerts to 0 for
  `v2_dnn_lstm` and 1.1 per 1,000 for `v2_dnn_only`. The cost is large:
  * `v2_dnn_lstm` catches 12 of 439 early fraud transactions and 5 of 151
    first frauds. Without an LSTM score its score rarely reaches its Critical
    cut-off. For this model set the policy is close to "no alerts for new
    customers".
  * `v2_dnn_only` keeps 67 of 439 and 34 of 151.
  * Production keeps 51 of 439 but still flags 20 legitimate per 1,000.
* **Measurement: Policy B with the two flags neutral.** This was not the
  proposed policy; it is reported because it was measured. For the
  candidates it gives 5.6 and 6.6 legitimate alerts per 1,000 (inside the
  budget) while keeping recall at 0.37 and 0.40. For production it removes
  nearly all alerts.
* **The first transaction (0 earlier).** 18 frauds. Production catches none;
  each candidate catches 4.

**Limit.** These fraud patterns are the existing fraud types placed in a new
customer's first transactions by the generator. The result shows how each
rule behaves on those patterns. It does not show that any rule works on real
new-customer fraud.

**No cold-start behaviour was changed.** The new-customer policy is still
the owner's decision.

## 10. Supplementary checks

### The alert budget on five further seeds

The customers that the extension does not touch in seeds 201–205 are exactly
the default generator's customers for those seeds. They are therefore five
more data seeds. This check was not pre-registered and is descriptive.
Rows with at least 10 earlier transactions, Policy B, point values:

| Seed | Rows | Fraud transactions | Recall: production / v2_dnn_lstm / v2_dnn_only | Legit alerts per 1,000: production / v2_dnn_lstm / v2_dnn_only |
|---|---|---|---|---|
| 201 | 79,113 | 488 | 0.287 / 0.387 / 0.363 | 10.61 / 11.59 / 11.17 |
| 202 | 76,828 | 469 | 0.281 / 0.484 / 0.456 | 11.55 / 11.69 / 11.39 |
| 203 | 76,632 | 465 | 0.271 / 0.495 / 0.434 | 10.94 / 9.72 / 9.92 |
| 204 | 79,173 | 486 | 0.292 / 0.432 / 0.411 | 10.72 / 9.25 / 8.79 |
| 205 | 79,081 | 481 | 0.212 / 0.403 / 0.345 | 10.14 / 9.86 / 9.89 |
| 201–205 pooled | 390,827 | 2389 | 0.269 / 0.440 / 0.401 | 10.78 / 10.42 / 10.23 |
| Late joiners, pooled | 52,585 | 283 | 0.159 / 0.463 / 0.428 | 12.09 / 13.90 / 13.16 |

* **Recall:** the candidates are above production in all five seeds, and
  `v2_dnn_lstm` is above `v2_dnn_only` in all five. This agrees with §4.
* **Legitimate alerts:** the candidates are above 10 per 1,000 in seeds 201
  and 202 and at 10.4 and 10.2 pooled. On seeds 101–105 they were 8.3–9.3.
* **Consequence.** The bootstrap intervals in §4 resample customers inside
  the five datasets. They do not include variation between datasets. Across
  all ten seeds the candidates' rate runs from 8.3 to 11.7. **The budget is
  met on the pre-registered data but is not robustly met.** A cut-off
  measured on one validation period does not hold the budget to within 1 per
  1,000 on new data.
* **Recent customers.** Late joiners with at least 10 earlier transactions
  are flagged at 13.9 and 13.2 per 1,000 by the candidates. A population
  with many recent customers would run over the budget.

### Latency

Live pipeline, one transaction at a time, each model set in its own process,
200 scorings after 20 warm-up. Measured on the machine that ran this
evaluation; the figures are not reproducible bit for bit.

| Model set | Median | 95th percentile |
|---|---|---|
| production | 317 ms | 455 ms |
| v2_dnn_lstm | 329 ms | 448 ms |
| v2_dnn_only | 263 ms | 369 ms |

* The time includes feature recomputation and the explanation.
* `v2_dnn_only` is about 55–65 ms faster (no LSTM). `v2_dnn_lstm` and
  production are the same within noise.

## 11. Gate status

Mapped to the checklist in `docs/step4c3d-promotion-readiness.md`.

| Item | Before | Now | Evidence |
|---|---|---|---|
| At least 100 fraud episodes | BLOCKED | **Evidence met** | 550 primary, 122 secondary |
| At least 5 seeds | BLOCKED | **Met for data seeds; not for training seeds** | 5 hold-out datasets (plus 5 supplementary). Each candidate is still one training run. |
| Ring fraud represented | BLOCKED | **Evidence met** | 20 rings, three levels reported |
| New-customer fraud represented | BLOCKED | **Evidence met** | 151 episodes; synthetic patterns only |
| Fixed evaluation policy | OPEN | **Met for this comparison** | Rule, cut-offs and margins fixed before the run |
| Candidate-versus-candidate uncertainty analysis | BLOCKED | **Done** | §4, §8. One endpoint separated, by a small margin |
| Production-versus-candidate analysis | BLOCKED | **Done** | §4, §8. Both candidates eligible under the rule; episode-level caveat |
| Threshold policy formally approved | OPEN | OPEN | Owner decision. §10 shows the budget is not robust |
| Thresholds recorded per model set | OPEN | OPEN | Depends on the approval above |
| New-customer policy approved | OPEN | OPEN | Evidence now exists (§9). The proposed policy has a large detection cost |
| Live data version selected | OPEN | OPEN | Not addressed by this step |
| Rollback triggers written | OPEN | OPEN | Not addressed by this step |
| Final regression at the promotion commit | BLOCKED | BLOCKED | No promotion commit exists |

The four READY items are unchanged. Marking a checklist item READY is the
owner's decision; this table reports the evidence only.

## 12. Is multi-seed candidate retraining still necessary?

**Yes, for two of its three purposes.**

| Purpose | Still needed? | Why |
|---|---|---|
| Show that the saved candidates beat production on fraud caught at Policy B | No | The difference is large (+0.12 to +0.16 recall), holds in all ten data seeds, and the saved files are what would be deployed. |
| Choose between `v2_dnn_lstm` and `v2_dnn_only` | **Yes** | The separation is +0.033 recall. Each candidate is a single training run with seed 42. A difference this small can come from the training run and not from the architecture. Only retraining with several seeds can tell. |
| Close the "5 seeds" gate for training, and test whether the alert budget holds | **Yes** | Cut-offs are measured per trained model. §10 shows the budget already drifts between data seeds. How much it also drifts between training runs is unknown. |

* It would train candidates only, into a separate directory. Production and
  the existing candidate files would stay untouched.
* It is **not approved** and was **not started**.

## 13. Limits

* All data is synthetic and comes from one generator. The hold-out differs
  from the training data in its random seed, not in its design.
* The cut-offs come from one validation period of one dataset.
* The candidates were trained on v2-like data; production was trained on v1.
  The hold-out is v2-like, which favours the candidates.
* The bootstrap intervals do not include variation between datasets or
  between training runs.
* The live application still scores v1 customers. Nothing here measures the
  candidates on v1.
* The JSON reports were produced in the cloud workspace (Linux). On another
  machine the scores can differ in the last decimal places; re-running the
  commands in §14 regenerates them.

## 14. State and reproduction

| Item | State |
|---|---|
| `MODEL_SET` default | `production` |
| Promotion | none |
| Models trained | none |
| `backend/models/saved/` | unchanged (7 checksums verified) |
| `backend/models/candidates/` | unchanged (15 checksums verified) |
| Alert bands 25 / 50 / 80 | unchanged |
| Cold-start behaviour | unchanged |
| Default generator output | byte-identical to 2.0.1 |
| HEAD | `582fe60`; nothing committed or pushed |

**Files added**

* `backend/app/evaluation/holdout.py`
* `backend/app/evaluation/ring_metrics.py`
* `backend/tests/test_holdout.py` (32 tests)
* `backend/tests/test_new_customer_generator.py` (12 tests)
* `backend/models/evaluation/v2_holdout/holdout_report.json`
* `backend/models/evaluation/v2_holdout/new_customer_report.json`
* `docs/step4c3e-evidence-report.md`

**Files changed**

* `data/v2/synth_v2/config.py`, `generator.py`, `output.py`, `schema.py`,
  `data/v2/generate.py`: the off-by-default extension
* `data/v2/README.md`: the extension described
* `.gitignore`: `data/v2_holdout/` ignored

**Tests:** the 44 new tests pass; the full backend suite passes (470 tests) in the cloud workspace (Python 3.13, TensorFlow 2.21.0, Keras 3.15.1). They have not been run on your machine yet.

**To reproduce** (from `backend`, with the project's virtual environment)

```
python -m app.evaluation.holdout --generate
python -m app.evaluation.holdout new-customer --generate
python -m app.evaluation.holdout latency
```

* The first two generate the datasets if they are missing (about 100 seconds
  per dataset) and write the two JSON reports. Each evaluation takes about 5
  minutes.
* A second run on the same machine gives a byte-identical report (verified).
* The datasets under `data/v2_holdout/` are not on your machine yet; they are
  reproducible from the seeds and are regenerated by these commands.

**Not started:** multi-seed retraining, Step 4C-3F, controlled promotion.
