# Step 4C-2e-c: candidate comparison (evidence only)

This step compares Candidate A (DNN + LSTM) and Candidate B (DNN only) on the saved v2 split. **It does not choose a model or recommend a production change.**

- **How to reproduce:** `cd backend && python -m app.evaluation.compare_candidates` writes `models/candidates/v2/comparison.json`. Every number below comes from that file.
- **Scores are not calibrated probabilities:** both models were trained with class weights of about 1:201. Thresholds come from the validation split only.

## 1. What was compared

| | |
|---|---|
| Dataset | v2 `transactions_with_features.csv`, SHA-256 `a09d0611…2b1ef08` |
| Split | saved `models/evaluation/v2/split_time.json`, SHA-256 `701301ae…90efcc` (train < 2026-05-03, validation < 2026-06-01, test after) |
| Features | the 9 production features, list hash `78f61bd7…052ed3e` |
| Candidate A | `models/candidates/v2/dnn_lstm/`; manifest `c9d033ae…`; DNN weights `4f1dd953…`; LSTM weights `d204d4cb…`; DNN inputs = 9 features + `risk_score` |
| Candidate B | `models/candidates/v2/dnn_only/`; manifest `e3e8f1d2…`; DNN weights `71516e88…`; DNN inputs = 9 features |
| Rows scored | validation 16,315 (119 fraud); **test 21,142 (61 fraud, 21,081 legitimate, 14 episodes)** |
| Thresholds | from each manifest (validation, F1-optimal): A **0.959282**, B **0.769814**. They were re-derived from the validation scores during the comparison and matched. |

**How the candidates were scored:**

- **Preprocessing:** the same as at training time. DNN inputs are scaled with the training mean and standard deviation, then clipped to ±6 (training used unclipped inputs, as in 4C-2d). For A, `risk_score` comes from its final LSTM.
- **Rows:** training rows were not scored.
- **Check against 4C-2d:** the scores are **identical** to the 4C-2d evaluation scores (maximum difference 0.0 for both candidates). The comparison therefore re-measures the evaluated models from their saved candidate files; it adds no new training evidence.

## 2. Overall test metrics (validation F1 thresholds)

| | A (DNN + LSTM) | B (DNN only) |
|---|---|---|
| PR-AUC | 0.1045 | 0.1283 |
| ROC-AUC | 0.8993 | 0.8965 |
| Precision | 0.184 | 0.120 |
| Recall | 0.148 | 0.393 |
| F1 | 0.164 | 0.184 |
| TN / FP / FN / TP | 21,041 / 40 / 52 / 9 | 20,905 / 176 / 37 / 24 |
| False-positive rate | 0.19% | 0.83% |
| False negatives | 52 | 37 |
| Alerts per 1,000 transactions | 2.32 | 9.46 |

## 3. Threshold and operating-point analysis

Each operating point's threshold was chosen on validation, targeting a false-positive rate of about 0.1%, 1% or 5% there. The table shows what happens on the test period.

| Operating point | Candidate | Threshold | Validation FPR | Test precision | Test recall | Test FPR | Test alerts / 1,000 | TP / FP |
|---|---|---|---|---|---|---|---|---|
| F1-optimal | A | 0.9593 | 0.29% | 0.184 | 0.148 | 0.19% | 2.32 | 9 / 40 |
| F1-optimal | B | 0.7698 | 1.06% | 0.120 | 0.393 | 0.83% | 9.46 | 24 / 176 |
| ~0.1% FPR | A | 0.9679 | 0.105% | 0.200 | 0.066 | 0.08% | 0.95 | 4 / 16 |
| ~0.1% FPR | B | 0.9119 | 0.105% | 0.400 | 0.164 | 0.07% | 1.18 | 10 / 15 |
| ~1% FPR | A | 0.7677 | 1.00% | 0.147 | 0.410 | 0.69% | 8.04 | 25 / 145 |
| ~1% FPR | B | 0.7744 | 1.00% | 0.122 | 0.393 | 0.82% | 9.27 | 24 / 172 |
| ~5% FPR | A | 0.5894 | 5.00% | 0.038 | 0.639 | 4.68% | 48.53 | 39 / 987 |
| ~5% FPR | B | 0.5765 | 5.00% | 0.035 | 0.639 | 5.06% | 52.27 | 39 / 1,066 |

**Reading:**

- **Alert volume drives the gap:** the large recall difference at the F1 thresholds comes mostly from A's validation threshold landing much higher (fewer alerts). At matched validation false-positive rates the two are close:
  - at about 1% and 5%, they catch the same or nearly the same fraud (25 vs 24, and 39 vs 39);
  - at about 0.1%, B catches more (10 vs 4).
- **Small counts:** all of these rest on 4–39 test fraud transactions.
- **Known issue:** the ~0.1% points reach 0.105% on validation (one extra false positive). `metrics.threshold_for_fpr` can admit a tied boundary value when scores are float32. The realised rates are reported as measured; the helper was not changed in this step.

## 4. Episode-level behaviour (test period, 14 episodes)

| | A | B |
|---|---|---|
| Episodes detected (any fraud transaction alerted) | 3 | 7 |
| First fraud detected | 1 (1/14 = 0.071, Wilson 95% 0.013–0.315) | 5 (5/14 = 0.357, 0.163–0.612) |
| Detected only after the first fraud | 2 | 2 |
| Never detected | 11 | 7 |
| Median delay, detected episodes | 3 fraud transactions / 1.4 h | 0 / 0.0 h |
| Median delay, episodes detected after onset | 16.9 h | 1.1 h |

**Per episode:**

- **Both** detect episodes 99 (small test charges → cash-out, both after onset), 107 (account takeover, both at the first fraud) and 108 (stolen card).
  - A detects episode 108 after 3 fraud transactions (32.5 h).
  - B detects it at the first fraud.
- **Only B** detects episodes 98 (device takeover), 103 (one large hit), 104 (small test charges → cash-out, after onset) and 105 (account takeover).
- **Only A:** there is no episode that only A detects.

## 5. Warning-period behaviour

There are 5 episodes with a legitimate warning period whose first fraud is out of sample: 4 in validation (74, 76, 84, 90) and 1 in test (105). The validation ones share the rows the thresholds were chosen on.

- **These are legitimate transactions:** an alert on a warning-period row is a false positive, not a fraud detection.
- **Warning alerts are separate from first-fraud detection:** the two are counted separately below.

| | A | B |
|---|---|---|
| Warning-period rows (9) at or above threshold | 0 (0%) | 2 (22%) |
| Episodes with an alert during the warning period | 0 of 5 | 2 of 5 (74, 90) |
| Of those, first fraud also alerted | — | 1 (74); in episode 90 the first fraud was missed |
| Episodes whose first fraud was alerted | 1 of 5 (74) | 3 of 5 (74, 76, 105) |
| Median score: 30 days before the warning period / warning rows / first fraud | 0.111 / 0.609 / 0.811 | 0.118 / 0.659 / 0.854 |
| Legitimate forgotten-password rows of other customers at or above threshold | 0.7% (865 rows) | 14.0% |

**Reading:**

- **Scores rise during warning periods for both candidates,** driven by the failed-login feature.
- **B crosses its threshold on some warning rows.** It does so on legitimate forgotten-password activity at a comparable rate (14%), so its warning alerts are not specific to credential attacks.
- **A raises no warning-period alerts.**
- **A's LSTM `risk_score` on warning rows** peaks at 16.7–18.3 in the four validation episodes and at 95.1 in test episode 105. It never reaches the LSTM's own alert threshold (96.5), and A's DNN did not alert on those rows.

## 6. False-positive analysis (test legitimate rows: 21,081)

**By warning signal** (production features on the flagged legitimate rows; a row can have several):

| Signal | Legitimate rows with signal | A: false positives (share of A's FPs; FPR within signal) | B: false positives (share; FPR) |
|---|---|---|---|
| All legitimate | 21,081 | 40 (100%; 0.19%) | 176 (100%; 0.83%) |
| Foreign location | 59 | 39 (97.5%; 66.1%) | 57 (32.4%; 96.6%) |
| Unusual hour | 2,711 | 13 (32.5%; 0.48%) | 89 (50.6%; 3.28%) |
| Unusual category | 2,571 | 12 (30.0%; 0.47%) | 56 (31.8%; 2.18%) |
| Failed logins in previous 24h | 798 | 6 (15.0%; 0.75%) | 83 (47.2%; 10.4%) |
| Another transaction within the hour (burst) | 2,632 | 8 (20.0%; 0.30%) | 40 (22.7%; 1.52%) |
| Amount ≥ 300% of customer average | 374 | 1 (2.5%; 0.27%) | 45 (25.6%; 12.0%) |
| New device | 30 | 0 | 0 |

**By legitimate-unusual reason** (`legit_context`):

| Context | Legitimate rows | A flagged (FPR) | B flagged (FPR) |
|---|---|---|---|
| No unusual context | 15,824 | 0 (0.00%) | 1 (0.01%) |
| Foreign travel | 59 | 39 (66.1%) | 57 (96.6%) |
| Forgotten password | 541 | 3 (0.6%) | 76 (14.0%) |
| Big purchase | 303 | 1 (0.3%) | 42 (13.9%) |
| Shopping burst | 382 | 4 (1.0%) | 28 (7.3%) |
| New category | 594 | 1 (0.2%) | 7 (1.2%) |
| Household device | 836 | 0 | 6 (0.7%) |
| Public Wi-Fi | 932 | 0 | 4 (0.4%) |
| VPN | 381 | 0 | 3 (0.8%) |
| Small purchase | 1,041 | 1 (0.1%) | 4 (0.4%) |
| Device upgrade | 334 | 0 | 1 (0.3%) |
| Domestic travel | 391 | 0 | 1 (0.3%) |
| Borrowed device | 7 | 0 | 0 |

**By customer type:**

| Customer type | Legitimate rows | A false positives (FPR) | B false positives (FPR) |
|---|---|---|---|
| Business traveller | 2,467 | 14 (0.57%) | 40 (1.62%) |
| Salaried commuter | 8,266 | 20 (0.24%) | 59 (0.71%) |
| Retiree | 1,330 | 6 (0.45%) | 17 (1.28%) |
| Household | 3,129 | 0 | 18 (0.58%) |
| Night shift | 1,714 | 0 | 11 (0.64%) |
| Student | 4,175 | 0 | 31 (0.74%) |

**Reading:**

- **Nearly all false positives are benign anomalies:** for both candidates they fall on legitimate unusual behaviour. There is almost none on rows with no unusual context.
- **A:** 39 of its 40 false positives are legitimate foreign-travel transactions.
- **B:** it also flags forgotten-password logins, big purchases and bursts at 7–14%. That, together with its lower threshold, explains its 4.4× larger false-positive count.

## 7. Fraud archetypes (test period)

"Caught" = fraud transactions alerted (recall), "first" = first frauds alerted, "episodes" = episodes detected.

| Type | Episodes | Fraud txns | A: caught (recall) / first / episodes | B: caught (recall) / first / episodes |
|---|---|---|---|---|
| Account takeover | 3 | 10 | 2 (0.20) / 1 / 1 | 3 (0.30) / 2 / 2 |
| Stolen card used online | 2 | 14 | 4 (0.29) / 0 / 1 | 9 (0.64) / 1 / 1 |
| Small test charges → cash-out | 4 | 25 | 3 (0.12) / 0 / 1 | 9 (0.36) / 0 / 2 |
| Device/phone takeover | 3 | 6 | 0 / 0 / 0 | 1 (0.17) / 1 / 1 |
| One large hit | 1 | 2 | 0 / 0 / 0 | 2 (1.00) / 1 / 1 |
| Fraud resembling normal activity | 1 | 4 | 0 / 0 / 0 | 0 / 0 / 0 |
| Ring fraud | 0 | 0 | — | — |

**Reading:**

- **Small numbers:** each type has 0–4 test episodes, so this is not a ranking of types or models.
- **Rings:**
  - The only out-of-sample ring (ring 4, validation, 3 victims, 4 fraud transactions) was caught by neither candidate.
  - On 1,424 legitimate test transactions on legitimately shared devices, A raised 0 false positives and B raised 7.

## 8. LSTM contribution (A vs B, test)

| | Value |
|---|---|
| PR-AUC difference A − B | **−0.024** (customer-cluster bootstrap 95% CI −0.114 to +0.034; 26% of resamples favour A) |
| ROC-AUC difference A − B | +0.003 |
| Fraud caught by both / only A / only B / neither | 9 / **0** / 15 / 37 |
| Episodes detected only by A / only by B | **0** / 4 |
| First frauds detected only by A / only by B | **0** / 4 |
| False positives only A / only B / both | 2 / 138 / 38 |
| Per-row score difference A − B (all test rows) | mean −0.023, median −0.001, 5th–95th percentile −0.176 to +0.054, mean absolute 0.050; A higher on 46.8% of rows |
| On legitimate rows / fraud rows | mean −0.023 / +0.015; A higher on 46.7% / 54.1% |
| Correlation of A and B scores | 0.861 |

**Answers:**

- **Detections unique to A:** none. On this test set no fraud transaction, episode or first fraud is detected only because of the LSTM input.
- **First-fraud detection:** the LSTM input does not improve it (1/14 vs 5/14).
- **False positives:** A raises fewer, mostly because of its higher threshold; it adds 2 of its own.
- **Ranking quality:** it is not measurably different. The PR-AUC difference's confidence interval includes zero, and the 4C-2d customer-grouped split pointed the other way (+0.046, CI −0.003 to +0.131).

**Customer-cluster bootstrap on the test period** (2,000 resamples, thresholds fixed):

| | A | B |
|---|---|---|
| Precision | 0.184 (0.000–0.500) | 0.120 (0.029–0.227) |
| Recall | 0.148 (0.000–0.289) | 0.393 (0.133–0.638) |
| PR-AUC | 0.104 (0.024–0.274) | 0.128 (0.033–0.309) |

## 9. Operational alert volume

These are the production `/predict` bands (Low < 25, Medium < 50, High < 80, Critical ≥ 80 on the score × 100, capped at 99.9), applied to the uncalibrated candidate scores. Test period, per 1,000 transactions:

| Band | A: all (legitimate / fraud rows) | B: all (legitimate / fraud rows) |
|---|---|---|
| Low | 848.5 (17,929 / 9) | 800.0 (16,906 / 7) |
| Medium | 86.3 (1,817 / 7) | 118.8 (2,501 / 10) |
| High | 58.9 (1,223 / 23) | 73.4 (1,530 / 22) |
| Critical | 6.3 (112 / 22) | 7.9 (144 / 22) |
| High + Critical | **65.3 per 1,000** (63.1 legitimate), catching 73.8% of fraud | **81.3 per 1,000** (79.2 legitimate), catching 72.1% of fraud |

- **The fixed bands would overload review for either candidate** (tens of alerts per 1,000 transactions).
- **The validation-chosen operating points (§3) give 1–10 alerts per 1,000.**
- **Threshold choice:** the bands are not a basis for comparing the candidates, and any production use would need thresholds from the manifest (decision deferred, as planned).

## 10. Reproducibility

- **Comparison:** run twice; `comparison.json` is byte-identical (SHA-256 `fa8c522c…`). It contains no timestamps, and the bootstrap is seeded (42).
- **Consistency with 4C-2d:**
  - the candidate scores are identical to the 4C-2d evaluation scores;
  - regenerating the 4C-2d `analysis.json` after the small refactor below gives a byte-identical file.
- **Production models:** the SHA-256 of all 7 files in `models/saved/` is unchanged before and after the comparison and the test suite.

## 11. Limitations

- **One synthetic dataset:** v2, generator 2.0.1. The difficulty comes from generator settings, not real traffic.
- **Small test set:**
  - 14 test episodes and 61 fraud transactions; one episode moves first-fraud recall by about 7 points;
  - no test rings;
  - one test warning-period episode.
- **Thresholds and calibration:**
  - Scores are uncalibrated.
  - Thresholds rest on 119 validation fraud rows; A's F1 threshold (0.959) and B's (0.770) sit in very different places. That drives most of the precision and recall contrast.
  - The ~0.1% false-positive-rate points overshoot slightly on validation (§3).
- **No new evidence:** both candidates are exactly the 4C-2d evaluation models.
- **Training vs scoring:** DNN training inputs are unclipped, while scoring clips them to ±6 (as evaluated).
- **This comparison is not a selection:** no winner is chosen and no production change is recommended here.

## Code touched in this step

- **New `backend/app/evaluation/compare_candidates.py`.**
- **`backend/app/evaluation/analysis.py`:** its helpers (`thresholds`, `lstm_ablation`, `first_fraud`, `fraud_types`, `legit_false_positives`, `rings`, `uncertainty`) now take the models to analyse as an argument. The defaults are unchanged, and the 4C-2d output is byte-identical.
- **New `backend/tests/test_compare_candidates.py`** (9 tests).
