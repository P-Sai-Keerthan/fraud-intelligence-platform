# Step 4C-2d — Leakage-safe evaluation on the v2 dataset

Every number below comes from these two commands, run from `backend/`:

```bash
python -m app.evaluation.run --dataset v2        # models/evaluation/v2/evaluation_report.json (+ splits, scores, evaluation models)
python -m app.evaluation.analysis --dataset v2   # models/evaluation/v2/analysis.json
```

The evaluation was run twice. Both runs produced identical reports, split files, per-transaction scores and model weights.

- **Production models:** none were trained or replaced (`models/saved/` is untouched).
- **App and v1 artefacts:** `/predict`, `/metrics` and the v1 evaluation files are unchanged.
- **Scope:** everything here is measured on synthetic data (see §15).

## 1. Objective

The aim is to find out whether the redesigned v2 data gives a meaningful evaluation where v1 could not. On v1, every single-transaction model scored 1.000, and the LSTM caught none of the 16 first frauds (0/16).

The evaluation should answer:

- how the proposed LSTM → DNN pipeline and the baselines perform on v2;
- whether the LSTM risk score adds anything;
- how well each model catches the first fraudulent transaction of an episode;
- how the models behave on legitimate warning-period rows, on each fraud type, on legitimate unusual behaviour, and on fraud rings.

## 2. Dataset

| | |
|---|---|
| File | `data/v2/transactions_with_features.csv` (generator 2.0.1, seed 42) |
| SHA-256 | `a09d06114ce7c0dec4b93ff9946ab9506f5c7d25317bb95396b596f2f7b1ef08` (matches `data/v2/manifest.json`) |
| Transactions / customers | 101,297 / 500 |
| Fraud | 471 transactions (0.465%) in 110 episodes: 22 account takeover, 22 stolen card used online, 17 small test charges → cash-out, 11 device/phone takeover, 11 one large hit, 11 fraud resembling normal activity, 16 ring episodes (4 rings) |
| Warning periods | 18 episodes, 60 legitimate transactions |
| Data audit | `data/v2/DATA_SANITY_REPORT.md` (no anomalies) |

## 3. Split methodology

**Primary split: time-based.**

- **Boundaries:** the same rule as v1, applied to v2's explicit episode groups.
- **Groups:** a ring's episodes form one group, and each group's span includes its warning period.
- **Straddling groups:** none straddled a boundary, so no group had to be moved.

| Split | Period | Transactions | Fraud | Fraud rate | Episodes |
|---|---|---|---|---|---|
| Train | 2026-01-12 → 2026-05-02 | 63,840 | 291 | 0.456% | 68 |
| Validation | 2026-05-03 → 2026-05-31 | 16,315 | 119 | 0.729% | 28 |
| Test | 2026-06-01 → 2026-07-06 | 21,142 | 61 | 0.289% | 14 |

**Secondary split: grouped by customer.**

- **Groups:** customers linked by a ring or a household form one group (441 groups).
- **Assignment:** groups are shuffled with seed 42, stratified by whether they contain fraud, and divided 60/20/20.
- **Test set:** 99 customers, 20,207 transactions, 73 fraud, 17 episodes.

Every transaction has at least 10 earlier transactions, so every row has an LSTM window and the evaluated rows are exactly the split rows.

Comparison with v1's primary split: v1 trained before 2026-05-06 and validated before 2026-06-01. It defined episodes with a 14-day rule (60 episodes, split 32/12/16), and 6 straddling episodes had to be moved.

## 4. Leakage controls

- **Model input:** only the identifiers, `is_fraud` and the 9 production features, in production order: `amount_zscore, hour_is_unusual, is_new_device, is_new_location, is_foreign_location, failed_logins_24h, category_is_unusual, txn_velocity_1h, amount_pct_of_avg`.
  - The v2 metadata is loaded separately and used only in the analysis.
  - `evaluate_split` refuses any frame that contains a ground-truth column. The analysis confirms none was present.
- **LSTM windows:** the 10 transactions strictly before each target, checked by `assert_past_only`.
- **LSTM → DNN handoff:** DNN training rows get LSTM scores from 3 out-of-fold models, grouped by customer. Validation and test rows are scored by an LSTM trained only on the training split.
- **Thresholds:** chosen on validation by maximising F1. The recall at fixed false-positive rates also uses validation-chosen thresholds. Test labels were never used for any choice.
- **Seeds:** seed 42 with TensorFlow op determinism. Two full runs gave identical results.
- **Same data:** the report's dataset hash equals the hash of the file the analysis read.
  - The old v1 rule recomputed during the analysis gives precision 0.069 and recall 0.193 on the full v2 dataset, the same as the sanity report.
  - On the test split it gives 0.073 and 0.180.

## 5. Model configurations

| Label | What it is |
|---|---|
| DNN + LSTM | The proposed pipeline: the DNN on the 9 features plus the LSTM `risk_score` (out-of-fold on training rows) |
| DNN only | The same DNN on the 9 features, without `risk_score` (ablation) |
| LSTM | The LSTM alone: probability that a transaction is fraud, from the 10 transactions before it |
| Logistic regression | On the 9 features |
| Rule | Existing baseline: `amount_pct_of_avg` when `hour_is_unusual`, otherwise 0, with the cutoff chosen on validation |

Architectures, hyperparameters and training recipe are unchanged from Step 4B. Only the evaluation copies in `models/evaluation/v2/time_split/` were trained.

## 6. Primary metrics (time split, test period)

The test period has 21,142 transactions, 61 of them fraud (14 episodes).

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | Threshold | TN / FP / FN / TP | FPR | Alerts / 1,000 |
|---|---|---|---|---|---|---|---|---|---|
| DNN + LSTM | 0.104 | 0.899 | 0.184 | 0.148 | 0.164 | 0.9593 | 21,041 / 40 / 52 / 9 | 0.19% | 2.32 |
| DNN only | 0.128 | 0.897 | 0.120 | 0.393 | 0.184 | 0.7698 | 20,905 / 176 / 37 / 24 | 0.83% | 9.46 |
| LSTM | 0.135 | 0.741 | 0.220 | 0.148 | 0.177 | 0.9648 | 21,049 / 32 / 52 / 9 | 0.15% | 1.94 |
| Logistic regression | 0.138 | 0.891 | 0.119 | 0.328 | 0.175 | 0.9712 | 20,933 / 148 / 41 / 20 | 0.70% | 7.95 |
| Rule | 0.027 | 0.730 | 0.089 | 0.180 | 0.120 | 199.06 | 20,969 / 112 / 50 / 11 | 0.53% | 5.82 |

**How to read these numbers:**

- **Baseline for PR-AUC:** a model with no information would score about 0.003, which is the test fraud rate.
- **Similar ranking quality:** the four learned models have PR-AUCs between 0.10 and 0.14. The confidence intervals in §14 overlap heavily, so these figures do not order the models reliably.
- **Thresholds drive the gaps:** most differences in precision and recall come from where each validation-chosen threshold landed, not from ranking quality.
- **Fixed false-positive rates** (thresholds chosen on validation for about 0.1% and 1%):

| Model | Recall at ~0.1% FPR | Recall at ~1% FPR |
|---|---|---|
| DNN + LSTM | 0.066 | 0.410 |
| DNN only | 0.164 | 0.393 |
| LSTM | 0.098 | 0.344 |
| Logistic regression | 0.131 | 0.328 |
| Rule | 0.016 | 0.197 |

## 7. v1 vs v2 (time split, test period)

| Metric | v1 | v2 |
|---|---|---|
| Test transactions | 18,904 | 21,142 |
| Fraud transactions | 201 | 61 |
| Fraud episodes | 16 | 14 |
| Fraud rate | 1.063% | 0.289% |
| DNN + LSTM PR-AUC | 1.000 | 0.104 |
| DNN + LSTM ROC-AUC | 1.000 | 0.899 |
| DNN + LSTM precision | 1.000 | 0.184 |
| DNN + LSTM recall | 1.000 | 0.148 |
| DNN + LSTM F1 | 1.000 | 0.164 |
| DNN-only PR-AUC | 1.000 | 0.128 |
| LSTM PR-AUC | 0.829 | 0.135 |
| Logistic regression PR-AUC | 1.000 | 0.138 |
| Rule PR-AUC | 1.000 | 0.027 |
| First-fraud recall, DNN + LSTM / LSTM | 16/16 / 0/16 | 1/14 / 0/14 |
| Train before / validation before | 2026-05-06 / 2026-06-01 | 2026-05-03 / 2026-06-01 |
| Episode definition | 14-day heuristic | explicit ids, rings grouped, warning periods included |

**What the comparison shows, and what it does not:**

- **The lower v2 numbers are not a worse model.** The architecture, features and training recipe are identical. The data changed.
  - On v1, every fraud was built from the signals the features measure, so a two-condition rule scored 1.000.
  - On v2, those signals also occur in legitimate behaviour, and some fraud has none of them.
- **v2 is a harder, less circular test, not a more realistic one in any proven sense.** Both datasets are synthetic, and v2's difficulty comes from generator settings (`data/v2/synth_v2/config.py`) chosen without reference to real bank data.
- **The two datasets cannot be compared score-for-score.** Their fraud rates differ (1.06% vs 0.29%), and PR-AUC depends strongly on the fraud rate.
- **Both test sets are small** (16 and 14 episodes).

## 8. LSTM ablation: DNN + LSTM vs DNN only

**Time split (primary), test period:**

| | DNN + LSTM | DNN only | Difference |
|---|---|---|---|
| PR-AUC | 0.104 | 0.128 | −0.024 (95% CI −0.114 to +0.034) |
| ROC-AUC | 0.899 | 0.897 | +0.003 |
| Precision | 0.184 | 0.120 | +0.064 |
| Recall | 0.148 | 0.393 | −0.246 |
| F1 | 0.164 | 0.184 | −0.020 |
| Alerts / 1,000 | 2.32 | 9.46 | −7.14 |
| False-positive rate | 0.19% | 0.83% | −0.65 pp |

**Where the two models disagree on individual transactions:**

- **Fraud caught:** 9 by both, 15 only by DNN only, 0 only by DNN + LSTM. Both missed 37.
- **False positives:** 38 shared, 138 only DNN only, 2 only DNN + LSTM.
- **Episodes detected:** DNN + LSTM 3, DNN only 7. There is no episode that only DNN + LSTM detected.
- **First frauds caught:** 4 only by DNN only, none only by DNN + LSTM.
- **Score correlation:** the two DNNs' test scores correlate at 0.86.

**Secondary split (grouped by customer)**, threshold-free comparison:

- PR-AUC 0.0875 with the LSTM vs 0.0416 without, a difference of +0.046 (95% CI −0.003 to +0.131).
- At the F1 thresholds, the DNN with the LSTM score had precision 0.170 and recall 0.110; without it, 0.095 and 0.096.

**Answers:**

- **Does the LSTM score improve performance?**
  - There is no consistent evidence either way: on the time split PR-AUC is 0.024 lower with it, and on the customer split 0.046 higher.
  - Both 95% intervals include zero, and ROC-AUC is essentially the same on the time split (0.899 vs 0.897).
- **Does it reduce false positives?**
  - At the validation-chosen thresholds, yes: 40 vs 176.
  - That is mainly a consequence of DNN + LSTM's validation threshold landing higher. The same pipeline also catches much less fraud, and at a fixed ~1% false-positive rate the recalls are similar (0.41 vs 0.39).
- **Does it detect more episodes?** No. It detected 3 test episodes against 7, and none that the DNN-only model missed.
- **Does it catch anything the DNN misses?** No: no fraud transaction, episode or first fraud was caught only with the LSTM score.
- **Does it add false positives?** Two, which DNN only did not flag.

## 9. First-fraud detection (time split, test period)

| Model | Episodes | First frauds caught | First-fraud recall (Wilson 95%) | Episodes detected | …only after fraud had started | Never detected | Median delay when detected (fraud txns / hours) | Median delay when caught after onset (hours) |
|---|---|---|---|---|---|---|---|---|
| DNN + LSTM | 14 | 1 | 0.071 (0.013–0.315) | 3 | 2 | 11 | 3 / 1.4 | 16.9 |
| DNN only | 14 | 5 | 0.357 (0.163–0.612) | 7 | 2 | 7 | 0 / 0.0 | 1.1 |
| LSTM | 14 | **0** | **0.000 (0.000–0.215)** | 3 | 3 | 11 | 1 / 40.8 | 40.8 |
| Logistic regression | 14 | 5 | 0.357 (0.163–0.612) | 7 | 2 | 7 | 0 / 0.0 | 2.5 |
| Rule | 14 | 3 | 0.214 (0.076–0.476) | 8 | 5 | 6 | 1 / 3.3 | 19.4 |

"Detected" means at least one fraud transaction of the episode was alerted. The delay counts fraud transactions (and hours) from the first fraud to the first alert.

**The LSTM's behaviour is unchanged from v1:**

- It caught no first fraud: 0/14, against 0/16 on v1. On the customer split it caught 1/17.
- The 3 episodes it detected were all detected after fraud had already started: after 1 fraud transaction, 1–49 hours in (median 40.8 hours).
- This is detection after onset, not detection before fraud.

## 10. Warning-period analysis

18 episodes have a legitimate warning period: credential-attack failed logins that show up on the customer's own legitimate transactions before the fraud. The rows compared are:

- (1) the customer's legitimate transactions in the 30 days before the warning period;
- (2) warning-period transactions;
- (3) the first fraud;
- (4) later fraud;
- (5) for comparison, legitimate forgotten-password transactions of customers who were never defrauded.

**LSTM, all splits.** The LSTM is out-of-sample everywhere: out-of-fold on training rows, and a model that never saw validation or test rows.

| Group | Rows (episodes) | Median | Mean | 90th pct | Share ≥ LSTM threshold |
|---|---|---|---|---|---|
| 1 Before warning | 499 (18) | 0.245 | 0.275 | 0.468 | 0.0% |
| 2 Warning period (legitimate) | 60 (17) | 0.298 | 0.357 | 0.743 | 0.0% |
| 3 First fraud | 18 (18) | 0.428 | 0.499 | 0.915 | 0.0% |
| 4 Later fraud | 68 (16) | 0.974 | 0.879 | 0.995 | 61.8% |
| 5 Forgotten-password rows, other customers | 2,077 | 0.273 | 0.342 | 0.695 | 0.4% |

**All models, validation and test rows only** (5 episodes, 9 warning rows; share of rows at or above each model's threshold):

| Group | DNN + LSTM | DNN only | LSTM | Rows |
|---|---|---|---|---|
| 1 Before warning | 0.0% | 1.4% | 0.0% | 69 |
| 2 Warning period (legitimate) | 0.0% | 22.2% | 0.0% | 9 |
| 3 First fraud | 20.0% | 60.0% | 0.0% | 5 |
| 4 Later fraud | 36.8% | 89.5% | 57.9% | 19 |
| 5 Forgotten-password rows, other customers | 0.7% | 14.0% | 0.5% | 865 |

**Reading:**

- **LSTM scores rise only a little in warning periods.** The median goes from 0.245 before to 0.298 during, and none of these rows reach the alert threshold.
- **Warning periods look like forgotten passwords to the LSTM.** Warning rows score close to forgotten-password rows of customers who were never defrauded (median 0.273). On these data the LSTM does not separate a credential attack from a forgotten password.
- **LSTM alerts start after fraud has begun.** Its scores reach the alert level only on later fraud transactions, whose windows already contain fraud.
- **The DNN-only model flags some warning rows,** because of their failed logins. It flags forgotten-password rows of unaffected customers at a similar rate (22% vs 14%).
  - These are false positives on legitimate transactions, not predictions.
- **Sample size:** out of sample for every model, there are only 5 episodes and 9 warning rows.

## 11. Fraud-type analysis (time split, test period)

"Caught" = fraud transactions alerted. "First" = first frauds caught. The median delay (hours) is over detected episodes.

| Type | Episodes | Fraud txns | DNN + LSTM caught / first | DNN only caught / first | LSTM caught / first | Log. reg. caught / first | Rule caught / first |
|---|---|---|---|---|---|---|---|
| Account takeover | 3 | 10 | 2 / 1 | 3 / 2 | 2 / 0 | 2 / 2 | 4 / 1 |
| Stolen card used online | 2 | 14 | 4 / 0 | 9 / 1 | 0 / 0 | 9 / 1 | 1 / 0 |
| Small test charges → cash-out | 4 | 25 | 3 / 0 | 9 / 0 | 7 / 0 | 6 / 1 | 1 / 0 |
| Device/phone takeover | 3 | 6 | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 0 | 4 / 1 |
| One large hit | 1 | 2 | 0 / 0 | 2 / 1 | 0 / 0 | 2 / 1 | 1 / 1 |
| Fraud resembling normal activity | 1 | 4 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| Ring fraud | 0 | 0 | — | — | — | — | — |

**Reading:**

- **Coverage:** four of the six types in the test period have 1–3 episodes, and ring fraud has none.
- **Not a ranking:** these figures describe individual episodes, not the relative difficulty of the fraud types.
- **Normal-looking fraud:** no model caught the one test episode, and only 1 of 14 such transactions was caught on validation and test combined (by logistic regression).
- **Supplementary validation + test view:** a wider view is saved in `analysis.json` (`fraud_types.validation_and_test_supplementary`). It is optimistic, because thresholds were chosen on the validation rows.

## 12. False positives on legitimate unusual behaviour (time split, test period)

The test period has 21,081 legitimate transactions. A row can carry several contexts.

| Context | Legitimate rows | DNN + LSTM | DNN only | LSTM | Log. reg. | Rule |
|---|---|---|---|---|---|---|
| All legitimate | 21,081 | 40 (0.19%) | 176 (0.83%) | 32 (0.15%) | 148 (0.70%) | 112 (0.53%) |
| No unusual context | 15,824 | 0 (0.00%) | 1 (0.01%) | 4 (0.03%) | 0 (0.00%) | 60 (0.38%) |
| Foreign travel | 59 | 39 (66.1%) | 57 (96.6%) | 25 (42.4%) | 27 (45.8%) | 0 (0.0%) |
| Big purchase | 303 | 1 (0.3%) | 42 (13.9%) | 1 (0.3%) | 95 (31.4%) | 40 (13.2%) |
| Forgotten password | 541 | 3 (0.6%) | 76 (14.0%) | 2 (0.4%) | 15 (2.8%) | 3 (0.6%) |
| Shopping burst | 382 | 4 (1.0%) | 28 (7.3%) | 4 (1.0%) | 26 (6.8%) | 3 (0.8%) |
| New category | 594 | 1 (0.2%) | 7 (1.2%) | 1 (0.2%) | 1 (0.2%) | 3 (0.5%) |
| Household device | 836 | 0 | 6 (0.7%) | 0 | 6 (0.7%) | 4 (0.5%) |
| Public Wi-Fi | 932 | 0 | 4 (0.4%) | 0 | 4 (0.4%) | 2 (0.2%) |
| VPN | 381 | 0 | 3 (0.8%) | 0 | 1 (0.3%) | 4 (1.0%) |
| Device upgrade | 334 | 0 | 1 (0.3%) | 2 (0.6%) | 4 (1.2%) | 2 (0.6%) |
| Domestic travel | 391 | 0 | 1 (0.3%) | 0 | 4 (1.0%) | 1 (0.3%) |
| Small purchase | 1,041 | 1 (0.1%) | 4 (0.4%) | 0 | 0 | 0 |
| Borrowed device | 7 | 0 | 0 | 0 | 1 (14.3%) | 0 |

**Reading:**

- **The learned models' false alarms come almost entirely from benign anomalies.** On legitimate rows with no unusual context they have false-positive rates of 0–0.03%.
- **Foreign travel dominates DNN + LSTM's false alarms.** 39 of its 40 false positives are legitimate transactions abroad, and every learned model flags a large share of those rows.
  - The features cannot tell a legitimate trip from a fraudster abroad: `is_foreign_location` is 1 for both, and nothing records that a customer is travelling.
- **Other main sources of false alarms:**
  - forgotten-password failed logins (DNN only 14%);
  - big purchases (logistic regression 31%, DNN only 14%);
  - shopping bursts.
- **This is the kind of benign anomaly v2 was designed to test.** The v1 data contained none, which is part of why every model scored 1.000 there.

## 13. Fraud rings

**Ring counts:** v2 has 4 rings. None is in the test period: rings 1–3 are in training and ring 4 is in validation. So there is no out-of-sample ring result on test.

| Ring | Split | Victims | Fraud txns | Out of sample | DNN + LSTM caught (victims) | DNN only | LSTM | Log. reg. | Rule |
|---|---|---|---|---|---|---|---|---|---|
| 1 | train | 4 | 8 | LSTM only | 6 (4) | 8 (4) | 0 (0) | 8 (4) | 2 (2) |
| 2 | train | 6 | 13 | LSTM only | 4 (3) | 7 (4) | 5 (3) | 11 (6) | 5 (5) |
| 3 | train | 3 | 7 | LSTM only | 0 (0) | 3 (1) | 2 (2) | 6 (3) | 2 (2) |
| 4 | validation | 3 | 4 | all (thresholds chosen on validation) | 0 (0) | 0 (0) | 0 (0) | 3 (2) | 2 (1) |

**Ring results:**

- **Training rings (1–3):** the DNN, logistic-regression and rule scores are in-sample, so their catches are not evidence of detection. Only the LSTM columns there are out-of-sample.
- **The one out-of-sample ring:** ring 4 was caught only by logistic regression (3 of 4 transactions) and the rule (2). DNN + LSTM, DNN only and LSTM caught none of it.

**Legitimate shared devices:** 1,424 legitimate test transactions were made on devices that several customers share legitimately (households, borrowed phones). DNN + LSTM flagged 0 of them, DNN only 7, LSTM 0, logistic regression 9 and the rule 5.

**Production ring detector** (`detect_fraud_rings`, unchanged: any device used by 2+ customers), run on the full v2 history:

- **Flags:** it flags 55 devices, of which 5 are ring devices, 47 are purely legitimate shared devices and 3 are customers' own phones that another customer also borrowed.
- **Precision:** 9% (5 of 55 flags are ring devices).
- **Rings found:** 3 of the 4 rings have a flagged ring device. Ring 4's members never shared a device, only a network or merchant, which the detector does not look at.
- **Interpretation:** a shared device is not a confirmed ring. On these data, most flags are households and borrowed phones.

## 14. Statistical uncertainty

**Method:**

- **Resampling:** a bootstrap over the test split that resamples whole customers (2,000 resamples, seed 42), so each customer's transactions and episodes stay together.
- **Thresholds:** held fixed at their validation values.
- **Intervals:** 2.5th–97.5th percentiles, plus Wilson score intervals for first-fraud recall.
- **Sample:** the test period has 500 customers, of whom only 14 had fraud. The intervals are therefore wide, and approximate at the extremes.

| Model | Precision (95% CI) | Recall (95% CI) | F1 (95% CI) | PR-AUC (95% CI) | First-fraud recall (Wilson 95%) |
|---|---|---|---|---|---|
| DNN + LSTM | 0.184 (0.000–0.500) | 0.148 (0.000–0.289) | 0.164 (0.000–0.322) | 0.104 (0.024–0.274) | 1/14 (0.013–0.315) |
| DNN only | 0.120 (0.029–0.227) | 0.393 (0.133–0.638) | 0.184 (0.050–0.322) | 0.128 (0.033–0.309) | 5/14 (0.163–0.612) |
| LSTM | 0.220 (0.000–0.413) | 0.148 (0.000–0.382) | 0.177 (0.000–0.369) | 0.135 (0.012–0.312) | 0/14 (0.000–0.215) |
| Logistic regression | 0.119 (0.024–0.234) | 0.328 (0.088–0.572) | 0.175 (0.039–0.321) | 0.138 (0.042–0.285) | 5/14 (0.163–0.612) |
| Rule | 0.089 (0.035–0.157) | 0.180 (0.085–0.328) | 0.120 (0.052–0.194) | 0.027 (0.010–0.066) | 3/14 (0.076–0.476) |

**Reading the intervals:**

- **Learned models vs rule:** the four learned models' PR-AUC intervals overlap almost completely. All of them sit clearly above the rule's interval and above the no-information level (about 0.003).
- **LSTM contribution:** the PR-AUC difference between DNN + LSTM and DNN only is −0.024 (95% CI −0.114 to +0.034); only 26% of resamples favour the LSTM version.
- **Intervals starting at 0:** DNN + LSTM and LSTM alone have only 9 true positives each, so some resamples contain none of them. Their lower bounds of 0 reflect that and should not be read as precise.

## 15. Limitations

- **Synthetic data:** v2 is generated by rules this project wrote, so its difficulty and its fraud–legitimate overlap come from `config.py`, not from real bank behaviour. The results measure behaviour on this generator only.
- **Small test set:** 14 test episodes and 61 fraud transactions. One episode moves first-fraud recall by about 7 percentage points.
  - There are no ring episodes and only 1 warning-period episode in the test period.
  - I did not change the split to enlarge the test set.
- **Less fraud in the test period:** the test fraud rate (0.29%) is lower than training (0.46%) and validation (0.73%). That comes from the generator's random episode timing; I did not adjust it. It lowers PR-AUC compared with the other periods.
- **Scores are not probabilities:** they come from class-weighted models and are not calibrated. Thresholds are F1-optimal on 119 validation fraud transactions, and small validation samples can put thresholds in quite different places. That is the main reason the models' precision and recall differ so much.
- **Out-of-sample scope:** the training-period ring and warning results are only out of sample for the LSTM.
- **Other analyses:** the customer-grouped secondary split tells a different story for the LSTM ablation (§8). Neither split shows a clear effect.
- **Models evaluated:** these are evaluation copies trained on v2. The production models behind `/predict` were trained on v1 and are not evaluated here.

## 16. Conclusions

- **The v1 artefact is gone:** on v2 no model or rule separates fraud perfectly. The old two-condition rule (unusual hour and amount ≥ 188% of average) falls to precision 0.07 and recall 0.18–0.19 (test period / full dataset), and every model's test PR-AUC is between 0.03 and 0.14.
- **The learned models perform alike:** DNN + LSTM, DNN only, LSTM and logistic regression rank fraud with similar quality (PR-AUC 0.10–0.14, overlapping intervals). All beat the amount/hour rule and the fraud rate by a wide margin.
- **The LSTM risk score shows no measurable contribution:**
  - On the time split, PR-AUC is 0.024 lower with it; on the customer split, 0.046 higher. Both intervals include zero.
  - At the validation-chosen threshold, DNN + LSTM raises fewer alerts but detects fewer episodes (3 vs 7) and fewer first frauds (1 vs 5).
  - It caught no fraud, episode or first fraud that the DNN-only model missed.
- **The LSTM on its own caught no first fraud (0/14),** as on v1 (0/16). Its alerts start after fraud has begun.
  - During legitimate warning periods its scores rise slightly, but not above the level of legitimate forgotten-password activity, and never to its alert threshold.
  - These results do not support describing the LSTM as detecting fraud before it happens.
- **False alarms come from benign anomalies:**
  - Almost all false positives fall on legitimate unusual behaviour, above all legitimate foreign travel (every learned model flags 42–97% of those rows).
  - For logistic regression and the DNN-only model, they also fall on forgotten-password logins and big purchases.
- **Fraud-type, ring and warning-period results are too small to generalise:** 0–4 test episodes per type, no test rings and one test warning period. The production ring detector's shared-device rule mostly flags legitimate sharing (5 of 55 flags are ring devices).

These are measurements of evaluation copies on one synthetic dataset. They do not show that any model is ready for production, and they do not show that fraud would be prevented.
