# Step 4C-3A: production decision audit

This is a read-only audit at commit `582fe60`. It compares the three model
sets the application can load: `production`, `v2_dnn_lstm` and `v2_dnn_only`.

Nothing was changed by this audit:

* no model was retrained, deployed or selected;
* `MODEL_SET` still defaults to `production`;
* no threshold or application behaviour was changed;
* `backend/models/saved/` is byte-identical before and after (§8).

**This document does not choose a winner.** §7 is a recommendation only. The
decision belongs to the project owner.

## 1. Evidence used

| Evidence | Source | Status |
|---|---|---|
| Candidate metrics, operating points, episodes, false positives, rings, LSTM contribution | `backend/models/candidates/v2/comparison.json` (SHA-256 `fa8c522c…`) | Saved artifact. Recomputed in memory from the saved models during this audit: identical. |
| Candidate scores equal the 4C-2d evaluation scores | `backend/models/evaluation/v2/scores_time_split.csv.gz` | Identical on every row for both candidates. |
| Production on the v2 test period | **New in this audit.** The existing production models scored the saved v2 time split. Scoring only: no training, nothing written to the repository. | See the caveats in §2. |
| Live v1 alert volume, thresholds, cold start | `docs/step4c2f-2-threshold-and-cold-start-analysis.md`, `docs/step4c2f-1-correctness-fixes.md` | Saved analysis. |
| Readiness items | `docs/step4c2e-e-production-readiness.md` | Reviewed in §5. |

**The test set.** The v2 time-split test period has 21,142 transactions: 61
fraud transactions in 14 episodes, and 21,081 legitimate transactions. Every
transaction has at least 10 earlier transactions.

## 2. Comparison: production, v2_dnn_lstm, v2_dnn_only

### 2.1 Threshold-free ranking (v2 test period)

| | production | v2_dnn_lstm | v2_dnn_only |
|---|---|---|---|
| PR-AUC | 0.128 | 0.104 | 0.128 |
| PR-AUC 95% CI (customer bootstrap) | 0.038 – 0.251 | 0.024 – 0.274 | 0.033 – 0.309 |
| ROC-AUC | 0.663 | 0.899 | 0.897 |

* **The PR-AUC intervals overlap almost completely.** The three model sets
  cannot be told apart on PR-AUC with this test set.
* **PR-AUC difference, v2_dnn_lstm minus v2_dnn_only:** −0.024, 95% CI −0.114
  to +0.034. 26% of resamples favour v2_dnn_lstm.
* **Production's ROC-AUC is much lower.** It ranks the bulk of legitimate v2
  transactions worse than the candidates do, which shows up as alert burden in
  §2.3.

### 2.2 Each model at its own validation-chosen F1 threshold

The candidate thresholds are the ones recorded in their manifests. Production
has no v2 threshold of its own. For a like-for-like row, this audit applied
the same rule (the F1-maximising threshold on the v2 validation period) to
production's scores. That production threshold is for analysis only: it is not
recorded anywhere and not applied by anything.

| | production | v2_dnn_lstm | v2_dnn_only |
|---|---|---|---|
| Threshold (score) | 0.9946 | 0.9593 | 0.7698 |
| Fraud caught (of 61) | 9 | 9 | 24 |
| False positives (of 21,081) | 21 | 40 | 176 |
| Precision | 0.300 | 0.184 | 0.120 |
| Precision 95% CI | 0.095 – 0.519 | 0.000 – 0.500 | 0.029 – 0.227 |
| Recall | 0.148 | 0.148 | 0.393 |
| Recall 95% CI | 0.054 – 0.250 | 0.000 – 0.289 | 0.133 – 0.638 |
| F1 | 0.198 | 0.164 | 0.184 |
| F1 95% CI | 0.070 – 0.311 | 0.000 – 0.322 | 0.050 – 0.322 |
| False-positive rate | 0.10% | 0.19% | 0.83% |
| Alerts per 1,000 transactions | 1.42 | 2.32 | 9.46 |
| Legitimate alerts per 1,000 | 0.99 | 1.89 | 8.32 |
| First fraud of an episode detected (of 14) | 3 | 1 | 5 |
| First-fraud 95% CI (Wilson) | 8% – 48% | 1% – 31% | 16% – 61% |
| Episodes detected at all (of 14) | 6 | 3 | 7 |

### 2.3 Each model under the live alert bands (what `/predict` does today)

`/predict` applies the legacy 25/50/80 bands to every model set. This is the
behaviour a switch would actually produce today.

| | production | v2_dnn_lstm | v2_dnn_only |
|---|---|---|---|
| **High + Critical (score ≥ 50)** | | | |
| Fraud caught (of 61) | 29 | 45 | 44 |
| False positives | 1,149 | 1,335 | 1,674 |
| Precision | 0.025 | 0.033 | 0.026 |
| Recall | 0.475 | 0.738 | 0.721 |
| Recall 95% CI | 0.268 – 0.700 | 0.515 – 0.916 | 0.521 – 0.889 |
| False-positive rate | 5.5% | 6.3% | 7.9% |
| Legitimate alerts per 1,000 | 54.4 | 63.1 | 79.2 |
| First fraud detected (of 14) | 5 | 10 | 10 |
| Episodes detected (of 14) | 11 | 12 | 13 |
| **Critical only (score ≥ 80)** | | | |
| Fraud caught (of 61) | 25 | 22 | 22 |
| False positives | 580 | 112 | 144 |
| Precision | 0.041 | 0.164 | 0.133 |
| Recall | 0.410 | 0.361 | 0.361 |
| Legitimate alerts per 1,000 | 27.4 | 5.3 | 6.8 |
| First fraud detected (of 14) | 5 | 5 | 5 |

### 2.4 Alert burden on the live (v1) customer history

The live application scores the v1 customers. These are alert volumes only:
v1 fraud is trivially separable, and production was trained on this data.

| Per 1,000 transactions on the live v1 history | production | v2_dnn_lstm | v2_dnn_only |
|---|---|---|---|
| Legitimate High + Critical under the live bands | 1.6 (in-sample) | 23.3 | 18.8 |
| All alerts at the manifest F1 threshold | — | 5.2 (0.0 legitimate) | 9.0 (0.06 legitimate) |

### 2.5 Where the false positives come from

Legitimate test rows flagged at each model's validation F1 threshold, by the
dataset's ground-truth context:

| Context (legitimate rows) | production (21) | v2_dnn_lstm (40) | v2_dnn_only (176) |
|---|---|---|---|
| Foreign travel (59) | 4 | 39 | 57 |
| Forgotten password (541) | 6 | 3 | 76 |
| Big purchase (303) | 11 | 1 | 42 |
| Burst of transactions (382) | 1 | 4 | 28 |
| No unusual context (15,824) | 0 | 0 | 1 |

A row can belong to more than one context, so the columns do not sum to the
totals.

* At score ≥ 50, production flags 612 legitimate rows that have no unusual
  context at all. v2_dnn_lstm flags 246 and v2_dnn_only flags 552.
* v2_dnn_lstm flags every one of the 59 legitimate foreign-travel rows at
  score ≥ 50, and 39 of them at its F1 threshold.

### 2.6 Other dimensions

| Dimension | production | v2_dnn_lstm | v2_dnn_only |
|---|---|---|---|
| Training data | v1 (trivially separable labels) | v2 synthetic | v2 synthetic |
| Training recipe | legacy: random split, in-sample stacking, no seed | time split, out-of-fold stacking, seed 42 | time split, seed 42 |
| Training and live features consistent | **No** (legacy-v1 feature values; 19 of 88,913 v1 windows change alert level) | Yes | Yes |
| Uses an LSTM | Yes | Yes | No |
| LSTM contribution demonstrated | not tested | **No** (see below) | not applicable |
| Provenance | file SHA-256 only; no manifest | manifest, weight checksums, dataset and split hashes | same |
| Evaluation of the deployed weights | none before this audit (`/metrics` shows v1 evaluation copies) | v2 time split | v2 time split |
| Cold start (fewer than 10 earlier transactions) | LSTM not run; risk score imputed (4C-2f-1) | same | first transaction: features imputed |
| Cold-start residual | flags from 1–9 earlier transactions are hypersensitive (36% Medium example) | same features | same features |
| `risk_score` meaning | LSTM trajectory score | LSTM behavioural-risk component | repeats the fraud score (compatibility) |
| Median latency (2-core CPU) | ~270 ms | ~263 ms | ~208 ms |

**LSTM contribution (candidates)**
* On the time split, v2_dnn_lstm caught no fraud transaction, no episode and
  no first fraud that v2_dnn_only missed. v2_dnn_only caught 15 fraud
  transactions and 4 episodes that v2_dnn_lstm missed.
* On the customer-grouped split the PR-AUC difference had the opposite sign:
  +0.046, 95% CI −0.003 to +0.131.
* At matched validation false-positive rates the two candidates' recall is
  close (0.41 and 0.39 at 1%).
* So no consistent LSTM contribution is established, in either direction.

**Behavioural and ring detection**
* **No fraud ring falls in the test period.** One ring falls in validation: 3
  victims and 4 fraud transactions. Both candidates caught none of them, and
  thresholds were chosen on those rows. Ring detection by the models is
  therefore untested out of sample and unpromising in sample.
* **The ring detector is not a model.** It flags any device used by 2 or more
  customers, and it is the same for every model set. On v2 it flags 55
  devices, of which 5 belong to rings (9% precision). On the live v1 data,
  shared devices are random collisions.
* **Warning periods.** v2_dnn_lstm raised no alert during any credential-attack
  warning period. v2_dnn_only crossed its threshold on 22% of warning rows,
  but also on 14% of legitimate forgotten-password rows, so the signal is not
  specific.
* **Shared legitimate devices.** Of 1,424 legitimate test rows on shared
  devices, v2_dnn_lstm flagged 0 and v2_dnn_only flagged 7.

## 3. What the evidence can and cannot support

**It can support**
* Under the live bands, every model set produces 54–79 legitimate
  High + Critical alerts per 1,000 v2 transactions. No model set is usable for
  alerting on v2-like data with the bands as they are.
* Production's scores on v2 are poorly ordered (ROC-AUC 0.66) and saturate:
  580 legitimate rows score 80 or more.
* The candidates are reproducible and documented: manifests, checksums and
  exact agreement with the evaluation.

**It cannot support**
* **A ranking of the three model sets.** Every interval in §2 overlaps. With
  14 episodes, one episode moves episode-level recall by about 7 points.
* **A claim that the LSTM helps or hurts.**
* **Any statement about real transactions.** Both datasets are synthetic, and
  the v2 generator was designed in this project.
* **Any statement about ring fraud.** The test period has no rings.
* **A threshold.** Thresholds were chosen on 28 validation episodes, the scores
  are uncalibrated, and the same threshold gives very different alert volumes
  on v1 and v2.

## 4. Threshold operating points (not approved, not applied)

| Operating point | v2_dnn_lstm threshold → v2 test alerts per 1,000 / precision / recall | v2_dnn_only threshold → v2 test alerts per 1,000 / precision / recall |
|---|---|---|
| Validation F1 | 0.9593 → 2.32 / 0.184 / 0.148 | 0.7698 → 9.46 / 0.120 / 0.393 |
| Validation FPR 0.1% | 0.9679 → 0.95 / 0.200 / 0.066 | 0.9119 → 1.18 / 0.400 / 0.164 |
| Validation FPR 1% | 0.7677 → 8.04 / 0.147 / 0.410 | 0.7744 → 9.27 / 0.122 / 0.393 |
| Validation FPR 5% | 0.5894 → 48.53 / 0.038 / 0.639 | 0.5765 → 52.27 / 0.035 / 0.639 |

Production has no recorded operating point. The 0.9946 used in §2.2 exists
only in this audit, and it sits just under the 0.999 score cap, which shows
how saturated production's scores are on v2.

## 5. Readiness items from the 4C-2e-e audit

| Item (4C-2e-e) | Status at `582fe60` | Blocks promotion? |
|---|---|---|
| MODEL_SET mechanism, production safety, rollback | Ready | No |
| Back-dated transactions scored with the wrong row (D8) | **Fixed** in 4C-2f-1 | No |
| Cold start: padded LSTM windows, first-transaction placeholders (D8) | **Fixed** in 4C-2f-1 | No |
| Cold start: unusual-hour and unusual-category flags with 1–19 earlier transactions | **Open.** Analysed and pinned by tests; not changed. | Yes, for any model set serving new customers |
| `risk_score` wording, PDF, `/metrics`, frontend (D5) | **Resolved** in 4C-2f-2 | No |
| Model-set provenance per transaction (D6) | **Resolved** in 4C-2f-2 (database). The history endpoint and timeline do not show it. | No (minor) |
| Candidate artifact distribution (D7) | **Resolved**: candidates and v2 evaluation artifacts are tracked in `582fe60` | No |
| Keras version | **Resolved**: `keras>=3.15.0,<3.16` pinned | No |
| Threshold policy (D4) | **Open.** No candidate threshold approved; bands are legacy; scores uncalibrated. | **Yes** |
| Alert volume of the candidates on the live v1 history | **Open** (19–23 legitimate High + Critical per 1,000 under the bands) | **Yes**, until D4 is decided |
| v1 feature-version mismatch in production (D8) | **Open.** Documented; the v1 CSV was not regenerated. | Affects "keep production" as much as promotion |
| Live history: v1 or v2 (D9) | **Open** | **Yes**: the candidates were evaluated on v2 but would serve v1 customers |
| Evaluation evidence (D1–D3) | **Open** (§3) | **Yes** |
| Home location under the 90-day rule | Open (design left it to be confirmed) | No |
| Timezone-aware and naive timestamps mixed for one customer | Open (pre-existing failure) | No |
| `threshold_for_fpr` admits tied scores | Open (documented) | Only if an FPR operating point is adopted |
| SHAP explanations vary between identical calls | Open (pre-existing) | No |

## 6. Unresolved risks and blockers

**Blockers for promoting either candidate**

1. **No approved threshold policy.** Under the live bands either candidate
   raises tens of legitimate High + Critical alerts per 1,000 on both
   datasets.
2. **The evidence does not separate the model sets.** All confidence
   intervals overlap, there are 14 test episodes, one training seed and
   synthetic data.
3. **Evaluation and serving data differ.** The candidates were evaluated on
   v2 and would score v1 customers. No labelled evaluation of candidate scores
   exists for the live customers.
4. **Ring fraud is untested.** No ring is in the test period, and the one
   validation ring was missed entirely by both candidates.
5. **Cold-start residual.** The unusual-hour and unusual-category flags fire on
   55–87% of a customer's second and third transactions. No model saw that in
   training.

**Risks of keeping production as it is**

1. **Legacy training.** Random split, in-sample stacking and no seed, on v1
   data whose labels are trivially separable.
2. **Training/serving feature mismatch.** Production was fitted on legacy-v1
   feature values, and live inference computes current values.
3. **No evaluation of the deployed weights on realistic data** until this
   audit. On the v2 test period it reaches ROC-AUC 0.66 and 54 legitimate
   High + Critical alerts per 1,000 under its own bands.
4. **No manifest.** Production has file checksums only.

**Risks common to every option**

* Scores are shown as percentages but are not calibrated probabilities.
* The ring detector is a shared-device rule with 9% precision on v2.
* All conclusions rest on synthetic data.

## 7. Decision framework (recommendation only)

**This section is a recommendation, not a decision. Nothing here has been
applied.**

The audit suggests treating promotion as a sequence of gates. Each gate is the
project owner's to open.

1. **Decide the threshold policy first (D4).** Until then no model set,
   including production, has a defensible alert volume on v2-like data.
   Options are listed in `docs/step4c2f-2-threshold-and-cold-start-analysis.md`.
2. **Decide which data the live application serves (D9).** A candidate
   evaluated on v2 should be judged on the customers it will score.
3. **Decide what evidence is enough.** With 14 test episodes the current
   intervals cannot rank the model sets. Ways to narrow them, each needing
   approval:
   * a larger or additional generated test period;
   * several training seeds;
   * a test period that contains rings.
4. **Decide the cold-start policy** for customers with 1–19 earlier
   transactions.
5. **Only then compare the options** on criteria fixed in advance, for example:
   * legitimate alerts per 1,000 at the chosen operating point;
   * first-fraud detection;
   * recall at a fixed false-positive rate;
   * explainability of the scores (`risk_score` has a real meaning only with
     an LSTM).
6. **If a candidate is promoted, do it reversibly:** set `MODEL_SET` for a
   trial period with provenance recorded (already in place), compare alert
   volumes against expectation, and keep `production` as the rollback.

On the present evidence, the audit's recommendation is to **keep
`MODEL_SET=production` as the default and promote nothing yet**. That is
because the gates above are open, not because production is shown to be
better: §2 and §6 show it is not.

## 8. Verification

* **Production model files.** The SHA-256 of all 7 files in
  `backend/models/saved/` was identical before and after the audit, in the
  session copy and on the user's machine:

  | File | SHA-256 (first 12) |
  |---|---|
  | `dnn_feature_mean.npy` | `cb12c7e19545` |
  | `dnn_feature_std.npy` | `5f526f065ae0` |
  | `dnn_fraud_model.keras` | `4fd3ab2e190f` |
  | `lstm_feature_mean.npy` | `cd7d3fda6554` |
  | `lstm_feature_std.npy` | `86b57e2ce8b3` |
  | `lstm_risk_model.keras` | `81c2ab738b6d` |
  | `shap_background.npy` | `ac8ffe3b5343` |

* **Candidate artifacts and datasets:** unchanged.
* **Consistency of the audit's numbers.** The audit's candidate figures equal
  `comparison.json` (for example 9/40 and 24/176 at the F1 thresholds, and
  65.27 and 81.26 per 1,000 under the bands). The production band figure
  (55.72 per 1,000) equals the 4C-2e-e audit.
* **`comparison.json`** was recomputed in memory from the saved candidates and
  matched the saved file. Candidate scores are identical to the 4C-2d
  evaluation scores.
* **Tests run** (read-only, relevant to the audit): `tests/test_model_sets.py`
  and `tests/test_observability.py`: 87 passed, 0 failed, 0 skipped.
* **`MODEL_SET`** default is still `production`. Nothing was committed or
  pushed.
