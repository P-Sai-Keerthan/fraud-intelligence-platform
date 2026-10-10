# Step 4C-3B: fraud-alert threshold policy analysis

This is a read-only analysis at commit `582fe60`. It defines three candidate
alert policies and evaluates each one for all three model sets (`production`,
`v2_dnn_lstm`, `v2_dnn_only`), using only saved models and saved scores.

Nothing was changed by this analysis:

* `MODEL_SET` still defaults to `production`;
* no threshold or band in the application was changed;
* no model was retrained, promoted or deployed;
* the production and candidate model files are byte-identical before and
  after (§9).

**This document recommends a policy, not a model.** §6 is a recommendation
only, and every number in it needs the project owner's approval before
anything is applied.

## 1. Ground rules

* **Scores are not probabilities.** Every score is a model output between 0
  and 99.9 from class-weighted training. A cut-off of 77 means "scores of 77
  or more", not "a 77% chance of fraud".
* **The 25/50/80 bands are legacy.** They are what `/predict` applies today,
  but they are not an approved threshold policy and were not derived from any
  evaluation.
* **A policy is defined by an alert budget, not by a score.** Each policy
  below fixes the share of legitimate transactions that may be flagged. The
  score cut-off that delivers it is then read off the v2 validation period,
  separately for each model set. The same budget gives very different
  cut-offs for different model sets (§2).
* **Cut-offs were chosen on validation only.** The test period was used only
  to check them.

**Data**

| Period | Transactions | Fraud transactions | Fraud episodes |
|---|---|---|---|
| v2 validation | 16,315 | 119 | 28 |
| v2 test | 21,142 | 61 | 14 |

Every transaction in both periods has at least 10 earlier transactions. Early
customer history is analysed separately in §5.

**Method notes**

* Candidate scores are the saved 4C-2d evaluation scores. Production scores
  come from the existing production models scoring the same rows, as in 4C-3A.
  There was no training.
* Cut-offs use a tie-safe rule: the lowest score such that the validation
  false-positive rate stays within the budget. They differ from the
  manifests' recorded operating points only in the fourth decimal place,
  because the manifest helper admits tied scores.
* Confidence intervals are 95%, from a customer-level bootstrap on the test
  period (2,000 resamples). First-fraud intervals are Wilson intervals.

## 2. The three policies and their cut-offs

| Policy | Budget: legitimate transactions flagged | production cut-off | v2_dnn_lstm cut-off | v2_dnn_only cut-off |
|---|---|---|---|---|
| A. Very low false-positive | at most 0.1% (about 1 per 1,000) | 99.51 | 96.81 | 91.19 |
| B. Balanced | at most 1% (about 10 per 1,000) | 96.80 | 76.77 | 77.44 |
| C. High recall | at most 5% (about 50 per 1,000) | 55.98 | 58.95 | 57.66 |

Cut-offs are scores on the 0–99.9 scale shown by the application.

* **Production's cut-offs are crowded against the 99.9 cap.** Policies A and
  B sit at 99.5 and 96.8. Its scores on v2 saturate, so small score changes
  move it between policies.
* **No fixed score suits all three model sets.** For example, 80 is inside
  policy B for both candidates but far below policy B for production.

## 3. Results by policy

### Policy A: very low false-positive (budget 0.1%)

| | production | v2_dnn_lstm | v2_dnn_only |
|---|---|---|---|
| **Validation** | | | |
| False-positive rate | 0.099% | 0.099% | 0.099% |
| Recall | 0.109 | 0.118 | 0.076 |
| Precision | 0.448 | 0.467 | 0.360 |
| F1 | 0.176 | 0.188 | 0.125 |
| **Test** | | | |
| Fraud caught (of 61) / false positives | 9 / 20 | 4 / 16 | 10 / 15 |
| Recall | 0.148 | 0.066 | 0.164 |
| Recall 95% CI | 0.054 – 0.250 | 0.000 – 0.137 | 0.020 – 0.306 |
| Precision | 0.310 | 0.200 | 0.400 |
| Precision 95% CI | 0.097 – 0.539 | 0.000 – 0.500 | 0.091 – 0.667 |
| F1 | 0.200 | 0.099 | 0.233 |
| False-positive rate | 0.095% | 0.076% | 0.071% |
| Legitimate alerts per 1,000 | 0.95 | 0.76 | 0.71 |
| First fraud detected (of 14) | 3 | 1 | 2 |
| First-fraud 95% CI | 8% – 48% | 1% – 31% | 4% – 40% |
| Episodes detected (of 14) | 6 | 3 | 4 |

### Policy B: balanced (budget 1%)

| | production | v2_dnn_lstm | v2_dnn_only |
|---|---|---|---|
| **Validation** | | | |
| False-positive rate | 0.99% | 0.99% | 0.99% |
| Recall | 0.193 | 0.336 | 0.345 |
| Precision | 0.125 | 0.199 | 0.203 |
| F1 | 0.152 | 0.250 | 0.256 |
| **Test** | | | |
| Fraud caught (of 61) / false positives | 18 / 189 | 25 / 145 | 24 / 172 |
| Recall | 0.295 | 0.410 | 0.393 |
| Recall 95% CI | 0.111 – 0.476 | 0.128 – 0.659 | 0.133 – 0.638 |
| Precision | 0.087 | 0.147 | 0.122 |
| Precision 95% CI | 0.024 – 0.162 | 0.034 – 0.278 | 0.030 – 0.231 |
| F1 | 0.134 | 0.217 | 0.187 |
| False-positive rate | 0.90% | 0.69% | 0.82% |
| Legitimate alerts per 1,000 | 8.94 | 6.86 | 8.14 |
| First fraud detected (of 14) | 4 | 5 | 5 |
| First-fraud 95% CI | 12% – 55% | 16% – 61% | 16% – 61% |
| Episodes detected (of 14) | 8 | 7 | 7 |

### Policy C: high recall (budget 5%)

| | production | v2_dnn_lstm | v2_dnn_only |
|---|---|---|---|
| **Validation** | | | |
| False-positive rate | 5.0% | 5.0% | 5.0% |
| Recall | 0.420 | 0.714 | 0.563 |
| Precision | 0.058 | 0.095 | 0.077 |
| F1 | 0.102 | 0.168 | 0.135 |
| **Test** | | | |
| Fraud caught (of 61) / false positives | 29 / 1,015 | 39 / 986 | 39 / 1,065 |
| Recall | 0.475 | 0.639 | 0.639 |
| Recall 95% CI | 0.268 – 0.700 | 0.364 – 0.864 | 0.407 – 0.824 |
| Precision | 0.028 | 0.038 | 0.035 |
| Precision 95% CI | 0.010 – 0.050 | 0.013 – 0.066 | 0.014 – 0.060 |
| F1 | 0.053 | 0.072 | 0.067 |
| False-positive rate | 4.8% | 4.7% | 5.1% |
| Legitimate alerts per 1,000 | 48.0 | 46.6 | 50.4 |
| First fraud detected (of 14) | 5 | 9 | 8 |
| First-fraud 95% CI | 16% – 61% | 39% – 84% | 33% – 79% |
| Episodes detected (of 14) | 11 | 11 | 11 |

### What the three tables show

* **The budgets carry over from validation to test.** Realised test
  false-positive rates are 0.07–0.10% for A, 0.69–0.90% for B and 4.7–5.1%
  for C. This holds on v2, where validation and test come from the same
  generator.
* **The cost of recall is steep.** Going from B to C roughly multiplies
  legitimate alerts by six, for about 1.5 times the fraud caught.
* **The intervals overlap for every pair of model sets under every policy.**
  The tables support a choice of policy; they do not support a choice of
  model.

## 4. Critical and High as two tiers

One way to use two policies together: **Critical** is policy A's cut-off,
and **High** is the range between policy B's and policy A's cut-offs. Test
period:

| | production | v2_dnn_lstm | v2_dnn_only |
|---|---|---|---|
| **Critical only** (policy A cut-off) | | | |
| Fraud caught / false positives | 9 / 20 | 4 / 16 | 10 / 15 |
| Precision | 0.310 | 0.200 | 0.400 |
| Legitimate alerts per 1,000 | 0.95 | 0.76 | 0.71 |
| **High only** (between the B and A cut-offs) | | | |
| Fraud caught / false positives | 9 / 169 | 21 / 129 | 14 / 157 |
| Precision | 0.051 | 0.140 | 0.082 |
| Legitimate alerts per 1,000 | 7.99 | 6.10 | 7.43 |
| **High + Critical** (policy B cut-off) | | | |
| Fraud caught / false positives | 18 / 189 | 25 / 145 | 24 / 172 |
| Recall | 0.295 | 0.410 | 0.393 |
| Legitimate alerts per 1,000 | 8.94 | 6.86 | 8.14 |

For comparison, the legacy bands on the same test period (from 4C-3A):

| Legacy bands | production | v2_dnn_lstm | v2_dnn_only |
|---|---|---|---|
| Critical (score ≥ 80): fraud caught / false positives | 25 / 580 | 22 / 112 | 22 / 144 |
| High + Critical (score ≥ 50): fraud caught / false positives | 29 / 1,149 | 45 / 1,335 | 44 / 1,674 |
| High + Critical: legitimate alerts per 1,000 | 54.4 | 63.1 | 79.2 |

The legacy High + Critical band is close to policy C (budget 5%) or looser,
for every model set.

**Alert burden on the live v1 customers** (legitimate alerts per 1,000, all
88,913 full-history windows; volume only, because v1 fraud is trivially
separable and production was trained on it):

| Policy | production | v2_dnn_lstm | v2_dnn_only |
|---|---|---|---|
| A | 0.00 | 0.00 | 0.00 |
| B | 0.00 | 0.11 | 0.04 |
| C | 1.25 | 11.45 | 5.87 |
| Legacy High + Critical | 1.55 | 23.34 | 18.76 |

The same cut-offs flag far fewer legitimate v1 customers than v2 customers.
A budget stated on v2 does not transfer to v1.

## 5. Early customer history: is one policy appropriate?

**No.** The same cut-off produces very different alert rates for customers
with fewer than 10 earlier transactions.

**What was measured**
* Every legitimate transaction in v2 (and in the live v1 data) was scored
  exactly as the live pipeline scores it, including the 4C-2f-1 cold-start
  handling below 10 earlier transactions. A batch replication was checked
  against the real pipeline on 20 early-history rows per model set: identical.
* The table shows legitimate alerts per 1,000 transactions in each
  history-length group.
* The **"flags neutral"** figures are a counterfactual for measurement only:
  the same rows scored with `hour_is_unusual` and `category_is_unusual` set to
  their training average. Nothing in the application was changed.

**v2 customers, policy B (budget: about 10 per 1,000)**

| Earlier transactions | Rows | production | production, flags neutral | v2_dnn_lstm | v2_dnn_lstm, flags neutral | v2_dnn_only | v2_dnn_only, flags neutral |
|---|---|---|---|---|---|---|---|
| 0 | 500 | 0 | — | 0 | — | 0 | — |
| 1 | 500 | 260 | 0 | 40 | 14 | 68 | 20 |
| 2 | 500 | 122 | 0 | 24 | 4 | 30 | 4 |
| 3–9 | 3,500 | 54 | 0 | 16 | 5 | 22 | 8 |
| 10–19 | 5,000 | 15 | 0 | 11 | 10 | 13 | 8 |
| 20 or more | 90,826 | 10 | — | 8 | — | 9 | — |

**v2 customers, policy A (budget: about 1 per 1,000)**

| Earlier transactions | production | v2_dnn_lstm | v2_dnn_only |
|---|---|---|---|
| 0 | 0 | 0 | 0 |
| 1 | 118 | 0 | 8 |
| 2 | 34 | 0 | 4 |
| 3–9 | 11 | 0 | 2 |
| 10–19 | 0.8 | 0.8 | 1.4 |
| 20 or more | 1.1 | 0.7 | 0.8 |

**v2 customers, policy C (budget: about 50 per 1,000)**

| Earlier transactions | production | v2_dnn_lstm | v2_dnn_only |
|---|---|---|---|
| 0 | 0 | 20 | 20 |
| 1 | 822 | 292 | 304 |
| 2 | 666 | 172 | 158 |
| 3–9 | 358 | 99 | 123 |
| 10–19 | 95 | 76 | 68 |
| 20 or more | 60 | 50 | 52 |

**Live v1 customers, policy B**

| Earlier transactions | production | v2_dnn_lstm | v2_dnn_only |
|---|---|---|---|
| 1 | 164 | 24 | 10 |
| 2 | 44 | 8 | 4 |
| 3–9 | 5.1 | 2.3 | 1.1 |
| 10–19 | 0.0 | 0.0 | 0.0 |
| 20 or more | 0.0 | 0.1 | 0.05 |

### Findings

1. **A customer's second and third transactions are flagged far above
   budget.** Under policy B, a v2 customer's second transaction is flagged
   260 times per 1,000 by production, 40 by v2_dnn_lstm and 68 by
   v2_dnn_only, against a budget of 10.
2. **The unusual-hour and unusual-category flags are the main cause.**
   * For production they are the whole cause: with the two flags neutral,
     production's early alerts under policies A and B fall to zero.
   * For the candidates they explain about two-thirds: under policy B the
     second-transaction rate falls from 40 to 14 (v2_dnn_lstm) and from 68 to
     20 (v2_dnn_only).
   * What remains for the candidates comes from other early-history features,
     such as a new device or city and amount ratios against a one-transaction
     average.
3. **From 10 earlier transactions the rates are close to budget** for the
   candidates (11–13 against 10). Production is still elevated at 10–19
   (15), again because of the two flags.
4. **A customer's first transaction is never flagged under A or B**, because
   since 4C-2f-1 its history-relative features are treated as missing. That
   is a property of the cold-start handling, not evidence that first
   transactions are safe.
5. **The benefit side cannot be measured.** Neither dataset contains a single
   fraud transaction with fewer than 10 earlier transactions. So the tables
   show the cost of alerting on early history, and nothing shows what it
   would catch.

## 6. Recommended policy (recommendation only; not applied)

**This is a recommendation for the project owner to accept, change or reject.
It names a policy, not a model, and none of it has been implemented.**

1. **Define alerts by budget, per model set.**
   * Replace fixed score bands with operating points chosen on validation
     data for whichever model set is loaded.
   * Record the points with the model. Candidate manifests already do this;
     production has none recorded.
2. **Maximum legitimate alert volume.**
   * Critical: about 1 legitimate alert per 1,000 transactions (policy A).
   * High + Critical together: about 10 per 1,000 (policy B).
   * These are proposals. The real limit is how many alerts reviewers can
     handle, which only the owner knows.
3. **Critical versus High.**
   * Critical (policy A cut-off): act immediately. Precision on test was
     0.20–0.40.
   * High (between the B and A cut-offs): queue for review. Precision on test
     was 0.05–0.14.
   * Policy C's range: record and display only, with no alert. At about 50
     legitimate alerts per 1,000 it is not reviewable.
4. **Minimum fraud recall target.**
   * Proposed acceptance target at the policy B point: at least 40% of fraud
     transactions and at least one-third of first frauds.
   * The candidates' point estimates on test are near this (0.39–0.41, and 5
     of 14). Production's are below (0.30, and 4 of 14). No interval is tight
     enough to confirm or rule out the target for any model set. The target
     should be confirmed on a larger test set before it is used as a gate.
5. **Cold-start policy: do not use one policy for all customers.**
   * Fewer than 10 earlier transactions: raise model alerts only at the
     Critical point (policy A), and mark everything else "limited history"
     rather than High.
   * Even policy A is not within budget for production on early history (118
     per 1,000 on the second transaction). So this rule is workable only
     together with a decision on the two flags.
   * The two flags: decide separately whether to neutralise
     `hour_is_unusual` and `category_is_unusual` below a history length (10
     or 20). The measured effect is in §5. This is a scoring change and needs
     its own approval and tests.
6. **Do not present scores as probabilities.** Keep the "not a calibrated
   probability" wording, and decide whether to calibrate before any
   percentage is used for decisions.

## 7. What cannot be decided from this evidence

**Because the data is synthetic**
* Whether any budget, cut-off or recall figure holds for real transactions.
  The fraud patterns are the ones the v2 generator was written to produce.
* Whether the false-positive contexts match reality. Foreign travel, big
  purchases and forgotten passwords dominate false positives because the
  generator includes them at chosen rates.
* How budgets transfer between populations. The same cut-offs give about 10
  legitimate alerts per 1,000 on v2 and almost none on v1.

**Because the test period has only 14 fraud episodes**
* Which model set is best under any policy. Every interval overlaps.
* Whether a recall target is met. One episode moves first-fraud detection by
  about 7 percentage points.
* Anything about fraud rings. No ring falls in the test period.
* Precision in the Critical tier. It rests on 4–10 caught fraud transactions
  and 15–20 false positives per model set.

**Because of how the data was built**
* The benefit of alerting on new customers. There are no fraud transactions
  with fewer than 10 earlier transactions in either dataset.
* Stability across training runs. Each candidate was trained with one seed.
* Stability over time. Validation and test are consecutive periods from one
  generator.

## 8. Information and decisions still required before promotion

**Decisions for the project owner**
1. **Alert budget:** how many legitimate alerts per 1,000 transactions
   reviewers can handle, for Critical and for High.
2. **Recall target:** the minimum fraud and first-fraud detection a model set
   must reach at that budget.
3. **Tier meaning:** what Critical and High trigger, and whether the
   policy C range is shown at all.
4. **Cold-start policy:** how customers with fewer than 10 earlier
   transactions are treated, and whether the two flags are neutralised below
   a history length.
5. **Which customers the live application serves** (v1 or v2). The budget
   must be set on that population.
6. **Whether scores are calibrated or relabelled.**

**Evidence still needed**
1. A larger test period or more generated data, so that intervals can
   separate the model sets and confirm a recall target.
2. A test period that contains fraud rings.
3. Fraud cases among new customers, so the cold-start policy has a measured
   benefit.
4. Several training seeds per candidate.
5. Operating points for production recorded with the model, if production is
   to be judged under the same policy.

**Engineering that would follow an approved policy** (not started)
1. Per-model-set operating points applied by `/predict` in place of the fixed
   bands, with tests.
2. The cold-start alert rule.
3. Any change to the two flags.
4. The frontend gauge colours and band labels, which still use 25/50/80.

## 9. Verification

* **Production model files:** all 7 files in `backend/models/saved/` have the
  same SHA-256 before and after (`cb12c7e19545…` through `ac8ffe3b5343…`).
* **Candidate files:** all 15 files under `backend/models/candidates/` are
  unchanged.
* **`MODEL_SET`:** default unchanged (`production`).
* **Thresholds:** none changed. The cut-offs in this document exist only in
  the document.
* **Deployment:** none.
* **Consistency checks**
  * The candidates' policy B and C results equal the operating points saved
    in `comparison.json`, apart from the tie-safe rounding noted in §1.
  * The legacy-band figures equal 4C-3A.
  * The cold-start replication equals the live pipeline on 60 sampled rows.
* **Tests run** (existing, relevant): `tests/test_model_sets.py`,
  `tests/test_observability.py`, `tests/test_cold_start.py` and
  `tests/test_cold_start_sensitivity.py`: 142 passed, 0 failed, 0 skipped.
* Nothing was committed or pushed.
