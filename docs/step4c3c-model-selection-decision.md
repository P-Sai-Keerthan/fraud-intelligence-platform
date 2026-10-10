# Step 4C-3C: model selection decision

This is a decision audit at commit `582fe60`. It compares `production`,
`v2_dnn_lstm` and `v2_dnn_only` against criteria fixed before the comparison,
and makes a provisional recommendation.

Nothing was changed:

* `MODEL_SET` still defaults to `production`;
* no threshold was changed and nothing was deployed;
* no model was retrained;
* production and candidate model files are byte-identical before and after
  (§9).

**No promotion is made or approved by this document** (§7). The recommendation
in §5 is provisional and belongs to the project owner to accept or reject.

Every statement below is labelled as one of three things:

* **Measured:** a number from saved models and saved scores.
* **Recommendation:** a judgement of this audit.
* **Unresolved:** something the evidence cannot settle.

## 1. Inputs

* `docs/step4c3a-production-decision-audit.md` (model-set comparison).
* `docs/step4c3b-threshold-policy-analysis.md` (alert policies, early
  history).
* **New in this step (measured, read-only):** paired confidence intervals for
  the differences between model sets at the Policy B points. They use the same
  customer-level bootstrap as before (2,000 resamples of the 500 test
  customers), applied to the difference between two model sets on the same
  resample.

**Policy B** is the 4C-3B "balanced" policy: at most 1% of legitimate
transactions flagged on validation, about 10 legitimate alerts per 1,000. It
is a proposal and has not been approved. Its cut-offs are scores, not
probabilities.

**Test period:** 21,142 transactions, 61 fraud transactions, 14 fraud
episodes. Every transaction has at least 10 earlier transactions.

## 2. Criteria fixed before selection

| # | Criterion | How it is judged |
|---|---|---|
| 1 | Policy B alert budget | Legitimate alerts per 1,000 on test at the Policy B cut-off: at most 10 |
| 2 | Recall target (proposed) | At least 40% of fraud transactions caught at the Policy B cut-off |
| 3 | First-fraud detection | First fraud transactions of the 14 episodes caught |
| 4 | False-positive burden | False positives at the Policy B cut-off |
| 5 | Precision | At the Policy B cut-off |
| 6 | PR-AUC | Threshold-free, test period |
| 7 | ROC-AUC | Threshold-free, test period |
| 8 | Cold-start behaviour | Legitimate alert rate with fewer than 10 earlier transactions |
| 9 | Complexity and latency | Number of models; median scoring time |
| 10 | Explainability | Whether every stated reason is an observable feature |
| 11 | Provenance and reproducibility | Manifest, checksums, reproducible training |
| 12 | Rollback safety | Can production be restored at once |

## 3. Measured evidence against the criteria

| # | Criterion | production | v2_dnn_lstm | v2_dnn_only |
|---|---|---|---|---|
| 1 | Legitimate alerts per 1,000 (budget 10) | 8.9 | 6.9 | 8.1 |
| | 95% CI | 7.6 – 10.4 | 4.4 – 9.6 | 5.8 – 10.8 |
| 2 | Recall (target 0.40) | 0.295 | 0.410 | 0.393 |
| | 95% CI | 0.111 – 0.476 | 0.128 – 0.659 | 0.133 – 0.638 |
| | Fraud caught (of 61) | 18 | 25 | 24 |
| 3 | First fraud caught (of 14) | 4 | 5 | 5 |
| | 95% CI (Wilson) | 12% – 55% | 16% – 61% | 16% – 61% |
| 4 | False positives (of 21,081) | 189 | 145 | 172 |
| 5 | Precision | 0.087 | 0.147 | 0.122 |
| | 95% CI | 0.024 – 0.162 | 0.034 – 0.278 | 0.030 – 0.231 |
| 6 | PR-AUC | 0.128 | 0.104 | 0.128 |
| | 95% CI | 0.038 – 0.251 | 0.024 – 0.274 | 0.033 – 0.309 |
| 7 | ROC-AUC | 0.663 | 0.899 | 0.897 |
| | 95% CI | 0.466 – 0.854 | 0.803 – 0.969 | 0.821 – 0.960 |
| 8 | Legitimate alerts per 1,000 on a customer's 2nd / 3rd / 4th–10th transaction (v2, Policy B) | 260 / 122 / 54 | 40 / 24 / 16 | 68 / 30 / 22 |
| 9 | Models loaded; median latency | 2 (LSTM + DNN); ~270 ms | 2 (LSTM + DNN); ~263 ms | 1 (DNN); ~208 ms |
| 10 | Reasons are observable features | no: `risk_score` (an LSTM output) can be a reason | no: same | yes: all 9 inputs are observable features |
| 11 | Manifest, weight checksums, seeded reproducible training | no (file checksums only; legacy unseeded training) | yes | yes |
| 12 | Rollback to production | — | restart without `MODEL_SET`; tested | same; tested |

**Other measured facts that bear on the choice**

* **Stricter cut-off (Policy A, about 1 legitimate alert per 1,000).**
  v2_dnn_lstm catches 4 of 61 fraud transactions with 16 false positives.
  v2_dnn_only catches 10 with 15. Production catches 9 with 20.
* **Looser cut-off (Policy C, about 50 per 1,000).** Both candidates catch 39
  of 61. Production catches 29.
* **LSTM contribution.** At their own F1 thresholds, v2_dnn_lstm caught no
  fraud transaction, episode or first fraud that v2_dnn_only missed. On the
  customer-grouped split the PR-AUC difference had the opposite sign to the
  time split.
* **Features.** Production was trained on legacy-v1 feature values while live
  scoring computes current values. The candidates' training and live features
  are the same.
* **Cold start.** With fewer than 10 earlier transactions the LSTM is not
  run, for production and v2_dnn_lstm alike. In that range v2_dnn_lstm behaves
  as a DNN with an average risk input.

### Paired differences at the Policy B points (new, measured)

A paired interval that excludes zero means the two model sets differ on that
measure in this test set. 18 comparisons were made, so one or two could
exclude zero by chance alone.

| Difference | v2_dnn_lstm − production | v2_dnn_only − production | v2_dnn_lstm − v2_dnn_only |
|---|---|---|---|
| ROC-AUC | **+0.236** (0.065 to 0.418) | **+0.234** (0.075 to 0.406) | +0.003 (−0.028 to 0.035) |
| PR-AUC | −0.023 (−0.115 to 0.108) | +0.001 (−0.100 to 0.149) | −0.024 (−0.114 to 0.034) |
| Recall | +0.115 (−0.049 to 0.246) | +0.098 (−0.043 to 0.212) | +0.016 (−0.111 to 0.149) |
| Legitimate alerts per 1,000 | −2.08 (−4.42 to 0.42) | −0.80 (−2.96 to 1.66) | **−1.28** (−2.28 to −0.32) |
| Precision | +0.060 (0.000 to 0.137) | +0.036 (−0.007 to 0.088) | +0.025 (−0.018 to 0.071) |
| First-fraud detection | +0.071 (−0.177 to 0.333) | +0.071 (−0.167 to 0.333) | 0.000 (−0.215 to 0.200) |

Bold marks intervals that exclude zero.

* **Both candidates rank v2 transactions better than production** (ROC-AUC).
  This is the only clear separation from production. It is expected:
  production was trained on v1, not v2.
* **v2_dnn_lstm raises about 1.3 fewer legitimate alerts per 1,000 than
  v2_dnn_only** at the Policy B cut-offs. Both cut-offs were set to the same
  validation budget, so this measures how each cut-off carried over to the
  test period, not which model ranks better.
* **Nothing else is separated.** Recall, first-fraud detection and PR-AUC do
  not distinguish any pair. No statistical superiority is claimed for any
  model set.

## 4. Reading the criteria

| # | Criterion | Reading (point estimates; "not separated" unless stated) |
|---|---|---|
| 1 | Budget | All three are within the budget. v2_dnn_lstm is lowest; lower than v2_dnn_only by a paired interval that excludes zero. |
| 2 | Recall target | v2_dnn_lstm meets 0.40 by point estimate (25 of 61). v2_dnn_only misses it by one transaction (24). Production is below (18). No interval confirms or rules out the target for any of them. |
| 3 | First fraud | Candidates 5 of 14, production 4. Not separated. |
| 4 | False positives | v2_dnn_lstm 145, v2_dnn_only 172, production 189. |
| 5 | Precision | v2_dnn_lstm highest. Not separated between the candidates. |
| 6 | PR-AUC | v2_dnn_only and production 0.128, v2_dnn_lstm 0.104. Not separated. |
| 7 | ROC-AUC | Candidates far above production (separated). Candidates tied. |
| 8 | Cold start | Every model set exceeds the budget on early history. Production is worst by a wide margin; v2_dnn_lstm is lowest. |
| 9 | Complexity and latency | v2_dnn_only is simplest and fastest. |
| 10 | Explainability | v2_dnn_only is the only one whose reasons are all observable features. With an LSTM, `risk_score` is a meaningful second signal on screen; with v2_dnn_only it repeats the fraud score. |
| 11 | Provenance | Candidates equal; production lacks it. |
| 12 | Rollback | Equal for both candidates. |

## 5. Provisional recommendation

**This section is a recommendation. It is provisional, it is not a promotion,
and it does not claim that any model set is statistically better than
another.**

### 5.1 Direction: a v2 candidate rather than the current production model

The audit recommends that the eventual production model be one of the v2
candidates, once the gates in §6 are met. The reasons are measured, but most
are not about detection performance:

* production ranks v2 transactions clearly worse (ROC-AUC, separated);
* its early-history alert rate is several times the candidates' (260 against
  40 and 68 per 1,000 on a second transaction);
* it was trained on feature values that live scoring no longer produces;
* it has no manifest and its training is not reproducible.

On detection at Policy B (recall, first fraud, PR-AUC), production is not
shown to be worse.

### 5.2 Provisional lead candidate: `v2_dnn_lstm`, with `v2_dnn_only` as the required challenger

Under the criteria fixed in §2, `v2_dnn_lstm` has the better point estimates
at Policy B:

* it is the only model set that meets the proposed 40% recall target by point
  estimate while staying inside the budget;
* it has the fewest false positives and the lowest legitimate alert rate (the
  one candidate-versus-candidate difference whose paired interval excludes
  zero);
* it has the highest precision and the lowest early-history alert rate.

**Why prefer it to `v2_dnn_only` without statistical separation**

* The preference rests on the pre-agreed criteria and on point estimates. It
  is a narrow lead: one fraud transaction in recall and 27 false positives.
* It keeps `risk_score` meaningful. The product, the PDF and the frontend
  describe a behavioural risk signal. With `v2_dnn_only` that field only
  repeats the fraud score.
* Being wrong costs little in detection. If the LSTM adds nothing,
  `v2_dnn_lstm` performs like `v2_dnn_only` at the price of a second model
  and about 55 ms.

**What argues the other way (measured)**

* **The LSTM's contribution is not established.** It caught nothing that
  `v2_dnn_only` missed, and the PR-AUC difference changes sign between the two
  splits.
* **`v2_dnn_lstm` is weaker at the strict cut-off.** Under Policy A it caught
  4 fraud transactions against 10. In the two-tier policy proposed in 4C-3B,
  that cut-off is the Critical tier.
* **`v2_dnn_only` is simpler.** It has one model, is faster, has no LSTM to
  skip for new customers, and gives reasons that are all observable features.
* **Its PR-AUC point estimate is higher** (0.128 against 0.104).

**Tie-break rule (recommendation).** `v2_dnn_lstm` stays the lead only if the
larger validation in gate 4 shows it at least matching `v2_dnn_only` at
Policy B on recall and legitimate alerts, and not worse in the Critical tier.
If it does not, choose `v2_dnn_only` on simplicity. If the two remain
indistinguishable, the simpler model should win.

### 5.3 What remains unresolved

* Whether either candidate is better than the other, or than production, at
  detecting fraud. The intervals on recall, first fraud and PR-AUC overlap for
  every pair.
* Whether the 40% recall target is met by any model set.
* Whether the 1.3-per-1,000 alert difference between the candidates would
  repeat on other data. It is one of 18 comparisons.
* Whether any of this holds on real transactions, on the live v1 customers,
  for ring fraud, or for new customers. The data is synthetic; the test period
  has 14 episodes and no rings; no fraud exists with fewer than 10 earlier
  transactions.
* Whether the result is stable across training runs. Each candidate was
  trained with one seed.

## 6. Promotion gates

`MODEL_SET` should not change until every gate is satisfied. The acceptance
figures marked "proposed" are recommendations for the owner to set.

| Gate | What must be true | Status today |
|---|---|---|
| 1. Threshold policy approval | The owner approves an alert policy: the budget, the tiers and what each tier triggers. The approved cut-offs for the chosen model set are recorded with the model. `/predict` applies them in place of the legacy 25/50/80 bands, with tests. | **Open.** Policy B is a proposal. Nothing is implemented. Production has no recorded operating points. |
| 2. New-customer policy | The owner decides how customers with fewer than 10 earlier transactions are alerted, and whether `hour_is_unusual` and `category_is_unusual` are neutralised below a history length. The decision is implemented and tested. | **Open.** Under Policy B the candidates flag 16–68 legitimate early transactions per 1,000 against a budget of 10. |
| 3. Data-version decision | The owner decides whether the live application serves v1 or v2 customers. The alert budget is confirmed on that population. | **Open.** The candidates were evaluated on v2 and would score v1 customers, where the same cut-offs flag almost no legitimate transactions. |
| 4. Larger or more realistic validation | A test set large enough to judge the criteria. Proposed: at least 100 fraud episodes; at least 5 training seeds per candidate; at Policy B, legitimate alerts within budget and recall of at least 40% with the lower confidence bound above production's point estimate. Real or externally sourced data if available. | **Open.** 14 episodes, one seed, synthetic data. |
| 5. Ring-fraud evidence | A test period containing fraud rings, with ring detection measured for the chosen model set and for the shared-device rule. | **Open.** No ring in the test period; both candidates missed the one validation ring; the rule has 9% precision on v2. |
| 6. Production rollback plan | A written plan covering: the trigger for rollback (for example alert volume above budget for a set period); who decides; the command (`MODEL_SET` unset or `production`, then restart); how rows scored by each model set are told apart afterwards. | **Partly ready.** The mechanism works and was tested end to end (restart in about 5 s; identical production scores; provenance stored per row). The written plan and triggers do not exist. |
| 7. Final regression test | At the promotion commit: the full backend suite passes; frontend lint and build pass; end-to-end smoke test of the selected model set and of rollback; SHA-256 of `backend/models/saved/` and of the selected candidate verified against the recorded values. | **Not yet applicable.** To be run at the promotion commit. The suite currently passes in full (426 tests at `582fe60`). |

## 7. NO PROMOTION YET

**No model set is promoted, and none should be on the present evidence.**

* Gates 1 to 5 are open. Gate 6 is only partly ready. Gate 7 can be run only
  at a promotion commit.
* `MODEL_SET` remains `production`.
* The provisional lead in §5 is a direction for the next phases. It is not a
  decision to deploy `v2_dnn_lstm`.
* Keeping production for now is not an endorsement of it. §5.1 lists why it
  should eventually be replaced.

## 8. Decisions required from the project owner

1. Accept, change or reject the provisional lead (`v2_dnn_lstm`, with
   `v2_dnn_only` as challenger) and the tie-break rule.
2. Approve an alert policy (gate 1): the budget, the tiers and the recall
   target.
3. Decide the new-customer policy (gate 2).
4. Decide the live data version (gate 3).
5. Approve how more evidence is produced (gates 4 and 5): more generated
   data, more seeds, a ring test period, or real data. Each involves
   generating data or training, which this phase did not do.
6. Approve a rollback plan (gate 6).

## 9. Verification

* **Production model files:** all 7 files in `backend/models/saved/` have the
  same SHA-256 before and after (`cb12c7e19545…` through `ac8ffe3b5343…`).
* **Candidate artifacts:** all 15 files under `backend/models/candidates/`
  unchanged.
* **`MODEL_SET`:** default still `production`.
* **Thresholds:** none changed; the 25/50/80 bands in the code are untouched.
* **Deployment:** none. No model was retrained.
* **HEAD:** `582fe60`.
* **Consistency:** the point estimates in §3 equal those in 4C-3A and 4C-3B.
* **Tests run** (existing, relevant, read-only): `tests/test_model_sets.py`
  and `tests/test_observability.py`: 87 passed, 0 failed, 0 skipped.
* Nothing was committed or pushed.
