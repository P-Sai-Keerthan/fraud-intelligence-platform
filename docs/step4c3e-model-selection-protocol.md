# Step 4C-3E.6: pre-registered model-selection protocol

Written at commit `582fe60` (nothing committed). Protocol version:
`4C-3E.6 v1`.

**NO PROMOTION.** This document defines rules. It selects nothing.

* `MODEL_SET` remains `production`.
* No production artifact was changed (`backend/models/saved/`).
* No seed-42 candidate artifact was changed (`backend/models/candidates/v2/`).
* No model was trained. No dataset was generated. No model was scored.
* Nothing was deployed. Nothing was committed or pushed.

Scores are model scores. They are not calibrated probabilities.

## 1. Why a protocol is needed

* The multi-seed evidence (`docs/step4c3e-multiseed-training-report.md`)
  shows that results depend strongly on the training run.
* The per-seed results on the existing hold-out (data seeds 101–105 and
  201–205) are already known. Choosing the best seed from them would be
  selection on the hold-out, and its result would be optimistic.
* So the rule for choosing an artifact must be fixed **before** any new data
  is generated or scored, and the final test must use data nobody has seen.

**Disclosure.** The author of this rule has seen every per-seed result on the
existing hold-out, and each model's recall on the validation period. Neither
is an input to the artifact rule. The rule's only inputs are development
datasets that **do not exist yet** (§3), so its outcome is unknown at the
time the rule is fixed.

## 2. Three separate decisions

| Stage | Decision | Evidence it may use | Status |
|---|---|---|---|
| A | Which **architecture** leads | The multi-seed comparison on the existing hold-out, by the existing rule | Already decided: `v2_dnn_lstm` |
| B | Which **trained artifact** represents each architecture | Development datasets only (§3) | Rule fixed here; not executed |
| C | **Final evaluation** of the selected artifacts | A fresh, untouched hold-out, scored once | Rule fixed here; data not generated |

## 3. Which data may be used for what

| Data | Seeds | May be used for | May never be used for |
|---|---|---|---|
| Training dataset, training period | 42 | Training (done) | Anything else |
| Training dataset, validation period | 42 | Each model's cut-offs, by the 4C-3B rule | Choosing a seed |
| Existing hold-out | 101–105 | Stage A (architecture); already reported | **Choosing a seed.** Final evaluation |
| Existing new-customer hold-out | 201–205 | Already reported; description only | **Choosing a seed.** Final evaluation |
| **Development datasets** (new) | **301–305** | **Stage B: choosing a seed** | Final evaluation |
| **Final hold-out** (new) | **401–405** | **Stage C only**, scored once | Everything before the selection record exists |
| **Final new-customer hold-out** (new) | **411–415** | **Stage C only**, scored once | Everything before the selection record exists |

* Development datasets: the default generator (version 2.0.1), 500 customers,
  under `data/v2_holdout/development/seed_<seed>/`. Untracked.
* Final hold-out: the default generator, 500 customers, under
  `data/v2_holdout/final/seed_<seed>/`. Final new-customer hold-out: the
  generator extension with the same settings as seeds 201–205 (20% late
  joiners, 30 new-customer episodes each). Untracked.
* The final hold-out **must remain ungenerated** until the stage B selection
  record has been written. The code refuses to write a selection record if a
  final hold-out directory already exists.
* The code refuses seeds 42, 101–105, 201–205, 401–405 and 411–415 as
  selection data, with no override.

**Why new development data, and not the validation period.** On the
validation period every model has exactly 9.87 legitimate alerts per 1,000 at
Policy B, because that is where its cut-off is set. The validation period
therefore cannot show how a model's alert burden behaves on new data. It also
holds only 119 fraud transactions. The generator can supply independent data
at no cost to the final test, as long as the seeds are kept apart.

## 4. Eligible models and metrics

| Item | Value |
|---|---|
| Eligible architectures | `v2_dnn_lstm`, `v2_dnn_only` |
| Eligible training seeds | 11, 12, 13, 14, 15 |
| Not eligible | The saved seed-42 candidates; any model trained after this protocol |
| Population | Primary: transactions with at least 10 earlier transactions |
| Cut-offs | Each model's own, by the 4C-3B validation rule (Policy B within 1%, Critical within 0.1%). Frozen; never re-tuned on development or final data |

| Role | Metric |
|---|---|
| **Primary metric** | Recall at the Policy B cut-off |
| Primary endpoints of the tie-break | Policy B recall; legitimate alerts per 1,000 at Policy B; Critical-tier recall |
| **Secondary metrics** | First-fraud detection; episodes with any alert; Critical-tier legitimate alerts per 1,000; precision; ring detection (ring, victim, transaction level); results by fraud type; PR-AUC and ROC-AUC; new-customer results; latency |

**No training seed is added.** If the five seeds do not yield an eligible
artifact, the answer is "none" (§8). Training more seeds until one passes is
not allowed under this protocol.

## 5. Stage A: architecture (the existing rule, unchanged)

`v2_dnn_lstm` remains the lead only if, for `v2_dnn_lstm − v2_dnn_only`:

| Condition | Test |
|---|---|
| Recall at Policy B | Lower bound of the 95% interval above −0.05 |
| Legitimate alerts per 1,000 at Policy B | Upper bound of the 95% interval below +1.0 |
| Critical-tier recall | Lower bound of the 95% interval above −0.05 |

Otherwise `v2_dnn_only` wins on simplicity.

**Outcome on the multi-seed evidence** (difference of the architectures'
means; interval with data and training-seed variation):

| Condition | Difference | Interval | Holds |
|---|---|---|---|
| Recall | +0.092 | [+0.029, +0.157] | Yes |
| Legitimate alerts per 1,000 | −0.22 | [−0.93, +0.53] | Yes |
| Critical-tier recall | +0.027 | [−0.015, +0.077] | Yes |

* **Lead: `v2_dnn_lstm`. Required challenger: `v2_dnn_only`.**
* Stage A is not re-opened by stage B. It is checked once more, between the
  two selected artifacts, on the final hold-out (§7).

## 6. Stage B: the artifact rule

Applied to each architecture separately. It is deterministic: no randomness,
no judgement, the same answer on every machine.

**Inputs.** For each training seed, five numbers measured on the five
development datasets pooled (primary population, the model's own cut-offs):

* recall at Policy B;
* legitimate alerts per 1,000 at Policy B;
* Critical-tier recall;
* first-fraud detection at Policy B;
* share of episodes with any alert at Policy B.

Values are taken at 6 decimals.

**Step 1: screens.** A seed is eligible only if all three hold.

| Screen | Test | Where the number comes from |
|---|---|---|
| Recall | Recall at Policy B ≥ 0.40 | The approved minimum recall (4C-3C, 4C-3E) |
| Alert burden | Legitimate alerts per 1,000 at Policy B ≤ 11.0 | The approved target of 10, plus the approved alert margin of 1.0 |
| Critical tier | Critical-tier recall ≥ (median of the five seeds − 0.05) | The approved Critical-tier margin |

**Step 2: robustness to training randomness.** The architecture passes
stage B only if **at least 3 of the 5** seeds are eligible. If fewer pass,
the architecture fails, even if one seed looks excellent.

**Step 3: select the most typical eligible seed.**

1. For each of the five metrics, take the **median** and the **standard
   deviation** (n − 1) over all five seeds of the architecture.
2. For each seed, compute its distance: the sum over the five metrics of
   |value − median| ÷ standard deviation. A metric whose standard deviation
   is 0 contributes 0. The distance is rounded to 6 decimals.
3. **Select the eligible seed with the smallest distance.**
4. Ties: the seed whose recall is closest to the median recall; then the
   lowest seed number.

The selected artifact is the directory
`backend/models/candidates_multiseed/v2/seed_<seed>/<architecture>/`. Its
weight checksums, file checksums and cut-offs are written into a selection
record. The record is written once and cannot be overwritten.

**What the rule does not do**

* It never ranks seeds by recall and never picks the maximum. Raising one
  seed's recall moves it away from the median and makes it less likely to be
  selected (tested).
* It does not read the existing hold-out or the final hold-out.

**Why this rule**

| Choice | Reason |
|---|---|
| The most typical run, not the best | The best of five on any dataset is partly luck, and its score falls on new data. A typical run's expected result on new data is the architecture's average, which is what stage A compared. |
| Median as the centre | One unusual run does not move it. |
| Five metrics with equal weight, each divided by its spread | The artifact should be typical on detection, alert burden, the Critical tier and early detection at once. Dividing by the spread puts the metrics on one scale without choosing weights. |
| Majority requirement | If most runs of an architecture miss the minimum, a passing run is the exception, and retraining would probably not reproduce it. |
| Screens before typicality | A typical run of a weak architecture must not go forward. |

**Minimum acceptable alert burden: what the 11.0 screen means.** It is a
screen against systematic over-alerting. It is **not** a claim that the
budget is met. On five pooled datasets, dataset-to-dataset noise in the alert
rate is roughly 0.4 per 1,000 (measured spread between datasets of about 1.0,
divided by √5). A pooled value above 11.0 is therefore unlikely to be noise.
The budget test itself is in stage C and is unchanged.

**Treatment of first-fraud detection.** It is secondary evidence.

* It is one of the five metrics that define "typical", so the selected
  artifact is typical of its architecture on first-fraud detection too.
* It is never a screen and never a target. No seed is preferred because its
  first-fraud detection is high.
* The known trade-off stays visible: on the multi-seed evidence
  `v2_dnn_only` detects the first fraud more often (−0.110 [−0.194, −0.021]
  for `v2_dnn_lstm − v2_dnn_only`). Stage C must report it with its interval
  (§7), and the decision record must state it in words.

**What goes forward**

| Stage B result | Goes to stage C |
|---|---|
| Both architectures pass | The `v2_dnn_lstm` artifact as primary candidate and the `v2_dnn_only` artifact as required challenger |
| Only one passes | That artifact alone |
| Neither passes | Nothing (§8) |

## 7. Stage C: the fresh hold-out

**Requirements**

1. Seeds 401–405 (and 411–415 for new-customer fraud). Generated only after
   the selection record exists.
2. The same harness, populations and bootstrap as the existing hold-out
   report: primary population; secondary population from 2026-06-01; 2,000
   resamples of customer groups within each dataset.
3. Size: at least 100 fraud episodes in each population and at least 20
   rings. With five default datasets this is met by construction (about 550
   episodes and 20 rings).
4. **Models scored: the selected artifact or artifacts, and `production`.**
   The other trained seeds and the saved seed-42 candidates may be scored
   afterwards for description only, after the decision has been recorded.
5. Cut-offs: exactly those in the selection record.
6. Scored **once**. Every metric in §4 is reported, whatever it shows.

**Decision rule on the final hold-out** (primary population, pooled)

*Against production (the approved 4C-3E rule, unchanged).* An artifact is
eligible only if all three hold:

| Check | Test |
|---|---|
| Alert budget | Upper bound of the 95% interval of legitimate alerts per 1,000 ≤ 10.0 |
| Recall | Recall at Policy B ≥ 0.40 |
| Better than production | Lower bound of the recall interval above production's point estimate |

*Between the two artifacts (the tie-break of §5).* If both are eligible, the
three conditions of §5 are applied to the difference between the two selected
artifacts. All hold: the `v2_dnn_lstm` artifact. Any fails: the
`v2_dnn_only` artifact, on simplicity. If one is eligible, that one. If none,
no artifact is eligible.

*Critical-tier requirement.* The third tie-break condition. In addition the
report must give each artifact's Critical-tier recall and Critical-tier
legitimate alerts per 1,000 against the Critical budget of about 1 per 1,000,
with intervals.

*Secondary evidence that must be reported, with intervals.* First-fraud
detection; episodes with any alert; ring detection; results by fraud type;
the alert rate in each of the five datasets and how many exceed 10; early
transactions (fewer than 10 earlier) on seeds 411–415 under the current
cold-start behaviour and under any new-customer policy approved before the
data is generated; latency.

**What a pass means.** "Eligible for a controlled-promotion decision." It is
not a promotion. The open items in §10 must still be closed.

**Training-seed variability in the final result.** One artifact per
architecture is tested. The final report must say so and must quote the
spread between training runs from the multi-seed report (for example a
standard deviation of about 0.05–0.06 in Policy B recall). A difference
between the two artifacts is a difference between two trained models. The
architecture comparison remains the one in §5.

## 8. If things fail

| Event | Consequence |
|---|---|
| An architecture fails stage B | It is not carried forward. The other one proceeds alone. |
| Neither architecture passes stage B | No artifact is selected. No final hold-out is generated. `production` stays. |
| The selected artifact is not eligible in stage C | No promotion. **No second seed is tried on the same hold-out.** |
| Neither artifact is eligible in stage C | No promotion. `production` stays. |
| Any later attempt | Needs a new written protocol and a new, unused set of final seeds. |

## 9. The alert budget is not guaranteed

This protocol does not assume the Policy B budget holds on new data.

| Observed (multi-seed report) | Value |
|---|---|
| Legitimate alerts per 1,000 on the validation period | 9.87 for every model, by construction |
| The same cut-offs on new data (primary population) | 7.5 to 13.2 across training seeds and datasets |
| Cells over 10 per 1,000 | 18 of 50 for each architecture |
| Early transactions (fewer than 10 earlier), current behaviour | 15 to 82 per 1,000 across training runs |

**What follows**

* **Stage B** screens out seeds that over-alert systematically (above 11.0
  pooled). It cannot remove variation between datasets.
* **Stage C** keeps the approved test (upper bound ≤ 10.0). A typical
  artifact has a real chance of failing it through dataset variation alone:
  on seeds 201–205, 6 of the 10 trained models were above 10 pooled. If that
  happens the result is "not eligible", as written.
* **Early transactions** are outside the primary population. This protocol
  does not control their alert burden. It only requires that it is reported.
  Promotion stays blocked until a new-customer policy is approved.

**Decisions the owner may take before any data is generated.** They are not
part of this protocol unless added in writing first (§11):

| Option | Effect |
|---|---|
| Keep the test as it is | The budget remains a hard test; failure through dataset variation is accepted as a real finding |
| Approve a band | For example "meets the target" at an upper bound ≤ 10 and "acceptable" at a pooled value ≤ 11. Easier to pass; a weaker guarantee |
| Calibrate cut-offs on pooled development data | Removes the dependence on one validation period. It does not remove variation between datasets. It changes the cut-off rule, so the cut-offs must then be frozen before stage C |

## 10. What stays open after a pass

* Threshold policy and alert budget approved by the owner.
* New-customer (cold-start) policy approved.
* Live data version (v1 or v2) chosen.
* Rollback triggers approved.
* The selected artifact wired to a model-set name, with a final regression
  run. Today the application cannot load a multi-seed directory at all.

## 11. No changes after the fact

1. Nothing in §3–§8 may be changed after the first development dataset has
   been generated, except by a written, dated amendment made **before** the
   stage it affects has produced any number.
2. Nothing may be changed after the final hold-out has been generated.
3. No metric, population, margin, cut-off, screen or tie-break may be added,
   dropped or redefined after final results are seen.
4. No artifact other than the selected ones may be substituted.
5. If a deviation happens anyway, the final result is reported as
   exploratory, not confirmatory, and cannot support a promotion decision.

## 12. Implementation and tests

| File | Purpose |
|---|---|
| `backend/app/evaluation/selection.py` | The rule in executable form: the fixed seeds, the data guard, `select_seed`, the stage B outcome, the one-time selection record |
| `backend/tests/test_selection.py` | 28 tests: fixed inputs and disjoint seed sets; the data guard; the rule against an independent re-statement of it on hand-built and random inputs; screens and boundaries; the majority requirement; ties; order independence; that the best-scoring seed is not selected for being best; the record is written once and never after a final hold-out exists; an end-to-end run on a small sample with quickly trained models |

* The command `python -m app.evaluation.selection --generate` would generate
  the development datasets and write the selection record. **It was not
  run.** It needs approval.
* Stage C has no code yet beyond the existing hold-out harness.
* No existing file was changed.

## 13. State

| Item | State |
|---|---|
| Promotion | **NO PROMOTION** |
| `MODEL_SET` | `production` |
| `backend/models/saved/` | unchanged |
| `backend/models/candidates/v2/` (seed 42) | unchanged |
| `backend/models/candidates_multiseed/` | unchanged; no seed selected |
| Development datasets (301–305) | not generated |
| Final hold-out (401–405, 411–415) | not generated, not scored |
| Deployment | none |
| Commit or push | none; HEAD `582fe60` |
