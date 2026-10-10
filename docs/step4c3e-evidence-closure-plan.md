# Step 4C-3E preconditions: evidence closure plan

This is a plan, written at commit `582fe60`. It defines the evaluation and
data-generation work needed to close the evidence gaps listed in
`docs/step4c3d-promotion-readiness.md`.

**Nothing in the repository was changed by this step except this document.**

* `MODEL_SET` still defaults to `production`. No model was promoted.
* No model was retrained and nothing was deployed.
* No dataset in the repository was generated or modified.
* No code was changed, so no implementation exists yet (§8 explains why).

The document separates five things throughout:

| Label | Meaning |
|---|---|
| **Current evidence** | What has been measured so far |
| **Missing evidence** | What the gates need and does not exist |
| **Proposed experiment** | What this plan would run, once approved |
| **Acceptance criteria** | What the result must show |
| **Final promotion gate** | Which 4C-3D checklist item it closes |

## 1. Summary of the plan

| Gap (4C-3D) | Proposed way to close it | Needs new code? | Needs training? |
|---|---|---|---|
| Fewer than 100 fraud episodes | Score the existing saved models on 5 fresh, independently seeded v2 datasets used purely as hold-out | Evaluation harness only | No |
| One seed | The same 5 datasets (data seeds). Optionally, 5 training seeds per candidate. | Harness; small training wrapper for the optional part | Only for the optional part (candidates, never production) |
| No ring fraud in test | The 5 hold-out datasets contain 20 rings | Harness | No |
| No new-customer fraud | A generator extension that places fraud in a customer's first 10 transactions | Generator change (off by default) | No |
| Decision made after seeing results | The decision rule in §6, fixed in this document before any model is scored | — | — |

**The key idea.** The candidates were trained on the seed-42 dataset only. A
dataset generated with a different seed has different customers and different
fraud, so **all of it** is out of sample for the saved candidates. No
retraining is needed to get a large test set.

## 2. Larger validation

### Current evidence

* The v2 test period has 14 fraud episodes and 61 fraud transactions.
* One dataset (seed 42) and one training seed per candidate.
* Every confidence interval on recall, first-fraud detection and PR-AUC
  overlaps between model sets (4C-3A, 4C-3C).

### Missing evidence

* At least 100 fraud episodes.
* At least 5 independent seeds.
* Intervals narrow enough to apply the decision rule.

### Proposed experiment

**Hold-out datasets**

* Generate 5 datasets with the existing generator, unchanged (version
  2.0.1): `python data/v2/generate.py --seed <s> --customers 500 --out <dir>`.
* **Seeds: 101, 102, 103, 104, 105.** They are fixed here, before any model is
  scored on them.
* **Location:** a new directory outside the tracked seed-42 data, proposed as
  `data/v2_holdout/seed_<s>/`.
* **Tracking:** each dataset is about 100,000 transactions. They are
  byte-reproducible from the seed, so the proposal is to keep them untracked
  and record each one's manifest and SHA-256. Tracking them is the owner's
  choice.
* The seed-42 dataset, the saved split and every model file stay untouched.

**What these seeds contain.** As a feasibility check, the raw transactions
(without features) for these five seeds were generated into the session
scratchpad and counted. No model was scored on them, and nothing was written
to the repository.

| Seed | Transactions | Fraud episodes | Episodes from 2026-06-01 | Rings | Ring episodes (victims) | Ring episodes from 2026-06-01 | Fraud transactions with fewer than 10 earlier |
|---|---|---|---|---|---|---|---|
| 101 | 101,033 | 110 | 16 | 4 | 16 | 0 | 0 |
| 102 | 104,638 | 110 | 30 | 4 | 16 | 6 | 2 |
| 103 | 100,485 | 110 | 25 | 4 | 16 | 3 | 0 |
| 104 | 101,636 | 110 | 24 | 4 | 16 | 7 | 0 |
| 105 | 102,238 | 110 | 26 | 4 | 16 | 4 | 0 |
| **Total** | **510,030** | **550** | **121** | **20** | **80** | **20** | **2** |

Raw generation took about 8 seconds per seed. Building features takes about
75–90 seconds per seed. This is not a huge dataset.

**Evaluation populations** (both fixed now)

* **Primary:** every transaction with at least 10 earlier transactions, over
  the whole 176 days. This gives about 550 fraud episodes.
* **Secondary (time-consistent):** the same, restricted to 2026-06-01 onward,
  which is the calendar period of the original test split. This gives 121
  episodes.

**Splitting and grouping**

* Nothing is fitted on the hold-out data, so it needs no train/validation
  split. Cut-offs come from the seed-42 validation period and are already
  fixed (§6).
* If anything is ever fitted on hold-out data (for example calibration), it
  must use the existing group-aware split: by customer, with ring members and
  households kept together (`app/evaluation/split.py`).

**Uncertainty**

* Keep the customer-level bootstrap (2,000 resamples).
* Two refinements, fixed now:
  * resample within each seed, so every resample has five datasets;
  * resample whole groups (a ring's victims, or a household) rather than
    single customers, so a ring is never split.
* Report each seed separately as well as pooled. The spread between seeds is
  itself evidence about stability.
* Use paired differences between model sets on the same resample, as in
  4C-3C.

**Optional second stage: training seeds** (needs separate approval)

* Train each candidate with 5 training seeds on the seed-42 data, using the
  existing candidate recipe, and score each on the hold-out.
* This measures how much results move between training runs.
* It trains candidates only. Production is never retrained. The existing
  candidate artifacts are not replaced: new runs go to a separate directory.

### Acceptance criteria

* At least 100 fraud episodes in the primary population (expected: about 550)
  and in the secondary population (expected: 121).
* All 5 seeds scored, with per-seed and pooled results.
* Every metric in §5 reported with a 95% interval.

### Final promotion gate

Closes "at least 100 fraud episodes", "at least 5 seeds",
"candidate-versus-candidate uncertainty analysis" and
"production-versus-candidate analysis" in the 4C-3D checklist.

## 3. Ring fraud

### Current evidence

* No ring falls in the seed-42 test period.
* The one validation ring (3 victims, 4 fraud transactions) was missed by
  both candidates.
* The shared-device rule flags 55 devices on seed 42, of which 5 belong to
  rings.

### Missing evidence

Any out-of-sample measurement of ring detection.

### Proposed experiment

* The 5 hold-out datasets contain 20 rings with 80 victim episodes, all out
  of sample for the saved models. 20 of those episodes fall in the secondary
  (June onward) population.
* Report three levels separately, per model set, at the fixed cut-offs:

  | Level | Measure |
  |---|---|
  | Ring | Rings with at least one victim alerted; rings with at least two victims alerted; time from the ring's first fraud to the first alert |
  | Victim (episode) | Victim episodes detected; first fraud of each victim detected |
  | Transaction | Ring fraud transactions caught, against non-ring fraud transactions caught |

* Report the shared-device rule separately: devices flagged, how many belong
  to rings, and how many rings it finds. It is the same for every model set.

### Acceptance criteria

* At least 20 rings in the primary population (expected: 20).
* Ring-level and transaction-level results reported separately, with
  intervals (Wilson for ring counts).
* **No claim of ring superiority** unless a paired interval between model
  sets excludes zero. With 20 rings this is unlikely, and the report must say
  so.

### Final promotion gate

Closes "ring fraud represented".

## 4. New-customer fraud

### Current evidence

* Neither v1 nor seed-42 v2 has any fraud with fewer than 10 earlier
  transactions.
* The five hold-out seeds have 2 such fraud transactions in total. That is
  too few to measure anything.
* The cost side is measured: under Policy B, legitimate early transactions
  are flagged at 16–68 per 1,000 by the candidates and 54–260 by production
  (4C-3B).

### Missing evidence

Fraud among new customers, so the proposed cold-start policy has a measured
benefit as well as a measured cost.

### Proposed experiment

**A generator extension (version 2.1.0), off by default**

* The current generator starts every customer on day 0 and places no fraud
  before day 21. So a customer almost always has more than 10 transactions
  before any fraud.
* Add two settings, both 0 by default:
  * a share of customers who join later in the period;
  * a number of fraud episodes placed within a customer's first 10
    transactions.
* With both at 0, the output must be byte-identical to version 2.0.1. This is
  a required test.
* Record the number of earlier transactions at first fraud in the episode
  ground truth. Ground-truth columns must never become model features; the
  existing guard covers this.

**Datasets**

* 5 seeds (proposed: 201–205), with a target of at least 100 new-customer
  fraud episodes in total.
* Kept separate from the hold-out in §2, so the main comparison is not mixed
  with data from a changed generator.

**What is evaluated, on transactions with fewer than 10 earlier**

| Variant | Description |
|---|---|
| Current behaviour | The live cold-start handling, with alerts at the Policy B cut-off |
| Proposed policy | Limited-history state: alerts only at the Critical cut-off |
| Measurement only | The proposed policy with the unusual-hour and unusual-category flags neutral |

For each variant and each model set, report:
* the false-positive rate and legitimate alerts per 1,000;
* fraud transactions caught and first frauds caught;
* results by number of earlier transactions: 0, 1, 2, 3–9.

**A limit that must be stated in the report.** The new-customer fraud
patterns will be ones written into the generator for this purpose. The result
will show how the policy behaves on those patterns. It will not show that the
policy works on real new-customer fraud.

### Acceptance criteria

* The generator's default output is byte-identical to version 2.0.1.
* At least 100 new-customer fraud episodes across the 5 seeds.
* False-positive rate and fraud detection reported for this population
  separately from customers with full history.

### Final promotion gate

Closes "new-customer fraud represented" and gives the evidence needed to
decide "new-customer policy approved".

## 5. Model comparison

**Model sets:** `production`, `v2_dnn_lstm`, `v2_dnn_only`, using the saved
files as they are.

**Cut-offs:** the Policy B and Critical cut-offs already measured on the
seed-42 validation period (§6). They are not re-tuned on hold-out data.

**Reported for each model set, per seed and pooled**

| Metric | At which cut-off | Interval |
|---|---|---|
| Legitimate alerts per 1,000 | Policy B | Bootstrap |
| False positives | Policy B | Count |
| Precision, recall, F1 | Policy B | Bootstrap |
| PR-AUC, ROC-AUC | Threshold-free | Bootstrap |
| First-fraud detection | Policy B | Wilson and bootstrap |
| Critical-tier detection: fraud caught, false positives, precision | Critical | Bootstrap |
| Ring-fraud detection (§3) | Policy B | Wilson |
| New-customer detection (§4) | Per variant | Bootstrap |
| Latency: median and 95th percentile scoring time | — | Repeated runs |

* **Paired differences** between each pair of model sets, with intervals, on
  every metric above.
* **Latency** is measured as in 4C-2e-e: each model set in its own process,
  the same sequence of scorings, warm-up excluded.

### Final promotion gate

Closes "candidate-versus-candidate uncertainty analysis" and
"production-versus-candidate analysis".

## 6. Pre-registered decision rule

**This rule is fixed here, before any model is scored on the hold-out data.**
The winner is not chosen after seeing the results.

**Fixed inputs**

| Input | Value |
|---|---|
| Hold-out data seeds | 101–105 (new-customer seeds 201–205) |
| Primary population | Transactions with at least 10 earlier, all 5 seeds pooled |
| Policy | Policy B, with the Critical tier at the Policy A point |
| Cut-offs (score, 0–99.9 scale; not probabilities) | See the table below |
| Uncertainty | Group-level bootstrap within seed, 2,000 resamples, 95% intervals |

| Tier | production | v2_dnn_lstm | v2_dnn_only |
|---|---|---|---|
| Critical | 99.51 | 96.81 | 91.19 |
| High + Critical (Policy B) | 96.80 | 76.77 | 77.44 |

**Primary endpoints** (three only; everything else is descriptive)

1. Recall at the Policy B cut-off.
2. Legitimate alerts per 1,000 at the Policy B cut-off.
3. Fraud caught at the Critical cut-off (Critical-tier recall).

**The rule between the candidates** (the existing provisional rule, made
precise)

`v2_dnn_lstm` remains the lead only if all three hold, using the paired
difference `v2_dnn_lstm − v2_dnn_only`:

| Condition | Proposed test |
|---|---|
| At least matches on recall at Policy B | The lower bound of the 95% interval is above −0.05 |
| At least matches on alert burden at Policy B | The upper bound of the 95% interval is below +1.0 legitimate alert per 1,000 |
| Not worse in the Critical tier | The lower bound of the 95% interval for Critical-tier recall is above −0.05 |

* If any condition fails, **`v2_dnn_only` wins on simplicity.**
* If all three hold but nothing favours either candidate, `v2_dnn_lstm`
  stays the lead under the existing provisional rule. The owner may still
  prefer the simpler model.
* The margins (0.05, 1.0 per 1,000, 0.05) are proposals. **The owner must
  approve or change them before the run.**

**The rule against production** (from 4C-3C gate 4)

A candidate is eligible for promotion on this evidence only if, at Policy B
on the primary population:

* its legitimate alerts per 1,000 are within the approved budget (the upper
  bound of the interval at or below the budget); and
* its recall is at least 0.40, with the lower bound of its interval above
  production's point estimate.

If no candidate meets this, there is no promotion and `production` stays.

**Claims**

* "Better" or "superior" may be used only where a paired 95% interval
  excludes zero on a primary endpoint.
* Differences on other metrics are reported as descriptive.
* With three primary endpoints and three pairs of model sets, there are nine
  primary comparisons. The report must state that one in twenty such
  intervals excludes zero by chance.

## 7. Data version

**This plan does not switch the live data.** The live application continues
to score the v1 customers.

**The unresolved choice**

| Option | What it implies |
|---|---|
| Keep live v1 | The candidates would serve a population they were not evaluated on. The same cut-offs flag almost no legitimate v1 transactions, so the budget would have to be re-set on v1. v1 has no realistic labelled fraud to check recall against. |
| Migrate or align live behaviour with v2 | The live history, the shared-device ring output and the home-device results all change. Several v1-specific tests change. This is a separate project from switching models. |

* The hold-out experiment in this plan evaluates the models on v2-like data.
  It can close the evidence gates. It cannot decide the data version.
* The owner must choose and document the data version before promotion (the
  4C-3D data version gate).

## 8. Implementation design (not implemented)

Nothing was implemented in this step. The work needs a new evaluation module
and a generator change, which is more than a small, safe edit. Per the
instruction for this phase, the step stops at the plan, the design and the
tests that would validate it.

**Proposed files**

| File | Purpose |
|---|---|
| `backend/app/evaluation/holdout.py` | Score a generated dataset with the three saved model sets at fixed cut-offs; compute §5 metrics per seed and pooled; write a JSON report |
| `backend/app/evaluation/ring_metrics.py` | Ring-level, victim-level and transaction-level detection (§3) |
| `backend/tests/test_holdout.py` | Tests below |
| `data/v2/synth_v2/` (generator 2.1.0) | The two off-by-default settings for new-customer fraud (§4) |
| `docs/step4c3e-evidence-report.md` | The results, written after the run |

**Tests the implementation must pass**

* The harness never writes to `backend/models/saved/` or
  `backend/models/candidates/`; checksums are compared before and after.
* The harness refuses a dataset whose seed is 42, or whose SHA-256 equals the
  training dataset's.
* Cut-offs are read from one fixed table and cannot be changed by the data
  being scored.
* Scoring matches the live pipeline on sampled rows, as verified for 4C-3B.
* The bootstrap is deterministic (fixed seed) and keeps ring members and
  households together.
* The ring metrics are correct on a small hand-built example.
* The report is byte-identical on a second run.
* The generator's default output is byte-identical to version 2.0.1.
* No ground-truth column can become a model feature (existing guard).

**Expected run time:** about 10 minutes to generate the 5 hold-out datasets
with features, and a few minutes to score them. The optional training stage
adds about 4.5 minutes per candidate training run.

## 9. Approvals needed before any of this runs

1. The plan itself, including seeds 101–105 and the two populations.
2. The decision-rule margins in §6.
3. Writing the evaluation harness and its tests.
4. Generating the 5 hold-out datasets, and where they are stored.
5. The generator extension for new-customer fraud.
6. The optional training-seed stage (candidates only).

## 10. State after this step

* **File created:** `docs/step4c3e-evidence-closure-plan.md`. No other file
  in the repository was created or changed.
* **Feasibility data:** five raw datasets were generated in the session
  scratchpad to count episodes and rings. They are not in the repository.
* **`MODEL_SET`:** `production`.
* **Promotion:** none.
* **Models:** none retrained; production and candidate files unchanged.
* **Thresholds:** unchanged.
* **Tests:** none run, because no code was changed.
* **HEAD:** `582fe60`. Nothing was committed or pushed.
