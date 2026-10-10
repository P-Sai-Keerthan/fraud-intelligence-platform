# Experiment 2 — reviewed protocol (SEQ-EARLY v1, **DRAFT, awaiting approval, not frozen**)

Supersedes `PROTOCOL_DRAFT.md` (v0, kept unchanged for the audit trail). Nothing here has been run on any
model. Items marked **[PENDING-ABLATION]** need the completed `lstm_value_ablation` files; **[USER]** are
decisions that belong to the project owner. Evidence tags: **[code]** read in `origin/Keerthan`;
**[measured]** infrastructure measurement on scratch seed 9 or random tensors (never a result about fraud);
**[simulated]** Monte-Carlo of paired outcomes (no data, no model); **[source]** a cited web source.

## 0. What changed from v0, and the evidence for each change

| # | v0 problem | v1 change | Evidence |
|---|---|---|---|
| 1 | One freeze before tuning; the *outcome* of model selection was not frozen before final data existed | **Two-stage freeze.** Freeze-1 = protocol + code + search spaces + tests. Freeze-2 = selected configurations, T\*, trained-model SHA-256, cut-offs, written **before any final dataset is generated** | design review; v0 §14 |
| 2 | "Customer split 60/40" | Existing public `build_customer_split`, fractions **60/20/20** (constants fixed in `split.py`); the last 20 % is a DEV hold-out used only for smoke tests | [code] `split.CUSTOMER_SPLIT_FRACTIONS = (0.6, 0.2, 0.2)`; v0 named a private function |
| 3 | Primary Δ undefined across 5 training seeds | Per event, hit = mean over the 5 seed-models of the alert indicator; Δ = difference of those means. CI is conditional on the trained models (training-data variability is **not** in it; stated as a limitation) | design review |
| 4 | A1 "reuse exact settings" conflicted with "tuned with the same budget" | Two rows: **A1-frozen** (the ablation's settings, replication, context only) and **A1-tuned** (same 40-trial budget). T\* ranges over {A1-tuned, A2, A3} | internal conflict in v0 §3/§6 |
| 5 | LAG-K only loosely defined | LAG-K ≡ RAW-SEQ flattened (identical per-step fields), so A3 and S2 have identical information | parity requirement |
| 6 | Forbidden list omitted ids | Add `customer_id`, `transaction_id`, and **raw id strings**; new **ID-relabelling invariance** test | [measured] on scratch seed 9: `corr(transaction_id, customer index) = 0.9999`; device-id *format* `DEV_dddd_[AB]` covers 85 % of legitimate but 31 % of fraud rows |
| 7 | Cut-off "tie-safe" assumed equal FPR across arms | **Validation-fitted randomised cut-off** (alert if score > t, or score = t with probability q; draw = hash of `transaction_id`) | [simulated] tie-heavy RF-like scores: existing `threshold_for_fpr` and `tie_safe_cutoff` realise 0.37–0.67 % when asked for 1 %; randomised cut-off realises 1.01 % (val) / 0.99 % (independent sample) |
| 8 | Lead time started at `precursor_start` | Lead time starts at the **first observable `credential_attack` login event**; denominators count only episodes with an opportunity; chance baseline and a trivial-rule reference are mandatory | [measured] scratch seed 9: median 20 h (max 99 h) between `precursor_start` and the first observable event; 64 of 95 window transactions follow it; 58 % of those already have `failed_logins_24h > 0` |
| 9 | Decision rule: "LB > 0 **and** Δ̂ ≥ 0.05" | Three-way rule (§9) | [simulated] v0 rule power at true Δ = 0.05 was **0.52** (it needs Δ̂ ≥ δ), 1.00 at 0.08 |
| 10 | Power table assumed discordance *d* without a source | Assumptions listed; *d* is **estimated on DEV validation at the Freeze-2 gate** (needs tuned models), with a pre-registered rule if it is too high (§10) | [simulated], analytic table |
| 11 | Search space too costly | Negative-sampling ratio {10, 50} for **all** arms (drop "all"); GRU search cap 12 epochs; budget-extension rule | [measured]+model: v0 space ≈ **41 h** of tuning, v1 ≈ **8 h** (§13) |
| 12 | A3 memory | A3 restricted to ratios {10, 50}; stop if RSS > 13 GB | [measured] LightGBM 200 k × 877 features peaked at 3.4 GB; linear extrapolation to 750 k rows ≈ 13 GB |
| 13 | Generator B timing unspecified | B's specification and code are frozen and committed **before FINAL-A is scored**; B-FINAL generated after B's own freeze; fixed wording limits | design review |
| 14 | Reuse of `metrics.episode_metrics` | Not reused as is: its docstring calls first-fraud recall an "early-warning rate" and its delay is measured after the first fraud. Wording lint test added | [code] `evaluation/metrics.py` |
| 15 | Generator determinism assumed | Verified: seed 42 regenerated here (Python 3.13, numpy 2.5.3, pandas 3.0.6) reproduces **all five** committed SHA-256 hashes (manifest: Python 3.10, numpy 2.2.6, pandas 2.3.3) | [measured] |
| 16 | Statistics unchecked | Cluster-bootstrap coverage, false-claim rate and power checked by simulation (§10) | [simulated] |

## 1. Question and claims

**Question.** With the same information and a comparable tuning budget, does a jointly trained sequence
classifier detect the **first fraudulent transaction of an episode** more often than the strongest tabular
alternative at the same false-positive budget?

* Allowed wording: "detection of the first fraudulent transaction *when it is attempted*".
* **Forbidden wording** (enforced by a lint test, §12): "early warning", "predicts fraud before it occurs",
  "pre-fraud prediction" anywhere except in the lead-time section, and there only with the §11 definitions.
  Generator-B results: never "bank", "real-world", "production performance".
* Recall on `subsequent` transactions is reported but is not evidence of early detection.
* Prior, stated before any result: on Generator A a sequence advantage is **unlikely**, because legitimate
  behaviour is almost order-free [code]. A win must come from the data, not the design.

## 2. Information contract at the instant transaction *i* of customer *c* is scored

Allowed: raw fields of *i* (timestamp, amount, merchant_category, device_id, location, `failed_logins_24h`) and the
same fields of *c*'s strictly earlier rows (canonical order `customer_id, timestamp, transaction_id`), and statistics
computed only from them. Scalers/encoders are fitted on DEV-train only.

Forbidden: `is_fraud` of any row; the nine metadata columns; **`customer_id`, `transaction_id`, and any parse of an id
string** (device-id format and the numeric part of `transaction_id` carry customer identity, order, and, for device
format, a fraud-correlated generator artefact); `login_failures.csv`, `episodes.csv`, `customers.csv`; other
customers' rows (no ring or household features); any later row. Device and location enter only as
customer-relative flags ("seen before", "is modal", "share of past rows"). The home city is the modal past
location, never `HOME_LOCATIONS`. **Documented exemption:** F9's `is_foreign_location` reads location *names*
by production design [code]; the ID-relabelling test therefore applies to AGG and RAW-SEQ, not F9.

## 3. Arms

| Id | Learner | Input | Role |
|---|---|---|---|
| R0 | rule | `failed_logins_24h` (score = the count) | Trivial reference; needed because login failures already carry much of the warning signal |
| A0 | RF | F9 | Floor |
| A1-frozen | RF | F9 + AGG | The ablation's RF with **its own settings**, no retuning **[PENDING-ABLATION]**; context only |
| A1-tuned | RF | F9 + AGG | Same features, same 40-trial budget as the others |
| A2 | LightGBM | F9 + AGG | |
| A3 | LightGBM | F9 + AGG + LAG-K | |
| S1 | GRU | RAW-SEQ | Information-restricted sequence arm (secondary) |
| S2 | GRU + MLP head | RAW-SEQ **and** F9 + AGG into the head | Same information as A3 (primary sequence arm) |

F9 = the nine production features from the unchanged `build_point_features`. AGG = past-only window aggregates
**[PENDING-ABLATION]** (fallback sketch in v0 §3, labelled "re-implemented"). RAW-SEQ step = log1p(amount),
log gap, hour and weekday sin/cos, category one-hot, device-seen/modal flags, location-seen/modal flags,
log1p(`failed_logins_24h`), is-current flag, padding mask; *K* = 32, current transaction last. Windows are built
lazily from per-customer arrays; the N×K×D tensor is never stored.

**T\*** = the tabular arm among {A1-tuned, A2, A3} with the highest mean E_val over its 5 refit seeds on DEV
validation; if the best two are within 0.005, the simpler one (A1 < A2 < A3). **Primary contrast: S2 vs T\*.**
T\* is the maximum of three noisy scores while S2 is a single arm, which is conservative against the sequence
model; this is stated in the report.

Framework: PyTorch for S1/S2 (measured 1.6–2.8× faster training than TF-CPU, §13); scikit-learn RF; LightGBM.
Disk-constrained fallback: TensorFlow-CPU 2.21.0 / Keras 3.15 (the project's own pins), same architectures.

## 4. Data and splits

Generator A = the existing v2 generator at the commit recorded in `FREEZE.json`, default configuration except
`n_customers = 2000`. **[measured, scratch seed 9]** one such dataset: 416,056 transactions, 1,904 fraud rows,
440 episodes (79 with a warning period, 425 warning-period transactions), 15 rings.

| Role | Seeds | Use |
|---|---|---|
| Scratch | 9, 42 | Timing and determinism only. Never reported. |
| DEV | 7101, 7102, 7103 | Component split 60 / 20 / 20: **train** / **validation** (tuning, E_val, cut-offs, T\*) / **DEV hold-out** (smoke tests only, never for selection) |
| FINAL-A | 8101 … 8105 | Generated only after Freeze-2. All rows are test. Scored once. No retraining. Thresholds from DEV validation |
| Reserved extension | 8106 … 8110 | Only if the §10 pre-freeze check demands more events |

All arms are scored on the identical `transaction_id` set, cold-start rows included; rows with < 10 earlier
transactions form a separate stratum. Expected sizes: DEV-train ≈ 749 k rows / 3.4 k fraud rows; DEV validation
≈ 250 k rows / ≈ 260 first-fraud events; FINAL-A ≈ 2,200 first-fraud events.

## 5. Seeds (check against the ablation's seeds **[PENDING-ABLATION]**)

Disjoint from `SPENT_SEEDS` 42, 101–105, 201–205, 401–405, 411–415 and 301–305, 501–505, 511–515.
DEV 7101–7103 · FINAL-A 8101–8105 (+8106–8110 reserved) · Generator B DEV/FINAL 7201–7203 / 8201–8205 ·
configuration sampling 9001 (same sample index for every arm) · final refits 31–35 · bootstrap 9101.
**[measured]** at review time no data exist for any reserved seed.

## 6. Tuning: equal budgets, one selection metric

* **E_val** = PR-AUC on DEV-validation rows ∈ {legitimate} ∪ {first-fraud}; subsequent-fraud rows are dropped.
  It drives early stopping, configuration choice, and T\* choice. During GRU search only, early stopping may use a
  25 % negative subsample with weight 4 (`average_precision` with sample weights); the final E_val is on the full set.
* **40 random configurations per arm** (A0: 20; R0: none), one training seed per trial, sampled with seed 9001.
  Best configuration refit with seeds 31–35. Budget is trial count; wall-clock is recorded.
* **Budget-extension rule:** if in any arm the best trial is among its last 10, add 20 trials to **every** arm.
  Best-so-far curves are published for every arm.
* Search spaces (frozen verbatim in `configs/search_spaces.json`) — negative ratio ∈ {10, 50} for every arm:

| Arm | Space |
|---|---|
| RF (A0, A1-tuned) | n_estimators {300, 600}; max_depth {None, 12, 20}; min_samples_leaf {1, 3, 5, 10, 20}; max_features {sqrt, 0.3, 0.5}; class_weight {None, balanced_subsample} |
| LightGBM (A2, A3) | num_leaves {15, 31, 63}; learning_rate logU[0.02, 0.2]; min_child_samples {10, 20, 50, 100}; feature_fraction U[0.5, 1]; bagging_fraction U[0.6, 1]; reg_lambda logU[1e-3, 10]; max_bin 63; positive weight {1, √ratio, ratio}; rounds by early stopping on E_val (cap 2,000) |
| GRU (S1, S2) | hidden {32, 64, 128}; layers {1, 2}; dropout {0, 0.1, 0.3}; lr logU[1e-3, 1e-2]; weight decay {0, 1e-5, 1e-4}; batch {256, 512}; search-phase cap 12 epochs; final refit cap 40 epochs with patience 5 |

* Pre-specified secondary: training-size curve (25 / 50 / 100 % of DEV-train components) for T\* and S2, fixed
  hyper-parameters, scored on DEV validation only.

## 7. Operating points and cut-offs

Per arm, fitted on DEV validation only: (a) **FPR = 1 %** (primary), (b) FPR = 0.1 %, (c) F1-maximising threshold.
For (a) and (b) use the **randomised cut-off**: (t, q) with t the (⌊target·n⌋+1)-th largest legitimate score and
q chosen so the expected validation FPR equals the target; at test time a row is alerted if score > t, or if
score = t and its hash-derived draw < q. Cut-off functions never receive test labels. The realised test FPR of each
arm is reported beside its recall. Recall-vs-FPR curves are descriptive only.

## 8. Endpoints

**Primary (one):** first-fraud recall at FPR = 1 %, S2 minus T\*, Δ pooled over FINAL-A (events weighted equally),
paired stratified cluster bootstrap (2,000 repetitions, resampling customer components within each dataset, seed 9101).

**Secondary (CIs, not tested):** first-fraud recall at 0.1 % and at the F1 cut-off; PR-AUC (all rows) and first-fraud
PR-AUC; ROC-AUC; precision/recall/F1; confusion matrices; legitimate false alerts per 1,000; overall transaction
recall; episode detection rate (*includes later transactions*); recall by position (1st, 2nd, 3rd+); delay after
first fraud (not lead time); cold-start stratum; per-seed and per-dataset tables; R0 and A1-frozen for context.

**Exploratory (labelled):** recall by fraud type (7 archetypes, no tests); second episodes of repeat victims; DEV
time-split sensitivity; permuted-history control; ablation without `failed_logins_24h`.

## 9. Decision rule (Generator A, FINAL-A only; δ = 0.05 **[USER]**)

* **Superiority detected:** lower bound of the 95 % CI of Δ > 0.
* **Meaningful advantage:** superiority detected **and** Δ̂ ≥ δ **and** realised FPR of S2 ≤ 1.25 × that of T\*
  **and** Δ > 0 in ≥ 4 of 5 datasets.
* **No meaningful advantage:** upper bound of the CI < δ (includes "tabular is better").
* Otherwise **inconclusive**. Secondary and exploratory results cannot rescue a failed primary.

Operating characteristics **[simulated]**, 5 datasets × 440 events, discordance ≈ 0.25, 300 simulations each. "Meaningful"
here is only the conjunction (LB > 0 and Δ̂ ≥ δ); the FPR guard and the 4-of-5 check were **not** simulated, so true
power for the full rule is somewhat lower:

| true Δ | P(superiority) | P(meaningful) | P(no meaningful) |
|---|---|---|---|
| 0 | 0.023 | 0.000 | 1.000 |
| 0.03 | 0.830 | 0.047 | 0.510 |
| 0.05 | 0.993 | 0.520 | 0.030 |
| 0.08 | 1.000 | 1.000 | 0.000 |
| 0.10 | 1.000 | 1.000 | 0.000 |

Reading: a true Δ exactly at δ is called "meaningful" only about half the time because the rule requires Δ̂ ≥ δ.
If you want ≥ 80 % power *at* δ, set the meaningful bar at Δ̂ ≥ 0.03 or use LB ≥ 0 only **[USER]**.

## 10. Power and sample-size justification

Assumptions (all explicit): paired outcomes per first-fraud event; discordance *d* = share of events where exactly one
model alerts; 15 % of events in rings of 3–6, ~20 % in repeat-victim or household pairs/triples; shared cluster effect
(latent ρ = 0.3). *d* has **no empirical source yet**; the table covers 0.15–0.40.

* Analytic (v0): Δ = 0.05 needs 469 / 783 / 1,254 events at *d* = 0.15 / 0.25 / 0.40.
* **[simulated]** with N = 2,200: SD(Δ̂) ≈ 0.0105, CI half-width ≈ 0.020; cluster-bootstrap coverage 93.0–96.3 % across
  scenarios (nominal 95 %, Monte-Carlo SE ≈ 1.3 points); false-claim rate at Δ = 0 is 2.3 %. N = 1,320 (3 datasets):
  P(superiority | Δ = 0.05) = 0.94. N = 440 (1 dataset): 0.57.
* In this simulation the cluster and naive bootstraps were equally wide, because a cluster effect shared by both models
  cancels in the paired difference. The cluster bootstrap stays as the safe default: it is correct if the models differ
  systematically within rings.
* **Why 5 FINAL datasets:** 3 would suffice at *d* ≈ 0.25; 5 keep power ≥ 0.95 up to *d* = 0.40 (analytic), support the
  ≥ 4-of-5 consistency check, and give ≈ 220–440 events per archetype for exploratory intervals (half-width ≈ 0.05–0.07 at p = 0.5).
* **Power gate (part of Freeze-2, so before any final data exists):** with the tuned, refit models, estimate *d* and the Δ of
  S2 vs T\* on **DEV validation** (never on final data). If the implied required N for Δ = 0.05 exceeds 80 % of 2,200,
  add FINAL seeds 8106–8110 before Freeze-2 is committed. The estimate itself is noisy (≈ 260 validation events), so it is
  recorded with its interval.
* Limitations: the CI conditions on the trained models and one DEV realisation; δ is a proposal.

## 11. Lead time (separate task; never Task A)

* Task A models are **not** evaluated for lead time; alerts on warning-period rows are false positives there.
* **Warning window** for episode *e*: customer's legitimate rows in `[t_signal, first_fraud_time)` where `t_signal` is the
  first `credential_attack` event of that customer in `[precursor_start, first_fraud_time)` (`login_failures.csv`,
  evaluation use only). `fraud_session` events are excluded. Episodes with no row in the window have **no opportunity**
  and are excluded from the denominator; the count is reported.
* Task B (exploratory): model trained with window rows labelled positive. That label uses generator ground truth and
  **cannot exist in production**. Report episodes with ≥ 1 alert, lead time = first fraud time − first alert,
  the **chance baseline** 1 − (1 − FPR)^|W_e|, and R0 (login-failure rule) at the same FPR.
* Generator B's explicit behavioural prodrome is the only place a stronger lead-time claim could be tested, and then only
  as a designed-assumption test.

## 12. Generalisation

1. **Generator B (synthetic generalisation test, not evidence of banking performance).** Independent specification
   (sequential legitimate behaviour; fraud with dynamics and a behavioural prodrome; regime changes that must not
   alert). Frozen and committed **before FINAL-A is scored**; B-FINAL generated only after B's freeze; models re-tuned
   with the same budget on B-DEV; the §9 rule applied separately; never pooled with A. Stated limits: it encodes
   sequential structure on purpose; its author is also the model builder.
2. **Public data.** Not downloadable here (egress policy: Kaggle, Hugging Face, UCI, Zenodo, figshare, Mendeley,
   download.pytorch.org return 403). ULB `creditcard.csv` has no entity id [source]. IEEE-CIS: real, but no user id
   (`card1` proxy), relative `TransactionDT`, chargeback labels, no episodes or warning periods [source]; at most
   first-fraud-per-proxy-entity recall, reported separately, never as lead time or platform validation. TabFormer and the
   Sparkov-based set are synthetic/simulated. **[USER]** supply files or allow the hosts.

## 13. Runtime and disk (measured where stated)

**[measured]** scratch seed 9, one core: generation 30 s (2,000 customers), `build_point_features` 441 s, write 6 s, peak
728 MB, 136 MB of CSV per dataset. Random-tensor benchmarks, 3 threads:

| GRU (K = 32, step 26, context 45) | PyTorch 2.14.1 train / infer samples/s | TF-CPU 2.21.0 train / infer samples/s |
|---|---|---|
| h32 × 1 | 22,487 / 65,499 | 13,904 / 27,714 |
| h64 × 1 | 11,186 / 38,338 | 6,257 / 19,010 |
| h128 × 1 | 6,780 / 18,021 | 3,182 / 9,766 |
| h64 × 2 | 5,645 / 22,655 | 2,982 / 10,435 |
| h128 × 2 | 3,325 / 9,397 | 1,191 / 4,387 |

Batch-1 latency 1.0–1.8 ms (torch) vs 3.4–7.4 ms (TF). RF: 0.35 / 0.83 / 1.99 s per tree at 100 k / 200 k / 400 k rows.
LightGBM: 0.024 s/round (200 k × 45) and 0.349 s/round (200 k × 877). **Modelled** tuning wall-clock (3 threads, sequential):
v0 space 41 h; v1 (ratios {10, 50}, 12-epoch cap, 40 trials) 8.3 h; lean (30 trials) 6.2 h. These are model outputs from the
micro-benchmarks (mean over sampled hyper-parameters, assuming 400 boosting rounds), **not** end-to-end timings; final refits
(5 seeds × 6 arms) are not yet estimated and will be measured in a DEV-only pilot after approval. Generator B repeats it all. Disk: PyTorch venv 5.8 GB (CUDA wheels), TF-CPU venv 1.8 GB; datasets 136 MB each (A+B ≈ 2.2 GB);
nothing large is cached. Free disk at review time ≈ 19 GB.

## 14. Leakage and integrity tests (all must pass before any final data is generated)

L1 future-perturbation (bit-identical) · L2 label-independence · L3 forbidden-column assertion · L4 strict ordering and
tie handling · L5 F9 equals production features on a sample · L6 split integrity and seed allow-list · L7 cut-off
provenance (poisoned test labels) · L8 fit-on-train-only · L9 identical `transaction_id` set across arms · L10 exactly one
`first` row per episode · L11 determinism · L12 permuted-label sanity · L13 final-access ledger · **L14 ID-relabelling
invariance** (random bijection of device_id/location strings leaves AGG and RAW-SEQ identical) · **L15 `customer_id` /
`transaction_id` absent from every input** · **L16 randomised cut-off hits the target FPR on tie-heavy scores and is
deterministic** · **L17 wording lint** · **L18 Freeze-2 hash check** (model artifacts and configs match before FINAL
generation) · **L19 regenerated seed 42 matches the committed manifest** · **L20 lazy windows equal materialised windows**.

## 15. Freeze procedure, layout, deviations

Freeze-1 commit (`PROTOCOL.md`, `FREEZE.json` with SHA-256 of protocol, code, configs, tests; tag
`seq-early-protocol-v1`) → DEV tuning, refit, T\* → Freeze-2 commit (`FREEZE2.json`: configurations, T\*, model hashes,
cut-offs; tag `seq-early-selection-v1`) → generate FINAL-A → score once → report. Final scripts refuse to run unless both
tags are ancestors of HEAD and hashes match. Anything changed later is a "Deviation" in `REPORT.md`.
Layout as in v0 §14.

## 16. Open items

[PENDING-ABLATION] files, aggregate definition, seeds · [USER] base-branch plan, δ, primary FPR, IEEE-CIS, Generator B
acceptance, framework choice (PyTorch 5.8 GB vs TF-CPU 1.8 GB).
