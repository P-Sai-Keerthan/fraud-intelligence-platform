# Experiment 2 — protocol draft v2 (SEQ-EARLY v2, **DRAFT, not frozen, awaiting approval**)

Supersedes nothing: `PROTOCOL_DRAFT.md` (v0) and `PROTOCOL_DRAFT_v1.md` stay unchanged for the audit trail. v2 keeps v1's design and fixes what became known since: it makes the specification self-contained, records what the missing ablation must provide,
and takes account of Experiment 3, the R-02 decision and the independent-data plan. **Nothing has been generated or run; no model has been trained.** Items marked **[USER]** need the owner. Evidence tags as in v1 ([code], [measured], [simulated], [source]) plus **[Exp3]** (verified result of Experiment 3, synthetic).

## 0. Research objective (unchanged)

Fairly compare **historical-aggregate tabular models (random forest / boosted trees)** against **LSTM/GRU sequence models** on (i) detection of the **first fraudulent transaction of an episode** at a fixed false-positive budget, and (ii) — as a *separate, directly measured* task — **warning lead time**.
No claim of "early detection" or "early warning" is allowed unless the corresponding quantity (first-fraud detection, or lead time as defined in §8) is measured directly under this protocol.

## 1. What changed since v1, and why

| # | Change | Reason / evidence |
|---|---|---|
| 1 | The missing-ablation requirements are listed exactly (§2), with a fallback that does not block the experiment | the ablation is still in no readable location (searched again: container, three remote branches, `git log -S`) |
| 2 | Context from Experiment 3 on Generator A | [Exp3] the stacked LSTM → forest beats the same forest **without** the LSTM input on nine point features (PR-AUC 0.507 vs 0.424, paired difference +0.083 [0.069, 0.096]). That ablation compares the LSTM with *no history aggregates*; it does **not** answer the Experiment 2 question (aggregates vs sequence), so v1's prior ("sequence advantage unlikely on A") is neither confirmed nor refuted |
| 3 | The default generator produces **no fraud on fewer than 10 earlier transactions** | [Exp3] 0 of 2,910 fraud rows. A cold-start stratum would be empty of fraud. v2 adds an optional **new-customer dataset** (generator 2.1.0 late-joiner extension, already in the repository) as a separately labelled secondary track **[USER]** |
| 4 | Operating points are defined by FPR targets, independent of the platform's alert tiers | R-02 resolved the platform's tiers separately; Experiment 2 never uses them. [Exp3] the platform's frozen policy cut-offs gave about 9.6 and 0.7 false alerts per 1,000 legitimate rows on fresh data, i.e. roughly v1's 1 % and 0.1 % points, but Experiment 2 fits its own cut-offs |
| 5 | Independent public data is a separate protocol | `experiments/independent_ieee_cis/PROTOCOL_DRAFT.md`; v1 §12.2 ("public data not downloadable") is superseded. IEEE-CIS cannot support lead time (no precursor events), so lead time stays a Generator-A/B task |
| 6 | Spent seeds extended | Experiment 3 consumed 9301–9305; added to the disallowed list |
| 7 | Resources re-checked | free disk is 15 GB: the PyTorch environment (5.8 GB) is borderline against v1's "≥ 15 GB free" advice, so **TensorFlow-CPU (1.8 GB) is the default**, PyTorch optional; `lightgbm` and `torch` are not installed |

## 2. What the missing `experiments/lstm_value_ablation/` must provide (required; none of it exists locally)

| Needed | Used for | If it never arrives |
|---|---|---|
| `README.md`, `REPORT.md`; what `REPORT_1.md` is | what was claimed and decided | claims stay "unverified"; nothing is reproduced from text |
| The **aggregate-feature code** (window lengths, statistics, tie handling, strictness of `<`) | the definition of AGG for A1–A3 and the arm A1-frozen | use the **AGG-v2 definition below** (pre-registered, re-implemented), drop arm A1-frozen, state that equivalence to the ablation is unknown **[USER]** |
| Split definition (customer/time), fractions | check that DEV/FINAL splits match what the ablation used | use v1's component split |
| **Seeds** (data, training, bootstrap) | collision check with 7101–7103, 8101–8110, 7201–7203, 8201–8205 and all spent seeds | assume nothing; re-draw only from unused seeds |
| Per-seed result files, decision rule text, library versions, dataset hashes/manifest | verify that reported numbers reproduce, and hash the ablation's data | no verification possible; report "not verifiable" |

**AGG-v2 (fallback, fixed now, past-only, strict inequality, current row excluded):** for windows {1 h, 24 h, 7 d, 30 d}: count, sum and mean of amount, std of amount (floored as in `build_point_features`), count of distinct devices, distinct locations, distinct categories, sum of `failed_logins_24h`; plus time since the previous row (log), time since the previous *different* device and location, share of rows in the current hour-of-day, maximum amount so far, rows so far; plus ratios of the current amount to the window means. All fitted statistics (scalers, quantile cut-points) on DEV-train only.

## 3. Data-generation assumptions

* **Generator A** = the repository's v2 generator, default configuration except `n_customers = 2000`, version recorded in `FREEZE.json`; deterministic (seed 42 regenerated bit-exactly) [measured]. Expected per dataset: ≈ 416 k transactions, ≈ 1,900 fraud rows, ≈ 440 episodes [measured on scratch seed 9].
* Its assumptions are the authors': seven fraud archetypes (account takeover, card testing and cash-out, device takeover, high-value single, ring, stolen card online, "normal-looking"), legitimate behaviour that is almost order-free, an explicit login-failure stream with `credential_attack` events before some episodes, homes in ten Indian cities. A sequence model can only win if the data contain order structure that aggregates miss; this is stated in the report, not hidden.
* **Generator B** (v1 §12.1): an independent specification with sequential legitimate behaviour and a behavioural prodrome; specification and code frozen and committed before FINAL-A is scored; reported separately, never pooled; "synthetic generalisation test", never banking.
* **New-customer track** (optional, **[USER]**): generator 2.1.0 `--late-joiner-share 0.2 --new-customer-fraud-episodes 30`, separately labelled; the only place where fraud on fewer than 10 earlier transactions exists.
* Seeds: scratch 9, 42 (never reported); DEV 7101–7103; FINAL-A 8101–8105 (reserve 8106–8110); B: DEV 7201–7203, FINAL 8201–8205; new-customer DEV 7301–7302, FINAL 8301–8303; sampling 9001; refits 31–35; bootstrap 9101. **Disallowed:** 42, 11–15, 101–105, 201–205, 301–305, 401–405, 411–415, 501–505, 511–515, **9301–9305**. At review time no data exist for any reserved seed.

## 4. Temporal and group splits

* **Primary (DEV):** customer-component split 60/20/20 (train / validation / DEV hold-out) with whole components (rings, households) kept together; within each dataset all rows are used as *history* for features regardless of split (a customer belongs to one split, so history never crosses splits).
* **FINAL-A datasets** are generated after Freeze-2 and are **entirely test**; thresholds come from DEV validation. No retraining on FINAL.
* **Temporal sensitivity (exploratory):** DEV time split by `build_time_split` (train before the first boundary, validation to the second, test after), reported separately; the platform's v2 time-split test had only 61 fraud rows, so no primary claim rests on it.
* Embargo: none for component splits. For the exploratory time split the existing `build_time_split` is used as is; how it treats episodes that straddle a boundary must be read from its code and recorded at Freeze-1.

## 5. Target definitions

* **Row label** `is_fraud` of the *scored* transaction; the scored row is the last step of its window (current transaction included for the jointly trained sequence arms; excluded for the platform-style stacked design).
* **Task A positives:** the **first fraudulent transaction of each episode** (marked in the generator metadata, used for evaluation only); **negatives:** legitimate rows. Subsequent fraud rows are excluded from the selection metric E_val (reported separately). 
* **Sequence-arm training target (S1, S2):** the label of the current transaction. The platform's stacked LSTM predicts the **next** row's label; if that design is included it is a separate arm **M2** with its out-of-fold scores (entity-grouped folds) and is not the primary comparator.

## 6. Arms, baselines, equal budgets (as v1 §3, §6)

R0 login-failure rule · A0 RF on the nine features · A1-tuned RF on F9 + AGG (A1-frozen only if the ablation arrives) · A2 LightGBM on F9 + AGG · A3 LightGBM on F9 + AGG + LAG-K · S1 GRU on RAW-SEQ · S2 GRU + MLP head with F9 + AGG (same information as A3).
**T\*** = best of {A1-tuned, A2, A3} by mean E_val over 5 refit seeds on DEV validation (tie within 0.005 → simpler). **Primary contrast: S2 − T\*.** 40 random configurations per arm (A0 20), sampling seed 9001, +20 for all arms if any best trial is among its last 10; search spaces as v1 §6.

## 7. Class imbalance

No resampling in the primary configuration. The **negative-sampling ratio {10, 50}** and `class_weight` (RF), positive weight (LightGBM: 1, √ratio, ratio) are hyper-parameters of every arm, tuned with the same trial count; early stopping on E_val; evaluation always on the **full, unsampled** validation and test rows. Prevalence ≈ 0.46 % of rows are fraud and ≈ 0.11 % are first-fraud events (440 episodes in ≈ 416 k rows, scratch seed 9) [measured]. Cut-offs by FPR on the full validation negatives, so sampling does not change the operating definition.

## 8. Metrics, operating points, and the lead-time calculation

**Detection (Task A)** — primary: first-fraud recall at **FPR = 1 %** (randomised, validation-fitted cut-off (t, q); realised test FPR reported beside it), S2 − T\*, pooled over FINAL-A with a paired, entity-component cluster bootstrap (2,000 replicates, seed 9101). Secondary: recall at 0.1 % and at the F1 cut-off, PR-AUC (all rows and first-fraud), ROC-AUC, precision/recall/F1, confusion matrices, false alerts per 1,000 legitimate, recall by position in the episode, cold-start stratum (empty of fraud on Generator A; filled only on the new-customer track), R0 and A1-frozen for context.
Decision rule as v1 §9 with δ **[USER]** (proposal 0.05; the rule's power at Δ = δ is about 0.5 [simulated], so consider Δ̂ ≥ 0.03).

**Warning lead time — separate task (Task B), never Task A, defined now:**

1. For episode *e* of customer *c*: *t_f(e)* = time of the first fraud transaction; *t_s(e)* = time of the first `credential_attack` login event of *c* in `[precursor_start(e), t_f(e))` (the first signal an institution could observe; `fraud_session` events excluded; evaluation-only file `login_failures.csv`).
2. Warning window *W_e* = the customer's **legitimate** rows with *t_s(e) ≤ t < t_f(e)*. Episodes with |*W_e*| = 0 have **no opportunity**; they are excluded from the denominator and counted.
3. A **Task-B model** is trained with rows in the warning windows of *training* episodes labelled positive. That label uses generator ground truth and **cannot exist in production**; Task-B results are labelled accordingly. Cut-off: FPR = 1 % on DEV validation negatives (windows excluded from the negatives' false-positive count in training only).
4. For each opportunity episode: *detected early* if any row in *W_e* is alerted; **lead time** *L(e) = t_f(e) − min{t_i : i ∈ W_e, alert_i}* (hours). Reported: share detected early (pooled, cluster bootstrap), median and inter-quartile range of *L*, over opportunity episodes only. Delay *after* the first fraud (existing `episode_metrics`) is a different quantity and is not called lead time.
5. **Chance baseline:** analytic *1 − (1 − FPR)^|W_e|* per episode averaged over episodes, and an empirical **permutation baseline** (score the same customer's rows outside warning windows with the model; the alert rate there is the false-alert rate that applies to windows). **Login-failure rule R0** at the same FPR as a trivial comparator (login failures already carry much of the signal). A model is said to give early detection beyond chance only if its early-detection share exceeds both baselines with the cluster-bootstrap interval excluding the difference 0, **and** exceeds R0.
6. Generator B's explicit behavioural prodrome is the only place a stronger lead-time statement could be tested, and only as a test of that design assumption.

## 9. Leakage and integrity checks (L1–L22; all pass before any final data exist)

L1 future-perturbation · L2 label-independence · L3 forbidden-column assertion · L4 strict ordering/ties · L5 F9 equals production features on a sample · L6 split integrity and seed allow-list · L7 cut-off provenance (poisoned test labels) · L8 fit on train only · L9 identical row set across arms · L10 one `first` row per episode · L11 determinism · L12 permuted-label sanity (0.45–0.55) · L13 final-access ledger · L14 ID-relabelling invariance · L15 `customer_id`/`transaction_id` absent from inputs · L16 randomised cut-off on tie-heavy scores · L17 wording lint · L18 Freeze-2 hash check · L19 regenerated seed 42 matches its manifest · L20 lazy windows equal materialised windows · **L21 warning-window labels never reach a Task-A model or any feature** · **L22 `login_failures.csv` / `episodes.csv` are used only for lead-time evaluation and Task-B training labels, never as Task-A inputs**.

## 10. Freeze, provenance, stop conditions

Two-stage freeze with tags and SHA-256 (v1 §15); per-run record of dataset hashes, seeds, configurations, versions, per-row scores and ledger (as in Experiment 3, which can be taken as the template: pre-registered protocol committed before data, independent recomputation script, second full run for reproducibility). Stop conditions as v1 REVIEW_CHECKLIST §8.

## 11. Open items

**[USER]:** restore the ablation (new branch; commands in REVIEW_CHECKLIST §9) or approve AGG-v2; δ and the 1 % FPR; the new-customer track; Generator B acceptance; base-branch plan; TensorFlow-CPU vs PyTorch; whether Experiment 2 runs after the IEEE-CIS track. **Unknown until measured:** discordance *d* between S2 and T\*; real tuning and refit times; bit-exact GRU determinism (L11).
