# Experiment 2 — proposed protocol (SEQ-EARLY v0, **DRAFT, not frozen**)

Nothing in this file has been run. Items marked **[PENDING-ABLATION]** need the completed
`lstm_value_ablation` files. Items marked **[USER]** are decisions that belong to the project owner.
Proposed numbers (margins, budgets, seeds) are proposals, open to change **before** the freeze (§14)
and not afterwards.

## 1. Question and what will, and will not, be claimed

**Question.** Given the same information and a comparable tuning budget, does a jointly trained
sequence classifier detect the **first fraudulent transaction of an episode** more often than the
strongest tabular alternative, at the same false-positive budget?

* Claimed if supported: "detection of the first fraudulent transaction *at the time it is attempted*".
* **Never claimed:** that any model "predicts fraud before it occurs". That needs a lead-time
  evaluation on a defined warning period (§10), and then only for that period.
* Recall on later transactions of an active episode (`fraud_stage = subsequent`) is reported but is
  **not** evidence of early detection: the model's history already contains the fraud itself.
* Prior expectation, stated before any result: on the existing generator the sequence model is
  **unlikely** to win, because legitimate behaviour there is almost order-free (AUDIT §1). A win would
  have to come from the data, not from the design.

## 2. Information contract: what a model may see when it scores transaction *i* of customer *c*

| Allowed | Forbidden |
|---|---|
| Raw fields of *i* itself: timestamp, amount, merchant_category, device_id, location, `failed_logins_24h` | `is_fraud` of **any** row, including old ones (labels arrive late in practice) |
| The same raw fields of every **strictly earlier** transaction of *c* (canonical order `customer_id, timestamp, transaction_id`) | The 9 metadata columns (`fraud_type, fraud_episode_id, fraud_stage, fraud_ring_id, is_precursor, legit_context, customer_segment, merchant_id, network_id`) |
| Statistics computed only from those rows | `episodes.csv`, `customers.csv`, `login_failures.csv` event stream (kept out of the primary comparison), any other customer's rows (no ring/household features), any row after *i* |
| Scalers/encoders fitted on the training rows only | ids as embeddings: device/location ids are per-customer strings, so they enter only as customer-relative flags ("seen before", "share of past rows") |

The home city is the modal location of *c*'s earlier rows, never the generator's `home_city` or the
hard-coded `HOME_LOCATIONS` list (AUDIT §2, finding 7).

## 3. Arms

| Id | Learner | Input | Role |
|---|---|---|---|
| A0 | Random forest | F9 | Floor: nine production features only |
| A1 | Random forest | F9 + AGG | **The existing aggregate RF** **[PENDING-ABLATION]**: reuse its exact feature code and settings; if unavailable, a documented re-implementation labelled "re-implemented", never "the existing" |
| A2 | LightGBM | F9 + AGG | Stronger tree learner, same inputs as A1 |
| A3 | LightGBM | F9 + AGG + LAG-K | A2 plus the last *K* raw events flattened: tests whether raw recent events alone help, without recurrence |
| S1 | GRU | RAW-SEQ (K steps, current transaction last) | Sequence model that must learn everything from raw events |
| S2 | GRU + MLP head | RAW-SEQ **and** F9 + AGG joined to the GRU state | Sequence model with the **same information** as A3 |

* F9 = the nine production features, computed by the **unchanged** `build_point_features`.
* AGG = past-only window aggregates **[PENDING-ABLATION]**. Fallback sketch: counts and mean/max/std of
  log-amount over 1 h / 24 h / 7 d / 30 d / all history, time since last transaction, number of distinct
  devices/locations/categories, share of past rows on the current device/location/category,
  max and count of `failed_logins_24h > 0` in the last 7 d/30 d.
* RAW-SEQ step = log1p(amount), log inter-event gap, hour and weekday as sin/cos, category one-hot,
  device-seen-before, device-is-modal, location-seen-before, location-is-modal, log1p(`failed_logins_24h`),
  is-current flag, padding mask. *K* = 32 (fixed; also LAG-K). Cold-start rows are padded and masked.
* The GRU is trained **end to end on the current-transaction label** (not on "next transaction", no
  out-of-fold stacking). The earlier LSTM stays untouched and is only a reference **[PENDING-ABLATION]**.
* Framework: PyTorch 2.14.1 CPU for S1/S2 (custom training loop, deterministic flags, exact parameter
  counts); scikit-learn 1.9.1; LightGBM 4.7.0. The project's TensorFlow is not used because the new
  models are written fresh; this is recorded as a design choice, not a result.

**Primary contrast:** S2 versus **T\***, where T\* is the tabular arm among {A1, A2, A3} with the highest
development-validation score (§6), chosen before any final data exists. S2 and A3 have identical
information, so S2-vs-A3 is the cleanest "representation" test; S2-vs-A2 shows whether raw events add
anything; S1 is a secondary, information-restricted arm. Comparing against the *best* tabular arm is
deliberately conservative.

## 4. Datasets and splits

Generator A = the existing v2 generator, **unmodified**, default configuration except
`n_customers = 2000` (the 4× scale is justified by the power table in AUDIT §5).

| Role | Seeds | Use |
|---|---|---|
| Scratch | 9 | Infrastructure timing only. Never reported. |
| DEV | 7101, 7102, 7103 | Train / validation / tuning / thresholds. Customer-component split 60 / 40, stratified by has-fraud, rings and households kept whole (`split._component_customer_split`). |
| FINAL-A | 8101 … 8105 | Confirmatory test. **Generated only after the freeze, scored once.** Every row is test data. Thresholds come from DEV validation. No retraining. |

Why a customer split: fresh seeds put the whole calendar in play, so the validation rows (from every
part of the timeline) must be representative of the final rows for the validation-fixed thresholds to
transfer. A time split of DEV is run as a sensitivity analysis (exploratory).

Populations: **all** transactions are scored by every arm, including cold-start rows. Rows with fewer
than 10 earlier transactions are reported as a separate stratum, because the earlier LSTM did not score them.

Expected size (planning, not measured): ~400 k transactions, ~440 episodes, ~1,900 fraud rows per dataset;
DEV ≈ 3 × that; FINAL-A ≈ 2,200 first-fraud events.

## 5. Seeds (all disjoint from `SPENT_SEEDS` 42, 101–105, 201–205, 401–405, 411–415 and from 301–305, 501–505, 511–515) **[PENDING-ABLATION]: check against the ablation's seeds**

| Purpose | Seeds |
|---|---|
| DEV data | 7101–7103 |
| FINAL-A data | 8101–8105 |
| Generator B DEV / FINAL (reserved, §11) | 7201–7203 / 8201–8205 |
| Configuration sampling | 9001 (same sample index for every arm) |
| Model training (final refits) | 31–35 |
| Bootstrap | 9101 |

## 6. Tuning: equal budgets, one selection metric

* **Common selection metric E_val** = PR-AUC on DEV-validation rows restricted to *legitimate rows ∪
  first-fraud rows* (subsequent-fraud rows removed so they count neither way). It targets the primary
  endpoint while staying a smooth ranking metric. It is used for early stopping, for choosing the best
  configuration of each arm, and for choosing T\*.
* **40 random configurations per arm** (A0: 20), sampled with seed 9001 from the spaces below, each
  trained once (one training seed) on DEV-train and scored on DEV-validation. The best configuration is
  refit with training seeds 31–35. Wall-clock for every arm is recorded; trial count, not time, is the budget.
* Search spaces (to be frozen verbatim in `configs/search_spaces.json`):

| Arm | Space |
|---|---|
| RF (A0, A1) | n_estimators {300, 600}; max_depth {None, 12, 20}; min_samples_leaf {1, 3, 5, 10, 20}; max_features {sqrt, 0.3, 0.5}; class_weight {None, balanced_subsample} |
| LightGBM (A2, A3) | num_leaves {15, 31, 63}; learning_rate logU[0.02, 0.2]; min_child_samples {10, 20, 50, 100}; feature_fraction U[0.5, 1]; bagging_fraction U[0.6, 1]; reg_lambda logU[1e-3, 10]; positive weight {1, √ratio, ratio}; rounds by early stopping on E_val (cap 2,000) |
| GRU (S1, S2) | hidden {32, 64, 128}; layers {1, 2}; dropout {0, 0.1, 0.3}; lr logU[1e-3, 1e-2]; weight decay {0, 1e-5, 1e-4}; batch {256, 512}; negative sampling per epoch {1:10, 1:50, all} with matching weight; max 40 epochs, early stopping on E_val (patience 5) |

* Pre-specified secondary: **training-size curve** for T\* and S2 at 25 / 50 / 100 % of DEV-train
  customers (components), fixed hyper-parameters, scored on DEV-validation only. It addresses the
  data-starvation confound in the earlier comparison.

## 7. Operating points

Per arm, from DEV validation only (`tie_safe_cutoff`, `select_threshold` from the existing code):
(a) **FPR = 1 %** — primary; (b) FPR = 0.1 %; (c) F1-maximising threshold. The test labels never enter any
cut-off (enforced by a test, §12). The realised test FPR of every arm is reported next to its recall,
because validation-fixed cut-offs transfer imperfectly. Full recall-vs-FPR curves are plotted as
descriptive analysis, and "recall at matched test FPR" (read from the curve) is labelled as such.

## 8. Endpoints

**Primary (one):** first-fraud recall at the FPR = 1 % cut-off, S2 versus T\*, difference Δ pooled over
FINAL-A (events weighted equally), with a paired cluster bootstrap (2,000 repetitions, resampling customer
components within each dataset, seed 9101).

**Secondary (reported with CIs, not tested):** first-fraud recall at FPR 0.1 % and at the F1 cut-off;
PR-AUC (all rows) and E_val-style first-fraud PR-AUC; ROC-AUC; precision, recall, F1; confusion matrices;
legitimate false alerts per 1,000; overall transaction recall; episode detection rate (any transaction
alerted — *includes later transactions*); recall by position in the episode (1st, 2nd, 3rd+); delay after
first fraud (descriptive; **not** lead time); cold-start stratum; per-dataset and per-seed tables.

**Exploratory (labelled as such in the report):** per-fraud-type first-fraud recall (7 archetypes, no tests,
multiplicity noted); second episodes of repeat victims; DEV time-split sensitivity; permuted-history
control (does shuffling the order of past events hurt S2?); ablation without `failed_logins_24h`.

## 9. Decision rule (applies to Generator A / FINAL-A only; proposed values)

* **Measurable advantage for early detection** only if all hold: (1) lower bound of the 95 % CI for Δ > 0;
  (2) Δ ≥ +0.05 (smallest difference worth acting on) **[USER]**; (3) realised FPR of S2 ≤ 1.25 × that of T\*
  (the gain is not just extra alerts); (4) Δ > 0 in at least 4 of the 5 datasets.
* **No meaningful advantage** if the upper bound of the CI is < +0.05 (this includes "tabular is better").
* Otherwise **inconclusive**, and reported as such.
* Secondary and exploratory results can inform the discussion but cannot rescue a failed primary.

## 10. Lead time / early warning (separate task, exploratory on Generator A)

Establishment test: the generator defines a warning period only for account_takeover, card_testing_cashout
and ring episodes with a precursor (≈16 % of episodes), whose observable content is login-failure bursts
(AUDIT §4). Therefore:

* Task A models are **not** evaluated for lead time. Their alerts on warning-period rows are false positives.
* Task B (exploratory): warning-window rows W_e = the customer's legitimate rows in
  `[precursor_start, first_fraud_time)`. A Task-B model is trained with W-rows labelled positive. **That label
  uses generator ground truth and is impossible in production**, where the compromise date is unknown.
  Report the share of episodes with ≥ 1 alert in W_e, the lead time (first fraud time − first alert in W_e),
  and the **chance baseline** 1 − (1 − FPR)^|W_e| for the same cut-off.
* Alerts in the minutes before the first fraud caused by `fraud_session` login failures are part of the fraud
  attempt and do not count as lead time (counted only if earlier than `precursor_start`).
* Wording allowed: "alerts in the generator-defined warning window"; never "predicts fraud".

## 11. Generalisation (Phase 4)

1. **Generator B (independent design) — proposal.** Written to a separate specification, **frozen before any
   model sees its data**, sharing no fraud rules, code or parameters with Generator A. Design principles:
   (i) legitimate behaviour with genuine sequential structure (habit chains over category/location/hour,
   recurring weekly/monthly payments, regime changes such as a move or new job that must *not* be alerted);
   (ii) fraud mechanisms with dynamics: gradual escalation ("bust-out") with a behavioural prodrome and
   an explicit warning window, mimicry ATO that copies the victim's hours and categories, slow low-amount
   drip, compromised-merchant fraud that is invisible in the victim's own history. Models are re-tuned with the same
   budget on B-DEV and evaluated once on B-FINAL, with the §9 rule applied separately.
   **Limits that will be stated in the report:** B encodes sequential structure on purpose, so a sequence-model
   win on B shows capacity, not real-world benefit; B and the models share an author, so B is independent of
   A's rules but not of the experimenter's assumptions. A and B together *bracket* the answer between an
   aggregation-friendly and a sequence-friendly synthetic world.
2. **Public data.** None can be downloaded from this environment (egress policy, AUDIT §6). By schema, ULB
   `creditcard.csv` is unsuitable (no entity). IEEE-CIS is the only real candidate, with a heuristic entity key
   (`card1`), a relative `TransactionDT`, delayed chargeback labels and no episode or warning definitions
   (AUDIT §7). If the user supplies it: derive episodes by a stated rule, report **only** first-fraud-per-proxy-entity
   recall and standard metrics, never lead time, in a separate table, and never call it real-world validation of
   the platform. TabFormer and the Sparkov-based set are *synthetic/simulated* and would be reported as
   "independent synthetic", not "real". **[USER]**
3. External and synthetic results are kept in separate tables and never pooled.

## 12. Leakage and integrity tests (all must pass before any final data is generated)

1. **Future perturbation:** for 1,000 sampled rows, replace every field of all later rows of that customer
   with random values; the feature vector, AGG vector and sequence tensor of the row must be bit-identical.
2. **Label independence:** flip/permute `is_fraud` everywhere; all model inputs are bit-identical.
3. **Forbidden columns:** `assert_no_ground_truth` on every input matrix, plus a name+hash check of the column list.
4. **Strict ordering:** per-customer timestamps strictly increasing; history excludes the current row; tie handling tested.
5. **Equivalence:** the F9 columns equal the production features exactly on a sample.
6. **Split integrity:** no customer component in two roles; no seed outside the allowed list; DEV and FINAL file
   hashes recorded at generation; FINAL data cannot be scored without the freeze tag.
7. **Threshold provenance:** cut-off functions never receive test labels (signature test with poisoned labels).
8. **Fit-on-train-only:** scalers, encoders, class weights recomputed from DEV-train only (test mutates nothing).
9. **Same population:** every arm's score table has the identical `transaction_id` set (hash compared).
10. **First-fraud labelling:** exactly one `first` row per episode, equal to the episode's earliest fraud row.
11. **Determinism:** same seed gives identical scores for every arm on CPU.
12. **Pipeline sanity:** training on permuted labels gives ROC-AUC ≈ 0.5 on validation.
13. **Access ledger:** every read of FINAL data is appended to `results/final_access_ledger.jsonl`; a second scoring
    run requires an explicit flag and is reported as a deviation.

## 13. Runtime and complexity (measured, not estimated)

Per arm: trainable parameters or tree/leaf counts; model size on disk; training wall-clock and CPU-seconds
(per trial and for the final refit); feature/sequence construction time; inference latency per 1,000
transactions on CPU at batch 1 and 256; peak memory. Hardware: 4 CPU cores, 15 GB, no GPU (recorded in the report).
The production feature builder (`build_point_features`, one pandas row at a time) is slow and sits on the
critical path of every arm, so its cost is reported too. **Measured so far (scratch seed 9, one core, infrastructure
only):** generating 500 customers took 12 s; the unchanged `build_point_features` took ≈143 s for 103 k rows. That
extrapolates to ≈10 min per 2,000-customer dataset, ≈80 min for the 8 datasets on one core, ≈20 min on 4 cores
(extrapolation, to be re-measured). Training times for the trees and the GRU are not yet measured: they need the
pilot on DEV data only, after approval.

## 14. Freeze procedure and deliverable layout

* The freeze is a commit containing this file renamed `PROTOCOL.md`, `configs/seeds.json`,
  `configs/search_spaces.json`, the code of `seqearly/` and the tests, plus `FREEZE.json` holding their SHA-256
  hashes; it is tagged `seq-early-protocol-v1`. The FINAL generation and scoring scripts refuse to run unless the
  tag is an ancestor of HEAD and the hashes match.
* Anything changed afterwards is listed under "Deviations" in `REPORT.md`, with the reason.

```
experiments/sequence_early_detection/
  README.md  AUDIT.md  PROTOCOL.md (after freeze)  FREEZE.json  REPORT.md  requirements.txt
  configs/   seeds.json  search_spaces.json
  seqearly/  data/ features/ models/ eval/ plots/
  scripts/   generate · tune · refit · score_final · make_tables · make_plots
  tests/     leakage and integrity tests (§12)
  results/   per-seed and per-dataset JSON/CSV, tables, figures, final_access_ledger.jsonl
```

## 15. Open items

* **[PENDING-ABLATION]** the ablation's README, REPORT, code, aggregate definition, seeds and decision rule.
* **[USER]** base branch (fast-forward this branch to `origin/Keerthan`, recommended); smallest effect of interest
  (0.05?); primary FPR (1 %?); whether to attempt IEEE-CIS; whether the generator-B design is acceptable.
