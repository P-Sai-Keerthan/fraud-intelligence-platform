# IEEE-CIS independent evaluation — proposed protocol (IEEE-v0, **DRAFT, not frozen, awaiting owner approval**)

Written **before any data were downloaded or inspected.** Nothing here has been run. Items marked **[USER]** are decisions that belong to the owner (also collected in `DECISIONS_FOR_APPROVAL.md`).
Facts about the dataset that are not verified are tagged as in `COMPATIBILITY_REPORT.md` ([S], [M], [A]); a **stage-0 audit** replaces every [M] by an observation before the protocol is frozen.

Relationship to other work: this protocol does **not** change the synthetic-data protocols, models or results (Experiment 1 holdouts, Experiment 2 draft, Experiment 3). It does not reuse any trained weights of the platform; code may be reused only through copied, tested functions listed in `MODEL_ADAPTATIONS.md`.

## 1. Question, claims, and what will not be claimed

**Question.** On a public, real card-transaction dataset with a *proxy* entity, evaluated strictly forward in time: (Q1) how do a historical-aggregate tabular model and a sequential model compare in ranking and in detection at a fixed false-positive budget;
(Q2) does adding a sequence-derived score to the tabular model help (the same ablation as Experiment 3); (Q3) how often is the first labelled fraud of a proxy entity detected; (Q4) what do Tree SHAP explanations of the trained tabular model look like.

**Allowed wording:** "on IEEE-CIS, with proxy entity E\*, chronologically held-out transactions…"; "detection of the first labelled fraud of a proxy entity".
**Forbidden wording** (checked by a lint test): "customer" for a proxy entity; "early warning", "predicts fraud before it occurs", "lead time"; "validated on real banking data"; "the platform's model performs … on IEEE-CIS" (no platform model is evaluated); any comparison of absolute numbers with the synthetic results as if they measured the same thing.
**Not claimed under any outcome:** same customers, banking environment, feature definitions or fraud process as the synthetic data; calibrated probabilities; explanation quality beyond "the SHAP values of the model that was trained".

## 2. Stages and gates (every gate is a commit; nothing later may change what an earlier stage froze)

| Stage | Content | Allowed to look at | Gate |
|---|---|---|---|
| **-1 Approval** | owner reads and accepts the dataset rules personally, downloads (or allows the host), states the files' location | nothing | **[USER]** written approval; hashes recorded |
| **0 Label-blind schema audit** | column list, dtypes, missingness, ranges of `TransactionDT`, id cardinalities, duplicate rows, join coverage, whether `TransactionDT` has ties; **no** label column is read except to confirm that it exists and is binary | features only | `AUDIT0.json` committed; every [M] item resolved or the protocol revised |
| **1 Entity choice (label-blind)** | candidate entities E1–E3 (+ variants) compared by the pre-registered diagnostics of §4 | features only | entity E\* fixed, `ENTITY.json` committed |
| **2 Freeze-1** | this protocol (as v1), code, configuration spaces, tests, split cut-points, seeds | — | tag `ieee-protocol-v1`; SHA-256 of protocol, code, configs |
| **3 Development** | tuning, refits, cut-off fitting, candidate comparison | labels of the **train** and **validation** segments only | all leakage tests pass first |
| **4 Freeze-2** | selected configurations, trained-model SHA-256, operating-point parameters, power/feasibility check | — | tag `ieee-selection-v1`; **before the final segment's labels are read** |
| **5 Final** | score the final segment **once**; write the access ledger; report | final segment | refuses to run unless both tags are ancestors of HEAD and hashes match |

## 3. Data handling and temporal design

* **Population.** The labelled training transaction table (with the identity table left-joined on `TransactionID`). The competition's unlabelled test file is not used.
* **Ordering.** Rows are ordered by `TransactionDT`; ties are ordered by a documented rule (`TransactionID` is used **only** to break ties in order, never as a feature, and the effect of the rule is checked by shuffling row order in a test).
* **Segments by time** (cut-points are values of `TransactionDT` fixed at Freeze-1 from stage-0 information only, i.e. from the time range, not from labels): **train** = first 60 % of the time span; **validation** = the next 20 %; **final** = the last 20 %; an **embargo** of 7 days is removed between segments (rows inside an embargo are excluded from every role but remain visible as *history*).
* **History is not a leak.** Entity features and sequences for a row use all strictly earlier rows of its entity regardless of segment (that information exists at scoring time). **Labels never enter a feature**, including the labels of earlier rows of the same entity (in production they arrive late).
* **Right-censoring.** Late chargebacks may leave recent fraud unlabelled [M]. Pre-registered sensitivity analysis: the final segment is also evaluated with its **last 14 days removed** (decided now, not after seeing results). If stage 0 shows the labelled file spans too short a period for any 60/20/20 + embargo design to leave ≥ 20 days per segment, the design is revised before Freeze-1 (**[USER]** D5).
* **Eligible rows.** Sequence-based arms score rows with at least 10 earlier rows of the entity (window K = 10, as in the platform). **All arms are scored on the identical row set** (L9); rows with fewer than 10 earlier rows form a separate *cold stratum* reported separately.
* **Feasibility gate (Freeze-2).** The sequence question is answered only if the eligible share of final-segment rows is ≥ 20 % **and** the final segment holds ≥ 300 eligible fraud rows; the first-fraud analysis only if ≥ 300 first-fraud events (entities) exist in the final segment. Otherwise those analyses are reported as *descriptive, underpowered*, and no superiority statement is made.

## 4. Entity choice (label-blind, pre-registered)

Candidates: E1 `card1`; E2 `card1` + `card2`–`card6` + `addr1`; E3 community-style identifier **only if** it can be built from past rows alone (its time component must not use a field of unknown timing); E4 sensitivity variants (drop `addr1`; add email domain).
Diagnostics, computed on features only: (a) **consistency** — within-entity agreement rate of `card4`, `card6`, `ProductCD` (modal-value share) and, if present, `addr2`; (b) **history** — median and 90th percentile of rows per entity and the share of rows with ≥ 10 earlier rows; (c) **size skew** — share of rows in entities above 1 % of the data (indicator of collisions); (d) **missingness splitting** — share of rows whose entity key has a missing component.
**Selection rule (fixed now):** among candidates with the share of eligible rows ≥ 20 % and no entity above 1 % of rows, pick the highest consistency (average of (a)); if none qualifies, the sequence analysis is declared infeasible. Sensitivity runs use every other qualifying candidate. Results are always labelled "proxy entity E\*".

## 5. Features

Defined in `FEATURE_MAPPING.md` (R/S/X classification, additional entity-relative features, sequence step). Scalers, encoders and category vocabularies are fitted on the **train** segment only. Forbidden inputs: `isFraud` of any row, `TransactionID`, entity key and any raw `card*` string, `C*`, `D*`, `M*`, `V*` (except in arm V0, labelled context), any parse of an id string.

## 6. Arms (identical features and history information where stated; equal tuning budgets)

| Arm | Learner | Input | Role |
|---|---|---|---|
| R0 | rule | `amount_pct_of_avg` when the relative hour is unusual (the project's existing baseline logic on the mapped features) | trivial reference |
| L0 | logistic regression (balanced) | adapted point features | linear baseline |
| T0 | random forest | adapted point features (no history aggregates beyond the nine-style features) | floor |
| T1 | random forest | point features + history aggregates (non-sequential "historical-aggregate" baseline) | tabular |
| T2 | gradient-boosted trees (LightGBM) | same as T1 | strongest tabular candidate |
| M2 | LSTM (next-row target, as in the platform) → random forest on point features + out-of-fold LSTM score | stacked design | **project's design retrained from scratch** |
| M3 | GRU/LSTM trained jointly on the current row's label with a tabular head | RAW-SEQ + point features | sequence classifier |
| V0 | LightGBM on all columns including Vesta's `C/D/M/V` | context only; leakage not verified | upper reference, never the comparator |

**T\*** = best of {T1, T2} by mean validation PR-AUC over 5 refit seeds (tie within 0.005 → simpler). **Primary contrast:** M3 minus T\* and, separately, M2 minus T1 (the same-model-family ablation of Experiment 3). Both are tuned with the same trial count (40 random configurations, sampling seed 12001); search spaces are frozen at Freeze-1; if the best trial is in an arm's last 10, +20 trials for every arm.
Imbalance: no resampling in the primary configuration; the negative-sampling ratio {10, 50, none} and `class_weight` are hyper-parameters of every arm.

## 7. Operating points and metrics

* **Ranking:** PR-AUC (primary), ROC-AUC.
* **Alerts:** cut-offs fitted on the **validation** segment only: FPR = 1 % (primary) and 0.1 %, using the randomised tie-handling cut-off from Experiment 2 (v1 §7); report the realised test FPR beside recall. Also precision, recall, F1, confusion matrix, false alerts per 1,000 legitimate rows. (The prevalence is about 3.5 % [S], so precision at a 1 % FPR will differ strongly from the synthetic 0.47 % setting; no cross-dataset comparison of precision.)
* **First labelled fraud of an entity:** recall of the first fraud row of each entity that has fraud in the segment, at the 1 % FPR cut-off (a row-level detection event, not a warning).
* **Strata:** eligible rows; cold stratum; by `ProductCD`; by entity-history length bin.
* **Uncertainty:** paired **entity-cluster bootstrap** (connected components if entities overlap), 2,000 replicates, seed 9102; intervals for every reported contrast.
* **Primary endpoint:** PR-AUC difference on eligible final rows: (a) M3 − T\*, (b) M2 − T1. **Decision rule** (as Experiment 2 §9, δ to be set **[USER]**, proposal 0.02 PR-AUC): *superiority detected* if the lower CI bound > 0; *meaningful* if also Δ̂ ≥ δ; *no meaningful advantage* if the upper bound < δ; otherwise inconclusive. Secondary and exploratory results cannot rescue a failed primary.

## 8. Warning lead time

**Not measured and not reported.** The dataset has no precursor events and no ground-truth warning window, and chargeback labels arrive late; an alert on an earlier non-fraud row of a later-defrauded entity cannot be separated from a false alert. If the owner later wants an exploratory "alert-to-first-fraud interval", it must be a separate, labelled analysis with the chance baseline 1 − (1 − FPR)^n, never called lead time or early warning, and is out of scope for this protocol.

## 9. Explanations

SHAP applies to the models actually trained here (exact Tree SHAP on the random-forest/LightGBM arm used for the stacked design M2 and on T\*). Checks: additivity (contributions + expected value = model output), stability of the top-5 features across the 5 refit seeds, a randomisation sanity check (SHAP of a model fitted on permuted labels should not agree with the real model's top features), and a table of the top features by frequency among true positives and false positives. SHAP of the sequence model itself is **not** claimed. Masked Vesta features are not offered as reasons.

## 10. Leakage and integrity tests (all must pass before any final labels are read)

IL1 future-perturbation: changing any later row of the file leaves earlier rows' features bit-identical · IL2 label-independence: replacing all labels by random bits leaves every feature identical · IL3 forbidden-column assertion · IL4 strict time order and tie-rule invariance (shuffled row order) · IL5 entity construction uses past rows only · IL6 segment integrity and embargo · IL7 cut-off provenance (poisoned final labels do not change cut-offs) · IL8 scalers/encoders fitted on train only · IL9 identical row set across arms · IL10 exactly one first-fraud row per entity with fraud · IL11 determinism (same seed, same scores) · IL12 permuted-label sanity (validation ROC-AUC within 0.45–0.55) · IL13 final-access ledger (one scoring) · IL14 ID-relabelling invariance (random bijection of category strings leaves features identical) · IL15 `TransactionID`/entity key absent from every input · IL16 randomised cut-off hits the target FPR on tie-heavy scores · IL17 wording lint · IL18 Freeze-2 hash check · IL19 raw-file SHA-256 equals the recorded one · IL20 lazy windows equal materialised windows · IL21 identity-presence ablation reported · IL22 duplicate-row check.

## 11. Provenance and reproducibility

Recorded for every run: raw-file SHA-256 and sizes, column-list hash, stage-0 audit JSON, entity definition JSON, split cut-points and their hash, feature-code hash, configuration JSON, seeds (sampling 12001, refits 41–45, bootstrap 9102), environment (`pip freeze`, Python, hardware), git commit, per-row scores (compressed; SHA-256 of decompressed content), metric JSON with intervals, ledger of every access to the final segment. The raw data and per-row scores are not committed (licence, size); hashes are.

## 12. Stop conditions

Any leakage test fails · the final segment is read before Freeze-2 · a second final scoring is attempted · the dataset rules forbid the intended use or publication of results · stage 0 contradicts a load-bearing [M] fact (e.g., no usable entity, time unit not seconds) → revise and ask · realised validation FPR deviates > 20 % from target · the sequence arms cannot reach the feasibility gate (report descriptively) · free disk < 5 GB or RSS > 13 GB (pause and ask).

## 13. Deviations

Anything changed after Freeze-1 is recorded as a *Deviation* in the report with its reason and whether any final label had been read. Changes after Freeze-2 require the owner's approval.
