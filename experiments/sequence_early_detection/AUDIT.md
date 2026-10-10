# Experiment 2 — Phase 1 audit

Status: **draft for review. No model has been trained and no result has been looked at.**
Everything below was read from code, documents and generator configuration, or measured as
infrastructure timing on a throw-away seed that will never be used for any result.

## 0. What this audit could and could not read

| Input named in the task | Available? | Consequence |
|---|---|---|
| `experiments/lstm_value_ablation/REPORT.md` and README | **No.** The attachments are Windows paths on the user's machine (`C:\Users\Keerthan\Downloads\…`); nothing arrived in the container (`/mnt/attach`, `/mnt/user-data/uploads` are empty). The folder is also absent from every remote branch I could list (`main`, `Keerthan`, `keerthan`, this branch). | I have **not** read the ablation's findings, decision rule, seeds, or the definition of its "historical-window aggregates". Nothing in this audit relies on them. |
| Existing aggregate-based Random Forest | **No.** A search of `backend/app/` on `origin/Keerthan` finds no historical-window aggregate features (only the 9 point features and the 10-step LSTM window). It presumably lives inside the ablation, which I cannot see. | The Phase 1 requirement "use the existing aggregate RF as the initial baseline" cannot be met yet. I will not guess its feature list. |
| Code base | **Yes, but not on this branch.** This branch (`claude/fair-sequence-fraud-detection-baiokj`) equals `main` (`23206d4`). The generator v2, split, metrics and holdout code live on `origin/Keerthan` (17 commits ahead; `main` is a strict ancestor, so a fast-forward is possible). | The audit below reads `origin/Keerthan` read-only. Nothing there was modified. |

Paths below are relative to `origin/Keerthan`.

## 1. Data generator (`data/v2/synth_v2/`, version 2.0.1, plus optional 2.1.0 new-customer extension)

* **Scale of a default dataset** (manifest, seed 42): 500 customers, 101,297 transactions, 471 fraud
  transactions (0.465 %), **110 fraud episodes**, 4 rings, **18 episodes with a warning period**,
  60 warning-period transactions. 176 days; no fraud before day 21.
* **Deterministic and cheap.** Every random draw is keyed by `(seed, domain, entity)`. Measured here:
  12 s to generate 500 customers without features, so 2,000-customer datasets are cheap to produce.
* **Fraud labels.** `is_fraud = 1` only on rows generated for an episode (`_fraud_transactions`).
  `fraud_stage` is `first` for the earliest fraud row of an episode, `subsequent` otherwise
  (`_finalize`). Ring episodes share a ring id. Card-testing episodes start with a small probe.
* **Warning period.** Exists for three archetypes only (`p_precursor`: account_takeover 0.5,
  card_testing_cashout 0.3, ring 0.3). It starts 1–7 days before the first fraud and contains only
  the customer's own **legitimate** transactions (`is_precursor = 1`, still `is_fraud = 0`).
  The only observable signal the generator puts there is **bursts of failed logins**
  (`credential_attack`, 1–3 bursts of 2–6 failures). It adds no behavioural drift.
* **`fraud_session` login failures** are written 1 min–2 h before an episode's first transaction
  (`_fraud_transactions`). They count in `failed_logins_24h` of the first fraud and are legitimately
  observable at authorization time, but they belong to the fraud attempt. They are **not** lead time.
* **Legitimate behaviour is almost order-free.** In `_legit_transactions`, each transaction's
  hour, device, network, category and amount are drawn independently given fixed per-customer
  parameters (hour weights, category Dirichlet weights, median/sigma). Order-dependence exists only
  through calendar structure (trips, salary days, device upgrade date), short bursts, and borrowed-device
  runs. There is no habit chain, recurring-bill or Markov structure.
  **Consequence (hypothesis, not a result):** a customer's baseline is essentially summarised by the
  empirical distribution of their past, which hand-built aggregates capture. This generator is
  *aggregation-friendly by construction*, so it is a hard place for a sequence model to show an
  advantage on the first fraud of an episode, whose only evidence is deviation from baseline plus
  recent login failures.
* **Stationary.** No drift in the legitimate population, so robustness to temporal drift cannot
  be assessed on this generator.
* **Group structure.** Rings (3–6 victims) and households (2–3 customers) are linked units. The
  repository's `datasets.build_grouping` and `split._component_customer_split` already keep them
  together.

## 2. Features and what each existing model sees at scoring time

Production features (`features/feature_engineering.py::FEATURE_COLUMNS`), all nine computed from the
customer's earlier rows only; history is updated *after* a row's features are computed (L115–119):

`amount_zscore, hour_is_unusual, is_new_device, is_new_location, is_foreign_location,
failed_logins_24h, category_is_unusual, txn_velocity_1h, amount_pct_of_avg`

| Pipeline | Input at the instant of scoring transaction *i* | What it cannot see |
|---|---|---|
| Nine-feature models (LR, DNN-only, RF-9) | the 9 features of *i* | everything not summarised by those 9 numbers |
| Existing LSTM (`evaluation/windows.py::build_windows`, `build_sequences` L131–161) | the 9 normalised features of the **10 previous** transactions. **Transaction *i* itself is excluded from its own window.** | the current transaction's attributes; raw amounts, gaps, devices, categories |
| LSTM → classifier (`evaluation/downstream.py`, protocol "4D v1") | 9 features of *i* + `risk_score` from the LSTM, which was fitted separately; the classifier trains on **out-of-fold** LSTM scores (`stacking.stacked_risk_scores`) | joint optimisation of the sequence encoder for the current-transaction label |

Cold start: rows with fewer than 10 earlier transactions are not run through the LSTM
(`holdout.score_frame`: risk score set to the training mean). `build_windows` silently drops them.
Any comparison must therefore state which rows it scores.

### Findings that limit how fair the earlier comparison could be (from the code visible to me)

1. **Information bottleneck.** The LSTM receives only normalised, already-summarised features, not raw
   amounts, inter-transaction gaps, device or location identities.
2. **No current-transaction input.** The LSTM's output cannot depend on the transaction being scored.
   It is a recent-history risk signal by design, so it cannot flag a first fraud by itself.
3. **Two-stage training.** The sequence encoder is not trained for the task it is later used for.
4. **Asymmetric tuning.** In `downstream.py` the classifiers get hyper-parameter grids
   (RF 12 settings, HGB 16, LR 8), whereas the LSTM architecture and hyper-parameters are fixed
   ("architectures and hyperparameters unchanged", `docs/EVALUATION.md` L54). I cannot say what the
   ablation did.
5. **Small data.** One 500-customer dataset gives ≈66 training episodes under the 60/20/20 split. A
   data-starved sequence model is a plausible confound for "LSTM underperformed". It is untested.
6. **Metric wording.** `evaluation/metrics.py` describes first-fraud recall as an "early-warning rate"
   and `episode_metrics.hours_to_first_alert` is measured **after** the first fraud. Neither is lead
   time. This experiment will not use the term "early warning" for them.
7. **Hard-coded geography.** `HOME_LOCATIONS` in `feature_engineering.py` is the generator's own
   domestic-city list, so `is_foreign_location` is coupled to the generator. It would be wrong on
   another generator or on real data; the new feature code must derive novelty from the customer's
   own history.

No label leakage was found in the production features or the v2 pipeline. `features/ground_truth.py`
lists nine metadata columns plus `is_fraud` as forbidden features, asserted at import time, and the
v2 sanity report records 0 mismatches for `failed_logins_24h`.

## 3. Splits and evaluation code that can be reused

* `evaluation/split.py`: canonical order `(customer_id, timestamp, transaction_id)`; time split
  (episode groups never straddle a boundary; a ring is one unit); customer split by **components**
  (rings and households stay together). Reusable as is.
* `evaluation/metrics.py`: `select_threshold` (validation F1), `threshold_for_fpr`, tie-safe cut-offs
  (`multiseed.tie_safe_cutoff`), PR-AUC / ROC-AUC, confusion matrices, `episode_metrics`.
  Existing policy budgets: FPR 0.1 % ("critical") and 1 % ("policy_b"); bootstrap = 2,000
  repetitions resampling customer components.
* Seed discipline: `downstream.SPENT_SEEDS = 42, 101–105, 201–205, 401–405, 411–415`; development
  301–305; confirmatory 501–505 / 511–515. The seeds the ablation consumed are unknown to me.

## 4. Can the generator support an early-warning (lead-time) evaluation?

**Partly, and only narrowly.**

* What is established: for 16 % of episodes (18 / 110 at default scale) the generator defines a
  start of the warning period and puts timestamped credential-attack bursts in it. A lead time
  (first alert inside `[precursor_start, first_fraud_time)` to first fraud) is therefore well defined
  for those episodes.
* What is not: the warning signal is a login-failure burst, detectable by a simple rule such as
  "failed logins ≥ 2". There is no behavioural drift for a sequence model to learn. Warning-period
  rows carry label 0, so a model trained on `is_fraud` is *trained not to alert* on them and every
  alert there counts as a false positive in the standard metrics. A valid lead-time analysis needs a
  **separate task with its own labels** and its own false-positive accounting.
* Power: ≈3–4 warning-period test episodes per default dataset.

**Recommendation.** Make first-fraud recall the *only* primary endpoint and describe it as
"detection of the first fraudulent transaction when it is attempted". Run lead time only as an
exploratory secondary analysis on the warning-period archetypes. A rigorous lead-time claim needs the
redesigned generator proposed in `PROTOCOL_DRAFT.md` §10–§11, and even then it would test designed
assumptions, not real-world precursors.

## 5. Statistical power (computed here, closed form, normal approximation)

Test sets in the earlier work had 16 episodes (12 on the customer split; `docs/EVALUATION.md` L120).
A single 500-customer dataset has about 22 first-fraud events in a 20 % test period.

Paired first-fraud events needed for 80 % power, two-sided α = 0.05
(*d* = share of events where exactly one model alerts):

| true difference Δ | d = 0.15 | d = 0.25 | d = 0.40 |
|---|---|---|---|
| 0.05 | 469 | 783 | 1,254 |
| 0.075 | 207 | 347 | 556 |
| 0.10 | 116 | 194 | 312 |
| 0.15 | 50 | 85 | 138 |

95 % half-width of one recall estimate near 0.5: ±0.21 at N = 22, ±0.10 at N = 100, ±0.047 at
N = 440, ±0.021 at N = 2,200. These are planning approximations that assume *d*; they are not results.

**Consequence:** a handful of 500-customer test sets cannot settle a ±5-point difference.
The protocol therefore proposes 2,000-customer datasets (≈440 episodes each) and 5 final datasets
(≈2,200 first-fraud events).

## 6. Environment

* Installed: numpy, pandas only. Installed during this audit into a scratch venv:
  scikit-learn 1.9.1, LightGBM 4.7.0, SciPy, Matplotlib. PyTorch 2.14.1 and TensorFlow-CPU 2.21.0 both
  resolve from PyPI for Python 3.13. 4 CPU cores, 15 GB RAM, no GPU.
* **Egress policy blocks** `kaggle.com`, `huggingface.co`, `archive.ics.uci.edu`, `zenodo.org`,
  `figshare.com`, `data.mendeley.com`, `download.pytorch.org` (HTTP 403 on CONNECT). No public
  dataset can be downloaded from inside this session. I have not tried to bypass this.

## 7. Public datasets: suitability for *this* task

Checked against web sources; column-level claims I could not confirm are marked.

| Dataset | Real? | Entity id for histories | Time | Labels | Verdict for per-customer sequence + first-fraud |
|---|---|---|---|---|---|
| ULB / Worldline "Credit Card Fraud" ([OpenML description](https://api.openml.org/d/42175), [Baselight](https://baselight.app/u/kaggle/dataset/mlg_ulb_creditcardfraud)) | real, anonymised | **none** (columns: `Time`, `V1–V28`, `Amount`, `Class`; inference from the listed columns) | 2 days, seconds from first transaction | 492 / 284,807 fraud | **Unsuitable.** No cardholder to build a history from; no episodes. |
| IEEE-CIS Fraud Detection (Vesta; [IEEE DataPort](https://ieee-dataport.org/documents/ieee-cis-fraud-detection)) | real | no user id; `card1` is the closest proxy ([arXiv 2604.13125](https://arxiv.org/pdf/2604.13125) discusses behavioural signals on it) | `TransactionDT` is a timedelta, not a timestamp | `isFraud`, 20,663 / 590,540 (3.5 %) | **Usable only with a heuristic entity key**; episodes and warning periods undefined; labels are chargeback-based (delayed). First-fraud-per-proxy-entity recall could be reported with heavy caveats. Requires a Kaggle login, so the user would have to supply the file. |
| IBM TabFormer "Synthesizing Credit Card Transactions" ([arXiv 1910.03033](https://arxiv.org/pdf/1910.03033)) | **synthetic** | user and card ids (column names not verified here) | multi-year | fraud flag | Independent *synthetic* source with long histories. Not real-world evidence. |
| Sparkov-based Kaggle set ([Baselight](https://baselight.app/u/kaggle/dataset/kartik2112_fraud_detection)) | **simulated** | `cc_num`; 1,000 customers, 800 merchants | 2019-01-01 → 2020-12-31 | `is_fraud`; `trans_date_trans_time` not confirmed by name | Independent *simulated* source. Not real-world evidence. |

No public dataset I know of provides real per-customer histories **and** explicit episodes **and** a
warning period. Real-data evidence would therefore be limited to a heuristic proxy (IEEE-CIS), and
lead-time claims on real data are not possible.

## 8. Decisions needed from the user (all listed again in the chat summary)

1. Provide `experiments/lstm_value_ablation/` (README, REPORT.md, REPORT_1.md, code, seed list) —
   committed to a branch of this repository is simplest.
2. Confirm the base branch: fast-forward this branch to `origin/Keerthan` (recommended), or keep
   it on `main` and vendor the needed generator/split code.
3. Approve the draft protocol, or amend it.
4. Real-data check: supply the IEEE-CIS files, or allow the hosts, or accept synthetic-only.
