# Step 4C-3D: promotion readiness

This document formalises the gates that must be satisfied before `MODEL_SET`
is allowed to change from `production`. It is documentation only, written at
commit `582fe60`.

Nothing was changed by this step:

* `MODEL_SET` still defaults to `production`;
* no model was promoted, retrained or deployed;
* the 25/50/80 alert bands and all runtime threshold behaviour are unchanged;
* production and candidate model files are byte-identical before and after
  (§9).

**Current position**

| Model set | Role |
|---|---|
| `production` | Loaded by default. Stays in place. |
| `v2_dnn_lstm` | Provisional lead candidate (4C-3C). Not promoted. |
| `v2_dnn_only` | Required challenger (4C-3C). Not promoted. |

No statistical superiority has been established between any two model sets.

**Sources:** `docs/step4c3a-production-decision-audit.md`,
`docs/step4c3b-threshold-policy-analysis.md` and
`docs/step4c3c-model-selection-decision.md`.

## 1. Threshold policy gate

**Proposed operational policy: Policy B** (from 4C-3B).

| Element | Proposed value |
|---|---|
| Legitimate-alert budget | about 1% of legitimate transactions |
| High + Critical together | about 10 legitimate alerts per 1,000 transactions |
| Critical only | about 1 legitimate alert per 1,000 (the Policy A point) |
| Policy C (about 5% budget, about 50 per 1,000) | display-only, for analysis; raises no alert |

**Cut-offs are model-specific score thresholds, not probabilities.** Each
model set needs its own cut-offs to deliver the same budget. The cut-offs
measured on the v2 validation period, on the 0–99.9 score scale, are:

| Tier | production | v2_dnn_lstm | v2_dnn_only |
|---|---|---|---|
| Critical (Policy A point) | 99.51 | 96.81 | 91.19 |
| High + Critical (Policy B point) | 96.80 | 76.77 | 77.44 |
| Display-only (Policy C point) | 55.98 | 58.95 | 57.66 |

**Status of the policy**

* Policy B is a proposal. It is **not** the active runtime policy.
* The application still applies the legacy 25/50/80 bands to every model set.
* Production has no operating points recorded with the model. The candidates'
  manifests record validation operating points, marked "not approved, not
  applied".

**Rule.** Changing `MODEL_SET` before the threshold policy is formally
approved is **prohibited**.

**To close this gate**

1. The project owner approves the budget, the tiers and what each tier
   triggers.
2. The approved cut-offs are recorded per model set, alongside the model.
3. `/predict` applies them in place of the legacy bands, with tests. This is a
   later engineering step and is not part of this document.

## 2. New-customer (cold-start) gate

This gate covers customers with fewer than 10 earlier transactions.

### Observed behaviour (measured)

* **How scoring works today (since 4C-2f-1).**
  * Below 10 earlier transactions the LSTM is not run. The DNN's risk input
    is set to its training average.
  * On a customer's first transaction, the six history-relative features are
    treated as missing.
* **Early transactions are flagged far above the budget.** Legitimate alerts
  per 1,000 under the Policy B cut-offs, on v2 customers:

  | Earlier transactions | production | v2_dnn_lstm | v2_dnn_only |
  |---|---|---|---|
  | 1 | 260 | 40 | 68 |
  | 2 | 122 | 24 | 30 |
  | 3–9 | 54 | 16 | 22 |
  | 10–19 | 15 | 11 | 13 |
  | 20 or more | 10 | 8 | 9 |

* **The unusual-hour and unusual-category flags are the main cause.**
  * With fewer than about 20 earlier transactions, any hour or category the
    customer has not used before is flagged.
  * In a measurement-only rescoring with the two flags neutral, production's
    early alerts under Policies A and B fell to zero, and the candidates'
    second-transaction rates fell from 40 to 14 and from 68 to 20.

### Proposed policy (not implemented)

* **Limited-history state:** treat a customer with fewer than 10 earlier
  transactions as being in a limited-history state.
* **Critical-only handling:** in that state, raise model alerts only at the
  Critical point. Mark everything else "limited history" rather than High.
* **No silent signal removal:** do not silently remove the hour and category
  signals. Whether to neutralise them below a history length is a separate
  scoring decision that needs its own approval and tests.

No runtime behaviour was implemented or changed in this step.

### Evidence still missing

* **The benefit cannot be measured.** Neither v1 nor v2 contains a single
  fraud transaction with fewer than 10 earlier transactions. The data shows
  what alerting on new customers costs. It shows nothing about what it would
  catch.
* So the current evidence cannot establish whether the proposed policy
  improves fraud detection.

**To close this gate:** the policy must be explicitly approved by the owner,
then implemented and tested, before any promotion.

## 3. Data version gate

**The unresolved choice**

| Option | What it means |
|---|---|
| Live v1 customer and history data | The application keeps scoring the existing v1 customers. |
| v2 data and feature semantics | The live history moves to the v2 customers that the candidates were trained and evaluated on. |

**Why it matters**

* The candidates were trained and evaluated on v2. The live system and its
  customer history are v1.
* The same cut-offs behave differently on the two populations. Under
  Policy B the candidates raise about 7–8 legitimate alerts per 1,000 on v2
  and almost none on v1 (0.04–0.11).
* An alert budget must therefore be set on the population the application
  will actually serve.
* Production was trained on legacy-v1 feature values, while live scoring
  computes current values (the 4C-2f-1 feature-version audit). That mismatch
  is part of this decision.
* Switching the live history to v2 would also change every customer, the
  shared-device ring output and the home-device results. It is a separate
  decision from switching models.

**Rule.** Promotion cannot proceed until the live data and feature version is
explicitly chosen and documented.

## 4. Validation evidence gate

**Current limitations**

* The v2 test period has only 14 fraud episodes (61 fraud transactions).
* Only one synthetic seed is represented: one generated dataset and one
  training seed per candidate.
* No ring fraud occurs in the test period.
* New-customer fraud is absent from both datasets.
* Statistical superiority between the candidates is **not** established. At
  Policy B, recall, first-fraud detection and PR-AUC do not separate any pair
  of model sets.

**Proposed evidence target** (from 4C-3C; **none of this has been achieved**)

| Target | Current state |
|---|---|
| At least 100 fraud episodes in the evaluation | 14 |
| At least 5 independent seeds | 1 |
| Ring-fraud coverage in the evaluation | none in the test period |
| New-customer fraud coverage | none in either dataset |
| Evaluation policy fixed before the comparison | proposed (Policy B and the 4C-3C criteria); not approved |
| Customer-level bootstrap uncertainty reported | method exists and is used; to be retained |

Producing this evidence means generating data and training models. Both need
separate approval; neither was done in this step.

## 5. Ring-fraud gate

**The current v2 test period contains no ring fraud.**

* The one ring that falls in the validation period (3 victims, 4 fraud
  transactions) was missed entirely by both candidates, and thresholds were
  chosen on those rows.
* The shared-device ring rule is not a trained model and is the same for
  every model set. On v2 it flags 55 devices, of which 5 belong to rings.

Therefore:

* ring detection cannot currently support a promotion decision;
* ring-specific evaluation must be added to the larger validation set
  (gate 4);
* no claim of ring-fraud superiority may be made for any model set.

## 6. Rollback runbook

**What already exists.** The rollback mechanism has been exercised end to
end (4C-2e-e and 4C-2f-2):

* restarting without `MODEL_SET` restores production in about 5 seconds;
* production's scores after rollback were identical to before the switch;
* every stored transaction keeps the model set and version that scored it.

**What was missing** is the written trigger and decision plan. This section
supplies a draft. It needs the owner's approval, and the trigger levels need
real numbers.

### Rollback triggers

Roll back if any of these occurs after a model-set change:

| # | Trigger | How it is detected |
|---|---|---|
| 1 | The legitimate-alert rate exceeds the approved budget | Alert counts per model set from the stored transactions |
| 2 | Recall regresses materially | Confirmed fraud cases compared with the alerts raised |
| 3 | First-fraud detection regresses materially | First transaction of confirmed fraud episodes |
| 4 | Model loading or startup failure | The backend fails to start, or logs a `ModelSetError` |
| 5 | Model provenance mismatch | `GET /model-info` version or checksums differ from the recorded values |
| 6 | Model-set or version mismatch in persisted transactions | Stored `model_set` / `model_version` differ from the loaded set |
| 7 | Unacceptable latency or resource usage | Scoring time or memory above the agreed limit |
| 8 | Customer-impact or operational issues | Reports from reviewers or customers |
| 9 | Unexplained scoring or reason-code behaviour | Scores or SHAP reasons that cannot be accounted for |

"Materially" is not yet defined. The owner must set the levels for triggers
1, 2, 3 and 7 before promotion.

### Rollback procedure

1. **Stop** the promotion or change activity. Make no further changes.
2. **Restore production:** unset `MODEL_SET`, or set `MODEL_SET=production`.
3. **Restart** the backend.
4. **Verify the model set:** `GET /model-info` reports `model_set` =
   `production` and the recorded production `model_version`.
5. **Verify scoring:** `POST /predict` returns a normal response with the 14
   expected fields.
6. **Verify provenance:** a newly scored transaction is stored with
   `model_set` = `production` and the production `model_version`. Rows scored
   during the candidate period keep their own values.
7. **Verify the frontend:** the Live Scan tab shows "Loaded model set
   production" and the production wording for Risk Score.
8. **Run the production regression tests:** at least
   `tests/test_model_sets.py`, `tests/test_observability.py`,
   `tests/test_predict.py` and `tests/test_persistence.py`; the full backend
   suite if time allows.
9. **Confirm the production model checksums** in `backend/models/saved/`
   match the recorded values (§9).
10. **Record the incident:** the trigger, the time, who decided, the evidence,
    and the decision on what happens next.

**Still open for this gate**

* The owner approves the triggers and sets their levels.
* The owner names who may decide a rollback.
* The procedure is rehearsed once as written, at the promotion commit.

## 7. Promotion checklist

Status key:

* **READY**: satisfied today.
* **OPEN**: not satisfied; the action is known.
* **BLOCKED**: cannot be satisfied until another gate closes or new data
  exists.

| Gate | Requirement | Status | Evidence / action needed |
|---|---|---|---|
| Threshold policy | Threshold policy formally approved | OPEN | Policy B is a proposal. The owner must approve the budget and tiers. |
| Threshold policy | Thresholds recorded per model set | OPEN | The candidates' manifests hold validation points marked "not approved". Production has none. Record the approved cut-offs per model set. |
| New-customer policy | New-customer policy approved | OPEN | The proposal is limited-history, Critical-only handling. The owner must approve; then implement and test. |
| Data version | Live data version selected | OPEN | The owner chooses v1 or v2 and documents it. |
| Validation evidence | At least 100 fraud episodes | BLOCKED | 14 today. Needs approved data generation. |
| Validation evidence | At least 5 seeds | BLOCKED | 1 today. Needs approved generation and training. |
| Ring fraud | Ring fraud represented | BLOCKED | None in the test period. Needs the larger validation set. |
| Validation evidence | New-customer fraud represented | BLOCKED | None in v1 or v2. Needs new data. |
| Validation evidence | Fixed evaluation policy | OPEN | Policy B and the 4C-3C criteria are proposed. Fix them before the new comparison. |
| Validation evidence | Candidate-versus-candidate uncertainty analysis | BLOCKED | Done on 14 episodes (4C-3C) and not separated. Must be repeated on the larger set. |
| Validation evidence | Production-versus-candidate analysis | BLOCKED | Done on 14 episodes (4C-3A, 4C-3C). Must be repeated on the larger set. |
| Rollback | Rollback triggers written | OPEN | Drafted in §6. The owner must approve them and set the levels. |
| Rollback | Rollback procedure tested | READY | The mechanism was exercised end to end. Rehearse the written procedure at the promotion commit. |
| Final regression | Final regression at the promotion commit | BLOCKED | Can run only at a promotion commit. The full suite passes at `582fe60`. |
| Final regression | Production model checksum verification | READY | Checksums recorded and verified unchanged (§9). Repeat at the promotion commit. |
| Final regression | `MODEL_SET` provenance verification | READY | Stored per transaction and exposed by `GET /model-info`; covered by tests. Repeat at the promotion commit. |
| Final regression | Frontend and model wording verification | READY | Model-aware wording in place and verified for all three model sets (4C-2f-2). Repeat at the promotion commit. |

**Summary:** 4 READY, 6 OPEN, 7 BLOCKED. Every item must be READY before
`MODEL_SET` may change.

## 8. Final decision

**NO PROMOTION YET.**

The current evidence supports continuing with v2 candidate validation, with
`v2_dnn_lstm` as the provisional lead and `v2_dnn_only` as the required
challenger.

No statistical superiority is claimed for any model set. `MODEL_SET` remains
`production`.

## 9. Verification

* **Production model files:** the SHA-256 of all 7 files in
  `backend/models/saved/` is unchanged. These are the recorded values:

  | File | SHA-256 |
  |---|---|
  | `dnn_feature_mean.npy` | `cb12c7e19545e838c185c3e3684f601c7ec3fdbd387add15d7d7f907befa75a8` |
  | `dnn_feature_std.npy` | `5f526f065ae0d635eecbabb0b9f636fc4dc63eed778ccb4a8e738fa5aacab155` |
  | `dnn_fraud_model.keras` | `4fd3ab2e190f7c5073d7d100c7f68114babf89b02b321e72a8fb4ebb3c6d3add` |
  | `lstm_feature_mean.npy` | `cd7d3fda6554f7f027d10faf5755a01a9fe2a7d0a67eb5a748d72682a3bcb7e6` |
  | `lstm_feature_std.npy` | `86b57e2ce8b31e366397d3a5b98cb9fbf44d07ecfabf717ad4647e539e927b34` |
  | `lstm_risk_model.keras` | `81c2ab738b6d109fa3ba94f4c8fa5f7303d6f8d114f7ec892e4f5f76abb3f588` |
  | `shap_background.npy` | `ac8ffe3b53434b13880af6a7c76b1746d5a9828701af04be14c54cef697b1519` |

* **Candidate artifacts:** all 15 files under `backend/models/candidates/`
  unchanged.
* **`MODEL_SET`:** default still `production`.
* **Thresholds:** unchanged. The 25/50/80 bands in the code are untouched.
* **Deployment:** none. No model was retrained.
* **Tests run** (existing, to confirm this documentation step changed no
  behaviour): `tests/test_model_sets.py`, `tests/test_observability.py`,
  `tests/test_predict.py` and `tests/test_persistence.py`: 120 passed, 0
  failed, 0 skipped.
* **HEAD:** `582fe60`. Nothing was committed or pushed.
