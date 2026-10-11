# Feature mapping — nine Behavioral Fraud DNA features versus IEEE-CIS

Principle: the existing nine features are **not forced** onto the public data. Each is classified as **R** (reproducible from available fields with the same logic), **S** (needs a defensible substitute whose meaning differs), or **X** (cannot be evaluated).
All statistics use only *earlier* rows of the same proxy entity (past-only, strict inequality on time, ties broken by `TransactionID` only for ordering, never as a feature). Field names are those the secondary sources report [S]/[M]; every mapping is **provisional until stage 0** confirms the columns exist and are populated.

| # | Existing feature [CODE] | IEEE-CIS counterpart | Class | Notes and risks |
|---|---|---|---|---|
| 1 | `amount_zscore`: (amount − mean of earlier amounts) / max(std, 0.1·mean, 1), clipped ±10 | `TransactionAmt` against the entity's earlier amounts | **R** | currency and scaling of `TransactionAmt` unknown; heavy-tailed, so the log of the amount is added as a **second, documented variant** (not a replacement of the original definition) |
| 2 | `hour_is_unusual`: share of earlier rows in this clock hour < 5 % | hour-of-day from `TransactionDT` modulo 86,400 | **S** | the reference datetime and time zone are unknown, so the hour is only a *constant offset* of the true local hour. "Unusual relative to the entity's own pattern" is still meaningful; absolute clock claims are not. Flagged "relative hour" |
| 3 | `is_new_device`: device never used before | `DeviceInfo` / `DeviceType` / selected `id_*` (browser, OS, device) | **S** | identity rows exist for only a subset [M]; for the rest the feature is *missing*, not "not new". Encoded as three states (new / seen / unknown), unknown share reported |
| 4 | `is_new_location`: city never used before | `addr1` (billing region code), `addr2` (country code), `dist1` | **S** | coarse, masked region codes, not cities; "new region code for this entity". Billing address is not the transaction location |
| 5 | `is_foreign_location`: city not in a hard-coded list of 10 Indian cities | `addr2` different from the entity's modal country code | **S** | the original list is generator-specific (documented limitation in the audit); the substitute is entity-relative, so it measures "unlike this entity's usual country" |
| 6 | `failed_logins_24h` | — | **X** | no login events exist. Dropped. Any proxy (e.g. `id_*` fields) would invent a meaning |
| 7 | `category_is_unusual`: share of earlier rows in this merchant category < 5 % | `ProductCD` (few categories) and optionally `card6`/`card4` | **S** | `ProductCD` is a product class, not a merchant category; with only a handful of values the 5 % rule behaves differently. Reported as "product class unusual" |
| 8 | `txn_velocity_1h`: earlier rows within 1 h | rows of the same entity in the preceding 3,600 s of `TransactionDT` | **R** | valid if `TransactionDT` is in seconds [M]; sensitive to entity quality (two people on one `card1` look like high velocity) |
| 9 | `amount_pct_of_avg`: 100·amount / mean of earlier amounts, clipped [0, 1000] | same | **R** | as feature 1 |
| – | `risk_score` (LSTM output ×100, out-of-fold for training rows) | recomputed on IEEE-CIS sequences | **R (retrained)** | the existing LSTM weights are **not used**; a new LSTM is trained on this dataset's sequence steps |

Summary: reproducible R = {1, 8, 9} (3 features); substitutes S = {2, 3, 4, 5, 7} (5 features); not evaluable X = {6} (1 feature).

## Additional features proposed for the *new* schema (not part of the original nine)

All past-only and entity-relative, defined before looking at labels:

* entity history length (earlier row count), time since the previous row (log seconds), count of rows in the last 24 h and 7 days of `TransactionDT`;
* mean/std of log amount over the last 10 and over all earlier rows; ratio of this amount to the entity's maximum so far;
* share of earlier rows with each `ProductCD` / `card6` / `card4`; "first time this email domain / region code for this entity" flags;
* missingness indicators for the identity fields (reported, and ablated, because identity presence may itself be informative).

## Sequence step for the LSTM / GRU (window of the last K = 10 earlier rows, current row excluded, as in the existing design)

Per step: log amount, log gap since the previous row, relative hour (sin/cos), `ProductCD` one-hot, new/seen flags for region code, country code, device (with an "unknown" state), email-domain-seen flag, step-mask for padding. The entity id, `TransactionID` and any raw card string are never inputs.

## Excluded from the primary arms

`C1`–`C14`, `D1`–`D15`, `M1`–`M9`, `V1`–`V339` (data-owner features of unknown time basis; context arm only), `TransactionID`, all raw `card*` strings (identity), `isFraud` of any row.

## Not reproducible, therefore not evaluable on this dataset

`failed_logins_24h`; the original `is_foreign_location` list; device-level behaviour for the majority of rows (if identity coverage is as low as expected); the Behavioral Similarity Score as defined (it would be computed on the substitute features and must be labelled "similarity on IEEE-CIS features", never compared numerically with the synthetic result).
