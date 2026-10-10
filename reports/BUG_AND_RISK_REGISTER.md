# Bug and risk register

Explainable Fraud Intelligence Platform — audit of branch `claude/fair-sequence-fraud-detection-baiokj`
(base: `origin/Keerthan` @ `5a3a7c8`, merged with approval as `843535a`). **Updated in the follow-up pass** (software commit `6c7eea9`,
Experiment 3 commits `b58fa52`, `e81b356`, `687dc7e` and the results commit that follows them): R-04, R-05, R-06 and R-09 are now fixed, R-11 and R-12 partly,
and the scientific findings in section C are extended with Experiment 3.

**Reading guide.** *Confirmed* = reproduced by a test or probe whose output is saved in `reports/evidence/`.
*Potential risk* = follows from the code but depends on a condition (stated). *Unverified* = not tested here.
Priorities: **P0** critical (immediate, severe), **P1** high, **P2** medium, **P3** low.
"Fixed" means the fix was verified by regression tests **and** by before/after probes or the browser run; the full suite passed after the first pass
(652 passed, 4 skipped, 0 failed) and again after the follow-up pass (**707 passed, 4 skipped, 0 failed**, run on the code of commit `6c7eea9`).

## Summary

| Priority | Confirmed defects fixed in the first pass (F-xx) | Open items fixed in the follow-up | Open items partly fixed | Open items still open |
|---|---|---|---|---|
| P0 | 0 | 0 | 0 | 0 |
| P1 | 1 (F-01) | 0 | 0 | 1 (R-01, conditional; documented requirements, no scheme chosen) |
| P2 | 5 (F-02, F-03, F-04, F-05, F-08) | 2 (R-04, R-05) | 0 | 2 (R-02, R-03) |
| P3 | 5 (F-06, F-07, F-09, F-10, F-11) | 2 (R-06, R-09) | 2 (R-11, R-12) | 6 (R-07, R-08, R-10, R-13, R-14, R-15) |
| **Total** | **11** | **4** | **2** | **9** (R-01, R-07 are potential risks) |

No P0 finding was identified. The project is a synthetic-data demonstrator with no authentication; the ratings
assume it is run locally. R-01 changes materially if the API is ever exposed on a network.

## A. Confirmed defects (all fixed)

### F-01 (P1) — `amount = Infinity` is accepted, stored, and permanently corrupts the customer
* **Component:** `schemas.TransactionInput`, `inference_pipeline`, database.
* **Evidence:** `evidence/output/probe_api2_BEFORE_original_code.txt` §2. `POST /predict` with `"amount": Infinity`
  (non-standard JSON that Python's parser accepts) returned **200** with `"amount": null`; the database stored `inf`;
  the customer's later `amount_zscore` values became `nan`, silently, while requests still returned 200. Replaying the
  database (what a restart does) reproduces the corrupted history. `1e308` was also accepted. `/predict/batch` already
  rejected `inf`/`nan`/`1e999` (tests exist), so the two paths disagreed.
* **Root cause:** `amount: float = Field(gt=0)` has no finiteness or upper bound; the history is persisted before use.
* **Consequence:** one unauthenticated request degrades one customer's scoring until the rows are deleted by hand.
* **Fix:** `allow_inf_nan=False`, `le=1_000_000_000`. **Verification:** 5 parametrised tests (`Infinity`, `-Infinity`,
  `NaN`, `2e9`, `1e308`) fail on the old code and pass now; AFTER probe returns 422 and `amount_zscore` stays finite.
* **Remaining limitation:** the 1e9 ceiling is a chosen constant (data maximum is about 1e6).

### F-02 (P2) — `amount = NaN` returns HTTP 500 instead of 422
* **Evidence:** BEFORE probe 1 §B (`[500] amount = NaN`). **Root cause:** the validation-error body echoes the offending
  input; `NaN` cannot be serialised to JSON, so building the 422 response raised. **Fix:** a
  `RequestValidationError` handler (`_json_safe`) that stringifies non-finite values; shape of `detail` unchanged.
* **Verification:** test `…[NaN]` fails before / passes after.

### F-03 (P2) — blank, whitespace-only and arbitrarily long identifiers are accepted
* **Evidence:** BEFORE probe 1 §B: `customer_id` of `""`, `"   "` and 100,000 characters returned 200; the customer count
  rose from 500 to 503; empty `device_id`, `location`, `merchant_category` were accepted.
* **Root cause:** string fields had no length constraints. **Consequence:** phantom customers in `/customers` and the
  fraud-ring view, unbounded database growth. **Fix:** stripped strings, 1–64 characters (`ShortText`); the batch path
  applies the same limits. **Verification:** 12 parametrised tests (4 fields × `""`, spaces, 65 chars) + whitespace-strip test.

### F-04 (P2) — a timezone-aware timestamp returns HTTP 500
* **Evidence:** BEFORE probe 1 §B: `"2026-07-01T10:00:00Z"` and `"…+05:30"` → 500. **Root cause:** stored histories use
  naive timestamps; mixing aware and naive values fails inside the history merge. ISO-8601 with `Z` is the standard
  browser/API format. **Fix:** validator converts aware values to naive UTC. **Verification:** both forms return 200 and
  the expected UTC time (`19:00+05:30` → `13:30`).

### F-05 (P2) — in-memory history can get ahead of the database
* **Evidence:** BEFORE probe 2 §1: `failed_logins_24h = 10**30` → 500 (integer overflow at commit) **but** the customer's
  in-memory history grew 97→98 rows while the database had 0; the next request built on the phantom row (99 vs 1), and a
  restart would silently discard it. **Root cause:** `score_transaction` mutates the shared history before the commit; any
  commit failure (overflow, locked or full database) or a mid-batch failure leaves the two stores inconsistent.
* **Fix:** `FraudIntelligencePipeline.remove_transaction` (rebuilds the history without the row) called when the commit
  fails, for single and batch scoring. `failed_logins_24h` is also bounded (≤10,000).
* **Verification:** unit test that removal restores exactly the recomputed history; test that a failed commit leaves memory
  unchanged for `/predict` and for a 3-row `/predict/batch`; AFTER probe 2 shows memory 97 / database 0 after a rejected request.
* **Remaining limitation:** a crash *between* scoring and rollback (process kill) is not covered; a restart rebuilds from the database.

### F-06 (P3) — unbounded query parameters
* **Evidence:** BEFORE probe 1 §C: `GET /customers?limit=-1` returned 502 of 503 ids (silently drops the last);
  `history?limit=-1` returned everything; `fraud-rings?min_customers=0` accepted. **Fix:** `Query(ge=…, le=…)`:
  customers 1–10,000, history 1–1,000, rings ≥1. The dashboard requests 500, within range. **Verification:** 5 tests.

### F-07 (P3) — PDF download filename: header parameter injection and a 500
* **Evidence:** BEFORE probe 1 §D: a `transaction_id` of `TXN"; evil=1` produced
  `Content-Disposition: attachment; filename="fraud_report_TXN"; evil=1.pdf"`; a non-Latin-1 id returned 500.
  **Root cause:** the client-supplied id was placed in the header unescaped. **Fix:** only `[A-Za-z0-9_.-]`, max 64
  characters. **Verification:** 4 tests (quote, Unicode, CRLF, path traversal).

### F-08 (P2) — batch upload has no size or row limit
* **Evidence:** measured 0.41 s per row on an idle machine (40 rows = 16.4 s; 400 rows = 236 s while other work ran). The
  endpoint is unauthenticated and holds a worker thread for the whole upload; the dashboard gives up after 5 minutes while
  the server keeps scoring and later commits. **Fix:** 2 MB and 500 rows (`BATCH_MAX_BYTES`, `BATCH_MAX_ROWS` override),
  HTTP 413 before anything is scored; same per-field limits as `/predict`. **Verification:** 3 tests.
* **Remaining limitation:** throughput itself is unchanged (see R-03).

### F-09 (P3) — `/health` always reports ok
* **Root cause:** returned a constant. **Fix:** runs `SELECT 1`; 503 `{"status": "database unavailable"}` on failure,
  `{"status": "ok"}` otherwise (the frontend treats any non-ok as offline). **Verification:** 2 tests.

### F-10 (P3) — vulnerable frontend dependencies
* **Evidence:** `npm audit --omit=dev` before: 2 high (axios 1.18.1 with 12 advisories; source-map-js). Most axios advisories concern
  the Node/fetch/HTTP2 adapters; this app uses axios in the browser with a fixed base URL, so practical exposure was low.
  **Fix:** axios `^1.20.0`, `npm audit fix`. **Verification:** `npm audit` → 0 vulnerabilities; `oxlint` exit 0; production build succeeds;
  end-to-end browser run works.

### F-11 (P3) — frontend path segments and download filename not encoded
* **Fix:** `encodeURIComponent` for customer ids in URLs, `params` for `limit`, sanitised download name (`frontend/src/api.js`).
  **Verification:** build and end-to-end run; no dedicated frontend unit tests exist (see R-12).

## B. Open items (status after the follow-up pass)

| ID | Pri | Type | Finding | Evidence | Why not fixed / next step |
|---|---|---|---|---|---|
| R-01 | P1* | Potential risk | **No authentication or authorisation** on any endpoint (scoring, history, rings, reports, batch). *P1 only if the API is reachable beyond the local machine; P3 for a local demo.* | `main.py` has no auth dependency | Needs a decision on the scheme. Requirements, three options and the open decisions are written in `docs/deployment-security-requirements.md`. Safe defaults added without choosing a scheme: CORS limited to GET/POST/OPTIONS and two headers, `X-Content-Type-Options: nosniff`, `Cache-Control: no-store` on API responses, a declared-`Content-Length` guard (413), and `API_DOCS=off` to hide the interactive docs. **Limits:** the length guard sees only a declared header (a chunked body is not counted; the proxy must enforce size), and none of this authenticates or rate-limits anything. |
| R-02 | P2 | Confirmed (documented) | **Alert bands (25/50/80) were designed for the earlier DNN and are applied to the random forest**, whose validated alert cut-off is a score of 4.39. A typical purchase scored 6.01 and is shown "Low Risk" although it exceeds the model's own operating point; reasons are listed under "Why was this transaction flagged?" for it. | `/model-info` states "legacy fixed bands … not derived from any evaluation of the loaded model"; probe 1 §A; `final_holdout_report.json` `cutoffs.policy_b.selected = 0.0439` | Changing it alters API semantics and every threshold-dependent test. Decision: adopt the validated policy cut-offs (`policy_b`, `critical`) as alert bands, or keep the bands and relabel them. |
| R-03 | P2 | Confirmed (documented) | **Per-request cost grows with history length**: features are recomputed from the full history on every request (≈1–2 ms per row; 6.1 s at 3,200 transactions) and all scored rows are held in memory and replayed at start-up. | `reports/assets/fig_latency.png`; `inference_pipeline.py` docstring | Architectural (incremental rolling statistics). Not needed at demo scale. |
| R-04 | P2 | **Fixed** | `/report/pdf` used to print client-supplied scores (a forged `fraud_probability`/`alert_level` for a real id was accepted with 200). | probe 1 §D (before) | `POST /report/pdf` now takes only a validated `transaction_id`, reads the stored row (404 if unknown) and prints stored values; the SHAP reasons are persisted in a new nullable `reasons_json` column (added by `ensure_schema`). Tests: tampered extra fields are ignored; malformed/missing ids are 422; unknown id 404; rows stored before the change print "explanation not stored". **Compatibility change:** extra fields in the body are ignored; the dashboard now sends only the id. |
| R-05 | P2 | **Fixed** | Behavioral Similarity was computed against placeholder baselines for customers with little history (98 % "deviation" for an ordinary first purchase). | probe 1 (before); `test_similarity_history.py` | `similarity_pct`/`deviation_pct` are `null` with `similarity_status = "insufficient_history"` and `history_transactions = n` below 10 earlier transactions (same threshold as the LSTM); the formula is unchanged above it. API schema, PDF ("n/a"), batch output and both dashboard cards updated; the UI shows "Not enough history" and no number, ring or bar. Tests for 0, 1, 9 (insufficient) and 10, 50 (sufficient) earlier transactions; 3 of 13 frontend tests fail on the old components. **Compatibility change:** clients must accept `null`. |
| R-06 | P3 | **Fixed** | Timestamps were unbounded (year 1 and 9999 accepted); a far-future transaction became the "newest" anchor for the home-device window. | probe 2 §3 | Accepted window 2000-01-01 to 2100-01-01 (back-dating inside it still works); 422 outside. 4 boundary tests. The window is a policy choice and is easy to change (`schemas.py`). |
| R-07 | P3 | Potential risk | `MODEL_SET=production` artifacts in `models/saved/` are not SHA-256 pinned (the default RF and seed-14 sets are, and are hashed before the pickle is loaded). Not the default. | `model_sets.py` | Add pins if that set is kept. |
| R-08 | P3 | Confirmed (documented) | The committed v1 seed CSV was produced by an earlier feature version (the std floor and clipping were added later); 43,442 of 93,913 `amount_zscore` values differ from a current-code recomputation. The `production` v1 model was trained on the old scale but scores with the new. My regression test initially tripped on exactly this. The default RF set was trained on v2 data built with current code. | `docs/step4c2f-1-v1-feature-version-audit.md`; test failure during the audit | Retraining is out of scope; document only. |
| R-09 | P3 | **Fixed** | v1 data generation was not reproducible (dates anchored on `datetime.now()`). | source | `generate_dataset(start_date=START_DATE, seed=RNG_SEED)` with a fixed default start date (2026-01-12) and explicit reseeding; test: two generations with the same arguments are identical, a different start date or seed differs. The committed v1 seed CSV is **not** regenerated or replaced. |
| R-10 | P3 | Confirmed | Reproducibility/hygiene: `keras` is a range (`>=3.15,<3.16`), so the requirements cannot be hashed; `psycopg2-binary` is installed but PostgreSQL is "untested"; there is no migration framework (`ensure_schema` issues `ALTER TABLE`); `_to_delete/` (two temp files) is tracked; histories use naive local time (`datetime.now()`). | `requirements.txt`, `db/migrations.py`, `git ls-files` | Low impact; list for housekeeping. |
| R-11 | P3 | **Partly fixed** | Frontend: validation errors were shown as raw `JSON.stringify(detail)`; one 689 kB JS chunk (build warning); gauge tint bands (25/50/80) repeat R-02. | build output; `App.jsx`, `severity.js` | Errors are now formatted (`apiError.js`: field and message, no "[object Object]"; verified in the browser run). Chunk warning and the gauge bands (tied to R-02) are unchanged. |
| R-12 | P3 | **Partly fixed** | No automated frontend tests. | `frontend/package.json` | `npm test` (vitest, static rendering) with **13 tests** covering the similarity states, the error formatter and the PDF request body; lint and build pass. Component interaction, the batch table and the charts are still not unit-tested. |
| R-13 | P3 | Confirmed (documented) | **Reports and history stored before the follow-up have no stored explanation:** transactions saved before `reasons_json` existed print "the explanation was not stored" in PDFs. | `test_report.py` | Expected; no back-fill (the reasons cannot be recomputed from the stored row without replaying the model). |
| R-14 | P3 | Confirmed (Experiment 3) | **The Behavioral Similarity formula is dominated by near-constant features:** in 94.5 % of scored rows at least one of the nine features has zero standard deviation in the customer's history and the formula replaces that standard deviation by 1, so a binary flag that is always 0 and suddenly 1 contributes a z-score of 1 while a continuous feature can contribute hundreds. Flooring the standard deviation at 0.5 did **not** help (ROC-AUC and PR-AUC lower, intervals exclude 0). | `experiments/validation_exp3/RESULTS.md` | Report only; the formula is unchanged by design (changing it needs its own protocol). |
| R-15 | P3 | Confirmed | **Per-row scoring in Experiment 3 and the API are consistent but the dashboard cannot create a new customer**, so the "Not enough history" UI state was verified in the browser by feeding the real API response of a new customer through the real UI (request interception), not by clicking through a new-customer flow. | `reports/evidence/` e2e notes | Add a customer-creation or manual-id path if that flow is wanted. |

## C. Scientific and evaluation findings (see `ML_EXPERIMENT_STATUS.md`)

* **Verified:** the reported v2 time-split metrics (PR-AUC, ROC-AUC, confusion matrix) are reproduced exactly from the committed raw scores; the
  default model's hold-out headline numbers are internally consistent (recall = TP/(TP+FN) etc.) and backed by a protocol hash and dataset hashes.
* **Limitations:** all data are synthetic; v1 data are trivially separable (`amount_pct_of_avg` alone has ROC-AUC 1.000); the v2 time-split
  test period has only 61 fraud rows (PR-AUC 0.10–0.14); the hold-out comes from the same generator as the training data; the Fraud Score is
  not calibrated (no calibration in the serving path); the LSTM does not warn before the first fraud (documented by the project itself).
* **Not verified here:** the hold-out result was not re-run (scored once by design; per-row scores are not committed).
* **Experiment 3 (synthetic, five fresh seeds 9301–9305; `experiments/validation_exp3/RESULTS.md`):** the Behavioral Similarity Score separates fraud from legitimate rows on the
  generator's own fraud types (ROC-AUC 0.866) but adds no measurable information beyond the classifier; the default random-forest score is close to calibrated on this
  generator and calibrators change the Brier score by about 1e-4; the LSTM adds detectable value over the same forest without it. All of this is in-distribution synthetic
  evidence; see the independent-validation note `experiments/validation_exp3/INDEPENDENT_VALIDATION_STATUS.md` for what is missing.
