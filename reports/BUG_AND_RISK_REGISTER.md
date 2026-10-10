# Bug and risk register

Explainable Fraud Intelligence Platform — audit of branch `claude/fair-sequence-fraud-detection-baiokj`
(base: `origin/Keerthan` @ `5a3a7c8`, merged with approval as `843535a`).

**Reading guide.** *Confirmed* = reproduced by a test or probe whose output is saved in `reports/evidence/`.
*Potential risk* = follows from the code but depends on a condition (stated). *Unverified* = not tested here.
Priorities: **P0** critical (immediate, severe), **P1** high, **P2** medium, **P3** low.
"Fixed" means the fix was verified by the regression tests in `backend/tests/test_audit_fixes.py` **and** by the
before/after probes; the full suite then passed (652 passed, 4 skipped, 0 failed).

## Summary

| Priority | Confirmed defects found and fixed (F-xx) | Open items (R-xx): confirmed, needing a decision or a larger change | of which only a potential risk |
|---|---|---|---|
| P0 | 0 | 0 | 0 |
| P1 | 1 (F-01) | 1 (R-01, conditional) | 1 |
| P2 | 5 (F-02, F-03, F-04, F-05, F-08) | 4 (R-02, R-03, R-04, R-05) | 0 |
| P3 | 5 (F-06, F-07, F-09, F-10, F-11) | 7 (R-06 … R-12) | 1 (R-07) |
| **Total** | **11** | **12** | **2** |

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

## B. Open items (not fixed — they need a decision or a larger change)

| ID | Pri | Type | Finding | Evidence | Why not fixed / next step |
|---|---|---|---|---|---|
| R-01 | P1* | Potential risk | **No authentication or authorisation** on any endpoint (scoring, history, rings, reports, batch). *P1 only if the API is reachable beyond the local machine; P3 for a local demo.* | `main.py` has no auth dependency; CORS allows only localhost by default | Needs a decision on the scheme (API key, OAuth, gateway). The un-merged branch `keerthan` contains a body-size limiter and per-IP rate limiter (`security.py`), not evaluated here. |
| R-02 | P2 | Confirmed (documented) | **Alert bands (25/50/80) were designed for the earlier DNN and are applied to the random forest**, whose validated alert cut-off is a score of 4.39. A typical purchase scored 6.01 and is shown "Low Risk" although it exceeds the model's own operating point; reasons are listed under "Why was this transaction flagged?" for it. | `/model-info` states "legacy fixed bands … not derived from any evaluation of the loaded model"; probe 1 §A; `final_holdout_report.json` `cutoffs.policy_b.selected = 0.0439` | Changing it alters API semantics and every threshold-dependent test. Decision: adopt the validated policy cut-offs (`policy_b`, `critical`) as alert bands, or keep the bands and relabel them. |
| R-03 | P2 | Confirmed (documented) | **Per-request cost grows with history length**: features are recomputed from the full history on every request (≈1–2 ms per row; 6.1 s at 3,200 transactions) and all scored rows are held in memory and replayed at start-up. | `reports/assets/fig_latency.png`; `inference_pipeline.py` docstring | Architectural (incremental rolling statistics). Not needed at demo scale. |
| R-04 | P2 | Confirmed | **`/report/pdf` prints client-supplied scores**; a forged `fraud_probability`/`alert_level` for a real transaction id is accepted (200). | probe 1 §D (both before and after) | Fix = look the scores up by `transaction_id` server-side (changes the documented contract "takes what /predict returns"). Matters only if reports are treated as records. |
| R-05 | P2 | Confirmed | **Behavioral Similarity is not meaningful for customers with no history**: a brand-new customer's ordinary first purchase scores 1.9 % (98 % "deviation"), because it is compared with an all-zero baseline; with one prior row std is forced to 1. | probe 1 (last lines); `similarity.py`, `inference_pipeline.py` | Needs a product decision on what the API returns when there is no baseline (null, neutral, or a flag). |
| R-06 | P3 | Confirmed | Timestamps are unbounded (year 1 and 9999 accepted); a far-future transaction becomes the "newest" anchor for the home-device window. | probe 2 §3 | Back-dating is a designed feature; a plausible window (e.g. 2000–2100) would be a policy choice. |
| R-07 | P3 | Potential risk | `MODEL_SET=production` artifacts in `models/saved/` are not SHA-256 pinned (the default RF and seed-14 sets are, and are hashed before the pickle is loaded). Not the default. | `model_sets.py` | Add pins if that set is kept. |
| R-08 | P3 | Confirmed (documented) | The committed v1 seed CSV was produced by an earlier feature version (the std floor and clipping were added later); 43,442 of 93,913 `amount_zscore` values differ from a current-code recomputation. The `production` v1 model was trained on the old scale but scores with the new. My regression test initially tripped on exactly this. The default RF set was trained on v2 data built with current code. | `docs/step4c2f-1-v1-feature-version-audit.md`; test failure during the audit | Retraining is out of scope; document only. |
| R-09 | P3 | Confirmed | **Data generation is not reproducible for v1**: `data/generate_synthetic_data.py` anchors dates on `datetime.now()`. v2 is deterministic (seed 42 regenerated here matched all five committed SHA-256 hashes). | source; `experiments/…/determinism_seed42.txt` | Pass a fixed start date. |
| R-10 | P3 | Confirmed | Reproducibility/hygiene: `keras` is a range (`>=3.15,<3.16`), so the requirements cannot be hashed; `psycopg2-binary` is installed but PostgreSQL is "untested"; there is no migration framework (`ensure_schema` issues `ALTER TABLE`); `_to_delete/` (two temp files) is tracked; histories use naive local time (`datetime.now()`). | `requirements.txt`, `db/migrations.py`, `git ls-files` | Low impact; list for housekeeping. |
| R-11 | P3 | Confirmed | Frontend: one 682 kB JS chunk (build warning); validation errors are shown as raw `JSON.stringify(detail)`; gauge tint bands (25/50/80) repeat R-02. | build output; `App.jsx`, `severity.js` | Cosmetic. |
| R-12 | P3 | Gap | No automated frontend tests; end-to-end checks in this audit were manual-scripted (Playwright run saved as screenshots). | `frontend/package.json` has no test script | Add component/API-contract tests. |

## C. Scientific and evaluation findings (see `ML_EXPERIMENT_STATUS.md`)

* **Verified:** the reported v2 time-split metrics (PR-AUC, ROC-AUC, confusion matrix) are reproduced exactly from the committed raw scores; the
  default model's hold-out headline numbers are internally consistent (recall = TP/(TP+FN) etc.) and backed by a protocol hash and dataset hashes.
* **Limitations:** all data are synthetic; v1 data are trivially separable (`amount_pct_of_avg` alone has ROC-AUC 1.000); the v2 time-split
  test period has only 61 fraud rows (PR-AUC 0.10–0.14); the hold-out comes from the same generator as the training data; the Fraud Score is
  not calibrated (no calibration in the serving path); the LSTM does not warn before the first fraud (documented by the project itself).
* **Not verified here:** the hold-out result was not re-run (scored once by design; per-row scores are not committed).
