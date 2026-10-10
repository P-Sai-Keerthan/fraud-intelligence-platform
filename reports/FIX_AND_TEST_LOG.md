# Fix and test log

Everything here was executed in this audit session unless it is marked **not executed**. Raw outputs are in `reports/evidence/output/`.

## 1. Environment

| Item | Value |
|---|---|
| OS / hardware | Linux (container), 4 CPU cores, 15 GB RAM, no GPU |
| Python / Node / npm | 3.13.16 / 22.22.0 / 10.9.4 |
| Backend packages (installed from the project's `requirements.txt` + `requirements-dev.txt`, 75 packages) | TensorFlow 2.21.0, Keras 3.15.1, scikit-learn 1.8.0, SHAP 0.52.0, FastAPI 0.139.0, SQLAlchemy 2.0.36, pandas 3.0.2, numpy 2.4.4, pydantic 2.10.4 |
| Frontend | React 19.2, Vite 8.1, axios (1.18.1 → 1.20.x), Tailwind 4.3; installed with `npm ci` |
| Repository | branch `claude/fair-sequence-fraud-detection-baiokj`; audit base = `origin/Keerthan` @ `5a3a7c8` merged as `843535a` (safety tag `backup-before-merge`) |
| Database for every test and probe | a throw-away SQLite file in a scratch directory; `backend/fraud_platform.db` was never created or touched |

## 2. Baseline (before any code change)

| Check | Command | Result |
|---|---|---|
| Install backend dependencies | `pip install -r requirements.txt -r requirements-dev.txt` | succeeded on Python 3.13 (2 min 35 s) |
| Backend tests | `cd backend && pytest -q` | **611 passed, 4 skipped, 0 failed**, 14 min 04 s. The 4 skips need `data/v2` CSVs that are intentionally not committed. Log: `pytest_BASELINE_before_changes.log` |
| Frontend install, lint, build | `npm ci`, `npm run lint`, `npm run build` | all succeeded (oxlint exit 0; build 1.3 s; one chunk-size warning, 682 kB) |
| Frontend dependency audit | `npm audit --omit=dev` | **2 high** (axios ≤1.19.0, source-map-js) |
| Python dependency audit | `pip-audit` on the exact installed set (69 packages) | no known vulnerabilities |
| Working tree after the test run | `git status` | clean |

## 3. Fixes (one row per finding; details in `BUG_AND_RISK_REGISTER.md`)

| ID | Files modified | Correction | Test and result |
|---|---|---|---|
| F-01, F-02 | `backend/app/schemas.py`, `backend/app/main.py` | `amount`: `allow_inf_nan=False`, `le=1e9`; validation-error handler that stringifies non-finite inputs | `test_non_finite_or_absurd_amount…[Infinity, -Infinity, NaN, 2e9, 1e308]` fail before, pass after |
| F-03 | `schemas.py`, `main.py` (batch) | `ShortText`: stripped, 1–64 characters for `customer_id`, `merchant_category`, `device_id`, `location`; same limits in batch rows | 12 parametrised tests + whitespace-strip test |
| F-04 | `schemas.py` | timestamp validator: aware → naive UTC | 2 tests (`Z`, `+05:30`) |
| F-05 | `backend/app/inference_pipeline.py`, `main.py` | `remove_transaction()`; `/predict` and `/predict/batch` roll the in-memory history back if persistence fails; `failed_logins_24h ≤ 10,000` | 4 tests (exact restore, new-customer removal, failed single commit, failed batch commit) + login bound test |
| F-06 | `main.py` | `Query(ge, le)` on `limit` and `min_customers` | 5 tests + 1 positive test |
| F-07 | `main.py` | filename restricted to `[A-Za-z0-9_.-]`, ≤64 chars | 4 tests (quote, Unicode, CRLF, traversal) |
| F-08 | `main.py` | `MAX_BATCH_BYTES = 2 MB`, `MAX_BATCH_ROWS = 500` (env-overridable), HTTP 413 | 3 tests |
| F-09 | `main.py` | `/health` runs `SELECT 1`; 503 on failure | 2 tests |
| F-10, F-11 | `frontend/package.json`, `package-lock.json`, `src/api.js` | axios `^1.20.0`; `npm audit fix`; `encodeURIComponent`; safe download filename | `npm audit` 0 vulnerabilities; lint exit 0; build OK; end-to-end run |

New test module: `backend/tests/test_audit_fixes.py` (41 tests).

**Do the new tests detect the defects?** The same module was run against an untouched copy of the original code (`git archive HEAD`):
**38 failed, 3 passed** (the 3 are control tests of behaviour that was already correct). Against the fixed code: **41 passed**.

One test failed on its first run for a reason worth recording: after removing a transaction the restored history differs from the *stored* v1 seed
features, because the committed seed CSV was produced by an earlier feature version (R-08). The code was right; the test's reference was wrong and was
changed to the current-code recomputation, which is exactly what live scoring uses.

## 4. After the fixes

| Check | Result |
|---|---|
| Full backend suite (`pytest -q`) | **652 passed, 4 skipped, 0 failed**, 14 min 20 s (= 611 + 41; no regression). Log: `pytest_AFTER_fixes.log` |
| Frontend `oxlint` / `npm run build` / `npm audit` | exit 0 / success / 0 vulnerabilities |
| Black-box probes, original vs fixed code | `probe_api_BEFORE_original_code.txt` vs `probe_api_AFTER_fixed_code.txt`; `probe_api2_*` (same pairing). Every rejected input is now 422 and leaves memory and database unchanged |
| End-to-end, real stack | `uvicorn` (default model set `v2_lstm_rf_seed14`, throw-away database) + `vite preview` (production build) driven by headless Chromium |

End-to-end results (screenshots in `reports/assets/ui_*.png`):

| Step | Observed |
|---|---|
| Dashboard loads, customer list populated (CUST_0001 selectable), model set shown | yes (`v2_lstm_rf_seed14`, status badge PRODUCTION) |
| "Typical Purchase" for CUST_0001 | **Low Risk**, fraud score 0.1, risk score 15, similarity 38.1 %, no reasons |
| "Suspicious Pattern" (₹75,000, electronics, unknown device, Lagos, 4 failed logins) | **Critical Risk**, fraud score 89.5, risk score 13, similarity 0.5 %, four SHAP reasons (Foreign Location, New Device, Multiple Failed Logins, Amount Far Above Average) |
| Download Report | PDF downloaded (2 pages, text matches the screen, names model set, version, seed) |
| Batch scoring with `test_data/batch_scoring/batch_mixed_transactions.csv` | scored; summary and table rendered; mix of Low/High/Critical rows |
| Fraud Rings and Model Performance tabs | rendered |
| Server errors during the run | none; one expected `404` (history before the first scan), which the UI handles |

## 5. Measurements taken during the audit

| Measurement | Value | Note |
|---|---|---|
| Single `/predict` latency | 0.38–0.46 s | 5 sequential calls, while the test suite was also running |
| Batch scoring rate | 0.41 s per row | 40 rows in 16.4 s on an otherwise idle machine; 400 rows took 236 s under load |
| Feature rebuild per request vs history length | 0.13 s (100 rows), 0.40 s (400), 0.99 s (800), 2.26 s (1,600), 6.14 s (3,200) | single runs, other load present: indicative |
| Seed data | 93,913 rows, 500 customers, 819 fraud rows (0.872 %), 60 fraud customers, no NaN/inf, no duplicate ids | |
| Saved models | LSTM 31,905 parameters (input 10×9); production DNN 3,585 parameters; random forest 200 trees, depth 12, min leaf 5, 10 inputs | |

## 6. Not executed (stated plainly)

* The 4 skipped backend tests (need `data/v2` CSVs generated with `python data/v2/generate.py`).
* The hold-out evaluation was **not re-run** (scored once by design; per-row scores are not committed).
* `python data/v2/sanity_report.py`, model retraining, and the evaluation commands in README §5 (about 11 minutes each or more) were not run.
* PostgreSQL (`DATABASE_URL`) was not tested; the README itself says it is untested.
* Authentication, rate limiting and TLS do not exist; there was nothing to test. Load and concurrency beyond the project's own 3 concurrency tests were not tested.
* No frontend unit tests exist, so none were run.
* The lower-case branch `origin/keerthan` (rate limiter, body-size limit, stricter CORS) was read but not run.
