# Final review — demo checklist

> **Step 4D update (8 October 2026).** The default model set is now
> `v2_lstm_rf_seed14`: the same seed-14 LSTM, followed by a **random forest**
> instead of a DNN, chosen by a pre-registered comparison and confirmed on a
> fresh hold-out (`docs/model_selection_report.md`). This checklist was
> written for the previous default (`production`, v1 LSTM -> DNN). Its click
> path still works, but the header, the Model Performance tab and the scores now
> show the random forest. The current start-up commands and demo sequence are in
> `docs/final-demo-verification.md`; to reproduce this checklist exactly, start
> the backend with `$env:MODEL_SET = "production"`.

A 6½-minute walkthrough, with what to open, click, type, point at and say.
Companion files: `final-review-demo-scenarios.md` (inputs),
`final-review-results.md` (numbers), `final-review-architecture.md` (diagram),
`final-review-viva.md` (questions).

The demo runs the **production** model set. Seed 14 is shown only as a
validated candidate that is not deployed.

## Start-up commands

Backend (PowerShell):

```
cd C:\Users\HP\Desktop\fraud-intelligence-platform\backend
.\venv\Scripts\Activate.ps1
Remove-Item Env:MODEL_SET -ErrorAction SilentlyContinue
Remove-Item Env:DATABASE_URL -ErrorAction SilentlyContinue
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Wait for these lines before opening the browser (model loading takes a while):

```
[pipeline] model set 'production' from ...\backend\models\saved; ...
[startup] Restored N previously scored transaction(s) from the database.
```

Frontend, in a second terminal:

```
cd C:\Users\HP\Desktop\fraud-intelligence-platform\frontend
npm run dev
```

Review URL: **http://localhost:5173**

### Optional: start the review on a clean database

The default database (`backend\fraud_platform.db`) holds 93 earlier test scans.
They change the scores of the customers that were scanned and add one
test-made ring to the Fraud Rings tab. None of the demo customers in the
scenarios file has been scanned, so the default database works. If you prefer
a clean slate, add this line before the `python -m uvicorn` line, and nothing
is deleted:

```
$env:DATABASE_URL = "sqlite:///./fraud_platform_review.db"
```

Decide once and rehearse with the same choice. With the clean database the
Fraud Rings tab shows exactly 20 rings in 9 clusters.

## Ten minutes before

- [ ] Laptop on power, notifications off, browser zoom 100%, window maximised.
- [ ] Both terminals running; backend log shows `model set 'production'`.
- [ ] Header shows **System ONLINE**, **Model set production**, green
      **PRODUCTION** chip. If it shows anything else, stop the backend, run the
      two `Remove-Item` lines and start it again.
- [ ] Note the clock hour and pick the **primary** customer for that hour from
      the scenarios file. Do not scan that customer before the review.
- [ ] Have `docs\final-review-seed14-model-performance.png`,
      `final-review-architecture.md` and `final-review-results.md` open in
      another window.
- [ ] Internet is not required. Without it the page uses fallback fonts; it
      still works.

## The sequence

### 1. Project introduction — 30 s

- **Screen:** the application, Live Scan tab, nothing scanned yet.
- **Point at:** the title, and the header chips (ONLINE, production, version).
- **Say:** "This is a fraud intelligence platform. It scores a banking
  transaction against that customer's own behaviour in real time, explains the
  score, and exposes groups of linked accounts. It is running our production
  model, and all data is synthetic."

### 2. Architecture — 45 s

- **Screen:** `final-review-architecture.md`, the scoring-path diagram.
- **Point at:** the two model boxes and the arrow from one into the other.
- **Say:** "Nine behavioural features are computed from the customer's past.
  An LSTM reads the previous ten transactions and gives a temporal risk
  signal, the Risk Score. A dense
  network takes the features of this transaction plus that Risk Score and
  gives the Fraud Score. SHAP explains it. The backend is FastAPI, the
  frontend is React."

### 3. Live Scan — 90 s

- **Screen:** Live Scan.
- **Click:** the Customer list, choose the primary customer for the hour.
- **Point at:** Customer Profile: usual device, home city, transactions in
  history.
- **Type (scenario A):** the customer's A amount and usual category; leave
  Device ID and Location blank; Failed Logins 0. Click **Scan Transaction**.
- **Point at:** the verdict card, the Fraud Score ring, Behavioral Similarity,
  and "Match" for device and location.
- **Say:** "A usual amount, on the usual phone, from the home city. The Fraud
  Score is a model score from 0 to 100, not a probability."
- **Type (scenario C):** Amount 85000; the C category; Device ID
  `DEV_UNKNOWN_7731`; Location `Singapore`; Failed Logins 4. Click **Scan
  Transaction**.
- **Point at:** the verdict card changing colour, the two score rings, the
  similarity ring dropping, "Different" for device and location.
- **Say:** "Same customer. A large purchase in a category they never use, from
  an unknown device abroad, after four failed logins."
- If there is time, run scenario B between A and C and say "same phone, same
  city, three times the usual amount".
- Do not use the **Typical Purchase** button for scenario A (see the scenarios
  file).

### 4. Explainable AI / SHAP — 45 s

- **Screen:** stay on Live Scan, scroll to "Why was this transaction flagged?".
- **Point at:** the top bar, its contribution value and "Primary driver"; then
  the Behavioral Analysis table below.
- **Say:** "These are SHAP values on the classifier. Each bar is how much one
  feature pushed this score up. An investigator sees the reason, not only a
  number."
- **Click:** **Download Report** on the verdict card; open the PDF.
- **Say:** "The same explanation as a report, with the model set and version
  that scored it."

### 5. Fraud Rings — 60 s

- **Click:** the **Fraud Rings** tab.
- **Point at:** the four tiles, then the cluster graphs.
- **Click:** the cluster labelled `DEV_UNKNOWN_1125 +5`.
- **Point at:** the Investigation summary: 7 customers, 6 shared devices, 12
  linked transactions, and the shared-device list.
- **Say:** "A genuine customer's phone belongs to them alone. Here seven
  accounts are connected through six devices. No account looks alarming alone;
  the link is the signal. This is a rule over device IDs, not a model, and the
  severity label is a reading aid based on cluster size."
- If the first cluster on screen is the one containing `DEV_UNKNOWN_9999`, say
  it comes from your own test scans.

### 6. Batch Scoring — 45 s

- **Click:** the **Batch Scoring** tab, then **Sample CSV** (downloads
  `sample_transactions.csv`).
- **Click:** the dropzone, choose that file, then **Upload & Score**.
- **Point at:** Pipeline status, the tiles, the alert distribution bar.
- **Click:** the **Critical Risk** filter, then **Download results**.
- **Say:** "The same model over a file of transactions, for back-office
  review. Results can be filtered by alert level and exported."

### 7. Model Performance — 60 s

- **Click:** the **Model Performance** tab.
- **Point at:** the PRODUCTION MODEL badge, the KPI row, then the Model
  information card: model set, version, "Model scores, not calibrated
  probabilities", alert bands.
- **Say:** "This is the production architecture evaluated on our first
  dataset with a time-based split. The classifier scores 1.000 here, and we
  do not take that at face value: a two-line rule scores the same on that
  data. It told us the first dataset was too easy, so we built a harder one."

### 8. Validated seed-14 candidate — 45 s

- **Screen:** `docs\final-review-seed14-model-performance.png` (the
  application's own Model Performance page with the candidate loaded), or the
  table in `final-review-results.md`. Do not switch the running model during
  the review.
- **Point at:** the EVALUATION CANDIDATE and NOT DEPLOYED badges; recall 0.558
  and 9.11 alerts per 1,000; production recall 0.279; "All gates passed"; the
  new-customer limitation card.
- **Say:** "On the harder dataset we retrained the same architecture with
  five random seeds and chose the most typical run, seed 14, by a rule fixed
  in advance. On 482,292 unseen transactions it caught 1,365 of 2,446 fraud
  transactions against 683 for production, with slightly fewer false alerts.
  It passed all three acceptance gates. It is a validated candidate and it is
  not deployed: it raises about 25.6 false alerts per 1,000 on new customers,
  and that has to be solved first."

### 9. Final conclusion — 30 s

- **Screen:** back to the application, Live Scan.
- **Say:** "The platform detects, explains and links. We evaluated it
  honestly: we found our first results were too good to be true, built a
  harder test, and selected a better model by a pre-registered procedure
  without deploying it prematurely. The limits are that the data is synthetic,
  the scores are not calibrated probabilities, and new customers need their
  own policy. The next step would be shadow testing on real transactions."

| Section | Time |
|---|---|
| 1. Introduction | 0:30 |
| 2. Architecture | 0:45 |
| 3. Live Scan | 1:30 |
| 4. Explainable AI / SHAP | 0:45 |
| 5. Fraud Rings | 1:00 |
| 6. Batch Scoring | 0:45 |
| 7. Model Performance | 1:00 |
| 8. Seed-14 candidate | 0:45 |
| 9. Conclusion | 0:30 |
| **Total** | **7:30** with every optional step; about **6:30** if scenario B and the PDF are skipped |

## If something goes wrong

| Symptom | What to do |
|---|---|
| Header shows OFFLINE | The backend is not running or still loading. Check the first terminal. |
| Customer list empty | Same cause. Wait for the `[startup] Restored …` line, then reload the page. |
| Backend exits with `ModelSetError` | `MODEL_SET` holds a wrong value. Run the two `Remove-Item` lines and start again. |
| Header shows EVALUATION · NOT DEPLOYED | The candidate was started by mistake. Stop the backend, run the two `Remove-Item` lines, start again. |
| A "normal" scan scores high | Read the SHAP reason aloud (usually an unusual hour or category for that customer) and move on. Do not rescan. |
| PDF does not open | The file is in the browser's Downloads; open it from there. |

## Showing the candidate live (only if an examiner asks, and only after the main demo)

```
$env:MODEL_SET = "v2_dnn_lstm_seed14"
$env:DATABASE_URL = "sqlite:///./fraud_platform_seed14.db"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The header then shows `v2_dnn_lstm_seed14` and **EVALUATION · NOT DEPLOYED**.
To return to production, stop the backend, run the two `Remove-Item` lines and
start it again. Never leave the machine configured for seed 14.
