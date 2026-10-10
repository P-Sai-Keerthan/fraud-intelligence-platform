# Batch Scoring test files

**Synthetic test data.** These files were written by hand-coded rules for testing
the Batch Scoring tab. They contain no real people, accounts, cards or payments.
The customer IDs and their usual devices and home cities come from the project's
own synthetic seed data (`data/transactions_with_features.csv`). The files carry
no fraud labels and no expected scores: the loaded model decides the result.

| File | Rows | Contents |
|---|---|---|
| `batch_normal_transactions.csv` | 15 | Everyday purchases: an amount near the customer's average, a category the customer uses, their own device, their home city, 0 or 1 failed logins |
| `batch_suspicious_transactions.csv` | 15 | 9 strongly unusual rows (large amount, new device and/or unfamiliar location, failed logins), 5 mildly unusual rows, 1 ordinary row |
| `batch_mixed_transactions.csv` | 30 | Main demonstration file: 12 ordinary, 9 mildly unusual, 9 strongly unusual, in shuffled order |

## Format

Columns, exactly as the backend (`POST /predict/batch`) reads them:

| Column | Required | Rule enforced by the backend |
|---|---|---|
| `customer_id` | yes | not blank |
| `amount` | yes | a number greater than 0 |
| `merchant_category` | yes | not blank |
| `device_id` | optional | blank or absent uses the customer's usual device |
| `location` | optional | blank or absent uses the customer's home city |
| `failed_logins_24h` | optional | a whole number, 0 or more; blank means 0 |

The backend accepts any non-blank merchant category. These files use only the
ten the application's form and seed data use: grocery, electronics, travel,
dining, utilities, entertainment, fashion, healthcare, fuel, online_retail.
There is no timestamp column: a batch is scored at the time of upload.
Encoding is UTF-8 without a byte-order mark; amounts are in rupees.

Each customer appears once across all three files (60 different customers).
None of them belongs to a seed-data fraud ring, has fraud in the seed history,
is one of the Live Scan demo customers in `docs/final-review-demo-scenarios.md`,
or had been scanned on this machine when the files were made (7 October 2026).

## Things to know before a demo

- **Scores depend on the time of upload.** One model feature is "unusual hour
  for this customer". Each customer is active in only a few hours of the day, so
  in any upload several ordinary rows are flagged for the hour alone. With the
  production model the "normal" file came out mostly Medium, not mostly Low
  (see below). That is the model's behaviour, not a fault in the file.
- **Every upload is saved.** Scored rows are added to those customers' histories
  and stored in the database, so uploading the same file twice gives different
  scores the second time. For a repeatable demo, upload each file once, or run
  the backend with a separate database
  (`$env:DATABASE_URL = "sqlite:///./fraud_platform_batch_test.db"`).
- **"Bangalore", not "Bengaluru".** The seed data spells the city "Bangalore".
  "Bengaluru" would be treated as a place the customer has never been.
- **Any city outside the model's ten home cities counts as foreign**, including
  Indian ones. Kochi and Guwahati are used here on purpose as unfamiliar
  locations.
- **Unknown devices are unique per row**, so uploading these files does not add
  new rings to the Fraud Rings tab.

## What was observed

Each file was uploaded once through the real Batch Scoring API with the
production model set and an empty throwaway database, at 03:43 on 7 October
2026. All three were accepted (HTTP 200, every row scored).

| File | Low | Medium | High | Critical |
|---|---|---|---|---|
| `batch_normal_transactions.csv` | 5 | 8 | 1 | 1 |
| `batch_suspicious_transactions.csv` | 1 | 1 | 0 | 13 |
| `batch_mixed_transactions.csv` | 4 | 9 | 4 | 13 |

Scoring the same rows with the time set to 10:30, 12:30, 15:30 and 17:30 gave
similar mixes: normal 2–6 Low and 8–11 Medium; suspicious 13–14 Critical; mixed
3–5 Low, 6–8 Medium, 2–4 High, 15–18 Critical. Your numbers will differ with the
hour, the loaded model set and the scans already in your database.
