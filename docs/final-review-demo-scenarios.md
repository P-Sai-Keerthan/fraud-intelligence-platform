# Final review — demo transaction scenarios

> **Step 4D update (8 October 2026).** The default model set is now
> `v2_lstm_rf_seed14`: the same seed-14 LSTM, followed by a **random forest**
> instead of a DNN, chosen by a pre-registered comparison and confirmed on a
> fresh hold-out (`docs/model_selection_report.md`). The "observed"
> scores in the table below were measured with the **previous default**
> (`MODEL_SET=production`) and do not apply to the random forest, whose scores
> are much lower (an ordinary purchase scores about 0.06, single unusual
> signals 0.2 to 16, a failed-login attack 43, several signals together 75; see section "Default model (Step 4D)" at the end and
> `docs/final-demo-verification.md`). Rehearse on the review machine.

**Read this first.** No input guarantees a particular score. The score of a
transaction depends on the loaded model set, on that customer's own history
(including every scan made earlier on this machine) and on the time of day the
scan is made. The outcomes quoted below were observed once, on 5 October 2026,
on a clean test copy of the project with the production model set and an empty
database. Treat them as what to expect, not as a promise, and rehearse on the
review machine.

## Three things that change the result

1. **Time of day.** Live Scan stamps the transaction with the current time.
   One feature is "unusual hour": an hour in which the customer made fewer than
   5% of their past transactions. A perfectly ordinary purchase at an hour that
   is unusual *for that customer* can score Medium, High or even Critical with
   the production model. So choose the customer by the hour of the review
   (table below).
2. **The "Typical Purchase" button is not customer-specific.** It always enters
   ₹1,500 / grocery. For a customer who never buys groceries, or at an hour
   unusual for them, that is not a typical purchase and it can score high
   (observed: Low for one customer, High or Critical for three others at the
   same moment). For scenario A type the values from the table instead.
3. **Every scan is remembered.** A scan is added to the customer's history and
   saved in the database, so it changes that customer's later scores, and it
   survives a restart. Use each demo customer once, in the order A → B → C.
   Rehearse on the *backup* customer and keep the *primary* one untouched for
   the review.

Also: the same unknown device ID used on two different customers creates a new
shared-device ring in the Fraud Rings tab (scenario D uses this on purpose).
The "Suspicious Pattern" button always enters `DEV_UNKNOWN_9999`, so earlier
testing with that button on several customers may already have added a ring on
your machine.

## Customer to use, by time of the review

All are customers of the seed data with more than 150 past transactions, no
fraud history and no ring membership. "Usual category" is their most frequent
one; "unusual category" is one they have never used.

| Review time | Role | Customer | Home city | Usual device | Usual category | A: amount (₹) | B: amount (₹) | C: unusual category | A observed | B observed | C observed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 9:00–9:59 | primary | CUST_0062 | Pune | DEV_0062_A | healthcare | 3,150 | 9,400 | travel | Low Risk (1.62) | Medium Risk (43.01) | Critical Risk (99.9) |
| 9:00–9:59 | backup | CUST_0210 | Pune | DEV_0210_A | fashion | 1,960 | 5,900 | electronics | Low Risk (2.11) | Medium Risk (43.72) | Critical Risk (99.9) |
| 10:00–10:59 | primary | CUST_0082 | Pune | DEV_0082_A | utilities | 3,760 | 11,300 | electronics | Low Risk (1.71) | Medium Risk (27.87) | Critical Risk (99.9) |
| 10:00–10:59 | backup | CUST_0311 | Mumbai | DEV_0311_A | utilities | 2,400 | 7,200 | electronics | Low Risk (3.32) | Low Risk (13.34) | Critical Risk (99.83) |
| 11:00–11:59 | primary | CUST_0048 | Chennai | DEV_0048_A | travel | 2,240 | 6,700 | electronics | Low Risk (2.62) | Low Risk (20.29) | Critical Risk (99.85) |
| 11:00–11:59 | backup | CUST_0285 | Pune | DEV_0285_A | online_retail | 2,540 | 7,600 | travel | Low Risk (3.51) | Low Risk (13.92) | Critical Risk (99.86) |
| 12:00–12:59 | primary | CUST_0318 | Kolkata | DEV_0318_A | travel | 2,770 | 8,300 | electronics | Low Risk (2.07) | Medium Risk (44.98) | Critical Risk (99.9) |
| 12:00–12:59 | backup | CUST_0364 | Mumbai | DEV_0364_A | fashion | 3,940 | 11,800 | electronics | Low Risk (1.45) | Medium Risk (43.64) | Critical Risk (99.9) |
| 13:00–13:59 | primary | CUST_0464 | Mumbai | DEV_0464_A | dining | 3,600 | 10,800 | travel | Low Risk (1.57) | Medium Risk (42.0) | Critical Risk (99.9) |
| 13:00–13:59 | backup | CUST_0091 | Delhi | DEV_0091_A | utilities | 3,920 | 11,800 | travel | Low Risk (1.5) | Medium Risk (42.18) | Critical Risk (99.9) |
| 14:00–14:59 | primary | CUST_0329 | Mumbai | DEV_0329_A | healthcare | 3,200 | 9,600 | electronics | Low Risk (1.88) | Medium Risk (44.38) | Critical Risk (99.9) |
| 14:00–14:59 | backup | CUST_0477 | Pune | DEV_0477_A | fuel | 2,560 | 7,700 | online_retail | Low Risk (1.46) | Medium Risk (30.61) | Critical Risk (99.9) |
| 15:00–15:59 | primary | CUST_0376 | Hyderabad | DEV_0376_A | grocery | 2,200 | 6,600 | electronics | Low Risk (1.58) | Medium Risk (40.52) | Critical Risk (99.9) |
| 15:00–15:59 | backup | CUST_0165 | Mumbai | DEV_0165_A | fuel | 2,110 | 6,300 | electronics | Low Risk (1.44) | Medium Risk (39.46) | Critical Risk (99.9) |
| 16:00–16:59 | primary | CUST_0369 | Chennai | DEV_0369_A | healthcare | 2,050 | 6,100 | electronics | Low Risk (2.53) | Medium Risk (26.95) | Critical Risk (99.88) |
| 16:00–16:59 | backup | CUST_0053 | Hyderabad | DEV_0053_A | electronics | 870 | 2,600 | travel | Low Risk (1.79) | Medium Risk (36.47) | Critical Risk (99.9) |
| 17:00–17:59 | primary | CUST_0351 | Kolkata | DEV_0351_A | travel | 1,370 | 4,100 | electronics | Low Risk (1.64) | Medium Risk (38.3) | Critical Risk (99.9) |
| 17:00–17:59 | backup | CUST_0408 | Bangalore | DEV_0408_A | online_retail | 3,900 | 11,700 | electronics | Low Risk (3.2) | Low Risk (19.07) | Critical Risk (99.85) |
| 18:00–18:59 | primary | CUST_0497 | Mumbai | DEV_0497_A | utilities | 620 | 1,900 | electronics | Low Risk (2.84) | Low Risk (17.27) | Critical Risk (99.87) |
| 18:00–18:59 | backup | CUST_0396 | Mumbai | DEV_0396_A | electronics | 2,410 | 7,200 | travel | Low Risk (3.18) | Low Risk (15.54) | Critical Risk (99.84) |

The number in brackets is the Fraud Score (0–100). The hour is the clock of the
computer running the backend. If the review crosses an hour boundary, switch to
the next row's customer. Outside 9:00–18:59 no customer has been checked.

## A. Normal transaction

A customer paying a usual amount, in a usual category, on their own phone, from
their home city.

| Field | Enter |
|---|---|
| Customer | the row for the current hour |
| Amount (₹) | the "A: amount" value (close to that customer's average spend) |
| Merchant Category | the customer's usual category |
| Device ID | leave blank (uses the usual device) |
| Location | leave blank (uses the home city) |
| Failed Logins (24h) | 0 |

Example at 3 pm: CUST_0376, a Hyderabad customer who mostly buys groceries,
pays ₹2,200 for groceries on DEV_0376_A.

What was observed: Low Risk, Fraud Score between 1 and 4, behavioral similarity
about 95%, no SHAP reasons listed, Device and Location shown as "Match".

## B. Moderate anomaly

The same customer, same phone, same city, same category, but about three times
their usual amount: a festival-season purchase, for instance.

| Field | Enter |
|---|---|
| Amount (₹) | the "B: amount" value (about 3× the average) |
| Merchant Category | the usual category |
| Device ID / Location | leave blank |
| Failed Logins (24h) | 0 |

Example at 3 pm: CUST_0376 pays ₹6,600 for groceries.

What was observed: Medium Risk for 13 of the 20 customers in the table (Fraud
Score 27–45) and Low Risk for the other 7 (Fraud Score 4–22). Similarity dropped
well below scenario A. The SHAP reasons were "Unusual Transaction Amount" and
"High Transaction Velocity" (it is the second scan within the hour). Point out
that the device and location still match, which is why this is not treated like
scenario C.

## C. Strong suspicious transaction

An account-takeover pattern: after several failed logins, a large purchase in a
category the customer never uses, from a phone and a place never seen before.

| Field | Enter |
|---|---|
| Amount (₹) | 85000 |
| Merchant Category | the "C: unusual category" value |
| Device ID | DEV_UNKNOWN_7731 |
| Location | Singapore |
| Failed Logins (24h) | 4 |

Example at 3 pm: CUST_0376's account is used for an ₹85,000 electronics
purchase from Singapore on an unknown device after four failed logins.

What was observed: Critical Risk for all 20 customers, Fraud Score 99.8–99.9,
similarity below 6%, Device and Location shown as "Different". The SHAP reasons
came from this set: New Device, Unusual Merchant Category, Amount Far Above
Average, High Transaction Velocity, New Location, Foreign Location.

Note on locations: the model treats any city outside its list of ten Indian
home cities (Hyderabad, Mumbai, Delhi, Bangalore, Chennai, Kolkata, Pune,
Ahmedabad, Jaipur, Lucknow) as foreign. An Indian city outside that list, such
as Kochi, would also be flagged as foreign. That is a simplification of the
synthetic data, worth admitting if asked.

## D. Fraud-ring investigation

### D1. An existing ring (no typing needed)

Open **Fraud Rings**. With the seed data and an empty database the page shows
20 rings in 9 clusters, 29 affected customers, 20 shared devices and 40
linked transactions; your numbers can be higher if earlier scans added rings.

**On the review machine the page will show more than that.** Its database
(checked on 5 October 2026) holds earlier test scans in which `DEV_UNKNOWN_9999`
was used on seven customers (CUST_0001, 0006, 0007, 0010, 0020, 0021, 0022).
Those appear as one extra ring, joined to the seed cluster that contains
CUST_0022, and that cluster will probably be listed first. It is real output of
the detector, but it was created by testing. Either say so ("this one is from
our own test scans, which shows a ring being picked up as soon as a device is
reused") or start the review on a clean database (see the checklist).

Click the seed-data cluster labelled `DEV_UNKNOWN_1125 +5`:

| | |
|---|---|
| Customers linked | 7: CUST_0003, CUST_0044, CUST_0057, CUST_0079, CUST_0279, CUST_0377, CUST_0445 |
| Shared devices | 6: DEV_UNKNOWN_4112, _1299, _2157, _6823, _9350, _1125 |
| Linked transactions | 12 |
| Size-based level shown | Critical (a reading aid based on cluster size, not a model output) |

Story: no single account looks alarming on its own, each shares one device with
one other account. Linking the devices shows seven accounts connected through
six devices, with CUST_0044 and CUST_0057 as the hubs.

### D2. A ring forming live (optional, 30 seconds)

After scenario C on the primary customer, select the backup customer of the same
hour and run scenario C again with the **same** device ID `DEV_UNKNOWN_7731`.
Open Fraud Rings and press Refresh: a new two-customer ring on
`DEV_UNKNOWN_7731` appears, because one device has now been used on two
accounts. This was observed in testing (the ring count went from 20 to 21).
Skip D2 if you want to keep the backup customer clean.

## If a result surprises you during the review

Say what the screen says and explain it from the SHAP reasons, which is the
point of the explainability panel. For example: "It is flagged mainly because
this hour is unusual for this customer." Do not rerun the same scan hoping for
a different number: each rerun is added to the history and moves the score.

## Default model (Step 4D): hand-built scenarios

Scored on 8 October 2026 by `python -m app.evaluation.downstream_sanity`
(`backend/models/evaluation/downstream/sanity_scenarios.json`). Every row
starts from the untouched seed history, dated 10 July 2026 at the customer's
most common hour. Fraud Scores are model scores (0–100), not probabilities.
For comparison, the same inputs are shown for the DNN with the same LSTM
(`v2_dnn_lstm_seed14`) and for the previous default (`production`).

| # | Scenario | Input | Risk Score | Fraud Score (random forest) | SHAP reasons (random forest) | DNN, same LSTM | Previous default |
|---|---|---|---|---|---|---|---|
| A | Normal low-risk purchase | CUST_0376, Rs. 2,200, grocery, DEV_0376_A, Hyderabad, 0 failed logins, 17:15 | 14.55 | **0.06** (Low Risk) | (none shown: score below 5) | 9.61 | 1.58 |
| B | Moderate behavioral deviation (3x usual amount) | CUST_0376, Rs. 6,600, grocery, DEV_0376_A, Hyderabad, 0 failed logins, 17:15 | 14.55 | **1.64** (Low Risk) | (none shown: score below 5) | 50.90 | 2.40 |
| C | New device only | CUST_0376, Rs. 2,200, grocery, DEV_NEW_A001, Hyderabad, 0 failed logins, 17:15 | 14.55 | **3.34** (Low Risk) | (none shown: score below 5) | 64.69 | 21.27 |
| D | New domestic location only | CUST_0376, Rs. 2,200, grocery, DEV_0376_A, Mumbai, 0 failed logins, 17:15 | 14.55 | **0.17** (Low Risk) | (none shown: score below 5) | 50.55 | 15.41 |
| E | Foreign location only | CUST_0376, Rs. 2,200, grocery, DEV_0376_A, Singapore, 0 failed logins, 17:15 | 14.55 | **16.24** (Low Risk) | Foreign Location +0.163, New Location +0.016, Elevated Behavioral Risk Score +0.002, Unusual Transaction Amount +0.001 | 85.88 | 84.30 |
| F | High amount only (Rs. 85,000) | CUST_0376, Rs. 85,000, grocery, DEV_0376_A, Hyderabad, 0 failed logins, 17:15 | 14.55 | **2.38** (Low Risk) | (none shown: score below 5) | 67.70 | 1.41 |
| G | Multiple suspicious signals | CUST_0376, Rs. 85,000, electronics, DEV_UNKNOWN_7731, Singapore, 0 failed logins, 17:15 | 14.55 | **74.56** (High Risk) | Foreign Location +0.282, New Device +0.220, Amount Far Above Average +0.120, New Location +0.098 | 91.71 | 99.90 |
| H | Failed-login attack pattern | CUST_0376, Rs. 25,000, electronics, DEV_UNKNOWN_8842, Hyderabad, 5 failed logins, 17:15 | 14.55 | **43.39** (Medium Risk) | New Device +0.216, Multiple Failed Logins +0.121, Amount Far Above Average +0.088, Unusual Transaction Amount +0.023 | 92.74 | 99.90 |
| N1 | Normal purchase, CUST_0062 | CUST_0062, Rs. 3,140, healthcare, DEV_0062_A, Pune, 0 failed logins, 12:15 | 9.99 | **0.05** (Low Risk) | (none shown: score below 5) | 7.78 | 1.61 |
| N2 | Normal purchase, CUST_0082 | CUST_0082, Rs. 3,760, utilities, DEV_0082_A, Pune, 0 failed logins, 01:15 | 16.00 | **0.06** (Low Risk) | (none shown: score below 5) | 10.14 | 1.71 |
| N3 | Normal purchase, CUST_0318 | CUST_0318, Rs. 2,750, travel, DEV_0318_A, Kolkata, 0 failed logins, 13:15 | 12.86 | **0.06** (Low Risk) | (none shown: score below 5) | 8.47 | 2.03 |
| N4 | Normal purchase, CUST_0464 | CUST_0464, Rs. 3,610, dining, DEV_0464_A, Mumbai, 0 failed logins, 00:15 | 14.73 | **0.06** (Low Risk) | (none shown: score below 5) | 9.74 | 1.57 |
| N5 | Normal purchase, CUST_0329 | CUST_0329, Rs. 3,200, healthcare, DEV_0329_A, Mumbai, 0 failed logins, 14:15 | 15.99 | **0.06** (Low Risk) | (none shown: score below 5) | 10.17 | 1.88 |

What this shows:

* **Ordinary purchases** score about 0.06 with the random forest, against about 8
  to 10 for the DNN with the same LSTM.
* **One unusual signal on its own** (more money, a new device, a new city, a
  large amount) stays low: 0.2 to 3.3. A foreign city alone reaches 16.2.
* **Combined signals rise steeply:** the failed-login attack (H) scores 43.4
  and the full account-takeover pattern (G) 74.6.
* **The scores are spread out.** The model does not output about 100 for
  everything.
* **Alert bands.** E (16.24) appears as "Low Risk" under the fixed bands,
  although it is above the random forest's validated alert cut-off (a score
  of 4.39); see `docs/model_selection_report.md`, limitation 3.
