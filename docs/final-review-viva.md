# Final review — viva questions and answers

Short answers to say aloud. Numbers are from `final-review-results.md`.
Three rules for every answer: the data is synthetic; the Fraud Score is not a
probability; seed 14 is validated but not deployed.

## Problem and approach

**1. What is the problem statement?**
Banks must decide within moments whether a card or online transaction is
genuine. Fixed rules miss fraud that looks different from past fraud and flag
many genuine customers, and a bare "fraud / not fraud" answer gives an
investigator nothing to act on. The problem is to detect suspicious
transactions from each customer's own behaviour and explain every alert.

**2. What is the objective?**
To build a working fraud intelligence platform that scores a transaction in
real time against the customer's behavioural history, explains the score
feature by feature, exposes groups of accounts linked by shared devices, and
reports its own performance honestly.

**3. Why fraud detection?**
It is a real cybersecurity problem with direct financial loss, it is hard
because fraud is rare (under 1% of transactions here), and both kinds of error
are costly: missed fraud loses money, false alerts block genuine customers.

**4. Why machine learning?**
What is "normal" differs for every customer and changes over time. A model
learns combinations of signals (amount, device, place, time, login failures)
relative to each customer, where hand-written rules need constant maintenance.
On our harder dataset a simple rule scored far below the learned models.

**5. Why LSTM?**
The LSTM is used to model the customer's recent transaction sequence. It
captures temporal behavioral patterns and produces a temporal risk signal that
is used by the downstream fraud classifier. Fraud often unfolds over several
transactions (test charges before a cash-out, a takeover followed by rising
amounts), which is why a sequence model is used. We do not claim that the LSTM
independently proves that fraud can always be predicted before it occurs.

**6. Why DNN?**
The final decision is about one transaction described by a small table of
numbers. A small dense network handles that well, is fast enough for real
time, and works with SHAP so that each decision can be explained.

**7. Why use two stages?**
They answer different questions. The LSTM summarises the customer's recent
transaction sequence as a temporal risk signal. The DNN asks "is this
transaction abnormal right now?" Feeding the first into the second lets the final score use both.
On the final hold-out the two-stage model caught more fraud with fewer false
alerts than the DNN alone (recall 0.558 against 0.419).

**8. What is the role of the LSTM?**
It produces the Risk Score, 0–100, from the 10 transactions before the current
one. It does not see the current transaction. Its score is one input to the
DNN.

**8a. Does the LSTM predict fraud before it happens?**
It is designed to capture temporal signals that may precede or accompany
suspicious behavior, but our evaluation does not justify claiming guaranteed
pre-fraud prediction. On v1 it flagged none of the 16 first-fraud
transactions, and on v2 the LSTM + DNN design is weaker than the DNN alone on
the first fraud of an episode. Our defensible claim is temporal behavioral
risk detection.

**9. What is the role of the DNN?**
It produces the Fraud Score for the current transaction from 10 inputs: the 9
behavioural features and the LSTM Risk Score. The alert level comes from this
score.

**10. What features are used?**
Nine, all relative to the customer's own past: amount z-score, amount as a
percentage of their average, unusual hour, new device, new location, foreign
location, unusual merchant category, failed logins in 24 hours, and
transactions in the last hour. Each uses only earlier transactions, so there is
no look-ahead.

## Explainability and investigation

**11. What is SHAP?**
SHAP (SHapley Additive exPlanations) assigns each input feature a contribution
to one particular prediction, based on Shapley values from game theory. We
apply it to the DNN and show the features that pushed the score up.

**12. Why is explainability important?**
An investigator has to justify blocking a customer, and a bank has to justify
it to regulators and to the customer. "New device, foreign location, four
failed logins" can be checked and acted on; a bare number cannot. It also lets
us see when the model is reacting to the wrong thing.

**13. What is behavioral similarity?**
A 0–100% measure of how close the transaction's features are to that
customer's own averages. We measure the distance in standard deviations per
feature, average it, and convert it so that normal behaviour is near 100% and
very unusual behaviour near 0%. It is a separate indicator, not an input to the
model.

**14. What is a fraud ring?**
A group of accounts operated or attacked together, for example several
accounts accessed from one stolen or shared phone.

**15. How are fraud rings detected?**
By a rule, not a model: a customer's devices normally belong to them alone, so
we list every device used by two or more customers. The interface joins rings
that share a customer into clusters and draws them as a graph of customers and
devices. Seven accounts in our data are linked through six devices this way.

## Metrics

**16. What is PR-AUC?**
The area under the precision–recall curve. It summarises the trade-off between
catching fraud and raising false alerts across all thresholds. It is the right
summary when fraud is rare, because it ignores the huge number of correctly
ignored genuine transactions. A random model scores about the fraud rate.

**17. What is ROC-AUC?**
The probability that the model scores a random fraud above a random genuine
transaction. 0.5 is random, 1.0 is perfect ranking. With rare fraud it can
look good while the alert list is still mostly false alarms, which is why we
report PR-AUC too.

**18. What is precision?**
Of the transactions we alerted on, the share that were really fraud.

**19. What is recall?**
Of all fraud transactions, the share we alerted on.

**20. What is F1-score?**
The harmonic mean of precision and recall. It is high only when both are.

**21. What is a false positive?**
A genuine transaction flagged as fraud. The cost is an annoyed customer and
wasted investigator time. We report it as legitimate alerts per 1,000
transactions.

**22. What is a false negative?**
A fraud transaction that was not flagged. The cost is the money lost.

**23. Is Fraud Score a probability?**
No. It is the model's output scaled to 0–100. A score of 80 does not mean an
80% chance of fraud.

**24. Why not call it probability?**
Because it is not calibrated. The models were trained with fraud weighted
far more heavily than genuine transactions to cope with the imbalance (about
200 times for the v2 models), which pushes outputs upward. The score ranks transactions correctly, but its
value is not a frequency. The interface and the PDF say "Fraud Score" for that
reason.

## Models, data and selection

**25. What is the production model?**
The model set the application loads by default: an LSTM and a DNN trained on
our first synthetic dataset (v1), stored in `backend/models/saved/`. It is the
model running in the demo.

**26. What is v2?**
Our second synthetic dataset and the models trained on it. It has about
100,000 transactions from 500 customers with 0.47% fraud in seven patterns
(account takeover, stolen card online, test charges then cash-out, device
takeover, one large hit, fraud that resembles normal activity, and rings),
plus genuine customers who sometimes behave unusually.

**27. Why did we create a new dataset?**
On v1 the DNN scored a perfect 1.000, and so did a two-condition rule and a
logistic regression. That showed v1 was too easy: every fraud was built from
exactly the signals the features measure. A perfect score there measures the
data, not the model. v2 was designed with overlap between fraud and genuine
behaviour so that an evaluation means something.

**28. Why did we use multiple training seeds?**
Training a neural network is partly random. Training the same design five
times (seeds 11–15) showed recall varying by about 0.05 between runs. One run
could be lucky or unlucky, so we compared the two architectures on the average
of five runs, not on a single one.

**29. Why was seed 14 selected?**
By a rule written down before any selection data was generated: among the five
runs, take the one closest to the median on five metrics. Seed 14 was the most
typical run, not the best one. Picking the best of five would partly reward
luck.

**30. Why was seed 14 not automatically deployed?**
Passing an evaluation makes it eligible, not deployed. Four things are still
open: no approved policy for new customers, its frozen alert cut-offs are not
yet applied in live scoring, all evidence is synthetic, and promotion is a
decision for the project owner. The default remains production.

**31. What did the final hold-out show?**
On 482,292 unseen transactions with 2,446 frauds, seed 14 alerted on 55.8% of
fraud transactions (1,365) with 9.11 false alerts per 1,000. The production
model alerted on 27.9% (683) with 10.60 per 1,000. Seed 14 passed all three
pre-registered gates.

**32. Why did DNN-only fail the final gate?**
The gate requires the upper 95% bound of legitimate alerts to be at most 10
per 1,000. DNN-only seed 14 measured 10.04 with an upper bound of 10.74, so it
could not be shown to stay within the budget. Its recall, 0.419, passed.

**33. What is the new-customer limitation?**
The models need 10 earlier transactions. For newer customers the LSTM cannot
run and the features have little history to compare against. On the
new-customer test, seed 14 raised about 25.6 false alerts per 1,000 against
9.11 for established customers. We do not claim the headline performance for
new customers, and a new-customer policy is an open blocker.

## Limits and next steps

**34. What are the limitations of the project?**
All data is synthetic. Scores are not calibrated probabilities. New customers
are handled poorly. Seed 14 is weaker than DNN-only on the first fraud of an
episode (0.605 against 0.690). Live alert levels use fixed bands, not the
evaluated cut-offs. Ring detection is a shared-device rule only. "Foreign"
means outside a list of ten cities. It runs on one machine with SQLite and has
no login system.

**35. Is the model tested on real banking data?**
No. Real transaction data is confidential and we had no access to it.
Everything is tested on synthetic data that we generated, with held-out data
kept separate from training. Results on real data would differ and would have
to be measured.

**36. What would be the next step in a real deployment?**
Run the candidate in shadow mode beside the current system on real
transactions, without acting on its alerts. Then calibrate the scores and set
thresholds from real alert budgets, define a new-customer policy, add
monitoring for drift and a retraining schedule, and add the security a bank
needs: authentication, audit logs and encryption. Only then promote it
gradually, with rollback ready.

## Likely follow-up questions

**"Your Model Performance page shows 1.000. Is the model perfect?"**
No. That page evaluates the production architecture on dataset v1, where a
simple rule also scores 1.000. We found that ourselves, documented it, and
built v2 because of it. On the v2 hold-out the production model's recall is
0.279.

**"Which model is running right now?"**
Production. The header shows the model set and version at all times.

**"Why did an ordinary transaction get a high score?"**
The explanation panel says why. Usually the hour is unusual for that customer,
or the category or amount is unusual for them. The score is relative to that
customer's own history, not to customers in general.

**"What is Policy B?"**
The alert threshold chosen so that about 1% of genuine transactions are
flagged, which is about 10 alerts per 1,000. It was fixed on development data
and frozen before the final test.
