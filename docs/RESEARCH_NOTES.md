# Research Notes: Writing This Up for Publication

This project was scoped from the start to support submission to an
IEEE / Springer / Scopus-indexed conference. Here's how to turn the working
system into a paper.

**Read `docs/EVALUATION.md` first.** It holds the corrected, leakage-safe
evaluation and its results. Every number and claim in a paper should come
from there. Two findings shape how the system can be described:

- **The LSTM is not an early-warning signal on the current data.** On the
  time-based test set it flags 0 of 16 first-fraud transactions (0 of 12 on
  the customer-grouped split). It flags an episode only after fraud has
  already started: median delay 1.5 fraud transactions, about 36 hours.
- **The LSTM adds no measurable value to the DNN on the current synthetic
  dataset.** The DNN without the LSTM `risk_score`, a logistic regression
  and a two-condition amount/hour rule all match the full model (PR-AUC,
  precision and recall all 1.000). The generator makes fraud trivially
  separable.

## Suggested paper structure

1. **Abstract** — 150-200 words. State the problem (rule-based fraud
   systems lack transparency and fail on evolving patterns), your approach
   (Behavioral Fraud DNA features + a sequence model (LSTM) whose score is
   passed to a DNN transaction classifier + SHAP explainability), and your
   headline result from the corrected evaluation (PR-AUC, precision, recall,
   F1) together with the baselines. Do not describe the LSTM as predicting
   fraud before it happens.

2. **Introduction** — motivate why explainability matters in banking fraud
   specifically (regulatory requirements, analyst trust, actionability of
   alerts), not just as a general ML nicety.

3. **Related Work** — cover three buckets and cite 3-5 papers in each:
   - Rule-based / traditional statistical fraud detection
   - Deep learning fraud detection (autoencoders, GNNs on transaction
     graphs, sequence models)
   - Explainable AI in finance (SHAP/LIME applications to credit scoring
     or fraud)
   Position the contribution on what the evidence supports: an
   explainable, per-customer behavioral detection pipeline with a
   leakage-safe, out-of-time evaluation and honest baselines. Sequence
   (trajectory) modeling was implemented and tested, but on the current
   data it did not improve detection over single-transaction models.
   Report that as a finding, not as a contribution.

4. **Proposed Methodology**
   - Describe the Behavioral Fraud DNA feature set (Section 6 of the main
     README has the full list) and why each feature was chosen.
   - Describe the two-stage architecture. An LSTM reads the customer's 10
     previous transactions and outputs a risk score (the predicted
     probability that the next transaction is fraud). That score is one of
     the DNN's 10 inputs, and the DNN classifies the current transaction.
     Explain the leakage-safe handoff: the DNN trains on *out-of-fold* LSTM
     scores (see `docs/EVALUATION.md`).
   - State the measured role of each part neutrally. On the current dataset
     the DNN's detection comes from the current transaction's own features;
     removing the LSTM score does not measurably change the results
     (identical on the time split; 3 vs 1 false positives out of 18,448
     legitimate transactions on the customer split). The LSTM detects
     fraud episodes that are already under way, not their first transaction.
     Do not present the LSTM -> DNN handoff as a novelty claim unless new
     data shows it adds value.
   - Describe the SHAP integration and how raw SHAP values are mapped to
     human-readable reasons.
   - Include an architecture diagram (LSTM -> risk_score -> DNN input ->
     SHAP -> explanation).

5. **Experimental Setup**
   - Dataset: be upfront it's synthetic/semi-synthetic (standard in this
     field since real bank data is never public — cite PaySim or IEEE-CIS
     papers as precedent for this practice).
   - Preprocessing: describe the rolling-window, no-lookahead feature
     computation.
   - Split: time-based (train before 2026-05-06, validation to 2026-06-01,
     test after), with each fraud episode kept inside one split, plus a
     customer-grouped split as a robustness check. Thresholds are chosen on
     validation only. See `docs/EVALUATION.md`. Do not use the old random
     80/20 stratified split; it mixed future transactions into training.
   - Class imbalance handling: class weighting (mention the imbalance
     ratio you observed, e.g. ~1:100).

6. **Results**
   - Report **PR-AUC, ROC-AUC, Precision, Recall and F1** for both the LSTM
     (predicting the next transaction's fraud label from the previous 10
     transactions) and the DNN (classifying the current transaction), from
     the corrected evaluation.
   - Report the fraud-episode metrics, especially **first-fraud recall**
     (LSTM: 0/16) and detection delay. These show what the LSTM does and
     does not do.
   - Include a confusion matrix for the DNN at your chosen alert threshold.
   - Include 2-3 concrete SHAP explanation examples (screenshot or table)
     showing a fraud case and a normal case side by side — this is often
     the most memorable figure in fraud-detection papers.
   - Always show the baselines from the corrected evaluation: the
     amount/hour rule, logistic regression, and the DNN without the LSTM
     score. On the current synthetic data they match the full model, so
     the deep-learning approach is not yet justified by these results. If
     you retrain on a real dataset (PaySim/IEEE-CIS), repeat the same
     comparison.

7. **Discussion / Limitations** — use the "Known limitations" list from
   the main README almost verbatim; reviewers respond well to explicit,
   honest limitations sections.

8. **Conclusion & Future Work** — mention: incremental/streaming feature
   computation for true production real-time use, learned feature
   weighting for the similarity score, graph-based features (shared
   devices/locations across customers, useful for detecting fraud rings).

## Metrics cheat-sheet (why these, not just accuracy)

With a fraud rate around 1%, a model that predicts "not fraud" for
everything gets ~99% accuracy while catching zero fraud. Report instead:

- **Precision** = of the transactions flagged as fraud, how many actually
  were. Low precision = too many false alarms for analysts to handle.
- **Recall** = of all actual fraud, how much was caught. Low recall = fraud
  slipping through.
- **F1** = harmonic mean of the two, useful as a single summary number.
- **AUC-ROC** = threshold-independent measure of separability; good for
  comparing models before you've picked an operating threshold.
- **False Positive Rate** = worth reporting separately in banking context,
  since false positives have real customer-friction cost (blocked
  legitimate transactions).

## A note on the synthetic dataset

Reviewers will ask about this — answer it before they do. In your
Experimental Setup section, state plainly: "Due to the unavailability of
public real-world banking transaction data with behavioral labels (device,
login, location history), we constructed a synthetic dataset in which each
fraud episode is a burst of anomalous transactions (unusual hour, amount
several times the customer's average, new device or foreign location,
failed logins), following established practice in the field (cf. PaySim,
Lopez-Rojas et al.)." Add the limitation too: the generator labels its
lower-severity "ramp-up" transactions as fraud as well, so the data contains
no unlabeled precursor period, and fraud is easy to separate from
legitimate behaviour. That is why simple baselines score perfectly. Then,
if you have time before the deadline, also run your pipeline on PaySim or
IEEE-CIS and report those numbers too — having both synthetic data (with
the per-customer behavioral fields this pipeline needs) and a public
benchmark (comparability with prior work) is the strongest possible
position for a reviewer.
