# Research Notes: Writing This Up for Publication

This project was scoped from the start to support submission to an
IEEE / Springer / Scopus-indexed conference. Here's how to turn the working
system into a paper.

## Suggested paper structure

1. **Abstract** — 150-200 words. State the problem (rule-based fraud
   systems lack transparency and fail on evolving patterns), your approach
   (Behavioral Fraud DNA + LSTM risk prediction + DNN detection + SHAP
   explainability), and your headline result (e.g. AUC-ROC, F1).

2. **Introduction** — motivate why explainability matters in banking fraud
   specifically (regulatory requirements, analyst trust, actionability of
   alerts), not just as a general ML nicety.

3. **Related Work** — cover three buckets and cite 3-5 papers in each:
   - Rule-based / traditional statistical fraud detection
   - Deep learning fraud detection (autoencoders, GNNs on transaction
     graphs, sequence models)
   - Explainable AI in finance (SHAP/LIME applications to credit scoring
     or fraud)
   Position your contribution as combining *behavioral trajectory
   modeling* (not just point-in-time classification) with explainability,
   which most existing fraud papers treat separately.

4. **Proposed Methodology**
   - Describe the Behavioral Fraud DNA feature set (Section 6 of the main
     README has the full list) and why each feature was chosen.
   - Describe the two-stage architecture: LSTM (temporal risk) feeding
     into DNN (point classification). Treat this as an engineering design
     choice, not a proven contribution: on the synthetic data, replacing the
     LSTM score with its mean did not change the DNN's recall or AUC, so any
     novelty claim needs an ablation on harder / real data first.
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
   - Train/test split: stratified by fraud label, 80/20.
   - Class imbalance handling: class weighting (mention the imbalance
     ratio you observed, e.g. ~1:100).

6. **Results**
   - Report **Precision, Recall, F1-score, ROC-AUC, PR-AUC and FPR** for both
     the LSTM (estimating the next transaction's fraud label from the recent
     sequence) and the DNN (scoring the current transaction). State that the
     scores are not calibrated probabilities, and that results on the
     synthetic dataset are near-perfect because its fraud patterns are highly
     separable.
   - Include a confusion matrix for the DNN at your chosen alert threshold.
   - Include 2-3 concrete SHAP explanation examples (screenshot or table)
     showing a fraud case and a normal case side by side — this is often
     the most memorable figure in fraud-detection papers.
   - If you retrain on a real dataset (PaySim/IEEE-CIS), compare against
     at least one baseline (e.g. plain logistic regression or a
     rule-based threshold system) to justify the deep learning approach.

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
login, location history), we constructed a semi-synthetic dataset with
realistic behavioral drift patterns leading up to fraud events, following
established practice in the field (cf. PaySim, Lopez-Rojas et al.)." Then,
if you have time before the deadline, also run your pipeline on PaySim or
IEEE-CIS and report those numbers too — having both synthetic (behavioral,
matches your novel contribution) and a public benchmark (comparability
with prior work) is the strongest possible position for a reviewer.
