# Decisions needed before anything is downloaded or run (IEEE-CIS)

| ID | Decision | Recommendation |
|---|---|---|
| D1 | Proceed with IEEE-CIS as the primary independent dataset, given that it has no customer id (proxy entity only) and that lead time cannot be measured? | Yes, with the claims in the protocol §1; accept that the sequence question may be declared unanswerable at stage 0/1 |
| D2 | **Access and terms.** You must read and accept the Kaggle competition rules personally and download the files, or allow the host in the environment's network settings. This session will not accept terms or create accounts | You download; give the path (or allow `kaggle.com` and the download host) and confirm in writing that the rules permit this use and publishing results |
| D3 | Where raw data live (outside git); retention; whether the files may be processed in this cloud environment at all | local disk or the session scratch directory; delete after the study |
| D4 | Primary endpoint and δ: PR-AUC difference, δ = 0.02 proposed; secondary at FPR 1 % | accept, or give another δ |
| D5 | Time design: 60/20/20 with 7-day embargoes; 14-day end trim as a sensitivity analysis for label censoring | accept; revise only after stage 0 shows the time span |
| D6 | Entity-selection rule (label-blind consistency, ≥ 20 % eligible rows, no entity above 1 % of rows) and the rule that an unanswerable sequence question is reported as such | accept |
| D7 | Whether Vesta's `C/D/M/V` columns are used only as a labelled context arm (V0) | accept |
| D8 | Learner stack: LightGBM (new dependency) + TensorFlow-CPU for sequences; PyTorch optional | accept LightGBM + TensorFlow |
| D9 | Relationship to Experiment 2: run IEEE-CIS first (it needs no missing ablation files), or finish Experiment 2 first? | independent data first: it is the main scientific gap |
| D10 | Missing `lstm_value_ablation` files for Experiment 2 (still not in any branch) | push them to a new branch (commands in Experiment 2 REVIEW_CHECKLIST §9) |

Until D1–D3 are answered nothing is downloaded, installed or run. After D2 the first action is the label-blind stage-0 audit, followed by a stop for review.
