# Final report

> **Out of date for the classifier (Step 4D, 8 October 2026).** This report
> describes the build before Step 4D, in which a DNN followed the LSTM. The
> default model is now the same LSTM followed by a random forest
> (`docs/model_selection_report.md`). The report's model sections, scores and
> screenshots must be regenerated before it is presented as the current
> system.

| File | Contents |
|---|---|
| `Fraud_Intelligence_Platform_Technical_Report.docx` | The project technical report, editable in Word |
| `Fraud_Intelligence_Platform_Technical_Report.pdf` | The same report as a 32-page PDF |

## What the report contains

A cover page, a table of contents, a terminology page and 25 numbered sections:
executive summary, problem statement, the team's understanding of the problem,
proposed solution, system architecture, technology stack, machine learning
approach, features, SHAP explainability, behavioral analysis, fraud-ring
detection, the four dashboard modules, datasets, model evaluation, why the
Seed-14 candidate was not promoted, production versus candidate, database, API,
security and reliability, testing, results, limitations, future enhancements,
team contribution and conclusion. It has 11 numbered tables and 9 figures
(two diagrams of the system, one of the fraud-ring idea, one results chart and
five screenshots of the running application).

## Where the content comes from

The report is based on this repository as of 8 October 2026 (commit
`9e7f265` plus the demo-readiness wording changes): the backend and frontend
source code, the model manifests, the evaluation reports under
`backend/models/evaluation/`, and the documents in `docs/`. Metrics are copied
from `final_holdout_report.json` and the step reports; nothing was estimated.
The screenshots were retaken on 8 October 2026 from the current application
running with the production model set on a clean database with synthetic data.

The report states throughout that the data is synthetic, that the Fraud Score
is a model score and not a calibrated probability, that the LSTM produces a
temporal risk signal and is **not** claimed to predict fraud before it happens,
and that the Seed-14 candidate is an evaluation candidate that is not deployed.

## What you need to fill in

| Place | What to add |
|---|---|
| Cover page | Names and register numbers of the three team members; guide name and designation; department; institution |
| Section 24, first lines | Team member names and register numbers |
| Section 24, Table 11 | Who did what, for each work area |
| Section 24, end | Signatures of the team members and the project guide |

The cover page shows "October 2026"; change it if the submission date differs.
If the team has more or fewer than three members, adjust the lines on the
cover page and in Section 24.

## If you edit the Word file

The table of contents is typed text with page numbers, not an automatic Word
field. If your edits move sections to other pages, correct the page numbers by
hand, and export a new PDF from Word (File > Save As > PDF) so the two files
stay the same.
