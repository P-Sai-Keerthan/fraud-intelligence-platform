# Audit reports

Produced by the project audit (code revision `140731d` plus these reports). These files document the application; they are not part of it.

| File | Content |
|---|---|
| `PROJECT_AUDIT_AND_TECHNICAL_DOCUMENTATION.pdf` | the complete technical documentation and audit (18 pages) |
| `PROJECT_AUDIT_AND_TECHNICAL_DOCUMENTATION.md` | its editable source; the PDF is rendered from exactly this text |
| `BUG_AND_RISK_REGISTER.md` | every finding with evidence, root cause, fix status and limits |
| `FIX_AND_TEST_LOG.md` | environment, baseline, fixes, tests actually run, tests not run |
| `ML_EXPERIMENT_STATUS.md` | what the ML evidence shows, what was re-checked, what is missing |
| `RECOMMENDED_IMPROVEMENTS.md` | prioritised next steps, including decisions that belong to the owner |
| `evidence/` | probe scripts and their raw before/after output, test logs, test inventory |
| `assets/` | figures, dashboard screenshots, tables generated from the committed result files |
| `tools/` | the scripts that build the figures, tables and PDF |

Rebuild (needs `matplotlib`, `scikit-learn`, `pandas`, `markdown`, `pygments`, `pypdf`, `playwright` with Chromium):

```bash
python reports/tools/build_report_assets.py . reports/assets   # figures and tables from backend/models/evaluation/*.json
python reports/tools/build_pdf.py .                            # writes the .md and .pdf from tools/main_template.md
```
