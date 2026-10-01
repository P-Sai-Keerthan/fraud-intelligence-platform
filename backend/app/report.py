"""
PDF Explanation Report
=========================
Renders a single prediction result (the same shape /predict returns) into a
formatted, downloadable PDF -- a presentable, hand-off-able artifact
explaining why a transaction was scored the way it was.

Model-set wording (4C-2f-2): the report names the model set that scored the
transaction (its stored provenance) and describes risk_score accordingly:
* production and v2_dnn_lstm: an LSTM score from the customer's previous
  transactions;
* v2_dnn_only: no sequence model -- risk_score repeats the DNN fraud score for
  compatibility, and the report never calls it trajectory risk;
* provenance not recorded: neutral wording, no model set is assumed.
Scores are described as model scores, not calibrated probabilities.
"""

import io

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
)

ALERT_COLORS = {
    "Low Risk": colors.HexColor("#1a9b6b"),
    "Medium Risk": colors.HexColor("#c98a10"),
    "High Risk": colors.HexColor("#d9702e"),
    "Critical Risk": colors.HexColor("#c9403a"),
}
BRAND = colors.HexColor("#0e7a8f")

FRAUD_SCORE_MEANING = "This transaction's DNN fraud score (0-100); a model score, not a calibrated probability"

# model set -> (Risk Score label, Risk Score meaning, footer)
MODEL_SET_WORDING = {
    "production": (
        "Risk Score",
        "Trajectory risk from prior activity, before this transaction: the production LSTM's score for the "
        "customer's 10 previous transactions",
        "Risk Score is the output of the production LSTM risk predictor and Fraud Probability the output of "
        "the production DNN fraud classifier; reasons are derived from SHAP feature attribution on the DNN. "
        "For customers with fewer than 10 earlier transactions the LSTM is not used and Risk Score is the "
        "training-average value.",
    ),
    "v2_dnn_lstm": (
        "Risk Score",
        "LSTM behavioral-risk component: the v2 LSTM's score for the customer's 10 previous transactions, "
        "used as an input to the DNN",
        "Risk Score is the LSTM behavioral-risk component and Fraud Probability the output of the DNN fraud "
        "classifier of model set v2_dnn_lstm (trained on synthetic dataset v2); reasons are derived from SHAP "
        "feature attribution on the DNN. For customers with fewer than 10 earlier transactions the LSTM is not "
        "used and Risk Score is the training-average value.",
    ),
    "v2_dnn_only": (
        "Risk Score (compatibility)",
        "Same value as Fraud Probability: this model set has no sequence model, so the risk score field "
        "repeats the DNN fraud score for compatibility",
        "Fraud Probability is the score of a DNN on 9 behavioural features (model set v2_dnn_only, trained on "
        "synthetic dataset v2). This model set has no sequence model; Risk Score repeats the DNN fraud score "
        "for compatibility and is not a separate signal. Reasons are derived from SHAP feature attribution on "
        "the DNN.",
    ),
}
UNRECORDED_WORDING = (
    "Risk Score",
    "Model output reported in the risk score field; its meaning depends on the model set, which was not "
    "recorded for this transaction",
    "Scores are model outputs; reasons are derived from SHAP feature attribution on the DNN. The model set "
    "that scored this transaction was not recorded, so no model-specific description is given.",
)


def build_pdf_report(prediction: dict, provenance: dict | None = None) -> bytes:
    """provenance: {"recorded", "model_set", "model_version", "reason"?, "metadata"?}
    from the transaction's stored row (main._provenance). None or not recorded:
    neutral wording, no model set assumed."""
    provenance = provenance or {"recorded": False, "model_set": None, "model_version": None,
                                "reason": "no provenance supplied"}
    model_set = provenance.get("model_set") if provenance.get("recorded") else None
    risk_label, risk_meaning, footer_text = MODEL_SET_WORDING.get(model_set, UNRECORDED_WORDING)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=LETTER,
        topMargin=0.7 * inch, bottomMargin=0.7 * inch,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("ReportTitle", parent=styles["Title"], textColor=BRAND, fontSize=20)
    h2_style = ParagraphStyle("ReportH2", parent=styles["Heading2"], textColor=BRAND, spaceBefore=14)
    body_style = styles["BodyText"]

    alert_level = prediction.get("alert_level", "Unknown")
    alert_color = ALERT_COLORS.get(alert_level, colors.grey)

    story = []
    story.append(Paragraph("Fraud Intelligence Report", title_style))
    story.append(Paragraph("Explainable Fraud Intelligence Platform", body_style))
    story.append(Spacer(1, 0.2 * inch))

    alert_para_style = ParagraphStyle(
        "AlertBannerText", parent=styles["Heading2"], textColor=colors.white, alignment=1,
    )
    alert_table = Table([[Paragraph(alert_level, alert_para_style)]], colWidths=[6 * inch])
    alert_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), alert_color),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(alert_table)
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Transaction Details", h2_style))
    txn_rows = [
        ["Transaction ID", prediction.get("transaction_id", "-")],
        ["Customer ID", prediction.get("customer_id", "-")],
        ["Timestamp", str(prediction.get("timestamp", "-"))],
        ["Amount", f"Rs. {prediction.get('amount', 0):,.2f}"],
        ["Merchant Category", str(prediction.get("merchant_category", "-")).replace("_", " ")],
        ["Device ID", prediction.get("device_id", "-")],
        ["Location", prediction.get("location", "-")],
        ["Failed Logins (24h)", str(prediction.get("failed_logins_24h", 0))],
    ]
    txn_table = Table(txn_rows, colWidths=[2.2 * inch, 3.8 * inch])
    txn_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#444444")),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -2), 0.5, colors.HexColor("#dddddd")),
    ]))
    story.append(txn_table)

    story.append(Paragraph("Model", h2_style))
    if model_set:
        model_rows = [["Model set", model_set], ["Model version", provenance.get("model_version") or "-"]]
        meta = provenance.get("metadata")
        if meta:
            model_rows.append(["Training data", f"synthetic dataset {meta['dataset']['version']}"])
            model_rows.append(["Model type", "LSTM risk score -> DNN" if meta["uses_lstm"]
                               else "DNN only (no sequence model)"])
    else:
        model_rows = [["Model set", "not recorded"],
                      ["Reason", provenance.get("reason") or "-"]]
    model_table = Table(model_rows, colWidths=[2.2 * inch, 3.8 * inch])
    model_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#444444")),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -2), 0.5, colors.HexColor("#dddddd")),
    ]))
    story.append(model_table)

    story.append(Paragraph("Risk Assessment", h2_style))
    cell_style = ParagraphStyle("Cell", parent=body_style, fontSize=9, leading=11)
    risk_rows = [
        ["Metric", "Value", "Meaning"],
        [Paragraph(f"<b>{risk_label}</b>", cell_style), f"{prediction.get('risk_score', 0):.1f} / 100",
         Paragraph(risk_meaning, cell_style)],
        ["Fraud Probability", f"{prediction.get('fraud_probability', 0):.1f}%",
         Paragraph(FRAUD_SCORE_MEANING, cell_style)],
        ["Behavioral Similarity", f"{prediction.get('similarity_pct', 0):.1f}%",
         Paragraph("How closely this matches the customer's normal behavior", cell_style)],
        ["Deviation", f"{prediction.get('deviation_pct', 0):.1f}%",
         Paragraph("How far this deviates from the customer's own historical norm", cell_style)],
    ]
    risk_table = Table(risk_rows, colWidths=[1.5 * inch, 1.1 * inch, 3.4 * inch])
    risk_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BRAND),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (1, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f7f7")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
    ]))
    story.append(risk_table)

    story.append(Paragraph("Explainable AI — Why This Score", h2_style))
    reasons = prediction.get("reasons") or []
    if reasons:
        reason_rows = [["Factor", "SHAP Contribution"]]
        for r in reasons:
            reason_rows.append([r.get("display_name", r.get("feature", "-")), f"{r.get('shap_value', 0):+.3f}"])
        reason_table = Table(reason_rows, colWidths=[3.6 * inch, 2.4 * inch])
        reason_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef3f3")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
        ]))
        story.append(reason_table)
    else:
        story.append(Paragraph("No significant fraud-indicating factors detected.", body_style))

    story.append(Spacer(1, 0.15 * inch))
    footer_style = ParagraphStyle("Footer", parent=styles["Normal"], textColor=colors.grey, fontSize=8)
    story.append(Paragraph(
        "Generated by the Explainable Fraud Intelligence Platform. " + footer_text,
        footer_style,
    ))

    doc.build(story)
    return buf.getvalue()
