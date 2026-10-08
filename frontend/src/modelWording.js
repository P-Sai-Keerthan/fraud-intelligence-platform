// Model-aware wording for the scores (Step 4C-2f-2).
//
// What risk_score means depends on the loaded model set (GET /model-info):
// * production / v2_dnn_lstm: an LSTM score from the customer's previous
//   transactions (temporal behavioral risk), an input to the DNN;
// * v2_dnn_only: there is no sequence model -- risk_score repeats the DNN
//   fraud score for API compatibility and must not be called trajectory risk.
// If the model set is unknown (request failed), the wording stays neutral.
// fraud_probability is a model score, not a calibrated probability.

const WORDING = {
  production: {
    riskLabel: 'Risk Score',
    riskShort: 'Temporal behavioral risk (LSTM)',
    riskCaption: "Temporal behavioral risk signal computed by the LSTM from this customer's previous 10 transactions",
    fraudCaption: "This transaction's fraud score (model output, not a calibrated probability) -- can be high even if the recent history looked normal",
    riskColumn: 'Risk Score',
    timelineRisk: 'Risk Score (LSTM)',
  },
  v2_dnn_lstm: {
    riskLabel: 'Risk Score',
    riskShort: 'Behavioral risk from previous transactions',
    riskCaption: "LSTM behavioral-risk component from this customer's previous transactions, before this transaction",
    fraudCaption: "This transaction's fraud score (model output, not a calibrated probability) -- can be high even if the recent history looked normal",
    riskColumn: 'Risk Score',
    timelineRisk: 'Risk Score (LSTM)',
  },
  // the selected v2_dnn_lstm artifact, training seed 14 (evaluation only, not deployed)
  v2_dnn_lstm_seed14: {
    riskLabel: 'Risk Score',
    riskShort: 'Behavioral risk from previous transactions',
    riskCaption: "LSTM behavioral-risk component from this customer's previous transactions, before this transaction",
    fraudCaption: "This transaction's fraud score (model output, not a calibrated probability) -- can be high even if the recent history looked normal",
    riskColumn: 'Risk Score',
    timelineRisk: 'Risk Score (LSTM)',
  },
  v2_dnn_only: {
    riskLabel: 'Risk Score (= fraud score)',
    riskShort: 'Repeats the fraud score (no sequence model)',
    riskCaption: 'This model set has no sequence model: Risk Score repeats the DNN fraud score for compatibility',
    fraudCaption: "This transaction's DNN fraud score (model output, not a calibrated probability)",
    riskColumn: 'Risk Score (= fraud score)',
    timelineRisk: 'Risk Score (= fraud score)',
  },
}

const NEUTRAL = {
  riskLabel: 'Risk Score',
  riskShort: 'Model risk output',
  riskCaption: 'Model risk output (model set unknown)',
  fraudCaption: "This transaction's fraud score (model output, not a calibrated probability)",
  riskColumn: 'Risk Score',
  timelineRisk: 'Risk Score',
}

export function scoreWording(modelInfo) {
  return WORDING[modelInfo?.model_set] || NEUTRAL
}

export function modelSetLabel(modelInfo) {
  if (!modelInfo?.model_set) return 'model set unknown'
  return `model set ${modelInfo.model_set} (${modelInfo.model_version})`
}
