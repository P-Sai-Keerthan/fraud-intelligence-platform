import { useEffect, useState } from 'react'
import { getMetrics } from '../api'

const METRIC_LABELS = [
  { key: 'precision', label: 'Precision' },
  { key: 'recall', label: 'Recall' },
  { key: 'f1_score', label: 'F1 Score' },
  { key: 'auc_roc', label: 'AUC-ROC' },
]

function StatTile({ label, value }) {
  return (
    <div className="rounded-lg border p-3 text-center" style={{ borderColor: 'var(--border)', background: 'var(--surface-raised)' }}>
      <div className="text-2xl" style={{ fontFamily: 'var(--font-mono)', color: 'var(--brand)', fontWeight: 600 }}>
        {value != null ? value.toFixed(3) : '—'}
      </div>
      <div className="text-[10px] uppercase tracking-wide mt-1" style={{ color: 'var(--text-muted)' }}>{label}</div>
    </div>
  )
}

function ConfusionMatrix({ cm }) {
  if (!cm) return null
  const cellStyle = (bg) => ({
    borderColor: 'var(--border)',
    background: bg,
  })
  return (
    <div className="grid grid-cols-2 gap-2 text-center text-xs" style={{ fontFamily: 'var(--font-mono)' }}>
      <div className="rounded-md border p-3" style={cellStyle('var(--risk-low-dim)')}>
        <div style={{ color: 'var(--risk-low)', fontWeight: 600, fontSize: '1.1rem' }}>{cm.true_negative}</div>
        <div style={{ color: 'var(--text-faint)' }} className="mt-1">True Negative</div>
      </div>
      <div className="rounded-md border p-3" style={cellStyle('var(--risk-medium-dim)')}>
        <div style={{ color: 'var(--risk-medium)', fontWeight: 600, fontSize: '1.1rem' }}>{cm.false_positive}</div>
        <div style={{ color: 'var(--text-faint)' }} className="mt-1">False Positive</div>
      </div>
      <div className="rounded-md border p-3" style={cellStyle('var(--risk-critical-dim)')}>
        <div style={{ color: 'var(--risk-critical)', fontWeight: 600, fontSize: '1.1rem' }}>{cm.false_negative}</div>
        <div style={{ color: 'var(--text-faint)' }} className="mt-1">False Negative</div>
      </div>
      <div className="rounded-md border p-3" style={cellStyle('var(--risk-low-dim)')}>
        <div style={{ color: 'var(--risk-low)', fontWeight: 600, fontSize: '1.1rem' }}>{cm.true_positive}</div>
        <div style={{ color: 'var(--text-faint)' }} className="mt-1">True Positive</div>
      </div>
    </div>
  )
}

function ModelBlock({ title, subtitle, data }) {
  if (!data) return null
  return (
    <div className="rounded-xl border p-5" style={{ borderColor: 'var(--border)', background: 'var(--surface)' }}>
      <h4 style={{ fontFamily: 'var(--font-display)', color: 'var(--text-primary)', fontWeight: 600 }} className="text-base">
        {title}
      </h4>
      <p className="text-xs mb-4" style={{ color: 'var(--text-muted)' }}>{subtitle}</p>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
        {METRIC_LABELS.map(({ key, label }) => (
          <StatTile key={key} label={label} value={data[key]} />
        ))}
      </div>

      <p className="text-[11px] uppercase tracking-wide mb-2" style={{ color: 'var(--text-faint)', fontFamily: 'var(--font-mono)' }}>
        Confusion Matrix &middot; test set of {data.test_set_size?.toLocaleString()} ({data.fraud_rate_pct}% fraud) &middot; threshold {data.threshold}
      </p>
      <ConfusionMatrix cm={data.confusion_matrix} />
    </div>
  )
}

export default function ModelPerformance() {
  const [metrics, setMetrics] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    getMetrics()
      .then(setMetrics)
      .catch(() => setError('Could not load model metrics. Is the backend running?'))
      .finally(() => setLoading(false))
  }, [])

  if (loading) {
    return <p className="text-sm" style={{ color: 'var(--text-muted)' }}>Loading evaluation report…</p>
  }
  if (error) {
    return <p className="text-sm" style={{ color: 'var(--risk-critical)' }}>{error}</p>
  }

  const modelSet = metrics?.model_set
  const noteStyle = { color: 'var(--text-muted)' }
  const tagStyle = { color: 'var(--text-faint)', fontFamily: 'var(--font-mono)' }

  // candidate model sets: the v1 report does not evaluate them (4C-2f-2)
  if (modelSet && modelSet !== 'production') {
    const ev = metrics?.candidate_evaluation
    const title = modelSet === 'v2_dnn_lstm' ? 'DNN Fraud Classifier (v2 candidate: LSTM risk score -> DNN)' : 'DNN Fraud Classifier (v2 candidate: DNN only)'
    return (
      <div className="space-y-5">
        <p className="text-[11px]" style={tagStyle}>Loaded model set {modelSet} ({metrics.model_version})</p>
        <p className="text-xs" style={noteStyle}>
          The v1 evaluation shown for the production model set does not evaluate this model set, so it is not shown.
          {ev?.available
            ? ' Below: this candidate on the v2 synthetic test period (candidate comparison), at its validation-chosen F1 threshold. Scores are not calibrated probabilities; this threshold is not applied by live scoring, which still uses the 25/50/80 alert bands.'
            : ` No candidate evaluation is available (${ev?.reason || 'unknown reason'}).`}
        </p>
        {ev?.available && (
          <ModelBlock
            title={title}
            subtitle={`v2 test period: ${ev.rows?.test?.toLocaleString()} transactions, ${ev.rows?.test_fraud_episodes} fraud episodes. PR-AUC ${ev.overall_test?.pr_auc}; first-fraud recall ${ev.episodes_test?.first_fraud_detected}/${ev.episodes_test?.first_fraud_transactions}.`}
            data={ev.overall_test}
          />
        )}
      </div>
    )
  }

  return (
    <div className="space-y-5">
      {modelSet && (
        <p className="text-[11px]" style={tagStyle}>Loaded model set {modelSet} ({metrics.model_version}) &middot; v1 evaluation</p>
      )}
      <p className="text-xs" style={noteStyle}>
        Time-based evaluation on dataset v1: evaluation copies of the production models trained on the earliest
        transactions, the decision threshold chosen on the following period, and these numbers measured on the
        latest period. Scores are not calibrated probabilities. Full methodology, baselines and first-fraud
        metrics: GET /metrics/report.
      </p>
      <ModelBlock
        title="LSTM Risk Predictor"
        subtitle="Predicts whether the transaction after a 10-step behavioral window will be fraudulent."
        data={metrics?.lstm_risk_predictor}
      />
      <ModelBlock
        title="DNN Fraud Classifier"
        subtitle="Classifies each transaction using its own features plus the LSTM risk score -- the real-time detection signal."
        data={metrics?.dnn_fraud_classifier}
      />
    </div>
  )
}
