import { useEffect, useState } from 'react'
import { getMetrics } from '../api'

const METRIC_LABELS = [
  { key: 'precision', label: 'Precision' },
  { key: 'recall', label: 'Recall' },
  { key: 'f1_score', label: 'F1 Score' },
  { key: 'auc_roc', label: 'ROC-AUC' },
  { key: 'pr_auc', label: 'PR-AUC' },
  { key: 'false_positive_rate', label: 'False Positive Rate', percent: true },
]

function StatTile({ label, value, percent }) {
  return (
    <div className="rounded-lg border p-3 text-center" style={{ borderColor: 'var(--border)', background: 'var(--surface-raised)' }}>
      <div className="text-2xl" style={{ fontFamily: 'var(--font-mono)', color: 'var(--brand)', fontWeight: 600 }}>
        {value == null ? '—' : percent ? `${(value * 100).toFixed(2)}%` : value.toFixed(3)}
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

      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 mb-4">
        {METRIC_LABELS.map(({ key, label, percent }) => (
          <StatTile key={key} label={label} value={data[key]} percent={percent} />
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
    return <p className="text-sm" style={{ color: 'var(--text-muted)' }}>Evaluating held-out test set…</p>
  }
  if (error) {
    return <p className="text-sm" style={{ color: 'var(--risk-critical)' }}>{error}</p>
  }

  return (
    <div className="space-y-5">
      <div
        className="rounded-lg border p-3 text-xs space-y-1.5"
        style={{ borderColor: 'var(--risk-medium)', background: 'var(--risk-medium-dim)', color: 'var(--text-muted)' }}
      >
        <p style={{ color: 'var(--text-primary)', fontWeight: 600 }}>Synthetic dataset &mdash; read these numbers with care</p>
        <p>{metrics?.evaluation_context?.dataset || 'Synthetic behavioral dataset; no real banking data.'}</p>
        <p>{metrics?.evaluation_context?.split || 'Random stratified 80/20 hold-out.'}</p>
        <p>
          {metrics?.evaluation_context?.interpretation ||
            'Strong results on this synthetic data are not evidence of real-world fraud-detection performance.'}
        </p>
        <p>The model outputs are risk scores, not calibrated probabilities. Threshold 0.5 is used for precision, recall, F1 and FPR.</p>
      </div>
      <ModelBlock
        title="LSTM Temporal Risk Model"
        subtitle="Uses a customer's recent 10-transaction sequence to estimate temporal fraud risk for the next transaction. In this dataset most fraud sequences already contain earlier fraud, so it is not evidence of predicting the first fraudulent event."
        data={metrics?.lstm_risk_predictor}
      />
      <ModelBlock
        title="DNN Fraud Risk Model"
        subtitle="Scores each transaction from its own behavioral features plus the LSTM temporal risk score."
        data={metrics?.dnn_fraud_classifier}
      />
    </div>
  )
}
