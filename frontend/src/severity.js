// Severity presentation for the four alert levels returned by the backend.
// The levels themselves (and the score bands behind them) are decided by the
// backend; this file only maps a level name to colours and an order.

export const SEVERITY = {
  'Critical Risk': { key: 'critical', short: 'Critical', color: 'var(--risk-critical)', dim: 'var(--risk-critical-dim)', rank: 3 },
  'High Risk': { key: 'high', short: 'High', color: 'var(--risk-high)', dim: 'var(--risk-high-dim)', rank: 2 },
  'Medium Risk': { key: 'medium', short: 'Medium', color: 'var(--risk-medium)', dim: 'var(--risk-medium-dim)', rank: 1 },
  'Low Risk': { key: 'low', short: 'Low', color: 'var(--risk-low)', dim: 'var(--risk-low-dim)', rank: 0 },
}

export const SEVERITY_ORDER = ['Critical Risk', 'High Risk', 'Medium Risk', 'Low Risk']

const UNKNOWN = { key: 'unknown', short: 'Unknown', color: 'var(--text-muted)', dim: 'rgba(127,139,161,0.1)', rank: -1 }

export function severityOf(level) {
  return SEVERITY[level] || UNKNOWN
}

// Gauge colour for a 0-100 score. The cut points are the alert bands the
// backend applies to the fraud score (25 / 50 / 80); they are used here only
// to tint the gauge, never to decide an alert.
export function scoreColor(score) {
  if (score == null) return 'var(--border-strong)'
  if (score < 25) return 'var(--risk-low)'
  if (score < 50) return 'var(--risk-medium)'
  if (score < 80) return 'var(--risk-high)'
  return 'var(--risk-critical)'
}

// "fraud_probability >= 80" (backend wording of an alert band) -> "fraud score ≥ 80"
export function bandText(raw) {
  if (!raw) return ''
  return String(raw)
    .replaceAll('fraud_probability', 'fraud score')
    .replaceAll('>=', '≥')
    .replaceAll('<=', '≤')
}

export function shortHash(value, n = 10) {
  return value ? `${String(value).slice(0, n)}…` : '—'
}

export function formatDateTime(value) {
  if (!value) return '—'
  const d = new Date(value)
  if (Number.isNaN(d.getTime())) return String(value)
  return d.toLocaleString([], { year: 'numeric', month: 'short', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit' })
}
