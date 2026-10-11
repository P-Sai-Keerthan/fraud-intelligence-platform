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

// Legacy cut points (25 / 50 / 80) used when the backend publishes no alert policy for the loaded model set.
const LEGACY_TIERS = [
  { level: 'Low Risk', min_score: 0 },
  { level: 'Medium Risk', min_score: 25 },
  { level: 'High Risk', min_score: 50 },
  { level: 'Critical Risk', min_score: 80 },
]

// Alert tiers of the loaded model set, from GET /model-info (thresholds.alert_policy.tiers); legacy bands otherwise.
export function alertTiers(modelInfo) {
  const tiers = modelInfo?.thresholds?.alert_policy?.tiers
  return Array.isArray(tiers) && tiers.length ? tiers : LEGACY_TIERS
}

// Gauge colour for a 0-100 *Fraud Score*: the colour of the alert level that score falls in. The tiers come from the
// backend's alert policy and are used here only to tint the gauge, never to decide an alert (R-02).
export function scoreColor(score, tiers = LEGACY_TIERS) {
  if (score == null) return 'var(--border-strong)'
  let chosen = tiers[0]
  for (const t of tiers) if (score >= t.min_score) chosen = t
  return severityOf(chosen.level).color
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
