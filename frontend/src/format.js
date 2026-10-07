// Small display helpers shared by the dashboard. Display only: nothing here changes a value
// that comes from the backend, and nothing recomputes a score.

// Rupees: always the "₹" symbol, Indian digit grouping (the PDF report is the one exception, because
// the built-in PDF fonts have no ₹ glyph; see backend/app/report.py). Summary statistics are rounded to whole
// rupees; pass { exact: true } for a transaction's own amount, which is shown with its real decimals.
export function rupees(x, { exact = false } = {}) {
  if (x == null || Number.isNaN(Number(x))) return '—'
  const n = Number(x)
  // exact: whole rupees stay "₹1,500", fractional ones always show two decimals ("₹1,500.50", not "₹1,500.5")
  return exact
    ? `₹${n.toLocaleString('en-IN', { minimumFractionDigits: Number.isInteger(n) ? 0 : 2, maximumFractionDigits: 2 })}`
    : `₹${Math.round(n).toLocaleString('en-IN')}`
}

// "1 transaction", "5 transactions"
export const plural = (n, singular, pluralForm = `${singular}s`) =>
  `${Number(n).toLocaleString()} ${Number(n) === 1 ? singular : pluralForm}`

export function parseTimestamp(iso) {
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? null : d
}

// 24-hour clock everywhere (the old minute-only 12-hour labels were ambiguous)
const dayMonth = { day: '2-digit', month: 'short' }
const hms = { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false }
const hm = { hour: '2-digit', minute: '2-digit', hour12: false }

export const formatDay = (d) => d.toLocaleDateString([], dayMonth) // 08 Oct
export const formatClock = (d, withSeconds = true) => d.toLocaleTimeString([], withSeconds ? hms : hm) // 14:32:05
export const formatDateTime = (d, withSeconds = true) => `${formatDay(d)}, ${formatClock(d, withSeconds)}`
export const formatFullDateTime = (d) =>
  `${d.toLocaleDateString([], { day: '2-digit', month: 'short', year: 'numeric' })}, ${formatClock(d)}`

// Risk bands for DISPLAY (gauge colour + spoken level). They are the same 25 / 50 / 80 cut-offs the backend
// uses for its alert levels (backend/app/models/dnn_model.py: below 25 Low, below 50 Medium, below 80 High,
// otherwise Critical), including the "below" comparison at the boundaries. Nothing is recomputed here.
const BANDS = [
  { below: 25, level: 'Low', color: 'var(--risk-low)' },
  { below: 50, level: 'Medium', color: 'var(--risk-medium)' },
  { below: 80, level: 'High', color: 'var(--risk-high)' },
  { below: Infinity, level: 'Critical', color: 'var(--risk-critical)' },
]
export const riskBand = (score) => BANDS.find((b) => score < b.below)
