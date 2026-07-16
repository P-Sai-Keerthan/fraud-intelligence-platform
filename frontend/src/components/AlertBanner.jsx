const ALERT_STYLES = {
  'Low Risk': { color: 'var(--risk-low)', bg: 'var(--risk-low-dim)' },
  'Medium Risk': { color: 'var(--risk-medium)', bg: 'var(--risk-medium-dim)' },
  'High Risk': { color: 'var(--risk-high)', bg: 'var(--risk-high-dim)' },
  'Critical Risk': { color: 'var(--risk-critical)', bg: 'var(--risk-critical-dim)' },
}

export default function AlertBanner({ level }) {
  if (!level) return null
  const style = ALERT_STYLES[level] || ALERT_STYLES['Low Risk']

  return (
    <div
      className="rounded-lg px-4 py-3 flex items-center gap-3 border"
      style={{ background: style.bg, borderColor: style.color + '55' }}
    >
      <span className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ background: style.color }} />
      <span className="text-sm font-medium" style={{ color: style.color, fontFamily: 'var(--font-display)' }}>
        {level}
      </span>
    </div>
  )
}
