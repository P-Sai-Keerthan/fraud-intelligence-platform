const RISK_COLOR_STOPS = [
  { max: 25, color: 'var(--risk-low)' },
  { max: 50, color: 'var(--risk-medium)' },
  { max: 80, color: 'var(--risk-high)' },
  { max: 100, color: 'var(--risk-critical)' },
]

function colorForScore(score) {
  const stop = RISK_COLOR_STOPS.find((s) => score <= s.max)
  return (stop || RISK_COLOR_STOPS[RISK_COLOR_STOPS.length - 1]).color
}

export default function RiskGauge({ label, value, sublabel, decimals = 0 }) {
  const clamped = Math.max(0, Math.min(100, value ?? 0))
  const radius = 54
  const circumference = 2 * Math.PI * radius
  const offset = circumference * (1 - clamped / 100)
  const color = colorForScore(clamped)

  return (
    <div className="flex flex-col items-center justify-center">
      <div className="relative w-36 h-36">
        <svg viewBox="0 0 130 130" className="w-full h-full -rotate-90">
          <circle cx="65" cy="65" r={radius} fill="none" stroke="var(--border)" strokeWidth="10" />
          <circle
            cx="65" cy="65" r={radius} fill="none"
            stroke={color} strokeWidth="10" strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            style={{ transition: 'stroke-dashoffset 0.6s ease, stroke 0.3s ease' }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span
            className="text-3xl"
            style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-primary)', fontWeight: 600 }}
          >
            {value != null ? clamped.toFixed(decimals) : '—'}
          </span>
          {sublabel && (
            <span className="text-[10px] mt-0.5" style={{ color: 'var(--text-faint)' }}>{sublabel}</span>
          )}
        </div>
      </div>
      <span className="text-xs uppercase tracking-wide mt-3" style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
        {label}
      </span>
    </div>
  )
}
