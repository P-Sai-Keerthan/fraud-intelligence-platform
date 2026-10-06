import { scoreColor } from '../severity'

// A thin progress ring with the value in the middle. `color` overrides the
// score-based tint (the similarity ring uses the brand colour).
export function ScoreRing({ value, color, size = 132, stroke = 9, children }) {
  const hasValue = value != null
  const clamped = Math.max(0, Math.min(100, value ?? 0))
  const radius = (size - stroke) / 2 - 2
  const circumference = 2 * Math.PI * radius
  const offset = circumference * (1 - clamped / 100)
  const tint = color || scoreColor(hasValue ? clamped : null)
  return (
    <div className="relative shrink-0" style={{ width: size, height: size }}>
      <svg viewBox={`0 0 ${size} ${size}`} className="w-full h-full -rotate-90" aria-hidden="true">
        <circle cx={size / 2} cy={size / 2} r={radius} fill="none" stroke="var(--border)" strokeWidth={stroke} />
        {hasValue && (
          <circle
            cx={size / 2} cy={size / 2} r={radius} fill="none"
            stroke={tint} strokeWidth={stroke} strokeLinecap="round"
            strokeDasharray={circumference} strokeDashoffset={offset}
            style={{ transition: 'stroke-dashoffset 0.6s ease, stroke 0.3s ease' }}
          />
        )}
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">{children}</div>
    </div>
  )
}

// Metric card body: label, ring with "value / 100", one-line meaning, fine print.
export default function RiskGauge({ label, value, decimals = 0, caption, note, loading = false }) {
  const hasValue = value != null
  const clamped = Math.max(0, Math.min(100, value ?? 0))
  const text = hasValue ? clamped.toFixed(decimals) : '—'
  return (
    <div className="flex flex-col items-center text-center h-full">
      <div className="eyebrow self-start">{label}</div>
      <div className="my-3" style={{ opacity: loading ? 0.45 : 1, transition: 'opacity 0.2s ease' }}>
        <ScoreRing value={value}>
          <span className="text-[30px] leading-none tnum" style={{ fontWeight: 600, color: 'var(--text-primary)' }}
            aria-label={hasValue ? `${label}: ${text} out of 100` : `${label}: no value yet`}>
            {text}
          </span>
          <span className="mono text-[11px] mt-1" style={{ color: 'var(--text-faint)' }}>/ 100</span>
        </ScoreRing>
      </div>
      {caption && <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>{caption}</p>}
      {note && <p className="text-[11px] mt-1" style={{ color: 'var(--text-muted)' }}>{note}</p>}
    </div>
  )
}
