import { riskBand } from '../format'

// value: a number = the score; null = "not available" (e.g. too little history); undefined = nothing scanned yet.
export default function RiskGauge({ label, value, sublabel, decimals = 0 }) {
  const hasValue = typeof value === 'number'
  const clamped = hasValue ? Math.max(0, Math.min(100, value)) : 0
  const radius = 54
  const circumference = 2 * Math.PI * radius
  const offset = circumference * (1 - clamped / 100)
  const band = hasValue ? riskBand(clamped) : null
  const color = band ? band.color : 'var(--risk-low)'

  // The gauge is purely visual, so it is exposed to assistive technology as ONE image with a text
  // equivalent such as "Fraud Risk Score: 97.4, Critical" (the same value and band the eye reads).
  const spokenName = hasValue
    ? `${label}: ${clamped.toFixed(decimals)}, ${band.level}`
    : `${label}: ${value === null ? 'not available' : 'no result yet'}`

  return (
    <div role="img" aria-label={spokenName} className="flex flex-col items-center justify-center">
      <div className="relative w-36 h-36">
        <svg viewBox="0 0 130 130" className="w-full h-full -rotate-90" aria-hidden="true" focusable="false">
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
            {hasValue ? clamped.toFixed(decimals) : '—'}
          </span>
          {sublabel && (
            <span className="text-[10px] mt-0.5" style={{ color: 'var(--text-faint)' }}>{sublabel}</span>
          )}
        </div>
      </div>
      <span className="text-xs uppercase tracking-wide mt-3" style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
        {label}
      </span>
      {band && (
        // the band is also written out, so the risk level is never conveyed by colour alone
        <span className="text-[11px] uppercase tracking-wide mt-0.5" style={{ color: band.color, fontFamily: 'var(--font-mono)' }}>
          {band.level}
        </span>
      )}
    </div>
  )
}
