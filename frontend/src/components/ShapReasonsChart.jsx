import { useState } from 'react'
import { EmptyState } from './ui'

function importance(share, isTop) {
  if (isTop) return { label: 'Primary driver', pips: 3 }
  if (share >= 0.2) return { label: 'Strong', pips: 2 }
  return { label: 'Contributing', pips: 1 }
}

function Pips({ n }) {
  return (
    <span className="inline-flex gap-[3px]" aria-hidden="true">
      {[0, 1, 2].map((i) => (
        <span key={i} className="rounded-sm" style={{ width: 4, height: 10, background: i < n ? 'var(--series-1)' : 'var(--border-strong)' }} />
      ))}
    </span>
  )
}

// Horizontal feature-contribution bars from the `reasons` of POST /predict.
// Every number shown is the backend's SHAP value, or a share computed from them.
export default function ShapReasonsChart({ reasons, hasPrediction }) {
  const [hover, setHover] = useState(null)

  if (!reasons || reasons.length === 0) {
    return hasPrediction ? (
      <EmptyState icon="check" title="No significant fraud-indicating factors" compact>
        The model returned no features that pushed this transaction&apos;s score up.
      </EmptyState>
    ) : (
      <EmptyState icon="search" title="No explanation yet" compact>
        Scan a transaction to see which behaviours pushed its fraud score up, and by how much.
      </EmptyState>
    )
  }

  const rows = [...reasons].sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value))
  const max = Math.max(...rows.map((r) => Math.abs(r.shap_value))) || 1
  const total = rows.reduce((sum, r) => sum + Math.abs(r.shap_value), 0) || 1

  return (
    <div>
      <ul className="space-y-1" aria-label="Feature contributions to the fraud score">
        {rows.map((r, i) => {
          const magnitude = Math.abs(r.shap_value)
          const share = magnitude / total
          const imp = importance(share, i === 0)
          const active = hover === r.feature
          return (
            <li
              key={r.feature} tabIndex={0}
              onMouseEnter={() => setHover(r.feature)} onMouseLeave={() => setHover(null)}
              onFocus={() => setHover(r.feature)} onBlur={() => setHover(null)}
              className="grid items-center gap-x-4 gap-y-1 rounded-lg px-3 py-2.5"
              style={{
                gridTemplateColumns: 'minmax(96px, 220px) minmax(40px, 1fr) auto',
                background: active ? 'rgba(56,189,248,0.05)' : 'transparent', transition: 'background 0.12s ease',
              }}
              title={`${r.display_name} (${r.feature}): SHAP ${r.shap_value >= 0 ? '+' : ''}${r.shap_value.toFixed(4)}`}
            >
              <div className="min-w-0">
                <div className="text-[13px] truncate" style={{ color: 'var(--text-primary)', fontWeight: 500 }}>{r.display_name}</div>
                <div className="mono text-[10.5px] truncate" style={{ color: 'var(--text-faint)' }}>{r.feature}</div>
              </div>
              <div className="h-[18px] flex items-center" aria-hidden="true">
                <div className="w-full h-3 rounded-r" style={{ background: 'var(--surface-sunken)', borderLeft: '1px solid var(--border-strong)' }}>
                  <div
                    className="h-full"
                    style={{
                      width: `${Math.max(2, (magnitude / max) * 100)}%`, background: 'var(--series-1)',
                      borderRadius: '0 4px 4px 0', opacity: active || hover == null ? 1 : 0.55,
                      transition: 'width 0.5s ease, opacity 0.15s ease',
                    }}
                  />
                </div>
              </div>
              <div className="flex items-center gap-3 justify-end">
                <span className="mono text-xs tnum" style={{ color: 'var(--text-primary)' }}>
                  {r.shap_value >= 0 ? '+' : '−'}{magnitude.toFixed(3)}
                </span>
                <span className="mono text-[11px] tnum w-10 text-right" style={{ color: 'var(--text-muted)' }}>{Math.round(share * 100)}%</span>
                <span className="hidden md:inline-flex items-center gap-2 w-[124px]">
                  <Pips n={imp.pips} />
                  <span className="text-[11px]" style={{ color: 'var(--text-secondary)' }}>{imp.label}</span>
                </span>
              </div>
            </li>
          )
        })}
      </ul>
      <p className="text-[11px] mt-3 px-3" style={{ color: 'var(--text-muted)' }}>
        Bars show each feature&apos;s SHAP contribution towards a higher fraud score (longest = largest). The percentage is the
        feature&apos;s share of the contributions listed here. The labels (Primary driver, Strong, Contributing) are
        derived in the interface from these values. SHAP explains the model&apos;s score; it does not prove fraud.
      </p>
    </div>
  )
}
