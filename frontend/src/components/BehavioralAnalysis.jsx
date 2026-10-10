import { EmptyState, Icon } from './ui'

function Compare({ icon, label, baseline, current }) {
  const known = baseline != null && current != null
  const match = known && String(baseline) === String(current)
  return (
    <div className="grid items-center gap-3 py-3" style={{ gridTemplateColumns: '104px minmax(0,1fr) minmax(0,1fr) 96px', borderBottom: '1px solid var(--border-subtle)' }}>
      <span className="flex items-center gap-2 text-xs" style={{ color: 'var(--text-muted)' }}>
        <Icon name={icon} size={14} style={{ color: 'var(--text-faint)' }} />{label}
      </span>
      <span className="mono text-xs truncate" style={{ color: 'var(--text-secondary)' }} title={baseline || ''}>{baseline ?? '—'}</span>
      <span className="mono text-xs truncate" style={{ color: 'var(--text-primary)' }} title={current || ''}>{current ?? '—'}</span>
      <span className="flex justify-end">
        {known ? (
          <span className="inline-flex items-center gap-1.5 text-[11px] rounded-md px-2 h-[22px] border"
            style={match
              ? { color: 'var(--text-primary)', borderColor: 'rgba(47,191,143,0.35)', background: 'var(--risk-low-dim)' }
              : { color: 'var(--text-primary)', borderColor: 'rgba(240,130,60,0.4)', background: 'var(--risk-high-dim)' }}>
            <Icon name={match ? 'check' : 'x'} size={11} style={{ color: match ? 'var(--risk-low)' : 'var(--risk-high)' }} />
            {match ? 'Match' : 'Different'}
          </span>
        ) : <span className="text-[11px]" style={{ color: 'var(--text-faint)' }}>—</span>}
      </span>
    </div>
  )
}

// Customer baseline (GET /customer/{id}/profile) next to the scanned
// transaction (POST /predict), and the backend's similarity / deviation split.
export default function BehavioralAnalysis({ profile, prediction }) {
  if (!profile && !prediction) {
    return (
      <EmptyState icon="user" title="No customer selected" compact>
        Select a customer to load their behavioural baseline, then scan a transaction to compare against it.
      </EmptyState>
    )
  }
  const similarity = prediction?.similarity_pct
  const deviation = prediction?.deviation_pct
  const hasSplit = similarity != null && deviation != null
  const insufficient = prediction?.similarity_status === 'insufficient_history' && !hasSplit

  return (
    <div>
      <div className="overflow-x-auto">
      <div style={{ minWidth: 460 }}>
      <div className="grid gap-3 pb-2" style={{ gridTemplateColumns: '104px minmax(0,1fr) minmax(0,1fr) 96px', borderBottom: '1px solid var(--border)' }}>
        <span />
        <span className="eyebrow">Customer baseline</span>
        <span className="eyebrow">Current transaction</span>
        <span className="eyebrow text-right">Deviation</span>
      </div>
      <Compare icon="device" label="Device" baseline={profile?.home_device} current={prediction?.device_id} />
      <Compare icon="pin" label="Location" baseline={profile?.home_location} current={prediction?.location} />
      </div>
      </div>

      <div className="mt-4">
        <div className="flex items-baseline justify-between mb-2">
          <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>Overall behavioural match</span>
          <span className="mono text-[11px]" style={{ color: 'var(--text-muted)' }}>
            {profile?.n_transactions != null ? `baseline: ${profile.n_transactions.toLocaleString()} transactions` : ''}
          </span>
        </div>
        <div className="flex h-3 w-full" style={{ gap: 2 }} role="img"
          aria-label={hasSplit ? `Similarity ${similarity.toFixed(1)} percent, deviation ${deviation.toFixed(1)} percent` : insufficient ? 'Not enough history for a behavioural baseline' : 'No scan yet'}>
          {hasSplit ? (
            <>
              <div style={{ width: `${similarity}%`, background: 'var(--brand)', borderRadius: '4px 0 0 4px', transition: 'width 0.5s ease', minWidth: similarity > 0 ? 3 : 0 }} />
              <div style={{ width: `${deviation}%`, background: 'var(--risk-high)', borderRadius: '0 4px 4px 0', transition: 'width 0.5s ease', minWidth: deviation > 0 ? 3 : 0 }} />
            </>
          ) : (
            <div className="w-full rounded" style={{ background: 'var(--border)' }} />
          )}
        </div>
        <div className="flex justify-between mt-2 text-xs">
          <span className="flex items-center gap-2" style={{ color: 'var(--text-secondary)' }}>
            <span className="rounded-sm" style={{ width: 9, height: 9, background: 'var(--brand)' }} />
            Similar to baseline
            <span className="mono tnum" style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{hasSplit ? `${similarity.toFixed(1)}%` : '—'}</span>
          </span>
          <span className="flex items-center gap-2" style={{ color: 'var(--text-secondary)' }}>
            <span className="mono tnum" style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{hasSplit ? `${deviation.toFixed(1)}%` : '—'}</span>
            Deviation
            <span className="rounded-sm" style={{ width: 9, height: 9, background: 'var(--risk-high)' }} />
          </span>
        </div>
      </div>
      {insufficient && (
        <p className="text-[11px] mt-3" data-testid="behavioral-insufficient" style={{ color: 'var(--text-muted)' }}>
          Not enough history for a behavioural match: this customer has {prediction.history_transactions ?? 0} earlier
          transaction{prediction.history_transactions === 1 ? '' : 's'} and at least 10 are needed. No score is shown rather than a misleading one.
        </p>
      )}
      {!prediction && (
        <p className="text-[11px] mt-3" style={{ color: 'var(--text-muted)' }}>Scan a transaction to compare it with this baseline.</p>
      )}
    </div>
  )
}
