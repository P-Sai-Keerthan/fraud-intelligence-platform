import { Icon, SeverityBadge, Skeleton } from './ui'
import { formatDateTime } from '../severity'

function Fact({ icon, label, children }) {
  return (
    <div className="flex items-center gap-2.5 py-2" style={{ borderBottom: '1px solid var(--border-subtle)' }}>
      <Icon name={icon} size={14} style={{ color: 'var(--text-faint)' }} />
      <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{label}</span>
      <span className="mono text-xs ml-auto text-right truncate" style={{ color: 'var(--text-primary)', maxWidth: '60%' }}>{children}</span>
    </div>
  )
}

// Customer picker plus what the backend knows about the selected customer:
// GET /customer/{id}/profile (usual device, home city, history size) and the
// newest scored transaction from GET /customer/{id}/history.
export default function CustomerSelector({ customers, value, onChange, profile, profileLoading, timeline, loadingCustomers }) {
  const latest = timeline && timeline.length > 0
    ? [...timeline].sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))[0]
    : null

  return (
    <div>
      <label htmlFor="customer-select" className="field-label">Customer</label>
      {loadingCustomers ? (
        <Skeleton style={{ height: 38 }} />
      ) : (
        <select id="customer-select" className="input mono" value={value} onChange={(e) => onChange(e.target.value)}>
          <option value="">Select a customer…</option>
          {customers.map((c) => (
            <option key={c} value={c}>{c}</option>
          ))}
        </select>
      )}

      {value && (
        <div className="mt-4 fade-up">
          <div className="flex items-center gap-3 mb-2">
            <span className="grid place-items-center w-10 h-10 rounded-full shrink-0"
              style={{ background: 'var(--brand-dim)', border: '1px solid rgba(56,189,248,0.3)', color: 'var(--brand)' }}>
              <Icon name="user" size={18} />
            </span>
            <div className="min-w-0">
              <div className="mono text-sm truncate" style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{value}</div>
              <div className="text-[11px]" style={{ color: 'var(--text-muted)' }}>Profile derived from transaction history</div>
            </div>
          </div>

          {profileLoading && !profile ? (
            <div className="space-y-2 mt-3">
              <Skeleton style={{ height: 14 }} /><Skeleton style={{ height: 14, width: '80%' }} /><Skeleton style={{ height: 14, width: '65%' }} />
            </div>
          ) : (
            <dl>
              <Fact icon="device" label="Usual device">{profile?.home_device || '—'}</Fact>
              <Fact icon="pin" label="Home city">{profile?.home_location || '—'}</Fact>
              <Fact icon="layers" label="Transactions in history">{profile?.n_transactions != null ? profile.n_transactions.toLocaleString() : '—'}</Fact>
              <Fact icon="scan" label="Scans recorded">{timeline ? timeline.length.toLocaleString() : '—'}</Fact>
            </dl>
          )}

          <div className="flex items-center justify-between gap-3 mt-3">
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Latest scan</span>
            {latest ? (
              <span className="flex items-center gap-2 min-w-0">
                <span className="mono text-[10.5px] truncate" style={{ color: 'var(--text-faint)' }}>{formatDateTime(latest.timestamp)}</span>
                <SeverityBadge level={latest.alert_level} />
              </span>
            ) : (
              <span className="text-xs" style={{ color: 'var(--text-faint)' }}>No scans yet</span>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
