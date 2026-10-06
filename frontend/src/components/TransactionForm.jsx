import { useState } from 'react'
import { ErrorState, Icon, Spinner } from './ui'

const MERCHANT_CATEGORIES = [
  'grocery', 'electronics', 'travel', 'dining', 'utilities',
  'entertainment', 'fashion', 'healthcare', 'fuel', 'online_retail',
]

const QUICK_SCENARIOS = {
  normal: {
    label: 'Typical Purchase',
    icon: 'cart',
    hint: 'Small grocery purchase on the usual device, from the home city',
    values: { amount: 1500, merchant_category: 'grocery', device_id: '', location: '', failed_logins_24h: 0 },
  },
  suspicious: {
    label: 'Suspicious Pattern',
    icon: 'bolt',
    hint: 'Large electronics purchase on an unknown device, from Lagos, after failed logins',
    values: { amount: 75000, merchant_category: 'electronics', device_id: 'DEV_UNKNOWN_9999', location: 'Lagos', failed_logins_24h: 4 },
  },
}

// `profile` is the selected customer's { home_device, home_location } from
// GET /customer/{id}/profile. Leaving Device ID / Location blank uses those,
// so a normal transaction matches the customer's real usual device and city;
// typing a different value (e.g. the suspicious scenario) overrides them.
export default function TransactionForm({ customerId, profile, onSubmit, loading }) {
  const [form, setForm] = useState({
    amount: 1500,
    merchant_category: 'grocery',
    device_id: '',
    location: '',
    failed_logins_24h: 0,
  })
  const [formError, setFormError] = useState('')
  const [scenario, setScenario] = useState('normal')

  const update = (field, value) => {
    setScenario('')
    setForm((f) => ({ ...f, [field]: value }))
  }

  const applyScenario = (key) => {
    setScenario(key)
    setForm((f) => ({ ...f, ...QUICK_SCENARIOS[key].values }))
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    if (!customerId) return
    const deviceId = form.device_id.trim() || profile?.home_device || ''
    const location = form.location.trim() || profile?.home_location || ''
    if (!deviceId || !location) {
      setFormError("This customer's usual device and city aren't loaded yet. Enter a Device ID and Location.")
      return
    }
    setFormError('')
    onSubmit({
      customer_id: customerId,
      amount: Number(form.amount),
      merchant_category: form.merchant_category,
      device_id: deviceId,
      location,
      failed_logins_24h: Number(form.failed_logins_24h),
    })
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4" noValidate={false}>
      <div>
        <span className="field-label">Quick scenario</span>
        <div className="flex flex-wrap gap-2">
          {Object.entries(QUICK_SCENARIOS).map(([key, s]) => {
            const active = scenario === key
            return (
              <button
                key={key} type="button" onClick={() => applyScenario(key)} title={s.hint} aria-pressed={active}
                className="btn btn-sm"
                style={{
                  justifyContent: 'flex-start', height: 36, padding: '0 10px', gap: 6, fontSize: 12, whiteSpace: 'nowrap', flex: '1 1 auto', minWidth: 'max-content',
                  color: active ? 'var(--text-primary)' : 'var(--text-secondary)',
                  background: active ? 'var(--brand-dim)' : 'var(--surface-sunken)',
                  borderColor: active ? 'rgba(56,189,248,0.5)' : 'var(--border)',
                }}
              >
                <Icon name={s.icon} size={14} style={{ color: active ? 'var(--brand)' : 'var(--text-muted)' }} />
                {s.label}
              </button>
            )
          })}
        </div>
      </div>

      <div className="divider" />

      <div className="grid grid-cols-2 gap-3">
        <div>
          <label htmlFor="tx-amount" className="field-label">Amount (₹)</label>
          <input
            id="tx-amount" type="number" min="1" step="0.01" required inputMode="decimal"
            className="input mono" value={form.amount}
            onChange={(e) => update('amount', e.target.value)}
          />
        </div>
        <div>
          <label htmlFor="tx-category" className="field-label">Merchant Category</label>
          <select
            id="tx-category" className="input" value={form.merchant_category}
            onChange={(e) => update('merchant_category', e.target.value)}
          >
            {MERCHANT_CATEGORIES.map((c) => (
              <option key={c} value={c}>{c.replace('_', ' ')}</option>
            ))}
          </select>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div>
          <label htmlFor="tx-device" className="field-label">Device ID</label>
          <input
            id="tx-device" type="text" placeholder={profile?.home_device || 'usual device'} autoComplete="off" spellCheck="false"
            className="input mono" value={form.device_id}
            onChange={(e) => update('device_id', e.target.value)}
          />
        </div>
        <div>
          <label htmlFor="tx-location" className="field-label">Location</label>
          <input
            id="tx-location" type="text" placeholder={profile?.home_location || 'home city'} autoComplete="off"
            className="input" value={form.location}
            onChange={(e) => update('location', e.target.value)}
          />
        </div>
      </div>
      {profile && (
        <p className="text-[11px] -mt-1.5 flex items-start gap-1.5" style={{ color: 'var(--text-muted)' }}>
          <Icon name="info" size={12} className="mt-0.5" />
          <span>
            Leave blank to use this customer&apos;s usual device (<span className="mono">{profile.home_device}</span>) and
            home city ({profile.home_location}).
          </span>
        </p>
      )}

      <div>
        <label htmlFor="tx-logins" className="field-label">Failed Logins (24h)</label>
        <input
          id="tx-logins" type="number" min="0" step="1" inputMode="numeric"
          className="input mono" value={form.failed_logins_24h}
          onChange={(e) => update('failed_logins_24h', e.target.value)}
        />
      </div>

      <button type="submit" disabled={!customerId || loading} className="btn btn-primary w-full" style={{ height: 42 }}>
        {loading ? <Spinner /> : <Icon name="scan" size={16} />}
        {loading ? 'Scanning…' : 'Scan Transaction'}
      </button>
      {!customerId && (
        <p className="text-[11px] text-center" style={{ color: 'var(--text-muted)' }}>Select a customer to enable scanning.</p>
      )}
      {formError && <ErrorState>{formError}</ErrorState>}
    </form>
  )
}
