import { useState } from 'react'

const MERCHANT_CATEGORIES = [
  'grocery', 'electronics', 'travel', 'dining', 'utilities',
  'entertainment', 'fashion', 'healthcare', 'fuel', 'online_retail',
]

const QUICK_SCENARIOS = {
  normal: {
    label: 'Typical purchase',
    values: { amount: 1500, merchant_category: 'grocery', device_id: '', location: '', failed_logins_24h: 0 },
  },
  suspicious: {
    label: 'Suspicious pattern',
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

  const update = (field, value) => setForm((f) => ({ ...f, [field]: value }))

  const applyScenario = (key) => {
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

  const inputClass =
    'w-full rounded-lg px-3 py-2 text-sm outline-none border transition-colors focus:border-[var(--brand)]'
  const inputStyle = { background: 'var(--surface)', borderColor: 'var(--border)', color: 'var(--text-primary)' }
  const labelClass = 'text-xs uppercase tracking-wide mb-1.5 block'
  const labelStyle = { color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="flex gap-2 mb-2">
        {Object.entries(QUICK_SCENARIOS).map(([key, s]) => (
          <button
            key={key}
            type="button"
            onClick={() => applyScenario(key)}
            className="text-xs px-3 py-1.5 rounded-md border transition-colors hover:border-[var(--brand)]"
            style={{ borderColor: 'var(--border)', color: 'var(--text-muted)' }}
          >
            {s.label}
          </button>
        ))}
      </div>

      <div>
        <label className={labelClass} style={labelStyle}>Amount (₹)</label>
        <input
          type="number" min="1" step="0.01" required
          className={inputClass} style={inputStyle}
          value={form.amount}
          onChange={(e) => update('amount', e.target.value)}
        />
      </div>

      <div>
        <label className={labelClass} style={labelStyle}>Merchant Category</label>
        <select
          className={inputClass} style={inputStyle}
          value={form.merchant_category}
          onChange={(e) => update('merchant_category', e.target.value)}
        >
          {MERCHANT_CATEGORIES.map((c) => (
            <option key={c} value={c}>{c.replace('_', ' ')}</option>
          ))}
        </select>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className={labelClass} style={labelStyle}>Device ID</label>
          <input
            type="text" placeholder={profile?.home_device || ''}
            className={inputClass} style={inputStyle}
            value={form.device_id}
            onChange={(e) => update('device_id', e.target.value)}
          />
        </div>
        <div>
          <label className={labelClass} style={labelStyle}>Location</label>
          <input
            type="text" placeholder={profile?.home_location || ''}
            className={inputClass} style={inputStyle}
            value={form.location}
            onChange={(e) => update('location', e.target.value)}
          />
        </div>
      </div>
      {profile && (
        <p className="text-[10px] -mt-2" style={{ color: 'var(--text-faint)' }}>
          Leave blank to use this customer's usual device ({profile.home_device}) and home city ({profile.home_location}).
        </p>
      )}

      <div>
        <label className={labelClass} style={labelStyle}>Failed Logins (24h)</label>
        <input
          type="number" min="0"
          className={inputClass} style={inputStyle}
          value={form.failed_logins_24h}
          onChange={(e) => update('failed_logins_24h', e.target.value)}
        />
      </div>

      <button
        type="submit"
        disabled={!customerId || loading}
        className="w-full rounded-lg py-2.5 text-sm font-medium transition-opacity disabled:opacity-40"
        style={{ background: 'var(--brand)', color: '#04202a', fontFamily: 'var(--font-display)' }}
      >
        {loading ? 'Scanning…' : 'Scan Transaction'}
      </button>
      {formError && <p className="text-xs" style={{ color: 'var(--risk-critical)' }}>{formError}</p>}
    </form>
  )
}
