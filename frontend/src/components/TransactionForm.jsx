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

export default function TransactionForm({ customerId, onSubmit, loading }) {
  const [form, setForm] = useState({
    amount: 1500,
    merchant_category: 'grocery',
    device_id: '',
    location: '',
    failed_logins_24h: 0,
  })

  const update = (field, value) => setForm((f) => ({ ...f, [field]: value }))

  const applyScenario = (key) => {
    setForm((f) => ({ ...f, ...QUICK_SCENARIOS[key].values }))
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    if (!customerId) return
    onSubmit({
      customer_id: customerId,
      amount: Number(form.amount),
      merchant_category: form.merchant_category,
      device_id: form.device_id || `DEV_${customerId}_A`,
      location: form.location || 'Hyderabad',
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
            type="text" placeholder={`DEV_${customerId || 'XXXX'}_A`}
            className={inputClass} style={inputStyle}
            value={form.device_id}
            onChange={(e) => update('device_id', e.target.value)}
          />
        </div>
        <div>
          <label className={labelClass} style={labelStyle}>Location</label>
          <input
            type="text" placeholder="Hyderabad"
            className={inputClass} style={inputStyle}
            value={form.location}
            onChange={(e) => update('location', e.target.value)}
          />
        </div>
      </div>

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
    </form>
  )
}
