import { useEffect, useRef, useState } from 'react'

const MERCHANT_CATEGORIES = [
  'grocery', 'electronics', 'travel', 'dining', 'utilities',
  'entertainment', 'fashion', 'healthcare', 'fuel', 'online_retail',
]

// The "Suspicious pattern" preset is unchanged and sent at the current time.
// The "Typical purchase" preset is NOT hard-coded: it is built from the selected
// customer's own history (see GET /customer/{id}/profile -> typical_scenario).
const SUSPICIOUS_VALUES = {
  amount: 75000, merchant_category: 'electronics', device_id: 'DEV_UNKNOWN_9999', location: 'Lagos', failed_logins_24h: 4,
}

const EMPTY_FORM = {
  amount: '', merchant_category: 'grocery', device_id: '', location: '', failed_logins_24h: 0,
  timestamp: '', // '' = "now"; otherwise an ISO time (the customer's usual hour)
}

function formatTime(iso) {
  const d = new Date(iso)
  return Number.isNaN(d.getTime())
    ? iso
    : d.toLocaleString([], { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' })
}

function typicalValues(profile) {
  const s = profile?.typical_scenario
  if (!s) return null
  return {
    amount: s.amount, merchant_category: s.merchant_category, device_id: s.device_id,
    location: s.location, failed_logins_24h: s.failed_logins_24h, timestamp: s.timestamp || '',
  }
}

export default function TransactionForm({ customerId, profile, onSubmit, loading }) {
  const [form, setForm] = useState(EMPTY_FORM)
  const [timeIsTypical, setTimeIsTypical] = useState(false)
  const [localError, setLocalError] = useState('')
  const prefilledFor = useRef('')

  const typical = typicalValues(profile)

  // When a customer's profile first loads, start from that customer's own typical
  // transaction (so a plain "Scan" is a normal one). Later profile refreshes (after
  // each scan) must not overwrite what the user typed.
  useEffect(() => {
    if (!profile || profile.customer_id !== customerId || prefilledFor.current === customerId) return
    prefilledFor.current = customerId
    const t = typicalValues(profile)
    if (t) {
      setForm({ ...EMPTY_FORM, ...t })
      setTimeIsTypical(Boolean(t.timestamp))
    } else {
      setForm(EMPTY_FORM)
      setTimeIsTypical(false)
    }
  }, [profile, customerId])

  // keep "customer's usual hour" scans in chronological order: after each scan the
  // profile is refreshed and the suggested time moves past the latest transaction
  useEffect(() => {
    const next = profile?.typical_scenario?.timestamp
    if (timeIsTypical && next) setForm((f) => (f.timestamp === next ? f : { ...f, timestamp: next }))
  }, [profile, timeIsTypical])

  const update = (field, value) => setForm((f) => ({ ...f, [field]: value }))

  const applyTypical = () => {
    if (!typical) return
    setForm({ ...EMPTY_FORM, ...typical })
    setTimeIsTypical(Boolean(typical.timestamp))
    setLocalError('')
  }

  const applySuspicious = () => {
    setForm((f) => ({ ...f, ...SUSPICIOUS_VALUES, timestamp: '' }))
    setTimeIsTypical(false)
    setLocalError('')
  }

  const useCurrentTime = () => {
    setTimeIsTypical(false)
    update('timestamp', '')
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    if (!customerId) return
    // blank device / location default to THIS customer's usual ones (from history)
    const device = form.device_id.trim() || profile?.primary_device || ''
    const location = form.location.trim() || profile?.primary_location || ''
    if (!device || !location) {
      setLocalError('Enter a device and a location (this customer has no history to default from).')
      return
    }
    setLocalError('')
    const payload = {
      customer_id: customerId,
      amount: Number(form.amount),
      merchant_category: form.merchant_category,
      device_id: device,
      location,
      failed_logins_24h: Number(form.failed_logins_24h),
    }
    if (form.timestamp) payload.timestamp = form.timestamp
    onSubmit(payload)
  }

  const inputClass =
    'w-full rounded-lg px-3 py-2 text-sm outline-none border transition-colors focus:border-[var(--brand)]'
  const inputStyle = { background: 'var(--surface)', borderColor: 'var(--border)', color: 'var(--text-primary)' }
  const labelClass = 'text-xs uppercase tracking-wide mb-1.5 block'
  const labelStyle = { color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="flex gap-2 mb-2">
        <button
          type="button"
          onClick={applyTypical}
          disabled={!typical}
          title={typical ? "Fills the form from this customer's own history" : 'Not enough history for this customer'}
          className="text-xs px-3 py-1.5 rounded-md border transition-colors hover:border-[var(--brand)] disabled:opacity-40"
          style={{ borderColor: 'var(--border)', color: 'var(--text-muted)' }}
        >
          Typical purchase
        </button>
        <button
          type="button"
          onClick={applySuspicious}
          className="text-xs px-3 py-1.5 rounded-md border transition-colors hover:border-[var(--brand)]"
          style={{ borderColor: 'var(--border)', color: 'var(--text-muted)' }}
        >
          Suspicious pattern
        </button>
      </div>

      <div>
        <label className={labelClass} style={labelStyle}>Amount (₹)</label>
        <input
          type="number" min="1" max="1000000" step="0.01" required
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
            type="text" placeholder={profile?.primary_device || 'device id'}
            className={inputClass} style={inputStyle}
            value={form.device_id}
            onChange={(e) => update('device_id', e.target.value)}
          />
        </div>
        <div>
          <label className={labelClass} style={labelStyle}>Location</label>
          <input
            type="text" placeholder={profile?.primary_location || 'city'}
            className={inputClass} style={inputStyle}
            value={form.location}
            onChange={(e) => update('location', e.target.value)}
          />
        </div>
      </div>

      <div>
        <label className={labelClass} style={labelStyle}>Failed Logins (24h)</label>
        <input
          type="number" min="0" max="100"
          className={inputClass} style={inputStyle}
          value={form.failed_logins_24h}
          onChange={(e) => update('failed_logins_24h', e.target.value)}
        />
      </div>

      <p className="text-[11px]" style={{ color: 'var(--text-faint)' }}>
        Transaction time:{' '}
        {form.timestamp ? (
          <>
            <span style={{ color: 'var(--text-muted)' }}>{formatTime(form.timestamp)}</span>{' '}
            (this customer's usual hour){' '}
            <button type="button" onClick={useCurrentTime} className="underline" style={{ color: 'var(--brand)' }}>
              use current time
            </button>
          </>
        ) : (
          <span style={{ color: 'var(--text-muted)' }}>current time</span>
        )}
      </p>

      {localError && <p className="text-xs" style={{ color: 'var(--risk-critical)' }}>{localError}</p>}

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
