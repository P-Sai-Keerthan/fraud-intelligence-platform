import { useId, useMemo, useState } from 'react'
import { inputClass, inputStyle, labelClass, labelStyle } from '../ui'

// A native <select> with 500 entries is hard to use, so a plain text filter sits above it. It only narrows
// the list the <select> shows; the customer API and the customer ids are untouched.
export default function CustomerSelector({ customers, value, onChange, loading = false }) {
  const uid = useId()
  const [query, setQuery] = useState('')

  const matches = useMemo(() => {
    const q = query.trim().toLowerCase()
    return q ? customers.filter((c) => c.toLowerCase().includes(q)) : customers
  }, [customers, query])
  // the selected customer always stays in the list, even if the filter no longer matches it
  const options = value && !matches.includes(value) ? [value, ...matches] : matches

  let status = ''
  if (loading) status = 'Loading customers…'
  else if (customers.length === 0) status = 'No customers available.'
  else if (query.trim()) status = matches.length === 0 ? `No customers match "${query.trim()}".` : `${matches.length} of ${customers.length} customers match.`
  else status = `${customers.length} customers.`

  return (
    <div className="space-y-3">
      <div>
        <label htmlFor={`${uid}-filter`} className={labelClass} style={labelStyle}>Find Customer</label>
        <input
          id={`${uid}-filter`}
          type="search"
          autoComplete="off"
          placeholder="Type to filter, e.g. CUST_04"
          className={inputClass}
          style={{ ...inputStyle, fontFamily: 'var(--font-mono)' }}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          aria-describedby={`${uid}-status`}
          disabled={loading || customers.length === 0}
        />
        <p id={`${uid}-status`} role="status" className="text-[11px] mt-1.5" style={{ color: 'var(--text-faint)' }}>
          {status}
        </p>
      </div>

      <div>
        <label htmlFor={`${uid}-select`} className={labelClass} style={labelStyle}>Customer Profile</label>
        <select
          id={`${uid}-select`}
          className={inputClass}
          style={{ ...inputStyle, fontFamily: 'var(--font-mono)' }}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          disabled={loading || customers.length === 0}
        >
          <option value="">Select a customer…</option>
          {options.map((c) => (
            <option key={c} value={c}>{c}</option>
          ))}
        </select>
      </div>
    </div>
  )
}
