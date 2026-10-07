import { useCallback, useEffect, useState } from 'react'
import { getFraudRings } from '../api'
import { plural } from '../format'

export default function FraudRings() {
  const [rings, setRings] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const load = useCallback((isActive = () => true) => {
    setLoading(true)
    setError('')
    getFraudRings()
      .then((data) => { if (isActive()) setRings(data.rings) })
      .catch(() => { if (isActive()) setError('Could not load fraud ring data. Is the backend running?') })
      .finally(() => { if (isActive()) setLoading(false) })
  }, [])

  // ignore the response if the tab was closed before it arrived (also keeps StrictMode's double mount tidy:
  // the two simultaneous calls share one request, see api.js)
  useEffect(() => {
    let active = true
    load(() => active)
    return () => { active = false }
  }, [load])

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <p className="text-xs max-w-2xl" style={{ color: 'var(--text-muted)' }}>
          Flags devices used by more than one distinct customer -- a single legitimate customer's own
          devices are namespaced to them, so a shared device across accounts is a strong signal of an
          organized ring rather than one customer acting oddly alone. Try it: run the &quot;Suspicious
          pattern&quot; scan on Live Scan for two different customers, then come back here.
        </p>
        <button
          type="button"
          onClick={() => load()}
          disabled={loading}
          className="text-xs px-3 py-1.5 rounded-md border shrink-0 transition-colors hover:border-[var(--brand)] disabled:opacity-40"
          style={{ borderColor: 'var(--border)', color: 'var(--text-muted)' }}
        >
          Refresh
        </button>
      </div>

      {loading && <p role="status" className="text-sm" style={{ color: 'var(--text-muted)' }}>Scanning customer histories…</p>}
      {error && <p role="alert" className="text-sm" style={{ color: 'var(--risk-critical)' }}>{error}</p>}

      {!loading && !error && rings?.length === 0 && (
        <p className="text-sm" style={{ color: 'var(--text-faint)' }}>No shared-device rings detected.</p>
      )}

      {!loading && !error && rings?.length > 0 && (
        <ul className="space-y-2">
          {rings.map((ring) => (
            <li
              key={ring.identifier}
              className="rounded-lg border p-4 flex items-center justify-between gap-4"
              style={{ borderColor: 'var(--risk-critical)', background: 'var(--risk-critical-dim)' }}
            >
              <div>
                <div className="text-sm" style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>
                  {ring.identifier}
                </div>
                <div className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
                  Shared across {plural(ring.customer_ids.length, 'customer')} &middot; {plural(ring.transaction_count, 'transaction')}
                </div>
                <div className="flex flex-wrap gap-1.5 mt-2">
                  {ring.customer_ids.map((cid) => (
                    <span
                      key={cid}
                      className="text-[10px] px-2 py-0.5 rounded"
                      style={{ fontFamily: 'var(--font-mono)', background: 'var(--surface)', color: 'var(--brand)' }}
                    >
                      {cid}
                    </span>
                  ))}
                </div>
              </div>
              <div
                className="text-xs px-3 py-1 rounded-full shrink-0"
                style={{ background: 'var(--risk-critical)', color: '#2a0a0a', fontWeight: 600 }}
              >
                Ring Detected
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
