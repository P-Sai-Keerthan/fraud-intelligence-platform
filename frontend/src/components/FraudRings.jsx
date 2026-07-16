import { useEffect, useState } from 'react'
import { getFraudRings } from '../api'

export default function FraudRings() {
  const [rings, setRings] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const load = () => {
    setLoading(true)
    setError('')
    getFraudRings()
      .then((data) => setRings(data.rings))
      .catch(() => setError('Could not load fraud ring data. Is the backend running?'))
      .finally(() => setLoading(false))
  }

  useEffect(load, [])

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
          onClick={load}
          className="text-xs px-3 py-1.5 rounded-md border shrink-0 transition-colors hover:border-[var(--brand)]"
          style={{ borderColor: 'var(--border)', color: 'var(--text-muted)' }}
        >
          Refresh
        </button>
      </div>

      {loading && <p className="text-sm" style={{ color: 'var(--text-muted)' }}>Scanning customer histories…</p>}
      {error && <p className="text-sm" style={{ color: 'var(--risk-critical)' }}>{error}</p>}

      {!loading && !error && rings?.length === 0 && (
        <p className="text-sm" style={{ color: 'var(--text-faint)' }}>No shared-device rings detected.</p>
      )}

      {!loading && !error && rings?.length > 0 && (
        <div className="space-y-2">
          {rings.map((ring) => (
            <div
              key={ring.identifier}
              className="rounded-lg border p-4 flex items-center justify-between gap-4"
              style={{ borderColor: 'var(--risk-critical)', background: 'var(--risk-critical-dim)' }}
            >
              <div>
                <div className="text-sm" style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>
                  {ring.identifier}
                </div>
                <div className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
                  Shared across {ring.customer_ids.length} customers &middot; {ring.transaction_count} transactions
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
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
