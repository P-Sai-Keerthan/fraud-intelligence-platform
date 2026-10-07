import { useRef, useState } from 'react'
import { predictBatch, formatApiError } from '../api'

const SUMMARY_ORDER = [
  { key: 'Critical Risk', color: 'var(--risk-critical)', dim: 'var(--risk-critical-dim)' },
  { key: 'High Risk', color: 'var(--risk-high)', dim: 'var(--risk-high-dim)' },
  { key: 'Medium Risk', color: 'var(--risk-medium)', dim: 'var(--risk-medium-dim)' },
  { key: 'Low Risk', color: 'var(--risk-low)', dim: 'var(--risk-low-dim)' },
]

const SAMPLE_CSV = `customer_id,amount,merchant_category,device_id,location,failed_logins_24h
CUST_0001,1500,grocery,,,0
CUST_0002,92000,electronics,DEV_UNKNOWN_1234,Lagos,5
CUST_0003,2200,dining,,,0
`

export default function BatchScoring() {
  const [file, setFile] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const inputRef = useRef(null)

  const handleUpload = async () => {
    if (!file) return
    setLoading(true)
    setError('')
    setResult(null)
    try {
      const data = await predictBatch(file)
      setResult(data)
    } catch (err) {
      setError(formatApiError(err, 'Batch scoring failed. Is the backend running?'))
    } finally {
      setLoading(false)
    }
  }

  const downloadSample = () => {
    const blob = new Blob([SAMPLE_CSV], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = 'sample_transactions.csv'
    link.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="space-y-5">
      <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
        Upload a CSV of up to 50 transactions to score them all at once. Required columns: customer_id, amount,
        merchant_category. Optional: device_id, location, failed_logins_24h. A blank device or location uses that
        customer's own usual one. Every row is checked first: if any row is invalid, nothing is scored.{' '}
        <button onClick={downloadSample} className="underline" style={{ color: 'var(--brand)' }}>
          Download a sample CSV
        </button>
      </p>

      <div className="flex items-center gap-3">
        <input
          ref={inputRef}
          type="file"
          accept=".csv"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          className="text-xs"
          style={{ color: 'var(--text-muted)' }}
        />
        <button
          onClick={handleUpload}
          disabled={!file || loading}
          className="text-xs px-4 py-2 rounded-lg font-medium transition-opacity disabled:opacity-40 shrink-0"
          style={{ background: 'var(--brand)', color: '#04202a', fontFamily: 'var(--font-display)' }}
        >
          {loading ? 'Scoring…' : 'Upload & Score'}
        </button>
      </div>

      {error && (
        <ul className="text-xs space-y-1" style={{ color: 'var(--risk-critical)' }}>
          {error.split(' | ').map((line) => (
            <li key={line}>{line}</li>
          ))}
        </ul>
      )}

      {result && (
        <div className="space-y-4">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {SUMMARY_ORDER.map(({ key, color, dim }) => (
              <div key={key} className="rounded-lg border p-3 text-center" style={{ borderColor: color, background: dim }}>
                <div className="text-2xl" style={{ fontFamily: 'var(--font-mono)', color, fontWeight: 600 }}>
                  {result.summary[key] ?? 0}
                </div>
                <div className="text-[10px] uppercase tracking-wide mt-1" style={{ color: 'var(--text-muted)' }}>{key}</div>
              </div>
            ))}
          </div>

          <div className="max-h-80 overflow-y-auto rounded-lg border" style={{ borderColor: 'var(--border)' }}>
            <table className="w-full text-xs">
              <thead className="sticky top-0" style={{ background: 'var(--surface-raised)' }}>
                <tr style={{ color: 'var(--text-muted)' }}>
                  <th className="text-left px-3 py-2 font-normal">Customer</th>
                  <th className="text-right px-3 py-2 font-normal">Amount</th>
                  <th className="text-right px-3 py-2 font-normal">Risk Score</th>
                  <th className="text-right px-3 py-2 font-normal">Fraud Prob.</th>
                  <th className="text-left px-3 py-2 font-normal">Alert</th>
                </tr>
              </thead>
              <tbody>
                {result.results.map((r) => {
                  const level = SUMMARY_ORDER.find((s) => s.key === r.alert_level)
                  return (
                    <tr key={r.transaction_id} style={{ borderTop: '1px solid var(--border-subtle)' }}>
                      <td className="px-3 py-2" style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>{r.customer_id}</td>
                      <td className="px-3 py-2 text-right" style={{ color: 'var(--text-muted)' }}>₹{r.amount.toLocaleString()}</td>
                      <td className="px-3 py-2 text-right" style={{ color: 'var(--text-muted)' }}>{r.risk_score.toFixed(1)}</td>
                      <td className="px-3 py-2 text-right" style={{ color: 'var(--text-muted)' }}>{r.fraud_probability.toFixed(1)}%</td>
                      <td className="px-3 py-2" style={{ color: level?.color }}>{r.alert_level}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
