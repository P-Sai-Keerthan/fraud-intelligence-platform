import { useMemo, useRef, useState } from 'react'
import { predictBatch } from '../api'
import { SEVERITY_ORDER, severityOf } from '../severity'
import { Badge, Card, EmptyState, ErrorState, Icon, SeverityBadge, Spinner, StatTile } from './ui'
import { formatApiError } from '../apiError'

const REQUIRED_COLUMNS = ['customer_id', 'amount', 'merchant_category']
const OPTIONAL_COLUMNS = ['device_id', 'location', 'failed_logins_24h']

const SAMPLE_CSV = `customer_id,amount,merchant_category,device_id,location,failed_logins_24h
CUST_0001,1500,grocery,,,0
CUST_0002,92000,electronics,DEV_UNKNOWN_1234,Lagos,5
CUST_0003,2200,dining,,,0
`

function download(name, text, type) {
  const blob = new Blob([text], { type })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = name
  document.body.appendChild(link)
  link.click()
  link.remove()
  URL.revokeObjectURL(url)
}

function formatBytes(n) {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / 1024 / 1024).toFixed(1)} MB`
}

// Reads the header row of the chosen file in the browser, so a missing column
// is reported before upload. The backend validates again and is the authority.
async function inspect(file) {
  if (!/\.csv$/i.test(file.name)) {
    return { ok: false, message: 'This is not a .csv file.' }
  }
  const text = await file.text()
  const lines = text.split(/\r?\n/).filter((l) => l.trim() !== '')
  if (lines.length === 0) return { ok: false, message: 'The file is empty.' }
  const header = lines[0].replace(/^﻿/, '').split(',').map((c) => c.trim().replace(/^"|"$/g, ''))
  const missing = REQUIRED_COLUMNS.filter((c) => !header.includes(c))
  if (missing.length) return { ok: false, message: `Missing required column${missing.length > 1 ? 's' : ''}: ${missing.join(', ')}`, header }
  if (lines.length < 2) return { ok: false, message: 'The file has a header but no transactions.', header }
  return { ok: true, rows: lines.length - 1, header, optional: OPTIONAL_COLUMNS.filter((c) => header.includes(c)) }
}

function Step({ state, label, detail }) {
  const map = {
    done: { color: 'var(--risk-low)', icon: 'check' },
    error: { color: 'var(--risk-critical)', icon: 'x' },
    active: { color: 'var(--brand)', icon: null },
    idle: { color: 'var(--text-faint)', icon: null },
  }
  const s = map[state]
  return (
    <div className="flex items-start gap-3 py-2.5" style={{ borderBottom: '1px solid var(--border-subtle)' }}>
      <span className="grid place-items-center rounded-full shrink-0 mt-0.5"
        style={{ width: 20, height: 20, border: `1.5px solid ${s.color}`, color: s.color }}>
        {state === 'active' ? <Spinner size={11} /> : s.icon ? <Icon name={s.icon} size={11} /> : <span className="rounded-full" style={{ width: 5, height: 5, background: s.color }} />}
      </span>
      <div className="min-w-0">
        <div className="text-xs" style={{ color: state === 'idle' ? 'var(--text-muted)' : 'var(--text-primary)', fontWeight: 500 }}>{label}</div>
        {detail && <div className="text-[11px] mt-0.5 break-words" style={{ color: 'var(--text-muted)' }}>{detail}</div>}
      </div>
    </div>
  )
}

function Distribution({ summary, total }) {
  const parts = SEVERITY_ORDER.map((level) => ({ level, n: summary[level] ?? 0, s: severityOf(level) })).filter((p) => p.n > 0)
  return (
    <div>
      <div className="flex w-full h-3.5" style={{ gap: 2 }} role="img"
        aria-label={SEVERITY_ORDER.map((l) => `${l}: ${summary[l] ?? 0}`).join(', ')}>
        {parts.map((p, i) => (
          <div key={p.level} title={`${p.level}: ${p.n} (${((p.n / total) * 100).toFixed(1)}%)`}
            style={{
              width: `${(p.n / total) * 100}%`, minWidth: 4, background: p.s.color,
              borderRadius: `${i === 0 ? 4 : 0}px ${i === parts.length - 1 ? 4 : 0}px ${i === parts.length - 1 ? 4 : 0}px ${i === 0 ? 4 : 0}px`,
            }} />
        ))}
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-x-4 gap-y-2 mt-3">
        {SEVERITY_ORDER.map((level) => {
          const n = summary[level] ?? 0
          return (
            <div key={level} className="flex items-center gap-2 text-xs">
              <span className="rounded-sm shrink-0" style={{ width: 9, height: 9, background: severityOf(level).color }} />
              <span style={{ color: 'var(--text-secondary)' }}>{level}</span>
              <span className="mono tnum ml-auto" style={{ color: 'var(--text-primary)' }}>{n.toLocaleString()}</span>
              <span className="mono tnum w-12 text-right" style={{ color: 'var(--text-muted)' }}>{total ? `${((n / total) * 100).toFixed(1)}%` : '—'}</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default function BatchScoring({ riskColumn = 'Risk Score' }) {
  const [file, setFile] = useState(null)
  const [check, setCheck] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [dragging, setDragging] = useState(false)
  const [filter, setFilter] = useState('all')
  const inputRef = useRef(null)

  const choose = async (f) => {
    setResult(null)
    setError('')
    setFilter('all')
    setFile(f ?? null)
    setCheck(null)
    if (f) {
      try {
        setCheck(await inspect(f))
      } catch {
        setCheck({ ok: false, message: 'The file could not be read.' })
      }
    }
  }

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

  const onDrop = (e) => {
    e.preventDefault()
    setDragging(false)
    const f = e.dataTransfer.files?.[0]
    if (f) choose(f)
  }

  const exportResults = () => {
    if (!result) return
    const header = ['transaction_id', 'customer_id', 'amount', 'risk_score', 'fraud_score', 'alert_level']
    const rows = result.results.map((r) => [r.transaction_id, r.customer_id, r.amount, r.risk_score, r.fraud_probability, r.alert_level].join(','))
    download('batch_scoring_results.csv', [header.join(','), ...rows].join('\n') + '\n', 'text/csv')
  }

  const summary = result?.summary || {}
  const total = result?.count ?? 0
  const highCritical = (summary['Critical Risk'] ?? 0) + (summary['High Risk'] ?? 0)
  const visible = useMemo(() => {
    if (!result) return []
    return filter === 'all' ? result.results : result.results.filter((r) => r.alert_level === filter)
  }, [result, filter])

  const validation = !file ? 'idle' : !check ? 'active' : check.ok ? 'done' : 'error'
  const processing = loading ? 'active' : error ? 'error' : result ? 'done' : 'idle'

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)] gap-5">
        <Card title="Upload transactions" eyebrow="Step 1" icon="upload"
          actions={<button type="button" onClick={() => download('sample_transactions.csv', SAMPLE_CSV, 'text/csv')} className="btn btn-ghost btn-sm"><Icon name="download" size={13} />Sample CSV</button>}>
          <div
            role="button" tabIndex={0} aria-label="Choose a CSV file to score"
            onClick={() => inputRef.current?.click()}
            onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); inputRef.current?.click() } }}
            onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            className="rounded-xl px-5 py-8 flex flex-col items-center text-center cursor-pointer transition-colors"
            style={{
              border: `1.5px dashed ${dragging ? 'var(--brand)' : 'var(--border-strong)'}`,
              background: dragging ? 'var(--brand-dim)' : 'var(--surface-sunken)',
            }}
          >
            <span className="grid place-items-center w-11 h-11 rounded-xl mb-3"
              style={{ background: 'var(--surface-raised)', border: '1px solid var(--border)', color: file ? 'var(--brand)' : 'var(--text-muted)' }}>
              <Icon name={file ? 'file' : 'upload'} size={20} />
            </span>
            {file ? (
              <>
                <p className="mono text-sm break-all" style={{ color: 'var(--text-primary)' }}>{file.name}</p>
                <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>{formatBytes(file.size)} &middot; click or drop to replace</p>
              </>
            ) : (
              <>
                <p className="text-sm" style={{ color: 'var(--text-primary)', fontWeight: 500 }}>Drop a CSV here, or click to browse</p>
                <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>One transaction per row</p>
              </>
            )}
            <input ref={inputRef} type="file" accept=".csv,text/csv" className="hidden"
              onChange={(e) => { choose(e.target.files?.[0]); e.target.value = '' }} />
          </div>

          <div className="flex flex-wrap items-center gap-x-2 gap-y-1.5 mt-4">
            <span className="text-[11px]" style={{ color: 'var(--text-muted)' }}>Required</span>
            {REQUIRED_COLUMNS.map((c) => <span key={c} className="chip">{c}</span>)}
            <span className="text-[11px] ml-2" style={{ color: 'var(--text-muted)' }}>Optional</span>
            {OPTIONAL_COLUMNS.map((c) => <span key={c} className="chip" style={{ color: 'var(--text-muted)' }}>{c}</span>)}
          </div>
          <p className="text-[11px] mt-2" style={{ color: 'var(--text-muted)' }}>
            A blank device_id or location uses the customer&apos;s usual device and home city.
          </p>

          <div className="flex items-center gap-3 mt-4">
            <button type="button" onClick={handleUpload} disabled={!file || loading} className="btn btn-primary">
              {loading ? <Spinner /> : <Icon name="scan" size={15} />}
              {loading ? 'Scoring…' : 'Upload & Score'}
            </button>
            {file && !loading && (
              <button type="button" onClick={() => choose(null)} className="btn btn-ghost">Clear</button>
            )}
          </div>
        </Card>

        <Card title="Pipeline status" eyebrow="Step 2" icon="activity">
          <Step state={file ? 'done' : 'idle'} label="File selected" detail={file ? `${file.name} · ${formatBytes(file.size)}` : 'No file chosen yet'} />
          <Step state={validation} label="Format check"
            detail={!file ? 'Header row is checked in the browser before upload'
              : !check ? 'Reading header…'
                : check.ok ? `${check.rows.toLocaleString()} row${check.rows === 1 ? '' : 's'} · required columns present${check.optional.length ? ` · optional: ${check.optional.join(', ')}` : ''}`
                  : `${check.message} The backend will make the final check.`} />
          <Step state={processing} label="Scoring"
            detail={loading ? 'The backend is scoring every row…' : error ? error : result ? `${total.toLocaleString()} transaction${total === 1 ? '' : 's'} scored` : 'Not started'} />
        </Card>
      </div>

      {error && <ErrorState>{error}</ErrorState>}

      {!result && !loading && !error && (
        <Card>
          <EmptyState icon="layers" title="No batch scored yet">
            Upload a CSV to see the alert distribution, the transactions that need attention and a downloadable result file.
          </EmptyState>
        </Card>
      )}

      {loading && (
        <Card>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            {[0, 1, 2, 3].map((i) => <div key={i} className="skeleton" style={{ height: 84 }} />)}
          </div>
          <div className="skeleton mt-4" style={{ height: 180 }} />
        </Card>
      )}

      {result && (
        <div className="space-y-5 fade-up">
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            <StatTile label="Transactions scored" value={total.toLocaleString()} hint="Rows returned by the backend" />
            <StatTile label="High / Critical" value={highCritical.toLocaleString()} accent="var(--risk-critical)"
              hint={total ? `${((highCritical / total) * 100).toFixed(1)}% of the batch · likely fraud` : ''} />
            <StatTile label="Medium" value={(summary['Medium Risk'] ?? 0).toLocaleString()} accent="var(--risk-medium)" hint="Needs review" />
            <StatTile label="Low" value={(summary['Low Risk'] ?? 0).toLocaleString()} accent="var(--risk-low)" hint="No alert · likely legitimate" />
          </div>

          <Card title="Alert distribution" eyebrow="Summary" icon="layers"
            actions={<Badge title="Counts are by alert level. They are not confirmed fraud labels.">By alert level</Badge>}>
            <Distribution summary={summary} total={total} />
            <p className="text-[11px] mt-3" style={{ color: 'var(--text-muted)' }}>
              &ldquo;Likely fraud&rdquo; and &ldquo;likely legitimate&rdquo; are read from the alert level of each transaction. They are not confirmed outcomes.
            </p>
          </Card>

          <Card title="Scored transactions" eyebrow="Results" icon="file"
            actions={<button type="button" onClick={exportResults} className="btn btn-secondary btn-sm"><Icon name="download" size={13} />Download results</button>}>
            <div className="flex flex-wrap items-center gap-2 mb-3" role="group" aria-label="Filter by alert level">
              {['all', ...SEVERITY_ORDER].map((key) => {
                const active = filter === key
                const n = key === 'all' ? total : (summary[key] ?? 0)
                return (
                  <button key={key} type="button" onClick={() => setFilter(key)} aria-pressed={active}
                    className="btn btn-sm" style={{
                      height: 28, fontFamily: 'var(--font-body)', fontWeight: 500,
                      color: active ? 'var(--text-primary)' : 'var(--text-muted)',
                      background: active ? 'var(--brand-dim)' : 'transparent',
                      borderColor: active ? 'rgba(56,189,248,0.45)' : 'var(--border)',
                    }}>
                    {key !== 'all' && <span className="rounded-full" style={{ width: 7, height: 7, background: severityOf(key).color }} />}
                    {key === 'all' ? 'All' : key}
                    <span className="mono tnum" style={{ color: 'var(--text-faint)' }}>{n}</span>
                  </button>
                )
              })}
            </div>
            <div className="max-h-[420px] overflow-auto rounded-lg border" style={{ borderColor: 'var(--border)' }}>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Transaction ID</th>
                    <th>Customer</th>
                    <th className="num" style={{ textAlign: 'right' }}>Amount</th>
                    <th className="num" style={{ textAlign: 'right' }}>{riskColumn}</th>
                    <th className="num" style={{ textAlign: 'right' }}>Fraud Score</th>
                    <th>Alert</th>
                  </tr>
                </thead>
                <tbody>
                  {visible.map((r) => (
                    <tr key={r.transaction_id}>
                      <td className="mono" style={{ color: 'var(--text-muted)' }}>{r.transaction_id}</td>
                      <td className="mono" style={{ color: 'var(--text-primary)' }}>{r.customer_id}</td>
                      <td className="num">₹{r.amount.toLocaleString()}</td>
                      <td className="num mono">{r.risk_score.toFixed(1)}</td>
                      <td className="num mono" style={{ color: 'var(--text-primary)' }}>{r.fraud_probability.toFixed(1)}</td>
                      <td><SeverityBadge level={r.alert_level} /></td>
                    </tr>
                  ))}
                  {visible.length === 0 && (
                    <tr><td colSpan={6} className="text-center" style={{ padding: '28px 14px', color: 'var(--text-muted)' }}>No transactions at this alert level.</td></tr>
                  )}
                </tbody>
              </table>
            </div>
            <p className="text-[11px] mt-2" style={{ color: 'var(--text-muted)' }}>
              Scores are on a 0–100 scale. The fraud score is a model score, not a calibrated probability.
            </p>
          </Card>
        </div>
      )}
    </div>
  )
}
