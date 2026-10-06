import { severityOf, bandText, formatDateTime } from '../severity'
import { Icon, Spinner } from './ui'

const VERDICT_TEXT = {
  'Critical Risk': 'Fraud score in the highest alert band.',
  'High Risk': 'Fraud score in the high alert band.',
  'Medium Risk': 'Fraud score in the medium alert band.',
  'Low Risk': 'Fraud score below every alert band.',
}

function Meta({ label, children }) {
  return (
    <div className="min-w-0">
      <div className="eyebrow" style={{ fontSize: 9.5 }}>{label}</div>
      <div className="mono text-xs mt-0.5 truncate" style={{ color: 'var(--text-primary)' }}>{children}</div>
    </div>
  )
}

// The transaction verdict: the alert level returned by POST /predict, with the
// transaction's identifiers. Strong red is reserved for Critical Risk.
export default function AlertBanner({ prediction, modelInfo, loading, onDownload, reportLoading }) {
  if (!prediction) {
    return (
      <div className="card px-5 py-5 flex items-center gap-4">
        <span className="grid place-items-center w-12 h-12 rounded-xl shrink-0"
          style={{ background: 'var(--surface-raised)', border: '1px solid var(--border)', color: 'var(--text-muted)' }}>
          {loading ? <Spinner size={18} /> : <Icon name="scan" size={22} />}
        </span>
        <div>
          <div className="eyebrow">Transaction verdict</div>
          <p className="display text-lg mt-0.5" style={{ fontWeight: 600, color: 'var(--text-secondary)' }}>
            {loading ? 'Scanning transaction…' : 'Awaiting scan'}
          </p>
          <p className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>
            Select a customer and scan a transaction to see its verdict, scores and explanation.
          </p>
        </div>
      </div>
    )
  }

  const level = prediction.alert_level
  const s = severityOf(level)
  const critical = s.key === 'critical'
  const band = bandText(modelInfo?.thresholds?.alert_bands?.[level])

  return (
    <div
      className="card fade-up overflow-hidden"
      style={{
        borderColor: `color-mix(in srgb, ${s.color} ${critical ? 60 : 38}%, var(--border))`,
        background: `linear-gradient(100deg, ${s.dim}, rgba(12,19,32,0.85) 55%)`,
      }}
      role="status" aria-live="polite"
    >
      <span className="absolute left-0 top-0 bottom-0" style={{ width: 4, background: s.color }} />
      <div className="pl-6 pr-5 py-4 flex flex-wrap items-center justify-between gap-x-6 gap-y-4">
        <div className="flex items-center gap-4 min-w-0">
          <span className="grid place-items-center w-12 h-12 rounded-xl shrink-0"
            style={{ background: s.dim, border: `1px solid color-mix(in srgb, ${s.color} 45%, transparent)`, color: s.color }}>
            <Icon name={s.key === 'low' ? 'shield' : 'alert'} size={24} />
          </span>
          <div className="min-w-0">
            <div className="eyebrow">Transaction verdict</div>
            <p className="display text-[26px] leading-tight" style={{ fontWeight: 700, color: critical ? s.color : 'var(--text-primary)' }}>
              {level}
            </p>
            <p className="text-xs mt-0.5" style={{ color: 'var(--text-secondary)' }}>
              {VERDICT_TEXT[level] || 'Alert level returned by the model.'}
              {band && <span className="mono" style={{ color: 'var(--text-muted)' }}> &middot; {band}</span>}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-6 flex-wrap">
          <Meta label="Transaction ID">{prediction.transaction_id}</Meta>
          <Meta label="Scanned">{formatDateTime(prediction.timestamp)}</Meta>
          <Meta label="Customer">{prediction.customer_id}</Meta>
          <button type="button" onClick={onDownload} disabled={reportLoading} className="btn btn-secondary"
            title="Download a PDF report for this transaction">
            {reportLoading ? <Spinner /> : <Icon name="download" size={15} />}
            {reportLoading ? 'Generating…' : 'Download Report'}
          </button>
        </div>
      </div>
    </div>
  )
}
