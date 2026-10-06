import { Icon } from './ui'

const STATUS = {
  online: { label: 'ONLINE', color: 'var(--risk-low)', live: true, title: 'Backend reachable (GET /health)' },
  offline: { label: 'OFFLINE', color: 'var(--risk-critical)', live: false, title: 'Backend not reachable (GET /health failed)' },
  checking: { label: 'CHECKING', color: 'var(--text-muted)', live: false, title: 'Checking the backend…' },
}

function Field({ label, children, title }) {
  return (
    <div className="flex flex-col gap-0.5 min-w-0" title={title}>
      <span className="eyebrow" style={{ fontSize: 9.5 }}>{label}</span>
      <span className="mono text-xs truncate" style={{ color: 'var(--text-primary)' }}>{children}</span>
    </div>
  )
}

export default function Header({ health = 'checking', modelInfo }) {
  const status = STATUS[health] || STATUS.checking
  const modelSet = modelInfo?.model_set
  const isProduction = modelSet === 'production'

  return (
    <header
      className="sticky top-0 z-30 border-b"
      style={{ borderColor: 'var(--border)', background: 'rgba(7, 11, 20, 0.86)', backdropFilter: 'blur(12px)' }}
    >
      <div className="max-w-[1480px] mx-auto px-5 lg:px-8 py-3.5 flex flex-wrap items-center justify-between gap-x-8 gap-y-3">
        <div className="flex items-center gap-3.5 min-w-0">
          <span
            className="grid place-items-center w-10 h-10 rounded-xl shrink-0"
            style={{
              background: 'linear-gradient(160deg, rgba(56,189,248,0.22), rgba(99,102,241,0.12))',
              border: '1px solid rgba(56,189,248,0.35)', color: 'var(--brand)',
            }}
          >
            <Icon name="shield" size={21} />
          </span>
          <div className="min-w-0">
            <h1 className="display text-[19px] leading-tight tracking-tight" style={{ fontWeight: 600 }}>
              Fraud Intelligence Platform
            </h1>
            <p className="text-xs mt-0.5 truncate" style={{ color: 'var(--text-muted)' }}>
              Behavioral Fraud DNA &middot; Explainable Real-Time Detection
            </p>
          </div>
        </div>

        <div className="flex items-center gap-5 flex-wrap">
          <div className="flex items-center gap-2" title={status.title} role="status" aria-live="polite">
            <span
              className={`rounded-full ${status.live ? 'status-dot-live' : ''}`}
              style={{ width: 8, height: 8, background: status.color }}
            />
            <div className="flex flex-col gap-0.5">
              <span className="eyebrow" style={{ fontSize: 9.5 }}>System</span>
              <span className="mono text-xs" style={{ color: 'var(--text-primary)' }}>{status.label}</span>
            </div>
          </div>

          <span className="hidden sm:block w-px h-8" style={{ background: 'var(--border)' }} />

          <Field label="Model set" title="The model set this backend loaded at startup (GET /model-info)">
            {modelSet || '—'}
          </Field>
          <Field label="Model version" title={modelInfo?.model_version || ''}>
            {modelInfo?.model_version || '—'}
          </Field>

          {modelSet && (
            <span
              className="chip"
              title={isProduction ? 'The production model set is loaded' : 'An evaluation candidate is loaded for controlled testing. It is not deployed; production is the default model set.'}
              style={isProduction
                ? { color: 'var(--risk-low)', borderColor: 'rgba(47,191,143,0.35)', background: 'var(--risk-low-dim)' }
                : { color: 'var(--risk-medium)', borderColor: 'rgba(242,180,31,0.35)', background: 'var(--risk-medium-dim)' }}
            >
              {isProduction ? 'PRODUCTION' : 'EVALUATION · NOT DEPLOYED'}
            </span>
          )}
        </div>
      </div>
    </header>
  )
}
