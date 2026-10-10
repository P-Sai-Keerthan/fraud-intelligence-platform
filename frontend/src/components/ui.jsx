import { severityOf } from '../severity'

/* ---------------------------------------------------------------- icons */
const PATHS = {
  shield: <><path d="M12 3 4.5 6v5.5c0 4.6 3.1 8 7.5 9.5 4.4-1.5 7.5-4.9 7.5-9.5V6L12 3Z" /><path d="m9 12 2.2 2.2L15.2 10" /></>,
  scan: <><path d="M4 8V6a2 2 0 0 1 2-2h2M16 4h2a2 2 0 0 1 2 2v2M20 16v2a2 2 0 0 1-2 2h-2M8 20H6a2 2 0 0 1-2-2v-2" /><path d="M4 12h16" /></>,
  layers: <><path d="m12 3 9 5-9 5-9-5 9-5Z" /><path d="m3 13 9 5 9-5" /></>,
  network: <><circle cx="12" cy="5" r="2.2" /><circle cx="5" cy="18" r="2.2" /><circle cx="19" cy="18" r="2.2" /><path d="M11 7 6.2 16M13 7l4.8 9M7.2 18h9.6" /></>,
  activity: <path d="M3 12h4l3-8 4 16 3-8h4" />,
  download: <><path d="M12 4v11M7.5 11 12 15.5 16.5 11" /><path d="M5 19h14" /></>,
  upload: <><path d="M12 16V5M7.5 9 12 4.5 16.5 9" /><path d="M5 19h14" /></>,
  file: <><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8l-5-5Z" /><path d="M14 3v5h5M9 13h6M9 17h4" /></>,
  refresh: <><path d="M20 11a8 8 0 0 0-14.5-4M4 13a8 8 0 0 0 14.5 4" /><path d="M5 3v4h4M19 21v-4h-4" /></>,
  alert: <><path d="M12 4 2.8 19.5h18.4L12 4Z" /><path d="M12 10v4.5M12 17.2v.3" /></>,
  check: <path d="m5 12.5 4.5 4.5L19 7.5" />,
  x: <path d="M6 6l12 12M18 6 6 18" />,
  info: <><circle cx="12" cy="12" r="9" /><path d="M12 11v5.5M12 7.7v.3" /></>,
  user: <><circle cx="12" cy="8" r="3.6" /><path d="M4.5 20c1-3.6 4-5.5 7.5-5.5s6.5 1.9 7.5 5.5" /></>,
  device: <><rect x="7" y="3" width="10" height="18" rx="2" /><path d="M11 17.5h2" /></>,
  pin: <><path d="M12 21s7-6.1 7-11.5A7 7 0 0 0 5 9.5C5 14.9 12 21 12 21Z" /><circle cx="12" cy="9.5" r="2.4" /></>,
  cpu: <><rect x="6" y="6" width="12" height="12" rx="2" /><path d="M9.5 9.5h5v5h-5zM9 3v3M15 3v3M9 18v3M15 18v3M3 9h3M3 15h3M18 9h3M18 15h3" /></>,
  bolt: <path d="M13 3 5 13.5h6L10 21l8.5-11H12.5L13 3Z" />,
  cart: <><path d="M4 5h2.2l2 10h9.3l1.8-7H7.2" /><circle cx="9.5" cy="19" r="1.3" /><circle cx="17" cy="19" r="1.3" /></>,
  clock: <><circle cx="12" cy="12" r="9" /><path d="M12 7v5.2l3.3 2" /></>,
  search: <><circle cx="11" cy="11" r="6.5" /><path d="m16 16 4.5 4.5" /></>,
}

export function Icon({ name, size = 16, className = '', style }) {
  return (
    <svg
      width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"
      className={`shrink-0 ${className}`} style={style} aria-hidden="true" focusable="false"
    >
      {PATHS[name]}
    </svg>
  )
}

/* ---------------------------------------------------------------- layout */
export function Card({ title, eyebrow, icon, actions, children, className = '', bodyClassName = '', as: Tag = 'section', ...rest }) {
  const hasHeader = title || eyebrow || actions
  return (
    <Tag className={`card ${className}`} {...rest}>
      {hasHeader && (
        <header className="flex flex-wrap items-start justify-between gap-x-3 gap-y-2 px-5 pt-4 pb-3">
          <div className="min-w-0 flex items-start gap-2.5">
            {icon && (
              <span className="mt-0.5 grid place-items-center w-7 h-7 rounded-lg shrink-0"
                style={{ background: 'var(--brand-dim)', color: 'var(--brand)' }}>
                <Icon name={icon} size={15} />
              </span>
            )}
            <div className="min-w-0">
              {eyebrow && <div className="eyebrow mb-0.5">{eyebrow}</div>}
              {title && (
                <h2 className="display text-[15px] leading-snug" style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{title}</h2>
              )}
            </div>
          </div>
          {actions && <div className="flex items-center gap-2 shrink-0">{actions}</div>}
        </header>
      )}
      <div className={`${hasHeader ? 'px-5 pb-5' : 'p-5'} ${bodyClassName}`}>{children}</div>
    </Tag>
  )
}

export function Badge({ children, tone = 'neutral', icon, className = '', title }) {
  const tones = {
    neutral: { color: 'var(--text-secondary)', border: 'var(--border)', bg: 'var(--surface-sunken)' },
    brand: { color: 'var(--brand)', border: 'rgba(56,189,248,0.35)', bg: 'var(--brand-dim)' },
    good: { color: 'var(--risk-low)', border: 'rgba(47,191,143,0.35)', bg: 'var(--risk-low-dim)' },
    warn: { color: 'var(--risk-medium)', border: 'rgba(242,180,31,0.35)', bg: 'var(--risk-medium-dim)' },
    danger: { color: 'var(--risk-critical)', border: 'rgba(239,77,90,0.4)', bg: 'var(--risk-critical-dim)' },
  }
  const t = tones[tone] || tones.neutral
  return (
    <span className={`chip ${className}`} title={title} style={{ color: t.color, borderColor: t.border, background: t.bg }}>
      {icon && <Icon name={icon} size={11} />}
      {children}
    </span>
  )
}

/* severity: a coloured dot + the level name in text ink, so colour is never the only cue */
export function SeverityBadge({ level, size = 'sm' }) {
  const s = severityOf(level)
  const big = size === 'md'
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-md border ${big ? 'px-2.5 h-7 text-xs' : 'px-2 h-[22px] text-[11px]'}`}
      style={{ borderColor: `color-mix(in srgb, ${s.color} 38%, transparent)`, background: s.dim, color: 'var(--text-primary)', fontWeight: 500, whiteSpace: 'nowrap' }}
    >
      <span className="rounded-full" style={{ width: 7, height: 7, background: s.color }} />
      {level || 'Unknown'}
    </span>
  )
}

export function Skeleton({ className = '', style }) {
  return <div className={`skeleton ${className}`} style={style} aria-hidden="true" />
}

export function EmptyState({ icon = 'info', title, children, compact = false }) {
  return (
    <div className={`flex flex-col items-center justify-center text-center ${compact ? 'py-6' : 'py-10'} px-4`}>
      <span className="grid place-items-center w-10 h-10 rounded-xl mb-3"
        style={{ background: 'var(--surface-raised)', border: '1px solid var(--border)', color: 'var(--text-muted)' }}>
        <Icon name={icon} size={18} />
      </span>
      {title && <p className="text-sm" style={{ color: 'var(--text-secondary)', fontWeight: 500 }}>{title}</p>}
      {children && <p className="text-xs mt-1 max-w-sm" style={{ color: 'var(--text-muted)' }}>{children}</p>}
    </div>
  )
}

export function ErrorState({ children, onRetry, className = '' }) {
  return (
    <div role="alert" className={`flex items-start gap-2.5 rounded-lg border px-3 py-2.5 ${className}`}
      style={{ borderColor: 'rgba(239,77,90,0.4)', background: 'var(--risk-critical-dim)' }}>
      <Icon name="alert" size={15} className="mt-0.5" style={{ color: 'var(--risk-critical)' }} />
      <p className="text-xs flex-1 break-words" style={{ color: 'var(--text-primary)' }}>{children}</p>
      {onRetry && <button type="button" onClick={onRetry} className="btn btn-ghost btn-sm">Retry</button>}
    </div>
  )
}

export function Spinner({ size = 14 }) {
  return (
    <svg className="spin" width={size} height={size} viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeOpacity="0.25" strokeWidth="3" />
      <path d="M21 12a9 9 0 0 0-9-9" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
    </svg>
  )
}

/* label / value row used by the technical panels */
export function MetaRow({ label, children, mono = true }) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-2" style={{ borderBottom: '1px solid var(--border-subtle)' }}>
      <dt className="text-xs shrink-0" style={{ color: 'var(--text-muted)' }}>{label}</dt>
      <dd className={`text-xs text-right min-w-0 break-words ${mono ? 'mono' : ''}`} style={{ color: 'var(--text-primary)' }}>{children}</dd>
    </div>
  )
}

export function StatTile({ label, value, hint, accent }) {
  return (
    <div className="panel px-4 py-3.5 min-w-0">
      <div className="eyebrow">{label}</div>
      <div className="mt-1.5 flex items-baseline gap-2">
        {accent && <span className="rounded-full shrink-0" style={{ width: 8, height: 8, background: accent, transform: 'translateY(-2px)' }} />}
        <span className="text-[26px] leading-none tnum truncate" style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{value}</span>
      </div>
      {hint && <div className="text-[11px] mt-1.5" style={{ color: 'var(--text-muted)' }}>{hint}</div>}
    </div>
  )
}
