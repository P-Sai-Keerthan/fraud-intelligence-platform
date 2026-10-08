import { useEffect, useMemo, useState } from 'react'
import { getFraudRings } from '../api'
import { buildClusters, layoutCluster, ringTypeLabel } from '../rings'
import { severityOf } from '../severity'
import { Badge, Card, EmptyState, ErrorState, Icon, MetaRow, SeverityBadge, Skeleton, Spinner, StatTile } from './ui'

const PAGE = 12
const shortId = (id) => String(id).replace(/^CUST_/, '').replace(/^DEV_(UNKNOWN_)?/, '')

function ClusterGraph({ cluster, width = 260, height = 150, labels = 'short' }) {
  const { nodes, edges } = useMemo(() => layoutCluster(cluster, width, height), [cluster, width, height])
  const s = severityOf(cluster.severity)
  const full = labels === 'full'
  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto" role="img"
      aria-label={`${cluster.devices.length} device${cluster.devices.length === 1 ? '' : 's'} shared by ${cluster.customers.length} customers`}>
      {edges.map((e, i) => (
        <line key={i} x1={nodes[e.from].x} y1={nodes[e.from].y} x2={nodes[e.to].x} y2={nodes[e.to].y}
          stroke={s.color} strokeOpacity="0.55" strokeWidth="1.4" />
      ))}
      {cluster.customers.map((c) => {
        const n = nodes[`c:${c}`]
        return (
          <g key={c}>
            <circle cx={n.x} cy={n.y} r={full ? 11 : 9} fill="var(--surface-raised)" stroke="var(--brand)" strokeWidth="1.5" />
            <circle cx={n.x} cy={n.y - 2} r={full ? 3 : 2.5} fill="var(--brand)" />
            <path d={`M${n.x - (full ? 5 : 4)} ${n.y + (full ? 6 : 5)} q${full ? 5 : 4} -${full ? 6 : 5} ${full ? 10 : 8} 0`} fill="none" stroke="var(--brand)" strokeWidth="1.5" strokeLinecap="round" />
            <text x={n.x} y={n.y + (full ? 26 : 22)} textAnchor="middle" fontSize={full ? 10.5 : 9.5} fill="var(--text-secondary)" fontFamily="var(--font-mono)">
              {full ? c : shortId(c)}
            </text>
          </g>
        )
      })}
      {cluster.devices.map((d) => {
        const n = nodes[`d:${d}`]
        const r = full ? 13 : 11
        return (
          <g key={d}>
            <rect x={n.x - r} y={n.y - r} width={r * 2} height={r * 2} rx="5" fill="var(--surface)" stroke={s.color} strokeWidth="1.8" />
            <rect x={n.x - 4} y={n.y - 6.5} width="8" height="13" rx="1.6" fill="none" stroke={s.color} strokeWidth="1.4" />
            {full && cluster.devices.length <= 2 && (
              <text x={n.x} y={n.y - r - 6} textAnchor="middle" fontSize="10.5" fill="var(--text-primary)" fontFamily="var(--font-mono)">{d}</text>
            )}
          </g>
        )
      })}
    </svg>
  )
}

function Legend() {
  return (
    <div className="flex flex-wrap items-center gap-x-5 gap-y-1.5 text-[11px]" style={{ color: 'var(--text-muted)' }}>
      <span className="flex items-center gap-1.5">
        <span className="rounded-full" style={{ width: 11, height: 11, border: '1.5px solid var(--brand)', background: 'var(--surface-raised)' }} />Customer
      </span>
      <span className="flex items-center gap-1.5">
        <span style={{ width: 11, height: 11, borderRadius: 3, border: '1.5px solid var(--text-secondary)', background: 'var(--surface)' }} />Shared device
      </span>
      <span className="flex items-center gap-1.5">
        <span style={{ width: 16, height: 1.5, background: 'var(--text-secondary)' }} />Customer used the device
      </span>
    </div>
  )
}

export default function FraudRings() {
  const [rings, setRings] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [selectedId, setSelectedId] = useState(null)
  const [showAll, setShowAll] = useState(false)
  const [loadedAt, setLoadedAt] = useState(null)

  const load = () => {
    setLoading(true)
    setError('')
    getFraudRings()
      .then((data) => { setRings(data.rings); setLoadedAt(new Date()) })
      .catch(() => setError('Could not load fraud ring data. Is the backend running?'))
      .finally(() => setLoading(false))
  }

  useEffect(load, [])

  const clusters = useMemo(() => (rings ? buildClusters(rings) : []), [rings])
  const selected = clusters.find((c) => c.id === selectedId) || clusters[0] || null
  const stats = useMemo(() => {
    if (!rings) return null
    return {
      rings: rings.length,
      customers: new Set(rings.flatMap((r) => r.customer_ids)).size,
      devices: new Set(rings.map((r) => r.identifier)).size,
      transactions: rings.reduce((n, r) => n + (r.transaction_count || 0), 0),
      clusters: clusters.length,
    }
  }, [rings, clusters])
  const visible = showAll ? clusters : clusters.slice(0, PAGE)

  const refresh = (
    <button type="button" onClick={load} disabled={loading} className="btn btn-ghost btn-sm">
      {loading ? <Spinner size={12} /> : <Icon name="refresh" size={13} />}Refresh
    </button>
  )

  if (error) {
    return <ErrorState onRetry={load}>{error}</ErrorState>
  }

  if (loading && !rings) {
    return (
      <div className="space-y-5">
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">{[0, 1, 2, 3].map((i) => <Skeleton key={i} style={{ height: 84 }} />)}</div>
        <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_380px] gap-5">
          <Skeleton style={{ height: 420 }} /><Skeleton style={{ height: 420 }} />
        </div>
      </div>
    )
  }

  if (rings && rings.length === 0) {
    return (
      <Card actions={refresh} title="No rings detected" eyebrow="Shared-device analysis" icon="network">
        <EmptyState icon="shield" title="No device is shared between customers">
          A ring appears here when the same device is used by two or more customers. To see one, run the
          &ldquo;Suspicious Pattern&rdquo; scan on Live Scan for two different customers, then refresh.
        </EmptyState>
      </Card>
    )
  }

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <StatTile label="Rings detected" value={stats.rings.toLocaleString()} accent="var(--risk-critical)"
          hint={`Shared devices, grouped into ${stats.clusters.toLocaleString()} cluster${stats.clusters === 1 ? '' : 's'}`} />
        <StatTile label="Affected customers" value={stats.customers.toLocaleString()} hint="Distinct customers linked to a ring" />
        <StatTile label="Shared devices" value={stats.devices.toLocaleString()} hint="Devices used by two or more customers" />
        <StatTile label="Linked transactions" value={stats.transactions.toLocaleString()} hint="Transactions made on those devices" />
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_380px] gap-5 items-start">
        <Card title="Shared-device clusters" eyebrow="Relationship map" icon="network" actions={refresh}>
          <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
            <Legend />
            {loadedAt && (
              <span className="mono text-[10.5px]" style={{ color: 'var(--text-faint)' }}>
                loaded {loadedAt.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
              </span>
            )}
          </div>
          <div className="grid grid-cols-2 md:grid-cols-3 2xl:grid-cols-4 gap-3" style={{ opacity: loading ? 0.5 : 1, transition: 'opacity 0.2s ease' }}>
            {visible.map((c) => {
              const active = selected?.id === c.id
              const s = severityOf(c.severity)
              return (
                <button
                  key={c.id} type="button" onClick={() => setSelectedId(c.id)} aria-pressed={active}
                  className="text-left rounded-xl border px-2.5 pt-2.5 pb-2 transition-colors"
                  style={{
                    borderColor: active ? `color-mix(in srgb, ${s.color} 70%, transparent)` : 'var(--border-subtle)',
                    background: active ? s.dim : 'var(--surface-sunken)',
                    boxShadow: active ? `0 0 0 1px color-mix(in srgb, ${s.color} 35%, transparent)` : 'none',
                  }}
                >
                  <div className="flex items-center justify-between gap-2 mb-1">
                    <span className="mono text-[10.5px] truncate" style={{ color: 'var(--text-secondary)' }}>{c.devices[0]}{c.devices.length > 1 ? ` +${c.devices.length - 1}` : ''}</span>
                    <span className="rounded-full shrink-0" title={c.severity} style={{ width: 7, height: 7, background: s.color }} />
                  </div>
                  <ClusterGraph cluster={c} />
                  <div className="flex items-center justify-between mt-1 text-[10.5px]" style={{ color: 'var(--text-muted)' }}>
                    <span>{c.customers.length} customers</span>
                    <span className="mono">{c.transactions} txns</span>
                  </div>
                </button>
              )
            })}
          </div>
          {clusters.length > PAGE && (
            <div className="flex justify-center mt-4">
              <button type="button" className="btn btn-ghost btn-sm" onClick={() => setShowAll((v) => !v)}>
                {showAll ? 'Show fewer' : `Show all ${clusters.length} clusters`}
              </button>
            </div>
          )}
        </Card>

        {selected && (
          <div className="xl:sticky xl:top-[92px] xl:max-h-[calc(100vh-108px)] xl:overflow-y-auto rounded-[14px]">
          <Card title="Investigation summary" eyebrow="Selected cluster" icon="search"
            actions={<Badge tone="warn" icon="alert">Shared-device link</Badge>}>
            <div className="panel p-2 mb-4">
              <ClusterGraph cluster={selected} width={340} height={210} labels="full" />
            </div>
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Size-based level (UI)</span>
              <SeverityBadge level={selected.severity} size="md" />
            </div>
            <dl>
              <MetaRow label="Signal" mono={false}>{selected.types.map(ringTypeLabel).join(', ')}</MetaRow>
              <MetaRow label="Pattern" mono={false}>{selected.customers.length} accounts linked by shared devices</MetaRow>
              <MetaRow label="Customers linked">{selected.customers.length}</MetaRow>
              <MetaRow label="Shared devices">{selected.devices.length}</MetaRow>
              <MetaRow label="Linked transactions">{selected.transactions}</MetaRow>
            </dl>

            <div className="mt-4">
              <div className="eyebrow mb-2">Shared device{selected.devices.length === 1 ? '' : 's'}</div>
              <ul className="space-y-1.5">
                {selected.rings.map((r) => (
                  <li key={r.identifier} className="panel px-3 py-2">
                    <div className="flex items-center gap-2">
                      <Icon name="device" size={13} style={{ color: 'var(--text-muted)' }} />
                      <span className="mono text-xs truncate" style={{ color: 'var(--text-primary)' }}>{r.identifier}</span>
                      <span className="mono text-[10.5px] ml-auto shrink-0" style={{ color: 'var(--text-muted)' }}>{r.transaction_count} txns</span>
                    </div>
                    <div className="flex flex-wrap gap-1.5 mt-2">
                      {r.customer_ids.map((cid) => <span key={cid} className="chip" style={{ color: 'var(--brand)' }}>{cid}</span>)}
                    </div>
                  </li>
                ))}
              </ul>
            </div>
            <p className="text-[11px] mt-4" style={{ color: 'var(--text-muted)' }}>
              Severity here reflects cluster size only (2 customers: Medium; 3 customers or 2 devices: High; 4+ customers or 3+ devices: Critical).
              It is a reading aid, not a model output.
            </p>
          </Card>
          </div>
        )}
      </div>
    </div>
  )
}
