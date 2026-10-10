import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts'
import { EmptyState } from './ui'

function TimelineTooltip({ active, payload, riskName }) {
  if (!active || !payload?.length) return null
  const row = payload[0].payload
  return (
    <div className="rounded-lg border px-3 py-2 text-xs" style={{ background: 'var(--surface-raised)', borderColor: 'var(--border-strong)', boxShadow: '0 10px 24px -12px #000' }}>
      <div className="mono mb-1" style={{ color: 'var(--text-muted)' }}>{row.fullTime}</div>
      <div className="flex items-center gap-2"><span style={{ width: 10, height: 2, background: 'var(--series-1)' }} /><span style={{ color: 'var(--text-secondary)' }}>{riskName}</span><span className="mono tnum ml-auto pl-4" style={{ color: 'var(--text-primary)' }}>{row.risk_score.toFixed(1)}</span></div>
      <div className="flex items-center gap-2"><span style={{ width: 10, height: 2, background: 'var(--series-2)' }} /><span style={{ color: 'var(--text-secondary)' }}>Fraud Score</span><span className="mono tnum ml-auto pl-4" style={{ color: 'var(--text-primary)' }}>{row.fraud_probability.toFixed(1)}</span></div>
      <div className="mt-1" style={{ color: 'var(--text-muted)' }}>{row.alert_level}</div>
    </div>
  )
}

export default function FraudEvolutionTimeline({ timeline, riskName = 'Risk Score' }) {
  if (!timeline || timeline.length === 0) {
    return (
      <EmptyState icon="activity" title="No scans recorded for this customer" compact>
        Each scan adds a point here, building the customer&apos;s score timeline.
      </EmptyState>
    )
  }

  // the API returns newest first; plot oldest -> newest, left to right
  const chronological = [...timeline].sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp))
  const data = chronological.map((t, i) => {
    const d = new Date(t.timestamp)
    return {
      index: i + 1,
      risk_score: t.risk_score,
      fraud_probability: t.fraud_probability,
      alert_level: t.alert_level,
      time: d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      fullTime: d.toLocaleString([], { month: 'short', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit' }),
    }
  })

  return (
    <div>
      <div className="flex items-center gap-5 mb-2 text-xs" style={{ color: 'var(--text-secondary)' }}>
        <span className="flex items-center gap-2"><span style={{ width: 14, height: 2, background: 'var(--series-1)', borderRadius: 2 }} />{riskName}</span>
        <span className="flex items-center gap-2"><span style={{ width: 14, height: 2, background: 'var(--series-2)', borderRadius: 2 }} />Fraud Score</span>
        <span className="mono text-[10.5px] ml-auto" style={{ color: 'var(--text-faint)' }}>0–100 &middot; {data.length} scan{data.length === 1 ? '' : 's'}</span>
      </div>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={data} margin={{ top: 8, right: 12, left: -18, bottom: 0 }}>
          <CartesianGrid stroke="var(--chart-grid)" vertical={false} />
          <XAxis
            dataKey="index" tickFormatter={(i) => data[i - 1]?.time || ''}
            tick={{ fill: 'var(--text-faint)', fontSize: 11 }} minTickGap={28}
            axisLine={{ stroke: 'var(--border)' }} tickLine={false}
          />
          <YAxis
            domain={[0, 100]} ticks={[0, 25, 50, 75, 100]}
            tick={{ fill: 'var(--text-faint)', fontSize: 11 }}
            axisLine={false} tickLine={false}
          />
          <Tooltip content={<TimelineTooltip riskName={riskName} />} cursor={{ stroke: 'var(--border-strong)', strokeWidth: 1 }} />
          <Line
            type="monotone" dataKey="risk_score" name={riskName} isAnimationActive={false}
            stroke="var(--series-1)" strokeWidth={2} dot={{ r: 4, fill: 'var(--series-1)', stroke: 'var(--surface)', strokeWidth: 2 }}
            activeDot={{ r: 5, stroke: 'var(--surface)', strokeWidth: 2 }}
          />
          <Line
            type="monotone" dataKey="fraud_probability" name="Fraud Score" isAnimationActive={false}
            stroke="var(--series-2)" strokeWidth={2} dot={{ r: 4, fill: 'var(--series-2)', stroke: 'var(--surface)', strokeWidth: 2 }}
            activeDot={{ r: 5, stroke: 'var(--surface)', strokeWidth: 2 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
