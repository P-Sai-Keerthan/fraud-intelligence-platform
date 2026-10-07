import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, CartesianGrid } from 'recharts'
import { formatClock, formatDay, formatFullDateTime, parseTimestamp } from '../format'

// Two-line x-axis label ("08 Oct" over "14:32:05"). The axis is keyed by the transaction's POSITION in the
// chronologically ordered list, never by its label, so two scans made within the same minute (or even the
// same second) can no longer collapse into identical, ambiguous tick labels.
function TimeTick({ x, y, payload, points }) {
  const point = points[payload.value - 1]
  if (!point) return null
  // the first / last label are anchored inward so they are not clipped by the chart edges
  const anchor = points.length > 1 && point.index === points.length ? 'end' : points.length > 1 && point.index === 1 ? 'start' : 'middle'
  return (
    <g transform={`translate(${x},${y})`}>
      <text textAnchor={anchor} fill="var(--text-faint)" fontSize={11}>
        <tspan x="0" dy="14">{point.day}</tspan>
        <tspan x="0" dy="13">{point.clock}</tspan>
      </text>
    </g>
  )
}

export default function FraudEvolutionTimeline({ timeline }) {
  if (!timeline || timeline.length === 0) {
    return (
      <div className="flex items-center justify-center text-center h-48 text-sm px-4" style={{ color: 'var(--text-faint)' }}>
        No transaction history available yet. Scan a transaction to start this customer's risk timeline.
      </div>
    )
  }

  // ordered exactly as the server returned them (chronological by transaction time); values are untouched
  const points = timeline.map((t, i) => {
    const when = parseTimestamp(t.timestamp)
    return {
      index: i + 1,
      risk_score: t.risk_score,
      fraud_probability: t.fraud_probability,
      day: when ? formatDay(when) : '',
      clock: when ? formatClock(when) : '',
      full: when ? formatFullDateTime(when) : String(t.timestamp),
    }
  })

  const last = points[points.length - 1]
  const summary =
    `Fraud risk timeline: ${points.length} scanned transaction${points.length === 1 ? '' : 's'}, from ${points[0].full} to ${last.full}. ` +
    `Latest: Fraud Risk Score ${last.fraud_probability.toFixed(1)}, ` +
    `Temporal Risk ${last.risk_score != null ? last.risk_score.toFixed(1) : 'not available'}.`

  return (
    <div role="img" aria-label={summary}>
      <ResponsiveContainer width="100%" height={236}>
        <LineChart data={points} margin={{ top: 8, right: 16, left: -16, bottom: 0 }}>
          <CartesianGrid stroke="var(--border-subtle)" vertical={false} />
          <XAxis
            dataKey="index"
            tick={<TimeTick points={points} />}
            height={44}
            minTickGap={28}
            axisLine={{ stroke: 'var(--border)' }} tickLine={false}
          />
          <YAxis
            domain={[0, 100]}
            tick={{ fill: 'var(--text-faint)', fontSize: 11 }}
            axisLine={false} tickLine={false}
          />
          <Tooltip
            contentStyle={{ background: 'var(--surface-raised)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 12 }}
            labelStyle={{ color: 'var(--text-primary)' }}
            labelFormatter={(index) => points[index - 1]?.full}
            formatter={(value, name) => [value == null ? 'not available' : Number(value).toFixed(1), name]}
          />
          {/* the two series are told apart by line style (solid vs dashed) as well as colour, and named here */}
          <Legend verticalAlign="top" height={26} wrapperStyle={{ fontSize: 11, color: 'var(--text-muted)' }} />
          <Line
            type="monotone" dataKey="risk_score" name="Temporal Risk (LSTM)"
            stroke="var(--brand)" strokeWidth={2} dot={{ r: 3, fill: 'var(--brand)' }}
            activeDot={{ r: 5 }}
          />
          <Line
            type="monotone" dataKey="fraud_probability" name="Fraud Risk Score (DNN)"
            stroke="var(--risk-critical)" strokeWidth={2} strokeDasharray="4 3" dot={{ r: 3, fill: 'var(--risk-critical)' }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
