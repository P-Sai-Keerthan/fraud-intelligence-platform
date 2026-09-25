import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts'

export default function FraudEvolutionTimeline({ timeline }) {
  if (!timeline || timeline.length === 0) {
    return (
      <div className="flex items-center justify-center h-48 text-sm" style={{ color: 'var(--text-faint)' }}>
        Scan a transaction to start building this customer's risk timeline.
      </div>
    )
  }

  // the API returns newest first; plot oldest -> newest, left to right
  const chronological = [...timeline].sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp))
  const data = chronological.map((t, i) => ({
    index: i + 1,
    risk_score: t.risk_score,
    fraud_probability: t.fraud_probability,
    time: new Date(t.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
  }))

  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={data} margin={{ top: 8, right: 16, left: -16, bottom: 0 }}>
        <CartesianGrid stroke="var(--border-subtle)" vertical={false} />
        <XAxis
          dataKey="time"
          tick={{ fill: 'var(--text-faint)', fontSize: 11 }}
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
        />
        <Line
          type="monotone" dataKey="risk_score" name="Risk Score"
          stroke="var(--brand)" strokeWidth={2} dot={{ r: 3, fill: 'var(--brand)' }}
          activeDot={{ r: 5 }}
        />
        <Line
          type="monotone" dataKey="fraud_probability" name="Fraud Probability %"
          stroke="var(--risk-critical)" strokeWidth={2} strokeDasharray="4 3" dot={{ r: 3, fill: 'var(--risk-critical)' }}
        />
      </LineChart>
    </ResponsiveContainer>
  )
}
