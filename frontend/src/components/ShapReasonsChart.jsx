import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'

export default function ShapReasonsChart({ reasons }) {
  if (!reasons || reasons.length === 0) {
    return (
      <div className="flex items-center justify-center h-40 text-sm" style={{ color: 'var(--text-faint)' }}>
        No significant fraud-indicating factors detected.
      </div>
    )
  }

  const data = reasons
    .map((r) => ({ name: r.display_name, value: Number(r.shap_value.toFixed(4)) }))
    .sort((a, b) => a.value - b.value)

  return (
    <ResponsiveContainer width="100%" height={Math.max(140, data.length * 44)}>
      <BarChart data={data} layout="vertical" margin={{ left: 8, right: 24, top: 4, bottom: 4 }}>
        <XAxis type="number" hide />
        <YAxis
          type="category" dataKey="name" width={160}
          tick={{ fill: 'var(--text-muted)', fontSize: 12, fontFamily: 'var(--font-body)' }}
          axisLine={false} tickLine={false}
        />
        <Tooltip
          contentStyle={{ background: 'var(--surface-raised)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 12 }}
          labelStyle={{ color: 'var(--text-primary)' }}
          itemStyle={{ color: 'var(--brand)' }}
          formatter={(value) => [value, 'SHAP contribution']}
        />
        <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={18}>
          {data.map((_, i) => (
            <Cell key={i} fill="var(--risk-high)" />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}
