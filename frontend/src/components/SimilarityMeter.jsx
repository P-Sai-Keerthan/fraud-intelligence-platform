export default function SimilarityMeter({ similarityPct, deviationPct, unavailable = false }) {
  const hasData = similarityPct != null
  return (
    <div>
      <div className="flex justify-between items-baseline mb-2">
        <span className="text-xs uppercase tracking-wide" style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
          Behavioral Similarity
        </span>
        <span className="text-sm" style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>
          {hasData ? `${similarityPct.toFixed(1)}%` : '—'}
        </span>
      </div>
      <div className="w-full h-2 rounded-full overflow-hidden" style={{ background: 'var(--border)' }}>
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{
            width: `${hasData ? similarityPct : 0}%`,
            background: 'linear-gradient(90deg, var(--risk-critical), var(--risk-medium), var(--brand))',
          }}
        />
      </div>
      {unavailable && (
        <p className="text-xs mt-1.5" style={{ color: 'var(--text-faint)' }}>
          Not available: a behavioral baseline needs at least 10 prior transactions for this customer.
        </p>
      )}
      {hasData && (
        <p className="text-xs mt-1.5" style={{ color: 'var(--text-faint)' }}>
          {deviationPct.toFixed(1)}% deviation from this customer's normal behavior
        </p>
      )}
    </div>
  )
}
