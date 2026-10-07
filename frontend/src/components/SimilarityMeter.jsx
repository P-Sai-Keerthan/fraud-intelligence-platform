export default function SimilarityMeter({ similarityPct, deviationPct, unavailable = false }) {
  const hasData = similarityPct != null

  // a real meter when there is a value; otherwise a plain labelled image saying why there is none
  const barA11y = hasData
    ? {
        role: 'meter',
        'aria-label': 'Behavioral Similarity',
        'aria-valuemin': 0,
        'aria-valuemax': 100,
        'aria-valuenow': similarityPct,
        'aria-valuetext': `${similarityPct.toFixed(1)} percent similar, ${deviationPct.toFixed(1)} percent deviation`,
      }
    : { role: 'img', 'aria-label': `Behavioral Similarity: ${unavailable ? 'not available' : 'no result yet'}` }

  return (
    <div>
      <div className="flex justify-between items-baseline mb-2">
        <span className="text-xs uppercase tracking-wide" style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }} aria-hidden="true">
          Behavioral Similarity
        </span>
        <span className="text-sm" style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }} aria-hidden="true">
          {hasData ? `${similarityPct.toFixed(1)}%` : '—'}
        </span>
      </div>
      <div className="w-full h-2 rounded-full overflow-hidden" style={{ background: 'var(--border)' }} {...barA11y}>
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
