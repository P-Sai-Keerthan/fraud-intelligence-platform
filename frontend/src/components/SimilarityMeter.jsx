import { ScoreRing } from './RiskGauge'

export default function SimilarityMeter({ similarityPct, deviationPct, loading = false }) {
  const hasData = similarityPct != null
  return (
    <div className="flex flex-col items-center text-center h-full">
      <div className="eyebrow self-start">Behavioral Similarity</div>
      <div className="my-3" style={{ opacity: loading ? 0.45 : 1, transition: 'opacity 0.2s ease' }}>
        <ScoreRing value={hasData ? similarityPct : null} color="var(--brand)">
          <span className="text-[30px] leading-none tnum" style={{ fontWeight: 600, color: 'var(--text-primary)' }}
            aria-label={hasData ? `Behavioral similarity ${similarityPct.toFixed(1)} percent` : 'Behavioral similarity: no value yet'}>
            {hasData ? similarityPct.toFixed(1) : '—'}
            {hasData && <span className="text-base" style={{ color: 'var(--text-muted)' }}>%</span>}
          </span>
          <span className="mono text-[11px] mt-1" style={{ color: 'var(--text-faint)' }}>match</span>
        </ScoreRing>
      </div>
      <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>
        {hasData && deviationPct != null
          ? <><span className="tnum" style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{deviationPct.toFixed(1)}%</span> deviation from this customer&apos;s normal behavior</>
          : 'Match with this customer’s normal behavior'}
      </p>
    </div>
  )
}
