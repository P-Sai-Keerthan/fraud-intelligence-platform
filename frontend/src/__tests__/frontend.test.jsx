// First frontend tests (audit finding R-12). Components are rendered to static HTML with react-dom/server, so no browser
// or DOM library is needed; they cover the parts the audit found fragile: error messages and the similarity display.
import { describe, expect, it } from 'vitest'
import { renderToStaticMarkup } from 'react-dom/server'
import { formatApiError } from '../apiError'
import { severityOf, scoreColor, bandText, alertTiers } from '../severity'
import AlertBanner from '../components/AlertBanner'
import { scoreWording } from '../modelWording'
import SimilarityMeter from '../components/SimilarityMeter'
import BehavioralAnalysis from '../components/BehavioralAnalysis'

const err = (status, detail) => ({ response: { status, data: detail === undefined ? {} : { detail } } })

describe('formatApiError', () => {
  it('passes string details through', () => {
    expect(formatApiError(err(404, 'No transaction history'))).toBe('No transaction history')
  })
  it('turns a 422 list into readable text, not raw JSON', () => {
    const msg = formatApiError(err(422, [
      { type: 'less_than_equal', loc: ['body', 'amount'], msg: 'Input should be less than or equal to 1000000000' },
      { type: 'string_too_short', loc: ['body', 'customer_id'], msg: 'String should have at least 1 character' },
    ]))
    expect(msg).toBe('amount: Input should be less than or equal to 1000000000; customer_id: String should have at least 1 character')
    expect(msg).not.toMatch(/[{}[\]]/)
  })
  it('falls back when there is no response', () => {
    expect(formatApiError(new Error('Network Error'))).toMatch(/backend running/)
    expect(formatApiError(err(500))).toMatch(/HTTP 500/)
  })
})

describe('severity helpers', () => {
  it('maps unknown levels to an explicit unknown state', () => {
    expect(severityOf('Nonsense').key).toBe('unknown')
    expect(severityOf('Critical Risk').rank).toBeGreaterThan(severityOf('High Risk').rank)
  })
  it('does not colour a missing score as a risk band', () => {
    expect(scoreColor(null)).toBe('var(--border-strong)')
    expect(scoreColor(10)).toBe('var(--risk-low)')
    expect(scoreColor(80)).toBe('var(--risk-critical)')
  })
  it('rewrites backend band wording', () => {
    expect(bandText('fraud_probability >= 80')).toBe('fraud score ≥ 80')
    expect(bandText(undefined)).toBe('')
  })
  it('falls back to neutral wording for an unknown model set', () => {
    expect(scoreWording({ model_set: 'unknown_set' }).riskLabel).toBe('Risk Score')
    expect(scoreWording(null).fraudCaption).toMatch(/not a calibrated probability/)
  })
})

describe('SimilarityMeter', () => {
  it('shows a score when the status is ok', () => {
    const html = renderToStaticMarkup(<SimilarityMeter similarityPct={56.8} deviationPct={43.2} similarityStatus="ok" historyTransactions={107} />)
    expect(html).toContain('56.8')
    expect(html).toContain('43.2%')
    expect(html).not.toContain('Not enough history')
  })
  it('says "not enough history" and shows no number or ring when there is no baseline', () => {
    const html = renderToStaticMarkup(<SimilarityMeter similarityPct={null} deviationPct={null} similarityStatus="insufficient_history" historyTransactions={3} />)
    expect(html).toContain('Not enough history')
    expect(html).toContain('3 earlier transactions')
    expect(html).toContain('at least 10')
    expect(html).not.toMatch(/>0(\.0)?<|>100(\.0)?<|deviation from/)
    expect(html).not.toContain('<circle')            // no progress ring is drawn for a missing score
  })
  it('uses the singular for exactly one earlier transaction', () => {
    const html = renderToStaticMarkup(<SimilarityMeter similarityPct={null} deviationPct={null} similarityStatus="insufficient_history" historyTransactions={1} />)
    expect(html).toContain('1 earlier transaction;')
  })
  it('still shows the neutral placeholder before any scan', () => {
    const html = renderToStaticMarkup(<SimilarityMeter />)
    expect(html).toContain('—')
    expect(html).not.toContain('Not enough history')
  })
})

describe('BehavioralAnalysis', () => {
  const profile = { home_device: 'DEV_X', home_location: 'Pune', n_transactions: 2 }
  it('explains a missing baseline and draws no similarity/deviation split', () => {
    const html = renderToStaticMarkup(<BehavioralAnalysis profile={profile} prediction={{
      similarity_pct: null, deviation_pct: null, similarity_status: 'insufficient_history', history_transactions: 2, device_id: 'D', location: 'L',
    }} />)
    expect(html).toContain('Not enough history for a behavioural match')
    expect(html).not.toMatch(/100\.0%|0\.0%/)
    expect(html).not.toMatch(/width:\d+(\.\d+)?%/)       // no similarity/deviation bar segments are drawn
    expect(html).toContain('aria-label="Not enough history for a behavioural baseline"')
  })
  it('draws the split for a real score', () => {
    const html = renderToStaticMarkup(<BehavioralAnalysis profile={profile} prediction={{
      similarity_pct: 40, deviation_pct: 60, similarity_status: 'ok', history_transactions: 50, device_id: 'D', location: 'L',
    }} />)
    expect(html).toContain('40.0%')
    expect(html).toContain('60.0%')
    expect(html).toMatch(/width:40%/)
    expect(html).toMatch(/width:60%/)
  })
})

// R-02: gauge colours and verdict wording follow the alert policy the backend publishes for the loaded model set.
const VALIDATED_INFO = {
  thresholds: {
    alert_policy: {
      kind: 'validated_cutoffs',
      tiers: [{ level: 'Low Risk', min_score: 0 }, { level: 'Medium Risk', min_score: 4.3931 }, { level: 'Critical Risk', min_score: 26.2307 }],
    },
    alert_bands: { 'Medium Risk': '4.39 <= fraud_probability < 26.23' },
  },
}

describe('alert policy (R-02)', () => {
  it('falls back to the legacy 25/50/80 tiers when no policy is published', () => {
    expect(alertTiers(null).map((t) => t.min_score)).toEqual([0, 25, 50, 80])
    expect(scoreColor(6)).toBe(severityOf('Low Risk').color)
    expect(scoreColor(30)).toBe(severityOf('Medium Risk').color)
    expect(scoreColor(85)).toBe(severityOf('Critical Risk').color)
  })
  it('colours a Fraud Score by the validated tiers', () => {
    const tiers = alertTiers(VALIDATED_INFO)
    expect(scoreColor(3, tiers)).toBe(severityOf('Low Risk').color)
    expect(scoreColor(6, tiers)).toBe(severityOf('Medium Risk').color)       // above the alert cut-off, was tinted "low" before
    expect(scoreColor(4.3931, tiers)).toBe(severityOf('Medium Risk').color)   // the cut-off itself alerts
    expect(scoreColor(30, tiers)).toBe(severityOf('Critical Risk').color)
    expect(scoreColor(null, tiers)).toBe('var(--border-strong)')
  })
  it('words the verdict as cut-offs for the validated policy and keeps the legacy wording otherwise', () => {
    const pred = { alert_level: 'Medium Risk', transaction_id: 'T1', customer_id: 'C1', timestamp: '2026-07-10T13:30:00' }
    const validated = renderToStaticMarkup(<AlertBanner prediction={pred} modelInfo={VALIDATED_INFO} />)
    expect(validated).toContain('at or above the alert cut-off')
    expect(validated).toContain('4.39 ≤ fraud score &lt; 26.23')
    const legacy = renderToStaticMarkup(<AlertBanner prediction={pred} modelInfo={{ thresholds: { alert_bands: { 'Medium Risk': '25 <= fraud_probability < 50' } } }} />)
    expect(legacy).toContain('medium alert band')
  })
})
