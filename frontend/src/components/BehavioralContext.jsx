// Behavioral Context / What Changed?
// Everything here is computed from the customer's EXISTING transaction history
// (GET /customer/{id}/profile) and the scanned transaction. It describes
// differences from past behavior; it never changes or re-scores a prediction,
// and it does not claim to show how behavior evolves over time.
import { plural, rupees } from '../format'

const USUAL_SHARE = 0.05 // same 5% rule the model's hour_is_unusual feature uses

const pct = (x) => `${Math.round(x * 100)}%`
const pad = (h) => String(h).padStart(2, '0')

function hourRanges(hours) {
  if (!hours || hours.length === 0) return '—'
  const sorted = [...hours].sort((a, b) => a - b)
  const ranges = []
  let start = sorted[0]
  let prev = sorted[0]
  for (const h of sorted.slice(1)) {
    if (h === prev + 1) { prev = h; continue }
    ranges.push([start, prev])
    start = prev = h
  }
  ranges.push([start, prev])
  return ranges.map(([a, b]) => `${pad(a)}:00–${pad(b + 1 === 24 ? 0 : b + 1)}:00`).join(', ')
}

function hourOf(iso) {
  // read the hour straight from the ISO string ("YYYY-MM-DDTHH:MM...") so it is
  // never shifted by the browser's timezone
  const h = Number(String(iso).slice(11, 13))
  return Number.isNaN(h) ? null : h
}

function describe(profile, prediction) {
  const known = (list, v) => (list || []).includes(v)
  const amount = prediction.amount
  const ratio = profile.typical_amount ? amount / profile.typical_amount : null
  const hour = hourOf(prediction.timestamp)
  const hourShare = hour != null && profile.hour_distribution?.length === 24 ? profile.hour_distribution[hour] : null

  const device =
    prediction.device_id === profile.primary_device ? 'primary'
      : known(profile.known_devices, prediction.device_id) ? 'seen' : 'new'
  const location =
    prediction.location === profile.primary_location ? 'primary'
      : known(profile.known_locations, prediction.location) ? 'seen' : 'new'

  const changes = []
  if (device === 'new') changes.push(`New device (${prediction.device_id}) — never used by this customer before; their usual device is ${profile.primary_device}.`)
  else if (device === 'seen') changes.push(`Not the usual device — ${prediction.device_id} was used before, but ${profile.primary_device} is the primary one.`)
  if (location === 'new') changes.push(`New location (${prediction.location}) — this customer has never transacted there; their usual location is ${profile.primary_location}.`)
  else if (location === 'seen') changes.push(`Not the usual location — ${prediction.location} was used before, but ${profile.primary_location} is the primary one.`)
  if (ratio != null && ratio >= 2) changes.push(`Amount is ${ratio.toFixed(1)}× this customer's typical amount (${rupees(profile.typical_amount)}).`)
  else if (ratio != null && ratio <= 0.5) changes.push(`Amount is well below this customer's typical amount (${rupees(profile.typical_amount)}).`)
  if (hourShare != null && hourShare < USUAL_SHARE) {
    changes.push(`Transaction time ${pad(hour)}:00 is outside this customer's preferred hours (${hourRanges(profile.preferred_hours)}).`)
  }
  return { ratio, hour, hourShare, device, location, changes }
}

function Row({ label, children }) {
  return (
    <div className="flex justify-between gap-4 py-1.5 text-xs" style={{ borderBottom: '1px solid var(--border-subtle)' }}>
      <dt style={{ color: 'var(--text-muted)' }}>{label}</dt>
      <dd className="text-right" style={{ color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>{children}</dd>
    </div>
  )
}

const STATE_TEXT = { primary: 'usual', seen: 'used before, not primary', new: 'new' }

const note = (text) => (
  <p className="text-sm" style={{ color: 'var(--text-faint)' }}>{text}</p>
)

// status: 'loading' | 'error' | anything else (ready / nothing selected)
export default function BehavioralContext({ profile, context, status }) {
  if (status === 'loading') return <p role="status" className="text-sm" style={{ color: 'var(--text-faint)' }}>Loading this customer's behavior…</p>
  if (status === 'error') return <p role="alert" className="text-sm" style={{ color: 'var(--risk-critical)' }}>Could not load this customer's behavior. Is the backend running?</p>
  if (!profile) return note('Select a customer to see their normal behavior.')
  if (profile.n_transactions === 0) return note('No transaction history available yet. There is no behavioral baseline to compare against.')

  const baseline = context?.baseline
  const prediction = context?.prediction
  const diff = baseline && prediction ? describe(baseline, prediction) : null

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <div>
        <p className="text-[11px] uppercase tracking-wide mb-2" style={{ color: 'var(--text-faint)', fontFamily: 'var(--font-mono)' }}>
          Customer's normal behavior · {plural(profile.n_transactions, 'past transaction')}
        </p>
        <dl>
          <Row label="Typical amount (median)">
            {rupees(profile.typical_amount)}
            <span style={{ color: 'var(--text-faint)' }}> ({rupees(profile.amount_p25)}–{rupees(profile.amount_p75)})</span>
          </Row>
          <Row label="Primary device">
            {profile.primary_device}
            {profile.devices?.[0] && <span style={{ color: 'var(--text-faint)' }}> ({pct(profile.devices[0].share)})</span>}
          </Row>
          <Row label="Primary location">
            {profile.primary_location}
            {profile.locations?.[0] && <span style={{ color: 'var(--text-faint)' }}> ({pct(profile.locations[0].share)})</span>}
          </Row>
          <Row label="Preferred hours">{hourRanges(profile.preferred_hours)}</Row>
          <Row label="Usual categories">{(profile.top_categories || []).map((c) => c.value.replace('_', ' ')).join(', ') || '—'}</Row>
          <Row label="Typical frequency">
            {profile.transactions_per_week != null ? `≈ ${profile.transactions_per_week} per active week` : '—'}
          </Row>
        </dl>
      </div>

      <div>
        <p className="text-[11px] uppercase tracking-wide mb-2" style={{ color: 'var(--text-faint)', fontFamily: 'var(--font-mono)' }}>
          This transaction vs normal
        </p>
        {!diff ? (
          <p className="text-xs" style={{ color: 'var(--text-faint)' }}>
            Scan a transaction to compare it with this customer's history.
          </p>
        ) : (
          <>
            <dl>
              <Row label="Amount">
                {rupees(prediction.amount, { exact: true })}
                {diff.ratio != null && <span style={{ color: 'var(--text-faint)' }}> ({diff.ratio.toFixed(1)}× typical)</span>}
              </Row>
              <Row label="Device">
                {prediction.device_id} <span style={{ color: 'var(--text-faint)' }}>({STATE_TEXT[diff.device]})</span>
              </Row>
              <Row label="Location">
                {prediction.location} <span style={{ color: 'var(--text-faint)' }}>({STATE_TEXT[diff.location]})</span>
              </Row>
              <Row label="Time">
                {diff.hour != null ? `${pad(diff.hour)}:00` : '—'}{' '}
                <span style={{ color: 'var(--text-faint)' }}>
                  ({diff.hourShare != null && diff.hourShare < USUAL_SHARE ? 'unusual hour' : 'usual hour'})
                </span>
              </Row>
            </dl>

            <p className="text-[11px] uppercase tracking-wide mt-4 mb-2" style={{ color: 'var(--text-faint)', fontFamily: 'var(--font-mono)' }}>
              What changed?
            </p>
            {diff.changes.length === 0 ? (
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                Nothing stands out: device, location, amount and time all match this customer's history.
              </p>
            ) : (
              <ul className="space-y-1.5 text-xs" style={{ color: 'var(--text-muted)' }}>
                {diff.changes.map((c) => (
                  <li key={c} className="flex gap-2">
                    <span style={{ color: 'var(--risk-high)' }} aria-hidden="true">•</span>
                    <span>{c}</span>
                  </li>
                ))}
              </ul>
            )}
          </>
        )}
      </div>

      {profile.history_status && profile.history_status !== 'established' && (
        <p className="md:col-span-2 text-xs" style={{ color: 'var(--risk-medium)' }}>
          Limited history ({plural(profile.n_transactions, 'transaction')}): these statistics are not yet a reliable
          behavioral baseline (at least 10 transactions are needed).
        </p>
      )}

      <p className="md:col-span-2 text-[10px]" style={{ color: 'var(--text-faint)' }}>
        Compared with this customer's own history{baseline ? ` (${plural(baseline.n_transactions, 'transaction')} before this scan)` : ''}.
        This describes differences from past behavior; it does not change the model's scores.
      </p>
    </div>
  )
}
