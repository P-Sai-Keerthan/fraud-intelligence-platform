import { useEffect, useState, useCallback, useRef } from 'react'
import Header from './components/Header'
import CustomerSelector from './components/CustomerSelector'
import TransactionForm from './components/TransactionForm'
import RiskGauge from './components/RiskGauge'
import AlertBanner from './components/AlertBanner'
import ShapReasonsChart from './components/ShapReasonsChart'
import FraudEvolutionTimeline from './components/FraudEvolutionTimeline'
import SimilarityMeter from './components/SimilarityMeter'
import ModelPerformance from './components/ModelPerformance'
import FraudRings from './components/FraudRings'
import BehavioralContext from './components/BehavioralContext'
import BatchScoring from './components/BatchScoring'
import { plural } from './format'
import {
  predictTransaction, getCustomerHistory, getCustomerProfile, listCustomers, downloadReportPdf, formatApiError,
} from './api'

const TABS = [
  { key: 'live', label: 'Live Scan' },
  { key: 'batch', label: 'Batch Scoring' },
  { key: 'rings', label: 'Fraud Rings' },
  { key: 'metrics', label: 'Model Performance' },
]

// These four sections swap the content below IN PLACE (no page or URL change), so this is a genuine tab
// interface: roving tabindex + arrow/Home/End keys, as in the WAI-ARIA tabs pattern.
function TabBar({ active, onChange }) {
  const buttons = useRef({})

  const onKeyDown = (e) => {
    const current = TABS.findIndex((t) => t.key === active)
    let next = null
    if (e.key === 'ArrowRight') next = (current + 1) % TABS.length
    else if (e.key === 'ArrowLeft') next = (current - 1 + TABS.length) % TABS.length
    else if (e.key === 'Home') next = 0
    else if (e.key === 'End') next = TABS.length - 1
    if (next === null) return
    e.preventDefault()
    onChange(TABS[next].key)
    buttons.current[TABS[next].key]?.focus()
  }

  return (
    <div
      role="tablist"
      aria-label="Dashboard sections"
      onKeyDown={onKeyDown}
      className="flex gap-1 mb-6 border-b"
      style={{ borderColor: 'var(--border)' }}
    >
      {TABS.map((tab) => {
        const isActive = active === tab.key
        return (
          <button
            key={tab.key}
            ref={(el) => { buttons.current[tab.key] = el }}
            type="button"
            role="tab"
            id={`tab-${tab.key}`}
            aria-selected={isActive}
            aria-controls={isActive ? `panel-${tab.key}` : undefined} // only the open panel exists in the page
            tabIndex={isActive ? 0 : -1}
            onClick={() => onChange(tab.key)}
            className="px-4 py-2.5 text-sm transition-colors relative -mb-px"
            style={{
              color: isActive ? 'var(--brand)' : 'var(--text-muted)',
              fontFamily: 'var(--font-display)',
              borderBottom: isActive ? '2px solid var(--brand)' : '2px solid transparent',
            }}
          >
            {tab.label}
          </button>
        )
      })}
    </div>
  )
}

function TabPanel({ tab, children }) {
  return (
    <div role="tabpanel" id={`panel-${tab}`} aria-labelledby={`tab-${tab}`}>
      {children}
    </div>
  )
}

const cardStyle = {
  background: 'var(--surface)',
  borderColor: 'var(--border)',
}

function Card({ title, children, className = '' }) {
  const Tag = title ? 'section' : 'div' // a <section> needs a heading; the untitled gauge cards are plain boxes
  return (
    <Tag className={`rounded-xl border p-5 ${className}`} style={cardStyle}>
      {title && (
        <h2
          className="text-sm mb-4 uppercase tracking-wide font-normal"
          style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}
        >
          {title}
        </h2>
      )}
      {children}
    </Tag>
  )
}

export default function App() {
  const [activeTab, setActiveTab] = useState('live')
  const [customers, setCustomers] = useState([])
  const [customersLoading, setCustomersLoading] = useState(true)
  const [selectedCustomer, setSelectedCustomer] = useState('')
  const [prediction, setPrediction] = useState(null)
  const [timeline, setTimeline] = useState([])
  const [timelineError, setTimelineError] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [customersError, setCustomersError] = useState('')
  const [reportLoading, setReportLoading] = useState(false)
  const [reportError, setReportError] = useState('')
  const [profile, setProfile] = useState(null)
  const [profileStatus, setProfileStatus] = useState('idle') // 'idle' | 'loading' | 'ready' | 'error'
  const [scanContext, setScanContext] = useState(null)
  const selectedRef = useRef('')

  useEffect(() => {
    let active = true
    listCustomers(500)
      .then((data) => { if (active) setCustomers(data.customer_ids) })
      .catch(() => {
        if (active) {
          setCustomersError('Could not reach the backend at /api. Make sure the FastAPI server is running (see README).')
        }
      })
      .finally(() => { if (active) setCustomersLoading(false) })
    return () => { active = false }
  }, [])

  const refreshHistory = useCallback(async (customerId) => {
    try {
      const data = await getCustomerHistory(customerId)
      if (selectedRef.current === customerId) { setTimeline(data.timeline); setTimelineError('') }
    } catch (err) {
      if (selectedRef.current !== customerId) return
      setTimeline([])
      // 404 is the NORMAL answer for a customer with no scanned transactions yet (the API's way of saying
      // "empty timeline"), so it is not an error. (The browser console still logs the 404 line itself;
      // that is the browser's network log, not an application error.) Anything else is a real failure.
      setTimelineError(err?.response?.status === 404 ? '' : "Could not load this customer's timeline.")
    }
  }, [])

  const refreshProfile = useCallback(async (customerId) => {
    try {
      const data = await getCustomerProfile(customerId)
      if (selectedRef.current === customerId) { setProfile(data); setProfileStatus('ready') }
    } catch {
      if (selectedRef.current === customerId) { setProfile(null); setProfileStatus('error') }
    }
  }, [])

  useEffect(() => {
    selectedRef.current = selectedCustomer
    setPrediction(null)
    setScanContext(null)
    setProfile(null)
    setTimeline([])
    setTimelineError('')
    setProfileStatus(selectedCustomer ? 'loading' : 'idle')
    if (selectedCustomer) {
      refreshHistory(selectedCustomer)
      refreshProfile(selectedCustomer)
    }
  }, [selectedCustomer, refreshHistory, refreshProfile])

  const handleSubmit = async (payload) => {
    const baseline = profile // the customer's behavior BEFORE this scan (for "What changed?")
    setLoading(true)
    setError('')
    try {
      const result = await predictTransaction(payload)
      if (selectedRef.current !== payload.customer_id) return // user switched customer while scanning
      setPrediction(result)
      setScanContext(baseline ? { baseline, prediction: result } : null)
      setReportError('')
      await Promise.all([refreshHistory(payload.customer_id), refreshProfile(payload.customer_id)])
    } catch (err) {
      setError(formatApiError(err, 'Prediction request failed. Is the backend running?'))
    } finally {
      setLoading(false)
    }
  }

  const handleDownloadReport = async () => {
    if (!prediction) return
    setReportLoading(true)
    setReportError('')
    try {
      await downloadReportPdf(prediction)
    } catch (err) {
      setReportError(formatApiError(err, 'Could not generate the report. Is the backend running?'))
    } finally {
      setReportLoading(false)
    }
  }

  // Spoken to screen-reader users when a scan starts/finishes (the visible result is a chart-like dashboard).
  const scanAnnouncement = loading
    ? 'Scanning transaction.'
    : prediction
      ? `Scan complete for ${prediction.customer_id}. Fraud Risk Score ${prediction.fraud_probability.toFixed(1)}, ${prediction.alert_level}.`
      : ''

  return (
    <div className="min-h-screen" style={{ background: 'var(--bg)' }}>
      <Header selectedCustomer={selectedCustomer} />

      <main className="max-w-7xl mx-auto px-6 py-8">
        <div role="status" aria-live="polite" className="sr-only">{scanAnnouncement}</div>

        <TabBar active={activeTab} onChange={setActiveTab} />

        {activeTab === 'live' && (
          <TabPanel tab="live">
            <div className="grid grid-cols-1 lg:grid-cols-[320px_1fr] gap-6">
              {/* left column: controls */}
              <div className="space-y-6">
                <Card title="Select Customer">
                  <CustomerSelector
                    customers={customers}
                    value={selectedCustomer}
                    onChange={setSelectedCustomer}
                    loading={customersLoading}
                  />
                  {customersError && (
                    <p role="alert" className="text-xs mt-2" style={{ color: 'var(--risk-critical)' }}>{customersError}</p>
                  )}
                </Card>

                <Card title="Scan a Transaction">
                  <TransactionForm customerId={selectedCustomer} profile={profile} onSubmit={handleSubmit} loading={loading} />
                  {!selectedCustomer && (
                    <p className="text-xs mt-3" style={{ color: 'var(--text-faint)' }}>
                      Select a customer above to enable scanning.
                    </p>
                  )}
                  {error && <p role="alert" className="text-xs mt-3" style={{ color: 'var(--risk-critical)' }}>{error}</p>}
                </Card>
              </div>

              {/* right column: results dashboard */}
              <div className="space-y-6">
                {prediction && (
                  <div className="flex items-center gap-3">
                    <div className="flex-1"><AlertBanner level={prediction.alert_level} /></div>
                    <button
                      type="button"
                      onClick={handleDownloadReport}
                      disabled={reportLoading}
                      className="text-xs px-4 py-2.5 rounded-lg border shrink-0 transition-colors hover:border-[var(--brand)] disabled:opacity-40"
                      style={{ borderColor: 'var(--border)', color: 'var(--text-primary)', background: 'var(--surface)', fontFamily: 'var(--font-display)' }}
                    >
                      {reportLoading ? 'Generating…' : 'Download Report'}
                    </button>
                  </div>
                )}
                {reportError && <p role="alert" className="text-xs" style={{ color: 'var(--risk-critical)' }}>{reportError}</p>}
                {prediction && prediction.history_status && prediction.history_status !== 'established' && (
                  <p
                    className="text-xs rounded-lg border px-3 py-2"
                    style={{ borderColor: 'var(--risk-medium)', background: 'var(--risk-medium-dim)', color: 'var(--text-muted)' }}
                  >
                    <strong style={{ color: 'var(--text-primary)' }}>Limited behavioral history</strong>
                    {' '}({plural(prediction.history_transactions, 'prior transaction')}).
                    Temporal risk and behavioral similarity need at least 10 prior transactions and are not available.
                    This score uses only the transaction-level signals that exist, so treat it as a limited-confidence
                    result, not as a behavioral assessment.
                  </p>
                )}

                <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
                  <Card>
                    <RiskGauge label="Temporal Risk" value={prediction?.risk_score} sublabel="/ 100" />
                    <p className="text-[10px] text-center mt-2" style={{ color: 'var(--text-faint)' }}>
                      {prediction && prediction.risk_score == null
                        ? 'Not available: the LSTM needs at least 10 prior transactions for this customer.'
                        : "LSTM score from this customer's previous transactions (before this one). A model score, not a probability."}
                    </p>
                  </Card>
                  <Card>
                    <RiskGauge label="Fraud Risk Score" value={prediction?.fraud_probability} sublabel="/ 100" decimals={1} />
                    <p className="text-[10px] text-center mt-2" style={{ color: 'var(--text-faint)' }}>
                      DNN score for this transaction. Not a calibrated probability; it reflects combinations of signals, not any single one.
                    </p>
                  </Card>
                  <Card>
                    <div className="h-full flex flex-col justify-center">
                      <SimilarityMeter
                        similarityPct={prediction?.similarity_pct}
                        deviationPct={prediction?.deviation_pct}
                        unavailable={Boolean(prediction) && prediction.similarity_pct == null}
                      />
                    </div>
                  </Card>
                </div>

                <Card title="Behavioral Context &middot; What Changed?">
                  <BehavioralContext profile={profile} context={scanContext} status={profileStatus} />
                </Card>

                <Card title="Explainable AI &middot; Why This Score">
                  <ShapReasonsChart reasons={prediction?.reasons} />
                  <p className="text-[10px] mt-3" style={{ color: 'var(--text-faint)' }}>
                    Bars show how much each factor raised the model's score (SHAP). A factor is only named as a
                    condition (e.g. "New Device") when this transaction actually shows it. SHAP explains the model,
                    not the cause of fraud.
                  </p>
                </Card>

                <Card title="Fraud Risk Timeline">
                  <FraudEvolutionTimeline timeline={timeline} />
                  {timelineError && (
                    <p role="alert" className="text-xs mt-3" style={{ color: 'var(--risk-critical)' }}>{timelineError}</p>
                  )}
                  <p className="text-[10px] mt-3" style={{ color: 'var(--text-faint)' }}>
                    Scores of the transactions scanned for this customer, ordered by transaction time.
                  </p>
                </Card>
              </div>
            </div>
          </TabPanel>
        )}

        {activeTab === 'batch' && (
          <TabPanel tab="batch">
            <Card title="Batch CSV Upload &amp; Bulk Scoring">
              <BatchScoring />
            </Card>
          </TabPanel>
        )}

        {activeTab === 'rings' && (
          <TabPanel tab="rings">
            <Card title="Fraud Ring Detection">
              <FraudRings />
            </Card>
          </TabPanel>
        )}

        {activeTab === 'metrics' && (
          <TabPanel tab="metrics">
            <Card title="Model Performance">
              <ModelPerformance />
            </Card>
          </TabPanel>
        )}
      </main>
    </div>
  )
}
