import { useEffect, useState, useCallback, useRef } from 'react'
import Header from './components/Header'
import CustomerSelector from './components/CustomerSelector'
import TransactionForm from './components/TransactionForm'
import RiskGauge from './components/RiskGauge'
import AlertBanner from './components/AlertBanner'
import ShapReasonsChart from './components/ShapReasonsChart'
import FraudEvolutionTimeline from './components/FraudEvolutionTimeline'
import SimilarityMeter from './components/SimilarityMeter'
import BehavioralAnalysis from './components/BehavioralAnalysis'
import ModelPerformance from './components/ModelPerformance'
import FraudRings from './components/FraudRings'
import BatchScoring from './components/BatchScoring'
import { Badge, Card, ErrorState, Icon } from './components/ui'
import {
  predictTransaction, getCustomerHistory, getCustomerProfile, listCustomers, downloadReportPdf, getModelInfo, getHealth,
} from './api'
import { scoreWording } from './modelWording'
import { formatApiError } from './apiError'

const TABS = [
  { key: 'live', label: 'Live Scan', icon: 'scan' },
  { key: 'batch', label: 'Batch Scoring', icon: 'layers' },
  { key: 'rings', label: 'Fraud Rings', icon: 'network' },
  { key: 'metrics', label: 'Model Performance', icon: 'activity' },
]

const HEALTH_POLL_MS = 30000

function TabBar({ active, onChange }) {
  const refs = useRef({})
  const onKeyDown = (e) => {
    const i = TABS.findIndex((t) => t.key === active)
    let next = null
    if (e.key === 'ArrowRight') next = TABS[(i + 1) % TABS.length]
    if (e.key === 'ArrowLeft') next = TABS[(i - 1 + TABS.length) % TABS.length]
    if (e.key === 'Home') next = TABS[0]
    if (e.key === 'End') next = TABS[TABS.length - 1]
    if (next) {
      e.preventDefault()
      onChange(next.key)
      refs.current[next.key]?.focus()
    }
  }
  return (
    <div
      role="tablist" aria-label="Sections" onKeyDown={onKeyDown}
      className="inline-flex max-w-full overflow-x-auto p-1 rounded-xl border gap-1"
      style={{ borderColor: 'var(--border)', background: 'rgba(10, 16, 27, 0.8)' }}
    >
      {TABS.map((tab) => {
        const isActive = active === tab.key
        return (
          <button
            key={tab.key} role="tab" type="button" id={`tab-${tab.key}`}
            aria-selected={isActive} aria-controls={`panel-${tab.key}`} tabIndex={isActive ? 0 : -1}
            ref={(el) => { refs.current[tab.key] = el }}
            onClick={() => onChange(tab.key)}
            className="display flex items-center gap-2 px-4 h-9 rounded-lg text-[13px] whitespace-nowrap transition-colors"
            style={{
              fontWeight: 600,
              color: isActive ? 'var(--text-primary)' : 'var(--text-muted)',
              background: isActive ? 'linear-gradient(180deg, rgba(56,189,248,0.18), rgba(56,189,248,0.08))' : 'transparent',
              boxShadow: isActive ? 'inset 0 0 0 1px rgba(56,189,248,0.38)' : 'none',
            }}
          >
            <Icon name={tab.icon} size={15} style={{ color: isActive ? 'var(--brand)' : 'var(--text-faint)' }} />
            {tab.label}
          </button>
        )
      })}
    </div>
  )
}

function PageHeading({ title, children }) {
  return (
    <div className="mb-5">
      <h2 className="display text-xl" style={{ fontWeight: 600 }}>{title}</h2>
      {children && <p className="text-[13px] mt-1 max-w-3xl" style={{ color: 'var(--text-muted)' }}>{children}</p>}
    </div>
  )
}

export default function App() {
  const [activeTab, setActiveTab] = useState('live')
  const [customers, setCustomers] = useState([])
  const [customersLoading, setCustomersLoading] = useState(true)
  const [selectedCustomer, setSelectedCustomer] = useState('')
  const [profile, setProfile] = useState(null)
  const [profileLoading, setProfileLoading] = useState(false)
  const [profileError, setProfileError] = useState('')
  const [prediction, setPrediction] = useState(null)
  const [timeline, setTimeline] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [customersError, setCustomersError] = useState('')
  const [reportLoading, setReportLoading] = useState(false)
  const [reportError, setReportError] = useState('')
  const [modelInfo, setModelInfo] = useState(null)
  const [health, setHealth] = useState('checking')
  const wording = scoreWording(modelInfo)

  // backend liveness for the header (GET /health), re-checked periodically
  useEffect(() => {
    let cancelled = false
    const check = () => {
      getHealth()
        .then((d) => { if (!cancelled) setHealth(d?.status === 'ok' ? 'online' : 'offline') })
        .catch(() => { if (!cancelled) setHealth('offline') })
    }
    check()
    const id = setInterval(check, HEALTH_POLL_MS)
    return () => { cancelled = true; clearInterval(id) }
  }, [])

  useEffect(() => {
    getModelInfo().then(setModelInfo).catch(() => setModelInfo(null))
  }, [])

  const loadCustomers = useCallback(() => {
    setCustomersLoading(true)
    setCustomersError('')
    listCustomers(500)
      .then((data) => setCustomers(data.customer_ids))
      .catch(() =>
        setCustomersError(
          'Could not reach the backend at /api. Make sure the FastAPI server is running (see README).'
        )
      )
      .finally(() => setCustomersLoading(false))
  }, [])

  useEffect(() => { loadCustomers() }, [loadCustomers])

  const refreshHistory = useCallback(async (customerId) => {
    try {
      const data = await getCustomerHistory(customerId)
      setTimeline(data.timeline)
    } catch {
      setTimeline([]) // 404 means no scored transactions yet - that's fine
    }
  }, [])

  useEffect(() => {
    if (selectedCustomer) {
      setPrediction(null)
      setError('')
      setTimeline([])
      refreshHistory(selectedCustomer)
    }
  }, [selectedCustomer, refreshHistory])

  // the selected customer's usual device + home city, used as the form's
  // defaults for a normal transaction
  useEffect(() => {
    setProfile(null)
    setProfileError('')
    if (!selectedCustomer) return
    let cancelled = false
    setProfileLoading(true)
    getCustomerProfile(selectedCustomer)
      .then((data) => { if (!cancelled) setProfile(data) })
      .catch(() => {
        if (!cancelled) {
          setProfileError("Could not load this customer's usual device and city. Enter them manually.")
        }
      })
      .finally(() => { if (!cancelled) setProfileLoading(false) })
    return () => { cancelled = true }
  }, [selectedCustomer])

  const handleSubmit = async (payload) => {
    setLoading(true)
    setError('')
    try {
      const result = await predictTransaction(payload)
      setPrediction(result)
      setReportError('')
      await refreshHistory(payload.customer_id)
      // the history count in the profile grows with every scan
      getCustomerProfile(payload.customer_id).then(setProfile).catch(() => {})
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
    } catch {
      setReportError('Could not generate the report. Is the backend running?')
    } finally {
      setReportLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex flex-col">
      <Header health={health} modelInfo={modelInfo} />

      <main className="flex-1 w-full max-w-[1480px] mx-auto px-5 lg:px-8 py-6">
        <div className="mb-6">
          <TabBar active={activeTab} onChange={setActiveTab} />
        </div>

        {activeTab === 'live' && (
          <div id="panel-live" role="tabpanel" aria-labelledby="tab-live"
            className="grid grid-cols-1 lg:grid-cols-[340px_minmax(0,1fr)] xl:grid-cols-[372px_minmax(0,1fr)] gap-5 items-start">
            {/* left column: who and what is being scanned */}
            <div className="space-y-5 sticky-tall">
              <Card title="Customer Profile" eyebrow="Investigation" icon="user">
                <CustomerSelector
                  customers={customers} value={selectedCustomer} onChange={setSelectedCustomer}
                  profile={profile} profileLoading={profileLoading} timeline={selectedCustomer ? timeline : null}
                  loadingCustomers={customersLoading}
                />
                {customersError && <ErrorState className="mt-3" onRetry={loadCustomers}>{customersError}</ErrorState>}
                {profileError && <ErrorState className="mt-3">{profileError}</ErrorState>}
              </Card>

              <Card title="Transaction Context" eyebrow="Input" icon="file">
                <TransactionForm customerId={selectedCustomer} profile={profile} onSubmit={handleSubmit} loading={loading} />
                {error && <ErrorState className="mt-3">{error}</ErrorState>}
              </Card>
            </div>

            {/* right column: verdict, scores, explanation */}
            <div className="space-y-5 min-w-0">
              <AlertBanner
                prediction={prediction} modelInfo={modelInfo} loading={loading}
                onDownload={handleDownloadReport} reportLoading={reportLoading}
              />
              {reportError && <ErrorState>{reportError}</ErrorState>}

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-5">
                <Card>
                  <RiskGauge
                    label={wording.riskLabel} value={prediction?.risk_score} loading={loading}
                    caption={wording.riskShort} note={<span title={wording.riskCaption}>From the previous 10 transactions</span>}
                  />
                </Card>
                <Card>
                  <RiskGauge
                    label="Fraud Score" value={prediction?.fraud_probability} decimals={1} loading={loading}
                    caption="Model score, not a calibrated probability." note={<span title={wording.fraudCaption}>For this transaction</span>}
                  />
                </Card>
                <Card>
                  <SimilarityMeter
                    similarityPct={prediction?.similarity_pct} deviationPct={prediction?.deviation_pct}
                    similarityStatus={prediction?.similarity_status} historyTransactions={prediction?.history_transactions} loading={loading}
                  />
                </Card>
              </div>

              <Card
                title="Why was this transaction flagged?" eyebrow="Explainable AI" icon="search"
                actions={<Badge tone="brand" icon="cpu">SHAP / Explainable AI</Badge>}
              >
                <ShapReasonsChart reasons={prediction?.reasons} hasPrediction={!!prediction} />
              </Card>

              <div className="grid grid-cols-1 2xl:grid-cols-2 gap-5">
                <Card title="Behavioral Analysis" eyebrow="Baseline vs. transaction" icon="user">
                  <BehavioralAnalysis profile={selectedCustomer ? profile : null} prediction={prediction} />
                </Card>
                <Card title="Fraud Evolution Timeline" eyebrow="Score history" icon="activity">
                  <FraudEvolutionTimeline timeline={timeline} riskName={wording.timelineRisk} />
                </Card>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'batch' && (
          <div id="panel-batch" role="tabpanel" aria-labelledby="tab-batch">
            <PageHeading title="Batch Scoring">
              Score a CSV of transactions in one pass and triage the results by alert level.
            </PageHeading>
            <BatchScoring riskColumn={wording.riskColumn} />
          </div>
        )}

        {activeTab === 'rings' && (
          <div id="panel-rings" role="tabpanel" aria-labelledby="tab-rings">
            <PageHeading title="Fraud Ring Investigation">
              Customers linked by a shared device. A legitimate customer&apos;s devices belong to them alone, so a device
              used by several accounts points to coordinated activity.
            </PageHeading>
            <FraudRings />
          </div>
        )}

        {activeTab === 'metrics' && (
          <div id="panel-metrics" role="tabpanel" aria-labelledby="tab-metrics">
            <PageHeading title="Model Performance">
              Evaluation of the loaded model set, and the technical details of what is running.
            </PageHeading>
            <ModelPerformance modelInfo={modelInfo} />
          </div>
        )}
      </main>

      <footer className="border-t" style={{ borderColor: 'var(--border-subtle)' }}>
        <div className="max-w-[1480px] mx-auto px-5 lg:px-8 py-3 flex flex-wrap items-center justify-between gap-2">
          <span className="mono text-[10.5px]" style={{ color: 'var(--text-faint)' }}>
            Fraud Intelligence Platform &middot; scores are model outputs, not calibrated probabilities
          </span>
          <span className="mono text-[10.5px]" style={{ color: 'var(--text-faint)' }}>
            {modelInfo ? `${modelInfo.model_set} · ${modelInfo.model_version}` : 'model set unknown'}
          </span>
        </div>
      </footer>
    </div>
  )
}
