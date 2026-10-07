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
import {
  predictTransaction, getCustomerHistory, getCustomerProfile, listCustomers, downloadReportPdf, formatApiError,
} from './api'

const TABS = [
  { key: 'live', label: 'Live Scan' },
  { key: 'batch', label: 'Batch Scoring' },
  { key: 'rings', label: 'Fraud Rings' },
  { key: 'metrics', label: 'Model Performance' },
]

function TabBar({ active, onChange }) {
  return (
    <div className="flex gap-1 mb-6 border-b" style={{ borderColor: 'var(--border)' }}>
      {TABS.map((tab) => {
        const isActive = active === tab.key
        return (
          <button
            key={tab.key}
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

const cardStyle = {
  background: 'var(--surface)',
  borderColor: 'var(--border)',
}

function Card({ title, children, className = '' }) {
  return (
    <div className={`rounded-xl border p-5 ${className}`} style={cardStyle}>
      {title && (
        <h3
          className="text-sm mb-4 uppercase tracking-wide"
          style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}
        >
          {title}
        </h3>
      )}
      {children}
    </div>
  )
}

export default function App() {
  const [activeTab, setActiveTab] = useState('live')
  const [customers, setCustomers] = useState([])
  const [selectedCustomer, setSelectedCustomer] = useState('')
  const [prediction, setPrediction] = useState(null)
  const [timeline, setTimeline] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [customersError, setCustomersError] = useState('')
  const [reportLoading, setReportLoading] = useState(false)
  const [reportError, setReportError] = useState('')
  const [profile, setProfile] = useState(null)
  const [scanContext, setScanContext] = useState(null)
  const selectedRef = useRef('')

  useEffect(() => {
    listCustomers(500)
      .then((data) => setCustomers(data.customer_ids))
      .catch(() =>
        setCustomersError(
          'Could not reach the backend at /api. Make sure the FastAPI server is running (see README).'
        )
      )
  }, [])

  const refreshHistory = useCallback(async (customerId) => {
    try {
      const data = await getCustomerHistory(customerId)
      if (selectedRef.current === customerId) setTimeline(data.timeline)
    } catch {
      if (selectedRef.current === customerId) setTimeline([]) // 404 means no scored transactions yet - that's fine
    }
  }, [])

  const refreshProfile = useCallback(async (customerId) => {
    try {
      const data = await getCustomerProfile(customerId)
      if (selectedRef.current === customerId) setProfile(data)
    } catch {
      if (selectedRef.current === customerId) setProfile(null)
    }
  }, [])

  useEffect(() => {
    selectedRef.current = selectedCustomer
    setPrediction(null)
    setScanContext(null)
    setProfile(null)
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
    } catch {
      setReportError('Could not generate the report. Is the backend running?')
    } finally {
      setReportLoading(false)
    }
  }

  return (
    <div className="min-h-screen" style={{ background: 'var(--bg)' }}>
      <Header selectedCustomer={selectedCustomer} />

      <main className="max-w-7xl mx-auto px-6 py-8">
        <TabBar active={activeTab} onChange={setActiveTab} />

        {activeTab === 'live' && (
          <div className="grid grid-cols-1 lg:grid-cols-[320px_1fr] gap-6">
            {/* left column: controls */}
            <div className="space-y-6">
              <Card title="Select Customer">
                <CustomerSelector customers={customers} value={selectedCustomer} onChange={setSelectedCustomer} />
                {customersError && (
                  <p className="text-xs mt-2" style={{ color: 'var(--risk-critical)' }}>{customersError}</p>
                )}
              </Card>

              <Card title="Scan a Transaction">
                <TransactionForm customerId={selectedCustomer} profile={profile} onSubmit={handleSubmit} loading={loading} />
                {!selectedCustomer && (
                  <p className="text-xs mt-3" style={{ color: 'var(--text-faint)' }}>
                    Select a customer above to enable scanning.
                  </p>
                )}
                {error && <p className="text-xs mt-3" style={{ color: 'var(--risk-critical)' }}>{error}</p>}
              </Card>
            </div>

            {/* right column: results dashboard */}
            <div className="space-y-6">
              {prediction && (
                <div className="flex items-center gap-3">
                  <div className="flex-1"><AlertBanner level={prediction.alert_level} /></div>
                  <button
                    onClick={handleDownloadReport}
                    disabled={reportLoading}
                    className="text-xs px-4 py-2.5 rounded-lg border shrink-0 transition-colors hover:border-[var(--brand)] disabled:opacity-40"
                    style={{ borderColor: 'var(--border)', color: 'var(--text-primary)', background: 'var(--surface)', fontFamily: 'var(--font-display)' }}
                  >
                    {reportLoading ? 'Generating…' : 'Download Report'}
                  </button>
                </div>
              )}
              {reportError && <p className="text-xs" style={{ color: 'var(--risk-critical)' }}>{reportError}</p>}

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
                <Card>
                  <RiskGauge label="Temporal Risk" value={prediction?.risk_score} sublabel="/ 100" />
                  <p className="text-[10px] text-center mt-2" style={{ color: 'var(--text-faint)' }}>
                    LSTM score from this customer's previous transactions (before this one). A model score, not a probability.
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
                    />
                  </div>
                </Card>
              </div>

              <Card title="Behavioral Context &middot; What Changed?">
                <BehavioralContext profile={profile} context={scanContext} />
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
                <p className="text-[10px] mt-3" style={{ color: 'var(--text-faint)' }}>
                  Scores of the transactions scanned for this customer, ordered by transaction time.
                </p>
              </Card>
            </div>
          </div>
        )}

        {activeTab === 'batch' && (
          <Card title="Batch CSV Upload &amp; Bulk Scoring">
            <BatchScoring />
          </Card>
        )}

        {activeTab === 'rings' && (
          <Card title="Fraud Ring Detection">
            <FraudRings />
          </Card>
        )}

        {activeTab === 'metrics' && (
          <Card title="Model Performance">
            <ModelPerformance />
          </Card>
        )}
      </main>
    </div>
  )
}
