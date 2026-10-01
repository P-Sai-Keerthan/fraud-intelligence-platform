import { useEffect, useState, useCallback } from 'react'
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
import BatchScoring from './components/BatchScoring'
import { predictTransaction, getCustomerHistory, getCustomerProfile, listCustomers, downloadReportPdf, getModelInfo } from './api'
import { scoreWording, modelSetLabel } from './modelWording'

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
  const [profile, setProfile] = useState(null)
  const [profileError, setProfileError] = useState('')
  const [prediction, setPrediction] = useState(null)
  const [timeline, setTimeline] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [customersError, setCustomersError] = useState('')
  const [reportLoading, setReportLoading] = useState(false)
  const [reportError, setReportError] = useState('')
  const [modelInfo, setModelInfo] = useState(null)
  const wording = scoreWording(modelInfo)

  useEffect(() => {
    getModelInfo().then(setModelInfo).catch(() => setModelInfo(null))
  }, [])

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
      setTimeline(data.timeline)
    } catch {
      setTimeline([]) // 404 means no scored transactions yet - that's fine
    }
  }, [])

  useEffect(() => {
    if (selectedCustomer) {
      setPrediction(null)
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
    getCustomerProfile(selectedCustomer)
      .then((data) => { if (!cancelled) setProfile(data) })
      .catch(() => {
        if (!cancelled) {
          setProfileError("Could not load this customer's usual device and city. Enter them manually.")
        }
      })
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
    } catch (err) {
      setError(
        err?.response?.data?.detail
          ? JSON.stringify(err.response.data.detail)
          : 'Prediction request failed. Is the backend running?'
      )
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
                {profileError && (
                  <p className="text-xs mt-3" style={{ color: 'var(--risk-critical)' }}>{profileError}</p>
                )}
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

              <p className="text-[10px]" style={{ color: 'var(--text-faint)', fontFamily: 'var(--font-mono)' }}>
                Loaded {modelSetLabel(modelInfo)}
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
                <Card>
                  <RiskGauge label={wording.riskLabel} value={prediction?.risk_score} sublabel="/ 100" />
                  <p className="text-[10px] text-center mt-2" style={{ color: 'var(--text-faint)' }}>
                    {wording.riskCaption}
                  </p>
                </Card>
                <Card>
                  <RiskGauge label="Fraud Probability" value={prediction?.fraud_probability} sublabel="%" decimals={1} />
                  <p className="text-[10px] text-center mt-2" style={{ color: 'var(--text-faint)' }}>
                    {wording.fraudCaption}
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

              <Card title="Explainable AI &middot; Why This Score">
                <ShapReasonsChart reasons={prediction?.reasons} />
              </Card>

              <Card title="Fraud Evolution Timeline">
                <FraudEvolutionTimeline timeline={timeline} riskName={wording.timelineRisk} />
              </Card>
            </div>
          </div>
        )}

        {activeTab === 'batch' && (
          <Card title="Batch CSV Upload &amp; Bulk Scoring">
            <BatchScoring riskColumn={wording.riskColumn} />
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
