import { useEffect, useState, useCallback } from 'react'
import Header from './components/Header'
import CustomerSelector from './components/CustomerSelector'
import TransactionForm from './components/TransactionForm'
import RiskGauge from './components/RiskGauge'
import AlertBanner from './components/AlertBanner'
import ShapReasonsChart from './components/ShapReasonsChart'
import FraudEvolutionTimeline from './components/FraudEvolutionTimeline'
import SimilarityMeter from './components/SimilarityMeter'
import { predictTransaction, getCustomerHistory, listCustomers } from './api'

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
  const [customers, setCustomers] = useState([])
  const [selectedCustomer, setSelectedCustomer] = useState('')
  const [prediction, setPrediction] = useState(null)
  const [timeline, setTimeline] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [customersError, setCustomersError] = useState('')

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

  const handleSubmit = async (payload) => {
    setLoading(true)
    setError('')
    try {
      const result = await predictTransaction(payload)
      setPrediction(result)
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

  return (
    <div className="min-h-screen" style={{ background: 'var(--bg)' }}>
      <Header selectedCustomer={selectedCustomer} />

      <main className="max-w-7xl mx-auto px-6 py-8 grid grid-cols-1 lg:grid-cols-[320px_1fr] gap-6">
        {/* left column: controls */}
        <div className="space-y-6">
          <Card title="Select Customer">
            <CustomerSelector customers={customers} value={selectedCustomer} onChange={setSelectedCustomer} />
            {customersError && (
              <p className="text-xs mt-2" style={{ color: 'var(--risk-critical)' }}>{customersError}</p>
            )}
          </Card>

          <Card title="Scan a Transaction">
            <TransactionForm customerId={selectedCustomer} onSubmit={handleSubmit} loading={loading} />
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
          {prediction && <AlertBanner level={prediction.alert_level} />}

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
            <Card>
              <RiskGauge label="Risk Score" value={prediction?.risk_score} sublabel="/ 100" />
              <p className="text-[10px] text-center mt-2" style={{ color: 'var(--text-faint)' }}>
                Trajectory risk from this customer's prior activity, before this transaction
              </p>
            </Card>
            <Card>
              <RiskGauge label="Fraud Probability" value={prediction?.fraud_probability} sublabel="%" decimals={1} />
              <p className="text-[10px] text-center mt-2" style={{ color: 'var(--text-faint)' }}>
                This specific transaction's fraud likelihood -- can be high even if prior trajectory was clean
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
            <FraudEvolutionTimeline timeline={timeline} />
          </Card>
        </div>
      </main>
    </div>
  )
}
