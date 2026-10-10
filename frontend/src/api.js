import axios from 'axios'

// In dev, Vite proxies /api -> http://localhost:8000 (see vite.config.js).
// For a production build, set VITE_API_BASE_URL to your deployed backend URL.
const baseURL = import.meta.env.VITE_API_BASE_URL || '/api'

export const api = axios.create({ baseURL, timeout: 15000 })

// Liveness of the backend (GET /health), used by the header status indicator.
export async function getHealth() {
  const { data } = await api.get('/health', { timeout: 5000 })
  return data
}

export async function predictTransaction(payload) {
  const { data } = await api.post('/predict', payload)
  return data
}

export async function getCustomerHistory(customerId) {
  const { data } = await api.get(`/customer/${encodeURIComponent(customerId)}/history`)
  return data
}

export async function getCustomerProfile(customerId) {
  const { data } = await api.get(`/customer/${encodeURIComponent(customerId)}/profile`)
  return data
}

export async function listCustomers(limit = 100) {
  const { data } = await api.get('/customers', { params: { limit } })
  return data
}

export async function getMetrics() {
  const { data } = await api.get('/metrics')
  return data
}

export async function getModelInfo() {
  const { data } = await api.get('/model-info')
  return data
}

export async function getFraudRings() {
  const { data } = await api.get('/fraud-rings')
  return data
}

export async function predictBatch(file) {
  const formData = new FormData()
  formData.append('file', file)
  // A batch is scored row by row (model + SHAP for each), so it routinely takes
  // longer than the 15 s default above; allow up to 5 minutes for this call only.
  const { data } = await api.post('/predict/batch', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 300000,
  })
  return data
}

export async function downloadReportPdf(prediction) {
  const response = await api.post('/report/pdf', prediction, { responseType: 'blob' })
  const url = URL.createObjectURL(new Blob([response.data], { type: 'application/pdf' }))
  const link = document.createElement('a')
  link.href = url
  link.download = `fraud_report_${String(prediction.transaction_id).replace(/[^A-Za-z0-9_.-]/g, '_')}.pdf`
  document.body.appendChild(link)
  link.click()
  link.remove()
  URL.revokeObjectURL(url)
}
