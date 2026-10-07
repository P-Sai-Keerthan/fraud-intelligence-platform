import axios from 'axios'

// In dev, Vite proxies /api -> http://localhost:8000 (see vite.config.js).
// For a production build, set VITE_API_BASE_URL to your deployed backend URL.
const baseURL = import.meta.env.VITE_API_BASE_URL || '/api'

export const api = axios.create({ baseURL, timeout: 15000 })

export async function predictTransaction(payload) {
  const { data } = await api.post('/predict', payload)
  return data
}

export async function getCustomerHistory(customerId) {
  const { data } = await api.get(`/customer/${customerId}/history`)
  return data
}

export async function getCustomerProfile(customerId) {
  const { data } = await api.get(`/customer/${encodeURIComponent(customerId)}/profile`)
  return data
}

// Turns a FastAPI error body into one readable sentence (422 detail is a list of
// {loc, msg}; 4xx from the batch endpoint is already a string).
export function formatApiError(err, fallback) {
  const detail = err?.response?.data?.detail
  if (!detail) return fallback
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    return detail
      .map((d) => `${(d.loc || []).filter((p) => p !== 'body').join('.') || 'request'}: ${d.msg}`)
      .join('; ')
  }
  return fallback
}

export async function listCustomers(limit = 100) {
  const { data } = await api.get(`/customers?limit=${limit}`)
  return data
}

export async function getMetrics() {
  // the first call re-scores the held-out test set (4-10 s on CPU), so allow more than the 15 s default
  const { data } = await api.get('/metrics', { timeout: 60000 })
  return data
}

export async function getFraudRings() {
  const { data } = await api.get('/fraud-rings')
  return data
}

export async function predictBatch(file) {
  const formData = new FormData()
  formData.append('file', file)
  // batch scoring takes ~0.6 s per row (up to 50 rows), so allow far more than the 15 s default
  const { data } = await api.post('/predict/batch', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 120000,
  })
  return data
}

export async function downloadReportPdf(prediction) {
  const response = await api.post('/report/pdf', prediction, { responseType: 'blob' })
  const url = URL.createObjectURL(new Blob([response.data], { type: 'application/pdf' }))
  const link = document.createElement('a')
  link.href = url
  link.download = `fraud_report_${prediction.transaction_id}.pdf`
  document.body.appendChild(link)
  link.click()
  link.remove()
  URL.revokeObjectURL(url)
}
