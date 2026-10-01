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
  const { data } = await api.get(`/customer/${customerId}/profile`)
  return data
}

export async function listCustomers(limit = 100) {
  const { data } = await api.get(`/customers?limit=${limit}`)
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
  const { data } = await api.post('/predict/batch', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
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
