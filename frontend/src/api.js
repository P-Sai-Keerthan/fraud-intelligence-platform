import axios from 'axios'

// In dev, Vite proxies /api -> http://localhost:8000 (see vite.config.js).
// For a production build, set VITE_API_BASE_URL to your deployed backend URL.
const baseURL = import.meta.env.VITE_API_BASE_URL || '/api'

export const api = axios.create({ baseURL, timeout: 15000 })

// Identical GETs that are in flight at the same moment share ONE request. React StrictMode runs every
// mount effect twice in development (and the Refresh buttons can be double-clicked), which used to send
// the same slow request (e.g. /metrics, 4-10 s) twice. Only read-only calls use this.
const inflight = new Map()
function shareInflight(key, request) {
  if (!inflight.has(key)) inflight.set(key, request().finally(() => inflight.delete(key)))
  return inflight.get(key)
}

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

export function listCustomers(limit = 100) {
  return shareInflight(`customers:${limit}`, async () => (await api.get(`/customers?limit=${limit}`)).data)
}

export function getMetrics() {
  // the first call re-scores the held-out test set (4-10 s on CPU), so allow more than the 15 s default
  return shareInflight('metrics', async () => (await api.get('/metrics', { timeout: 60000 })).data)
}

export function getFraudRings() {
  return shareInflight('rings', async () => (await api.get('/fraud-rings')).data)
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
  // only the id is sent: every value printed in the report comes from the server's own record of the prediction
  let response
  try {
    response = await api.post('/report/pdf', { transaction_id: prediction.transaction_id }, { responseType: 'blob' })
  } catch (err) {
    // with responseType 'blob' an error body arrives as a Blob; decode it so formatApiError can read the message
    if (err?.response?.data instanceof Blob) {
      try {
        err.response.data = JSON.parse(await err.response.data.text())
      } catch {
        // leave the body as is; the caller falls back to a generic message
      }
    }
    throw err
  }
  const url = URL.createObjectURL(new Blob([response.data], { type: 'application/pdf' }))
  const link = document.createElement('a')
  link.href = url
  link.download = `fraud_report_${prediction.transaction_id}.pdf`
  document.body.appendChild(link)
  link.click()
  link.remove()
  URL.revokeObjectURL(url)
}
