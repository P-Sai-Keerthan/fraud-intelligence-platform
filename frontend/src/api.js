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

export async function listCustomers(limit = 100) {
  const { data } = await api.get(`/customers?limit=${limit}`)
  return data
}
