// Turns an axios error into a message a person can read.
// FastAPI returns `detail` as a string (404, 400, 413 ...) or, for validation errors (422), as a list of
// {type, loc, msg, input}; the list used to be shown as raw JSON.
export function formatApiError(err, fallback = 'The request failed. Is the backend running?') {
  const detail = err?.response?.data?.detail
  if (detail == null || detail === '') {
    if (err?.response?.status) return `${fallback} (HTTP ${err.response.status})`
    return fallback
  }
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    const parts = detail.map((d) => {
      const where = Array.isArray(d?.loc) ? d.loc.filter((x) => x !== 'body').join('.') : ''
      const msg = d?.msg ?? 'invalid value'
      return where ? `${where}: ${msg}` : msg
    })
    return parts.join('; ')
  }
  return String(detail)
}
