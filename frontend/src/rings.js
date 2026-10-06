// Groups the rings returned by GET /fraud-rings into clusters: two rings that
// share a customer belong to the same cluster. Everything here is derived from
// the backend's rings (identifier, customer_ids, transaction_count).

export const RING_TYPE_LABEL = {
  shared_device: 'Shared device',
  shared_location: 'Shared location',
}

export function ringTypeLabel(type) {
  return RING_TYPE_LABEL[type] || String(type || 'Shared identifier').replaceAll('_', ' ')
}

// Display severity of a cluster, from its size only. It is a reading aid for
// the analyst, not a model output.
export function clusterSeverity(cluster) {
  const customers = cluster.customers.length
  const devices = cluster.devices.length
  if (customers >= 4 || devices >= 3) return 'Critical Risk'
  if (customers >= 3 || devices >= 2) return 'High Risk'
  return 'Medium Risk'
}

export function buildClusters(rings) {
  const parent = new Map()
  const find = (x) => {
    while (parent.get(x) !== x) {
      parent.set(x, parent.get(parent.get(x)))
      x = parent.get(x)
    }
    return x
  }
  const union = (a, b) => {
    const ra = find(a)
    const rb = find(b)
    if (ra !== rb) parent.set(ra < rb ? rb : ra, ra < rb ? ra : rb)
  }
  rings.forEach((ring) => {
    const d = `d:${ring.identifier}`
    if (!parent.has(d)) parent.set(d, d)
    ring.customer_ids.forEach((c) => {
      const k = `c:${c}`
      if (!parent.has(k)) parent.set(k, k)
      union(d, k)
    })
  })
  const groups = new Map()
  rings.forEach((ring) => {
    const root = find(`d:${ring.identifier}`)
    if (!groups.has(root)) groups.set(root, [])
    groups.get(root).push(ring)
  })
  const clusters = [...groups.values()].map((members) => {
    const customers = [...new Set(members.flatMap((r) => r.customer_ids))].sort()
    const devices = members.map((r) => r.identifier).sort()
    const cluster = {
      id: devices[0],
      rings: members,
      devices,
      customers,
      transactions: members.reduce((n, r) => n + (r.transaction_count || 0), 0),
      types: [...new Set(members.map((r) => r.ring_type))],
    }
    cluster.severity = clusterSeverity(cluster)
    return cluster
  })
  const rank = { 'Critical Risk': 3, 'High Risk': 2, 'Medium Risk': 1 }
  clusters.sort((a, b) =>
    rank[b.severity] - rank[a.severity] || b.customers.length - a.customers.length ||
    b.transactions - a.transactions || a.id.localeCompare(b.id))
  return clusters
}

// Node positions for one cluster inside a w x h box: devices near the centre,
// customers on an ellipse around them. Deterministic.
export function layoutCluster(cluster, w, h) {
  const cx = w / 2
  const cy = h / 2
  const nodes = {}
  const nd = cluster.devices.length
  cluster.devices.forEach((d, i) => {
    if (nd === 1) nodes[`d:${d}`] = { x: cx, y: cy }
    else {
      const a = (2 * Math.PI * i) / nd - Math.PI / 2
      nodes[`d:${d}`] = { x: cx + Math.cos(a) * w * 0.14, y: cy + Math.sin(a) * h * 0.16 }
    }
  })
  const nc = cluster.customers.length
  cluster.customers.forEach((c, i) => {
    const a = (2 * Math.PI * i) / nc + (nc === 2 ? 0 : -Math.PI / 2) + (nd > 1 ? Math.PI / nc : 0)
    nodes[`c:${c}`] = { x: cx + Math.cos(a) * w * 0.36, y: cy + Math.sin(a) * h * 0.34 }
  })
  const edges = cluster.rings.flatMap((r) => r.customer_ids.map((c) => ({ from: `d:${r.identifier}`, to: `c:${c}` })))
  return { nodes, edges }
}
