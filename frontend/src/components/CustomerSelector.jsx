export default function CustomerSelector({ customers, value, onChange }) {
  return (
    <div>
      <label
        className="text-xs uppercase tracking-wide mb-1.5 block"
        style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}
      >
        Customer Profile
      </label>
      <select
        className="w-full rounded-lg px-3 py-2 text-sm outline-none border transition-colors focus:border-[var(--brand)]"
        style={{ background: 'var(--surface)', borderColor: 'var(--border)', color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}
        value={value}
        onChange={(e) => onChange(e.target.value)}
      >
        <option value="">Select a customer…</option>
        {customers.map((c) => (
          <option key={c} value={c}>{c}</option>
        ))}
      </select>
    </div>
  )
}
