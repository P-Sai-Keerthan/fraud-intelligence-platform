export default function Header({ selectedCustomer }) {
  return (
    <header className="relative overflow-hidden border-b" style={{ borderColor: 'var(--border)' }}>
      {/* signature element: a drifting double-helix motif, representing
          "Behavioral Fraud DNA" - kept subtle so it reads as texture, not decoration */}
      <svg
        className="absolute inset-0 w-[140%] h-full opacity-[0.07] pointer-events-none"
        style={{ animation: 'helix-drift 18s linear infinite' }}
        aria-hidden="true"
        focusable="false"
        viewBox="0 0 800 160"
        preserveAspectRatio="none"
      >
        <path
          d="M0,80 C50,20 100,140 150,80 C200,20 250,140 300,80 C350,20 400,140 450,80 C500,20 550,140 600,80 C650,20 700,140 750,80 C800,20 850,140 900,80"
          fill="none" stroke="var(--brand)" strokeWidth="1.5"
        />
        <path
          d="M0,80 C50,140 100,20 150,80 C200,140 250,20 300,80 C350,140 400,20 450,80 C500,140 550,20 600,80 C650,140 700,20 750,80 C800,140 850,20 900,80"
          fill="none" stroke="var(--brand)" strokeWidth="1.5"
        />
      </svg>

      <div className="relative max-w-7xl mx-auto px-6 py-6 flex items-center justify-between">
        <div>
          <h1
            className="text-2xl tracking-tight"
            style={{ fontFamily: 'var(--font-display)', fontWeight: 600, color: 'var(--text-primary)' }}
          >
            Fraud Intelligence Platform
          </h1>
          <p className="text-sm mt-1" style={{ color: 'var(--text-muted)' }}>
            Behavioral Fraud DNA &middot; Explainable Real-Time Detection
          </p>
        </div>
        {selectedCustomer && (
          <div
            className="px-4 py-2 rounded-lg border text-sm"
            style={{ borderColor: 'var(--border)', background: 'var(--surface)', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}
          >
            active profile: <span style={{ color: 'var(--brand)' }}>{selectedCustomer}</span>
          </div>
        )}
      </div>
    </header>
  )
}
