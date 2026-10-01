import React from 'react'
import StatCard from '../components/StatCard'
import Chart from '../components/Chart'
import ResultsTable from '../components/ResultsTable'
import DecisionGuide from '../components/DecisionGuide'
import './pages.css'

function formatPct(v) {
  if (v == null || v === '') return null
  if (typeof v === 'string' && v.includes('%')) return v
  const n = Number(v)
  if (!Number.isFinite(n)) return String(v)
  return `${n.toFixed(n % 1 ? 1 : 0)}%`
}

export default function Overview({ data, loading, onRefresh, onUploadClick }) {
  if (loading && !data) {
    return <div className="loading-block">Loading BurnTestr results…</div>
  }

  if (!data) {
    return (
      <div className="empty-block">
        <p className="eyebrow">SIH 26170 · burn-in screening</p>
        <h1 className="page-title">
          <span className="brand-wordmark">BurnTestr</span>
        </h1>
        <p className="lede">
          No screening results yet. Upload a burn-in CSV, or confirm the API on port 8000 is serving seed results.
        </p>
        <button type="button" className="btn primary" onClick={onUploadClick}>Upload CSV</button>
      </div>
    )
  }

  const { summary, decisions_by_lot, parameter_stats, recent_scores, components, meta } = data
  const pie = [
    { name: 'Accept', value: summary.accepted_count ?? summary.accept ?? 0 },
    { name: 'Review', value: summary.review_count ?? summary.review ?? 0 },
    { name: 'Reject', value: summary.rejected_count ?? summary.reject ?? 0 },
  ]

  return (
    <div className="page">
      <header className="hero">
        <div className="hero-copy">
          <p className="eyebrow">SIH 26170 · burn-in screening</p>
          <h1 className="hero-title">
            <span className="brand-wordmark">BurnTestr</span>
          </h1>
          <p className="lede">
            Lot-relative anomaly detection and early-drift screening for aerospace burn-in.
            {meta?.source ? ` Showing results from ${meta.source}.` : ''}
          </p>
          <div className="hero-actions">
            <button type="button" className="btn primary" onClick={onUploadClick}>Upload CSV</button>
            <button type="button" className="btn ghost" onClick={onRefresh}>Refresh</button>
          </div>
        </div>
        <dl className="hero-meta">
          <div>
            <dt>Updated</dt>
            <dd className="mono">{meta?.updated_at ? new Date(meta.updated_at).toLocaleString() : '—'}</dd>
          </div>
          <div>
            <dt>Components</dt>
            <dd className="mono">{summary.total_components ?? summary.total ?? 0}</dd>
          </div>
        </dl>
      </header>

      <div className="stats-grid">
        <StatCard label="Total screened" value={summary.total_components ?? summary.total} />
        <StatCard
          label="Accept"
          value={summary.accepted_count ?? summary.accept}
          subValue={formatPct(summary.accepted_pct)}
          decisionKey="ACCEPT"
        />
        <StatCard
          label="Review"
          value={summary.review_count ?? summary.review}
          subValue={formatPct(summary.review_pct)}
          decisionKey="REVIEW"
        />
        <StatCard
          label="Reject"
          value={summary.rejected_count ?? summary.reject}
          subValue={formatPct(summary.rejected_pct)}
          decisionKey="REJECT"
        />
      </div>

      <DecisionGuide />

      <div className="charts-grid">
        <section className="panel">
          <h3>Decision mix</h3>
          <Chart type="pie" data={pie} />
        </section>
        <section className="panel">
          <h3>Decisions by lot</h3>
          <Chart type="bar" data={(decisions_by_lot || []).slice(0, 12)} />
        </section>
      </div>

      {parameter_stats?.length > 0 && (
        <section className="panel">
          <h3>Parameter snapshot (168h)</h3>
          <div className="param-grid">
            {parameter_stats.map((p) => (
              <div key={p.parameter} className="param-item">
                <div className="param-name">{p.parameter}</div>
                <div className="param-vals mono">
                  mean {Number(p.mean).toFixed(2)} · max {Number(p.max).toFixed(2)}
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      <ResultsTable
        rows={components?.length ? components : recent_scores || []}
        title="Current results"
      />
    </div>
  )
}
