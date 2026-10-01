import React from 'react'
import StatCard from '../components/StatCard'
import Chart from '../components/Chart'
import ResultsTable from '../components/ResultsTable'
import './pages.css'

export default function Overview({ data, loading, onRefresh, onUploadClick }) {
  if (loading && !data) {
    return <div className="loading-block">Loading BurnTestr results…</div>
  }

  if (!data) {
    return (
      <div className="empty-block">
        <h2 className="page-title">No results yet</h2>
        <p>Upload a burn-in CSV to score components, or start the API so seed results can load.</p>
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
      <div className="hero">
        <div>
          <p className="eyebrow">Current screening results</p>
          <h1 className="hero-title">
            <span className="brand-wordmark">BurnTestr</span>
            <span className="hero-rest"> overview</span>
          </h1>
          <p className="lede">
            Lot-relative anomaly detection and drift screening for aerospace burn-in.
            Showing persisted results{meta?.source ? ` from ${meta.source}` : ''}.
          </p>
          <div className="hero-actions">
            <button type="button" className="btn primary" onClick={onUploadClick}>Upload CSV</button>
            <button type="button" className="btn ghost" onClick={onRefresh}>Refresh</button>
          </div>
        </div>
        <div className="hero-aside">
          <div className="aside-label">Updated</div>
          <div className="aside-value mono">{meta?.updated_at ? new Date(meta.updated_at).toLocaleString() : '—'}</div>
          <div className="aside-label">Components</div>
          <div className="aside-value mono">{summary.total_components ?? summary.total ?? 0}</div>
        </div>
      </div>

      <div className="stats-grid">
        <StatCard label="Total components" value={summary.total_components ?? summary.total} />
        <StatCard label="Accepted" value={summary.accepted_count ?? summary.accept} subValue={summary.accepted_pct} tone="accept" />
        <StatCard label="Review" value={summary.review_count ?? summary.review} subValue={summary.review_pct} tone="review" />
        <StatCard label="Rejected" value={summary.rejected_count ?? summary.reject} subValue={summary.rejected_pct} tone="reject" />
      </div>

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
        title="Current results table"
      />
    </div>
  )
}
