import React from 'react'
import Chart from '../components/Chart'
import ResultsTable from '../components/ResultsTable'
import './pages.css'

export default function Results({ data, onBack }) {
  const summary = data.summary || {}
  const pie = [
    { name: 'Accept', value: summary.accept ?? summary.accepted_count ?? 0 },
    { name: 'Review', value: summary.review ?? summary.review_count ?? 0 },
    { name: 'Reject', value: summary.reject ?? summary.rejected_count ?? 0 },
  ]

  return (
    <div className="page">
      <button type="button" className="btn ghost back" onClick={onBack}>← Back to overview</button>
      <p className="eyebrow">Scored batch</p>
      <h1 className="page-title">
        <span className="brand-wordmark">BurnTestr</span> results
      </h1>
      <p className="lede">
        Decisions from the latest upload{data.meta?.source ? ` (${data.meta.source})` : ''}.
      </p>

      <div className="stats-grid">
        <div className="stat-card"><div className="stat-label">Total</div><div className="stat-value">{summary.total ?? summary.total_components}</div></div>
        <div className="stat-card tone-accept"><div className="stat-label">Accept</div><div className="stat-value">{summary.accept ?? summary.accepted_count}</div></div>
        <div className="stat-card tone-review"><div className="stat-label">Review</div><div className="stat-value">{summary.review ?? summary.review_count}</div></div>
        <div className="stat-card tone-reject"><div className="stat-label">Reject</div><div className="stat-value">{summary.reject ?? summary.rejected_count}</div></div>
      </div>

      <div className="charts-grid">
        <section className="panel">
          <h3>Decision distribution</h3>
          <Chart type="pie" data={pie} />
        </section>
        <section className="panel">
          <h3>By lot</h3>
          <Chart type="bar" data={(data.decisions_by_lot || []).slice(0, 12)} />
        </section>
      </div>

      {(data.parameter_insights || data.parameter_stats)?.length > 0 && (
        <section className="panel">
          <h3>Parameter insights</h3>
          <div className="param-grid">
            {(data.parameter_insights || data.parameter_stats).map((p) => (
              <div key={p.parameter} className="param-item">
                <div className="param-name">{p.parameter}</div>
                <div className="param-vals mono">
                  {Number(p.min).toFixed(2)} → {Number(p.max).toFixed(2)} (mean {Number(p.mean).toFixed(2)})
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      <ResultsTable rows={data.components || []} title="Component decisions" />
    </div>
  )
}
