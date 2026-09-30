import React from 'react'
import Chart from '../components/Chart'
import './Results.css'

export default function Results({ data, onBack }) {
  const { summary, components, parameter_insights } = data

  return (
    <div className="results-page">
      <div className="results-header">
        <button className="back-btn" onClick={onBack}>
          ← Back to Dashboard
        </button>
        <h2>Analysis Results</h2>
      </div>

      {/* Summary */}
      <div className="results-summary">
        <div className="summary-card">
          <div className="summary-value">{summary.total}</div>
          <div className="summary-label">Total Components</div>
        </div>
        <div className="summary-card accept">
          <div className="summary-value">{summary.accept}</div>
          <div className="summary-label">Accepted</div>
        </div>
        <div className="summary-card review">
          <div className="summary-value">{summary.review}</div>
          <div className="summary-label">Review</div>
        </div>
        <div className="summary-card reject">
          <div className="summary-value">{summary.reject}</div>
          <div className="summary-label">Rejected</div>
        </div>
      </div>

      {/* Charts */}
      <div className="results-charts">
        <div className="chart-card">
          <h3>Decision Distribution</h3>
          <Chart
            type="pie"
            data={[
              { name: 'Accept', value: summary.accept },
              { name: 'Review', value: summary.review },
              { name: 'Reject', value: summary.reject },
            ]}
          />
        </div>
      </div>

      {/* Components Table */}
      <div className="components-section">
        <h3>Component Decisions</h3>
        <div className="table-scroll">
          <table className="components-table">
            <thead>
              <tr>
                <th>Component</th>
                <th>Lot</th>
                <th>Decision</th>
                <th>Score A</th>
                <th>Score B</th>
                <th>Confidence</th>
              </tr>
            </thead>
            <tbody>
              {components?.slice(0, 20).map((comp, idx) => (
                <tr key={idx}>
                  <td><code>{comp.id}</code></td>
                  <td>{comp.lot_id}</td>
                  <td>
                    <span className={`badge badge-${comp.decision.toLowerCase()}`}>
                      {comp.decision}
                    </span>
                  </td>
                  <td>{comp.score_a?.toFixed(2)}</td>
                  <td>{comp.score_b?.toFixed(2)}</td>
                  <td>{(comp.confidence * 100)?.toFixed(0)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Parameter Insights */}
      {parameter_insights && (
        <div className="insights-section">
          <h3>Parameter Insights</h3>
          <div className="insights-grid">
            {parameter_insights.map((insight, idx) => (
              <div key={idx} className="insight-card">
                <div className="insight-name">{insight.parameter}</div>
                <div className="insight-stats">
                  <div className="stat">
                    <span className="label">Mean:</span>
                    <span className="value">{insight.mean?.toFixed(2)}</span>
                  </div>
                  <div className="stat">
                    <span className="label">Range:</span>
                    <span className="value">{insight.min?.toFixed(2)} → {insight.max?.toFixed(2)}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
