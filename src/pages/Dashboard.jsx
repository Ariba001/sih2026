import React, { useState, useEffect } from 'react'
import axios from 'axios'
import StatCard from '../components/StatCard'
import Chart from '../components/Chart'
import './Dashboard.css'

export default function Dashboard() {
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetchDashboardData()
  }, [])

  const fetchDashboardData = async () => {
    try {
      setLoading(true)
      const response = await axios.get('/api/dashboard')
      setStats(response.data)
      setError(null)
    } catch (err) {
      console.error('Failed to fetch dashboard:', err)
      setError('Unable to connect to API. Is the server running?')
      // Set mock data for demonstration
      setStats(getMockData())
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return <div className="loading">Loading dashboard...</div>
  }

  if (!stats) {
    return <div className="error">{error || 'No data available'}</div>
  }

  const { summary, decisions_by_lot, parameter_stats, recent_scores } = stats

  return (
    <div className="dashboard">
      <div className="dashboard-header">
        <h2>System Overview</h2>
        <button className="refresh-btn" onClick={fetchDashboardData}>
          🔄 Refresh
        </button>
      </div>

      {error && <div className="alert alert-info">{error}</div>}

      {/* Summary Stats */}
      <div className="stats-grid">
        <StatCard
          label="Total Components"
          value={summary.total_components}
          icon="📦"
        />
        <StatCard
          label="Accepted"
          value={summary.accepted_count}
          subValue={summary.accepted_pct}
          icon="✅"
          color="success"
        />
        <StatCard
          label="Review"
          value={summary.review_count}
          subValue={summary.review_pct}
          icon="⚠️"
          color="warning"
        />
        <StatCard
          label="Rejected"
          value={summary.rejected_count}
          subValue={summary.rejected_pct}
          icon="❌"
          color="danger"
        />
      </div>

      {/* Charts */}
      <div className="charts-grid">
        <div className="chart-card">
          <h3>Decision Distribution</h3>
          <Chart
            type="pie"
            data={[
              { name: 'Accept', value: summary.accepted_count },
              { name: 'Review', value: summary.review_count },
              { name: 'Reject', value: summary.rejected_count },
            ]}
          />
        </div>

        <div className="chart-card">
          <h3>Decisions by Lot</h3>
          <Chart
            type="bar"
            data={decisions_by_lot?.slice(0, 10) || []}
          />
        </div>

        <div className="chart-card full-width">
          <h3>Parameter Statistics</h3>
          <div className="param-stats">
            {parameter_stats?.map(stat => (
              <div key={stat.parameter} className="stat-item">
                <div className="stat-name">{stat.parameter}</div>
                <div className="stat-values">
                  <span>Mean: {stat.mean?.toFixed(2)}</span>
                  <span>Max: {stat.max?.toFixed(2)}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Recent Scores */}
      <div className="recent-scores">
        <h3>Recent Scores</h3>
        <table className="scores-table">
          <thead>
            <tr>
              <th>Component ID</th>
              <th>Decision</th>
              <th>Score A</th>
              <th>Score B</th>
              <th>Confidence</th>
            </tr>
          </thead>
          <tbody>
            {recent_scores?.slice(0, 5).map((score, idx) => (
              <tr key={idx}>
                <td>{score.component_id}</td>
                <td>
                  <span className={`badge badge-${score.decision.toLowerCase()}`}>
                    {score.decision}
                  </span>
                </td>
                <td>{score.score_a?.toFixed(2)}</td>
                <td>{score.score_b?.toFixed(2)}</td>
                <td>{(score.confidence * 100)?.toFixed(0)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function getMockData() {
  return {
    summary: {
      total_components: 1250,
      accepted_count: 950,
      accepted_pct: '76%',
      review_count: 200,
      review_pct: '16%',
      rejected_count: 100,
      rejected_pct: '8%',
    },
    decisions_by_lot: [
      { lot_id: 'LOT_001', accepted: 45, review: 5, rejected: 0 },
      { lot_id: 'LOT_002', accepted: 48, review: 2, rejected: 0 },
      { lot_id: 'LOT_003', accepted: 40, review: 8, rejected: 2 },
      { lot_id: 'LOT_004', accepted: 35, review: 10, rejected: 5 },
      { lot_id: 'LOT_005', accepted: 42, review: 6, rejected: 2 },
    ],
    parameter_stats: [
      { parameter: 'Iddq', mean: 15.2, max: 28.5 },
      { parameter: 'Leakage', mean: 8.3, max: 45.2 },
      { parameter: 'Prop_Delay', mean: 12.1, max: 32.8 },
    ],
    recent_scores: [
      { component_id: 'C_001', decision: 'Accept', score_a: 0.2, score_b: 0.1, confidence: 0.95 },
      { component_id: 'C_002', decision: 'Accept', score_a: 0.15, score_b: 0.08, confidence: 0.98 },
      { component_id: 'C_003', decision: 'Review', score_a: 0.85, score_b: 0.87, confidence: 0.72 },
      { component_id: 'C_004', decision: 'Reject', score_a: 1.2, score_b: 1.1, confidence: 0.88 },
      { component_id: 'C_005', decision: 'Accept', score_a: 0.25, score_b: 0.12, confidence: 0.92 },
    ],
  }
}
