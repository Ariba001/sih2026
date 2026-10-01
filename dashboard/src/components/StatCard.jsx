import React from 'react'

export default function StatCard({ label, value, subValue, tone = 'neutral' }) {
  return (
    <div className={`stat-card tone-${tone}`}>
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value ?? '—'}</div>
      {subValue != null && <div className="stat-sub">{subValue}</div>}
    </div>
  )
}
