import React from 'react'
import './StatCard.css'

export default function StatCard({ label, value, subValue, icon, color = 'primary' }) {
  return (
    <div className={`stat-card stat-card-${color}`}>
      <div className="stat-icon">{icon}</div>
      <div className="stat-content">
        <div className="stat-label">{label}</div>
        <div className="stat-value">{value}</div>
        {subValue && <div className="stat-sub">{subValue}</div>}
      </div>
    </div>
  )
}
