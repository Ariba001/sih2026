import React from 'react'
import { decisionMeta, normalizeDecision } from '../decisions'

export default function StatCard({ label, value, subValue, tone = 'neutral', decisionKey }) {
  const meta = decisionKey ? decisionMeta(decisionKey) : null
  const toneClass = decisionKey
    ? `tone-${normalizeDecision(decisionKey).toLowerCase()}`
    : `tone-${tone}`

  return (
    <div className={`stat-card ${toneClass}`}>
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value ?? '—'}</div>
      {subValue != null && subValue !== '' && <div className="stat-sub">{subValue}</div>}
      {meta && <p className="stat-desc">{meta.short}. {meta.action}</p>}
    </div>
  )
}
