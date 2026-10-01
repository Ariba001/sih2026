import React from 'react'
import { DECISION_GUIDE } from '../decisions'

const ORDER = ['ACCEPT', 'REVIEW', 'REJECT']

export default function DecisionGuide({ compact = false }) {
  return (
    <section className={`decision-guide ${compact ? 'compact' : ''}`} aria-label="Decision meanings">
      <div className="decision-guide-head">
        <h2>What each decision means</h2>
        {!compact && (
          <p>
            BurnTestr combines lot-relative anomaly detection (Module A), early drift (Module B),
            and hard datasheet/delta rules into one screening call.
          </p>
        )}
      </div>
      <div className="decision-guide-grid">
        {ORDER.map((key) => {
          const d = DECISION_GUIDE[key]
          return (
            <article key={key} className={`decision-def tone-${key.toLowerCase()}`}>
              <header>
                <span className="decision-label">{d.label}</span>
                <span className="decision-short">{d.short}</span>
              </header>
              {!compact && <p className="decision-meaning">{d.meaning}</p>}
              <p className="decision-action"><span>Operator:</span> {d.action}</p>
            </article>
          )
        })}
      </div>
    </section>
  )
}
