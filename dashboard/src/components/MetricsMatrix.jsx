import React from 'react'
import { screeningMetrics as metrics } from '../data/screeningMetrics'

export default function MetricsMatrix() {
  const { eval: ev, highlights, detectionRows, maeRows, moduleSummary, baselineCompare, source } = metrics
  const lots = (ev.testLots || []).join(', ')

  return (
    <section className="metrics-section" aria-label="Screening performance">
      <div className="metrics-head">
        <div>
          <p className="eyebrow">Evaluation · held-out test</p>
          <h2>Screening performance</h2>
          <p className="metrics-lede">
            Recall and model metrics on frozen lots {lots} ({ev.n} parts, {ev.nDefective} defective).
            Cost {ev.costFn}·FN + {ev.costFp}·FP. {source}.
          </p>
        </div>
      </div>

      <div className="metrics-highlights">
        {highlights.map((h) => (
          <div key={h.key} className="metrics-hl">
            <div className="metrics-hl-label">{h.label}</div>
            <div className="metrics-hl-value mono">{h.value}</div>
            {h.detail && <div className="metrics-hl-detail mono">{h.detail}</div>}
            {h.note && <div className="metrics-hl-note">{h.note}</div>}
          </div>
        ))}
      </div>

      <div className="metrics-tables">
        <div className="metrics-table-block">
          <h3>Detection matrix</h3>
          <div className="metrics-table-scroll">
            <table className="metrics-table">
              <thead>
                <tr>
                  <th>System</th>
                  <th>Recall</th>
                  <th>Caught</th>
                  <th>Precision</th>
                  <th>F2</th>
                  <th>Overkill</th>
                  <th>Escapes</th>
                  <th>Cost</th>
                </tr>
              </thead>
              <tbody>
                {detectionRows.map((r) => (
                  <tr key={r.system} className={r.highlight ? 'is-highlight' : undefined}>
                    <td>{r.system}</td>
                    <td className="mono">{r.recallLabel}</td>
                    <td className="mono">{r.caught || '—'}</td>
                    <td className="mono">{r.precisionLabel}</td>
                    <td className="mono">{r.f2Label}</td>
                    <td className="mono">{r.overkillLabel}</td>
                    <td className="mono">{r.escapes}</td>
                    <td className="mono">{r.cost}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="metrics-table-block">
          <h3>Module B · Value_168h MAE</h3>
          <div className="metrics-table-scroll">
            <table className="metrics-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Overall MAE</th>
                  <th>nMAE</th>
                  <th>Iddq</th>
                  <th>Leakage</th>
                  <th>Prop_Delay</th>
                </tr>
              </thead>
              <tbody>
                {maeRows.map((r) => (
                  <tr key={r.model} className={r.highlight ? 'is-highlight' : undefined}>
                    <td>{r.model}</td>
                    <td className="mono">{r.overallMaeLabel}</td>
                    <td className="mono">{r.nmaeLabel}</td>
                    <td className="mono">{r.iddqLabel}</td>
                    <td className="mono">{r.leakageLabel}</td>
                    <td className="mono">{r.propDelayLabel}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div className="metrics-module-grid">
        {moduleSummary.map((m) => (
          <article key={m.module} className="metrics-module">
            <header>
              <span className="metrics-module-name">{m.module}</span>
              <span className="metrics-module-role">{m.role}</span>
            </header>
            <dl>
              {m.recall != null && (
                <div>
                  <dt>Recall</dt>
                  <dd className="mono">{m.recall}</dd>
                </div>
              )}
              {m.rejectReviewRecall != null && (
                <div>
                  <dt>REJECT+REVIEW</dt>
                  <dd className="mono">{m.rejectReviewRecall}</dd>
                </div>
              )}
              {m.overkill != null && (
                <div>
                  <dt>Overkill</dt>
                  <dd className="mono">{m.overkill}</dd>
                </div>
              )}
              {m.mae != null && (
                <div>
                  <dt>MAE</dt>
                  <dd className="mono">{m.mae}</dd>
                </div>
              )}
              {m.escapes != null && (
                <div>
                  <dt>Escapes</dt>
                  <dd className="mono">{m.escapes}</dd>
                </div>
              )}
            </dl>
          </article>
        ))}
      </div>

      <p className="metrics-footnote mono">
        vs v1.0 — REJECT recall {baselineCompare.rejectRecallV1} → {baselineCompare.rejectRecallV2};
        REJECT+REVIEW {baselineCompare.rejectReviewV1} → {baselineCompare.rejectReviewV2};
        Ridge MAE {baselineCompare.maeRidgeV1} → {baselineCompare.maeRidgeV2}.
      </p>
    </section>
  )
}
