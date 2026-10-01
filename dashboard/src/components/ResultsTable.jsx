import React, { useMemo, useState } from 'react'
import { describeRow, normalizeDecision } from '../decisions'

const PAGE = 25

function pct(n) {
  const v = Number(n)
  if (!Number.isFinite(v)) return '—'
  return `${Math.round(v * 100)}%`
}

function score(n) {
  const v = Number(n)
  if (!Number.isFinite(v)) return '—'
  return v.toFixed(3)
}

export default function ResultsTable({ rows = [], title = 'Component decisions' }) {
  const [q, setQ] = useState('')
  const [filter, setFilter] = useState('ALL')
  const [page, setPage] = useState(0)

  const filtered = useMemo(() => {
    const needle = q.trim().toLowerCase()
    return rows.filter((r) => {
      const decision = normalizeDecision(r.decision)
      if (filter !== 'ALL' && decision !== filter) return false
      if (!needle) return true
      return (
        String(r.component_id || r.id || '').toLowerCase().includes(needle) ||
        String(r.lot_id || '').toLowerCase().includes(needle) ||
        describeRow(r).toLowerCase().includes(needle)
      )
    })
  }, [rows, q, filter])

  const pages = Math.max(1, Math.ceil(filtered.length / PAGE))
  const slice = filtered.slice(page * PAGE, page * PAGE + PAGE)

  return (
    <section className="table-panel">
      <div className="table-toolbar">
        <h3>{title}</h3>
        <div className="table-controls">
          <input
            type="search"
            placeholder="Search component, lot, or description"
            value={q}
            onChange={(e) => { setQ(e.target.value); setPage(0) }}
            aria-label="Search results"
          />
          <select
            value={filter}
            onChange={(e) => { setFilter(e.target.value); setPage(0) }}
            aria-label="Filter by decision"
          >
            <option value="ALL">All decisions</option>
            <option value="ACCEPT">Accept</option>
            <option value="REVIEW">Review</option>
            <option value="REJECT">Reject</option>
          </select>
        </div>
      </div>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Component</th>
              <th>Lot</th>
              <th>Decision</th>
              <th>Description</th>
              <th>A</th>
              <th>B</th>
              <th>Conf.</th>
            </tr>
          </thead>
          <tbody>
            {slice.map((r, i) => {
              const decision = normalizeDecision(r.decision)
              const label = decision.charAt(0) + decision.slice(1).toLowerCase()
              return (
                <tr key={`${r.component_id || r.id}-${i}`}>
                  <td className="mono">{r.component_id || r.id}</td>
                  <td className="mono">{r.lot_id}</td>
                  <td>
                    <span className={`badge badge-${decision.toLowerCase()}`}>{label}</span>
                  </td>
                  <td className="desc-cell">{describeRow(r)}</td>
                  <td className="num-cell">{score(r.score_a)}</td>
                  <td className="num-cell">{score(r.score_b)}</td>
                  <td className="num-cell">{pct(r.confidence)}</td>
                </tr>
              )
            })}
            {!slice.length && (
              <tr>
                <td colSpan={7} className="empty-row">No matching components</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      <div className="pager">
        <button type="button" disabled={page <= 0} onClick={() => setPage((p) => p - 1)}>Prev</button>
        <span>{page + 1} / {pages} · {filtered.length} rows</span>
        <button type="button" disabled={page >= pages - 1} onClick={() => setPage((p) => p + 1)}>Next</button>
      </div>
    </section>
  )
}
