import React, { useMemo, useState } from 'react'

const PAGE = 25

export default function ResultsTable({ rows = [], title = 'Component decisions' }) {
  const [q, setQ] = useState('')
  const [filter, setFilter] = useState('ALL')
  const [page, setPage] = useState(0)

  const filtered = useMemo(() => {
    const needle = q.trim().toLowerCase()
    return rows.filter((r) => {
      const decision = String(r.decision || '').toUpperCase()
      if (filter !== 'ALL' && decision !== filter) return false
      if (!needle) return true
      return (
        String(r.component_id || r.id || '').toLowerCase().includes(needle) ||
        String(r.lot_id || '').toLowerCase().includes(needle)
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
            placeholder="Filter component or lot"
            value={q}
            onChange={(e) => { setQ(e.target.value); setPage(0) }}
          />
          <select value={filter} onChange={(e) => { setFilter(e.target.value); setPage(0) }}>
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
              <th>A score</th>
              <th>B score</th>
              <th>Confidence</th>
            </tr>
          </thead>
          <tbody>
            {slice.map((r, i) => {
              const decision = String(r.decision || '').toLowerCase()
              return (
                <tr key={`${r.component_id || r.id}-${i}`}>
                  <td className="mono">{r.component_id || r.id}</td>
                  <td>{r.lot_id}</td>
                  <td><span className={`badge badge-${decision}`}>{r.decision}</span></td>
                  <td className="mono">{Number(r.score_a).toFixed(3)}</td>
                  <td className="mono">{Number(r.score_b).toFixed(3)}</td>
                  <td className="mono">{Math.round(Number(r.confidence || 0) * 100)}%</td>
                </tr>
              )
            })}
            {!slice.length && (
              <tr>
                <td colSpan={6} className="empty-row">No matching components</td>
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
