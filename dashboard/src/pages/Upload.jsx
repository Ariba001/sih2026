import React, { useRef, useState } from 'react'
import './pages.css'

export default function Upload({ onUpload, loading }) {
  const inputRef = useRef(null)
  const [drag, setDrag] = useState(false)

  const takeFile = (file) => {
    if (!file) return
    if (!file.name.toLowerCase().endsWith('.csv')) {
      alert('Please choose a CSV file')
      return
    }
    onUpload(file)
  }

  return (
    <div className="page">
      <p className="eyebrow">Score new lots</p>
      <h1 className="page-title">
        Upload to <span className="brand-wordmark">BurnTestr</span>
      </h1>
      <p className="lede">
        Drop a wide-schema burn-in CSV. Columns are validated, Modules A/B run, and results refresh
        with Accept / Review / Reject plus per-component explanations.
      </p>

      <div
        className={`dropzone ${drag ? 'drag' : ''} ${loading ? 'busy' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setDrag(true) }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDrag(false)
          takeFile(e.dataTransfer.files?.[0])
        }}
        onClick={() => !loading && inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => e.key === 'Enter' && inputRef.current?.click()}
      >
        <div className="drop-title">{loading ? 'Scoring…' : 'Drop CSV here'}</div>
        <p>or click to browse · Lot_ID, Component_ID, Param_Name, Value_0h / 24h / 96h / 168h</p>
        <input
          ref={inputRef}
          type="file"
          accept=".csv,text/csv"
          hidden
          disabled={loading}
          onChange={(e) => takeFile(e.target.files?.[0])}
        />
        <button
          type="button"
          className="btn primary"
          disabled={loading}
          onClick={(e) => {
            e.stopPropagation()
            inputRef.current?.click()
          }}
        >
          {loading ? 'Processing…' : 'Select CSV'}
        </button>
      </div>

      <section className="panel schema-panel">
        <h3>Required schema</h3>
        <pre className="schema mono">{`Lot_ID,Component_ID,Param_Name,Value_0h,Value_24h,Value_96h,Value_168h
L01,L01-0001,Iddq,4.5,4.7,5.2,5.8
L01,L01-0001,Leakage,2.1,2.3,2.8,3.5
L01,L01-0001,Prop_Delay,12.1,12.3,12.5,12.8`}</pre>
      </section>
    </div>
  )
}
