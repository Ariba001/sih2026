import React, { useEffect, useState } from 'react'
import { fetchCurrentResults, fetchHealth, uploadCsv } from './api'
import Overview from './pages/Overview'
import Upload from './pages/Upload'
import Results from './pages/Results'
import './App.css'

export default function App() {
  const [page, setPage] = useState('overview')
  const [results, setResults] = useState(null)
  const [health, setHealth] = useState(null)
  const [loading, setLoading] = useState(true)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState(null)

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      const [h, r] = await Promise.all([
        fetchHealth().catch(() => null),
        fetchCurrentResults(),
      ])
      setHealth(h)
      setResults(r)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to load results')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    load()
  }, [])

  const handleUpload = async (file) => {
    setUploading(true)
    setError(null)
    try {
      const data = await uploadCsv(file)
      setResults(data)
      setPage('results')
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Upload failed')
    } finally {
      setUploading(false)
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="topbar-inner">
          <a className="brand" href="#overview" onClick={(e) => { e.preventDefault(); setPage('overview') }}>
            <img src="/logo.svg" alt="" className="brand-mark" width="34" height="34" />
            <span className="brand-wordmark">BurnTestr</span>
          </a>
          <nav className="nav" aria-label="Primary">
            <button type="button" className={page === 'overview' ? 'nav-link active' : 'nav-link'} onClick={() => setPage('overview')}>
              Overview
            </button>
            <button type="button" className={page === 'upload' ? 'nav-link active' : 'nav-link'} onClick={() => setPage('upload')}>
              Upload
            </button>
            <button
              type="button"
              className={page === 'results' ? 'nav-link active' : 'nav-link'}
              onClick={() => setPage('results')}
              disabled={!results}
            >
              Results
            </button>
          </nav>
          <div className="status-chip" title={health?.status || 'unknown'}>
            <span className={`dot ${health?.model_loaded ? 'ok' : 'warn'}`} />
            {health?.model_loaded ? 'API ready' : 'API offline'}
          </div>
        </div>
      </header>

      <main className="main">
        {error && (
          <div className="banner error" role="alert">
            {String(error)}
            <button type="button" onClick={() => setError(null)}>Dismiss</button>
          </div>
        )}

        {page === 'overview' && (
          <Overview
            data={results}
            loading={loading}
            onRefresh={load}
            onUploadClick={() => setPage('upload')}
          />
        )}
        {page === 'upload' && (
          <Upload onUpload={handleUpload} loading={uploading} />
        )}
        {page === 'results' && results && (
          <Results data={results} onBack={() => setPage('overview')} />
        )}
      </main>

      <footer className="footer">
        <span className="brand-wordmark footer-mark">BurnTestr</span>
        <span>MIL-STD-883 / ESCC burn-in screening · Smart India Hackathon 26170</span>
      </footer>
    </div>
  )
}
