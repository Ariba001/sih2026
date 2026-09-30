import React, { useState, useEffect } from 'react'
import axios from 'axios'
import Dashboard from './pages/Dashboard'
import Upload from './pages/Upload'
import Results from './pages/Results'
import './App.css'

export default function App() {
  const [currentPage, setCurrentPage] = useState('dashboard')
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)

  const handleUpload = async (file) => {
    setLoading(true)
    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await axios.post('/api/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })

      setResults(response.data)
      setCurrentPage('results')
    } catch (error) {
      console.error('Upload failed:', error)
      alert('Failed to upload file: ' + (error.response?.data?.detail || error.message))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <nav className="navbar">
        <div className="navbar-content">
          <h1 className="logo">⚙️ Burn-In Anomaly Detection</h1>
          <ul className="nav-links">
            <li>
              <button
                className={currentPage === 'dashboard' ? 'active' : ''}
                onClick={() => setCurrentPage('dashboard')}
              >
                Dashboard
              </button>
            </li>
            <li>
              <button
                className={currentPage === 'upload' ? 'active' : ''}
                onClick={() => setCurrentPage('upload')}
              >
                Upload Data
              </button>
            </li>
          </ul>
        </div>
      </nav>

      <main className="container">
        {currentPage === 'dashboard' && <Dashboard />}
        {currentPage === 'upload' && <Upload onUpload={handleUpload} loading={loading} />}
        {currentPage === 'results' && results && (
          <Results
            data={results}
            onBack={() => setCurrentPage('dashboard')}
          />
        )}
      </main>
    </div>
  )
}
