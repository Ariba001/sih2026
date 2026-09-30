import React, { useRef } from 'react'
import './Upload.css'

export default function Upload({ onUpload, loading }) {
  const fileInputRef = useRef(null)

  const handleFileSelect = (event) => {
    const file = event.target.files?.[0]
    if (file) {
      if (!file.name.endsWith('.csv')) {
        alert('Please select a CSV file')
        return
      }
      onUpload(file)
    }
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    e.currentTarget.classList.add('drag-over')
  }

  const handleDragLeave = (e) => {
    e.currentTarget.classList.remove('drag-over')
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.currentTarget.classList.remove('drag-over')
    const file = e.dataTransfer.files?.[0]
    if (file) {
      if (!file.name.endsWith('.csv')) {
        alert('Please drop a CSV file')
        return
      }
      onUpload(file)
    }
  }

  return (
    <div className="upload-page">
      <div className="upload-container">
        <h2>Upload Burn-In Data</h2>
        <p className="subtitle">Upload a CSV file with component measurements for analysis</p>

        <div
          className="upload-dropzone"
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <div className="upload-content">
            <div className="upload-icon">📁</div>
            <h3>Drag and drop your CSV file here</h3>
            <p>or click to browse</p>
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv"
              onChange={handleFileSelect}
              disabled={loading}
              style={{ display: 'none' }}
            />
            <button
              className="browse-btn"
              onClick={() => fileInputRef.current?.click()}
              disabled={loading}
            >
              {loading ? 'Processing...' : 'Select File'}
            </button>
          </div>
        </div>

        {loading && (
          <div className="upload-progress">
            <div className="spinner"></div>
            <p>Analyzing data...</p>
          </div>
        )}

        <div className="upload-info">
          <h4>Required CSV Format</h4>
          <p>Your CSV should include these columns:</p>
          <ul>
            <li><code>Lot_ID</code> - Identifier for the lot</li>
            <li><code>Component_ID</code> - Component identifier</li>
            <li><code>Param_Name</code> - Parameter (Iddq, Leakage, Prop_Delay)</li>
            <li><code>Value_0h</code> - Initial measurement</li>
            <li><code>Value_24h</code> - 24-hour measurement</li>
            <li><code>Value_96h</code> - 96-hour measurement</li>
            <li><code>Value_168h</code> - 168-hour measurement (final)</li>
          </ul>

          <h4 style={{ marginTop: '1.5rem' }}>Example Row</h4>
          <pre className="example-code">
{`Lot_ID,Component_ID,Param_Name,Value_0h,Value_24h,Value_96h,Value_168h
LOT_001,C_0001,Iddq,15.2,15.8,16.1,16.5
LOT_001,C_0001,Leakage,8.3,8.5,8.7,9.1
LOT_001,C_0001,Prop_Delay,12.1,12.2,12.3,12.4`}
          </pre>
        </div>
      </div>
    </div>
  )
}
