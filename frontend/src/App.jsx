import React, { useState, useRef } from 'react'

const API_URL = '/api'

function App() {
  const [image, setImage] = useState(null)
  const [preview, setPreview] = useState(null)
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [activeTab, setActiveTab] = useState('overlay')
  const fileInputRef = useRef(null)

  const handleFileChange = (e) => {
    const file = e.target.files[0]
    if (!file) return
    
    if (!file.type.startsWith('image/')) {
      setError('Please select an image file')
      return
    }
    
    setError(null)
    setImage(file)
    setPreview(URL.createObjectURL(file))
    setResults(null)
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
    const file = e.dataTransfer.files[0]
    if (file && file.type.startsWith('image/')) {
      setImage(file)
      setPreview(URL.createObjectURL(file))
      setResults(null)
      setError(null)
    }
  }

  const analyze = async () => {
    if (!image) return
    
    setLoading(true)
    setError(null)
    
    const formData = new FormData()
    formData.append('file', image)
    
    try {
      const response = await fetch(`${API_URL}/analyze`, {
        method: 'POST',
        body: formData
      })
      
      if (!response.ok) {
        const err = await response.json()
        throw new Error(err.detail || 'Analysis failed')
      }
      
      const data = await response.json()
      setResults(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const renderImage = (base64, alt) => {
    if (!base64) return <div className="placeholder">Not available</div>
    return <img src={`data:image/png;base64,${base64}`} alt={alt} />
  }

  if (!preview) {
    return (
      <div className="app">
        <header>
          <h1>BRISC Brain Tumor Analysis</h1>
          <p>Upload a T1-weighted contrast-enhanced MRI for AI-powered classification, segmentation & explainability</p>
        </header>
        <main>
          <UploadZone 
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            fileInputRef={fileInputRef}
            handleFileChange={handleFileChange}
            error={error}
          />
        </main>
      </div>
    )
  }

  return (
    <div className="app">
      <header>
        <h1>BRISC Brain Tumor Analysis</h1>
        <p>Upload a T1-weighted contrast-enhanced MRI for AI-powered classification, segmentation & explainability</p>
      </header>
      <main className="layout">
        <aside className="sidebar">
          <div className="panel upload-panel">
            <h3>Input MRI</h3>
            <img src={preview} alt="Uploaded MRI" />
            <div className="actions">
              <button className="btn secondary" onClick={() => fileInputRef.current?.click()}>
                Change Image
              </button>
              <button 
                className="btn primary" 
                onClick={analyze} 
                disabled={loading}
              >
                {loading ? 'Analyzing...' : 'Run Analysis'}
              </button>
            </div>
            <input 
              ref={fileInputRef} 
              type="file" 
              accept="image/*" 
              onChange={handleFileChange} 
              hidden 
            />
            {error && <div className="error">{error}</div>}
          </div>
        </aside>
        
        <section className="results">
          {results ? (
            <>
              <div className="panel classification-panel">
                <h3>Classification</h3>
                <ClassificationChart probs={results.classification} />
              </div>
              
              <div className="panel tabs-panel">
                <div className="tabs">
                  <button 
                    className={activeTab === 'overlay' ? 'active' : ''} 
                    onClick={() => setActiveTab('overlay')}
                  >
                    GradCAM++ Overlay
                  </button>
                  <button 
                    className={activeTab === 'heatmap' ? 'active' : ''} 
                    onClick={() => setActiveTab('heatmap')}
                  >
                    Heatmap Only
                  </button>
                  <button 
                    className={activeTab === 'segmentation' ? 'active' : ''} 
                    onClick={() => setActiveTab('segmentation')}
                  >
                    Segmentation Mask
                  </button>
                </div>
                <div className="tab-content">
                  {activeTab === 'overlay' && (
                    <div className="image-container">
                      <h4>GradCAM++ Overlay (Model Attention)</h4>
                      {renderImage(results.gradcam_overlay, 'GradCAM++ Overlay')}
                    </div>
                  )}
                  {activeTab === 'heatmap' && (
                    <div className="image-container">
                      <h4>Raw Attention Heatmap</h4>
                      {renderImage(results.gradcam_heatmap, 'GradCAM++ Heatmap')}
                    </div>
                  )}
                  {activeTab === 'segmentation' && (
                    <div className="image-container">
                      <h4>BGSF Segmentation Mask</h4>
                      {renderImage(results.segmentation_mask, 'Segmentation Mask')}
                    </div>
                  )}
                </div>
              </div>
              
              <div className="panel summary-panel">
                <h3>AI Summary</h3>
                <pre className="summary-text">{results.ai_summary}</pre>
                <div className="meta">
                  Processing time: {results.processing_time_ms.toFixed(0)}ms
                </div>
              </div>
            </>
          ) : (
            <div className="placeholder-large">
              <h3>Results will appear here</h3>
              <p>Click "Run Analysis" to start</p>
            </div>
          )}
        </section>
      </main>
    </div>
  )
}

function UploadZone({ onDragOver, onDragLeave, onDrop, onClick, fileInputRef, handleFileChange, error }) {
  return (
    <div 
      className="upload-zone" 
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
      onClick={onClick}
    >
      <input ref={fileInputRef} type="file" accept="image/*" onChange={handleFileChange} hidden />
      <div className="upload-content">
        <span className="icon">[UPLOAD]</span>
        <p>Drag & drop MRI image here, or click to browse</p>
        <span className="hint">Supports: PNG, JPG, DICOM (converted)</span>
      </div>
      {error && <div className="error">{error}</div>}
    </div>
  )
}

function ClassificationChart({ probs }) {
  const entries = Object.entries(probs).sort((a, b) => b[1] - a[1])
  const maxProb = Math.max(...Object.values(probs))
  
  return (
    <div className="classification-chart">
      {entries.map(([label, prob]) => (
        <div key={label} className="prob-bar">
          <span className="label">{label.replace('_', ' ')}</span>
          <div className="bar-container">
            <div 
              className="bar" 
              style={{ width: `${prob * 100}%`, background: prob === maxProb ? '#2e7d32' : '#1976d2' }}
            />
            <span className="value">{(prob * 100).toFixed(1)}%</span>
          </div>
        </div>
      ))}
    </div>
  )
}

export default App