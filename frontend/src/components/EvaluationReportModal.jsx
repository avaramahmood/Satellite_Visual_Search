import { useState } from "react"
import ReactMarkdown from "react-markdown"
import "./EvaluationReportModal.css"

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000"

export function EvaluationReportModal({ sessionId, onClose }) {
  const [groundTruth, setGroundTruth] = useState("")
  const [loading, setLoading]         = useState(false)
  const [pdfLoading, setPdfLoading]   = useState(false)
  const [report, setReport]           = useState(null)
  const [error, setError]             = useState(null)

  const handleGenerate = async () => {
    if (!sessionId) return
    setLoading(true)
    setError(null)
    setReport(null)

    try {
      const form = new FormData()
      form.append("session_id", sessionId)
      if (groundTruth.trim()) form.append("ground_truth_count", groundTruth.trim())

      const res = await fetch(`${API_BASE}/api/report/generate`, {
        method: "POST",
        body: form,
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }))
        throw new Error(err.detail || `Server error ${res.status}`)
      }
      const data = await res.json()
      setReport(data.report)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleDownloadPdf = async () => {
    setPdfLoading(true)
    setError(null)
    try {
      const form = new FormData()
      form.append("session_id", sessionId)
      if (groundTruth.trim()) form.append("ground_truth_count", groundTruth.trim())

      const res = await fetch(`${API_BASE}/api/report/pdf`, { method: "POST", body: form })
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }))
        throw new Error(err.detail || `Server error ${res.status}`)
      }
      const blob = await res.blob()
      const url  = URL.createObjectURL(blob)
      const a    = document.createElement("a")
      a.href     = url
      a.download = `report_${sessionId.slice(0, 8)}.pdf`
      a.click()
      URL.revokeObjectURL(url)
    } catch (err) {
      setError(err.message)
    } finally {
      setPdfLoading(false)
    }
  }

  return (
    <div className="report-modal-backdrop" onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div className="report-modal">
        {/* Header */}
        <div className="report-modal-header">
          <div className="report-modal-title-row">
            <span className="report-modal-icon">📊</span>
            <h2 className="report-modal-title">Model Evaluation Report</h2>
          </div>
          <button className="report-modal-close" onClick={onClose}>✕</button>
        </div>

        {/* Session context badge */}
        <div className="report-session-badge">
          <span className="report-session-dot" />
          <span className="report-session-text">Session: {sessionId}</span>
        </div>

        {/* Controls */}
        {!report && (
          <div className="report-controls">
            <label className="report-gt-label">
              Ground Truth Count
              <span className="report-gt-hint">(optional — leave blank to estimate from confidence)</span>
            </label>
            <div className="report-gt-row">
              <input
                type="number"
                min="1"
                value={groundTruth}
                onChange={(e) => setGroundTruth(e.target.value)}
                placeholder="e.g. 42"
                className="report-gt-input"
                disabled={loading}
              />
              <button
                className="report-generate-btn"
                onClick={handleGenerate}
                disabled={loading}
              >
                {loading ? (
                  <>
                    <span className="report-spinner" />
                    Generating…
                  </>
                ) : (
                  "Generate Report"
                )}
              </button>
            </div>
            {error && <p className="report-error">{error}</p>}
          </div>
        )}

        {/* Report output */}
        {report && (
          <div className="report-output">
            <div className="report-output-toolbar">
              <span className="report-output-label">Evaluation Report</span>
              <div className="report-toolbar-actions">
                <button
                  className="report-pdf-btn"
                  onClick={handleDownloadPdf}
                  disabled={pdfLoading}
                  title="Download as PDF"
                >
                  {pdfLoading ? "Generating PDF…" : "⬇ PDF"}
                </button>
                <button
                  className="report-regenerate-btn"
                  onClick={() => setReport(null)}
                >
                  ↺ Regenerate
                </button>
              </div>
            </div>
            <div className="report-markdown-body">
              <ReactMarkdown>{report}</ReactMarkdown>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
