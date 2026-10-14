import { useState, useRef, useEffect } from "react"
import { Input } from "@/components/ui/input"
import { EvaluationReportModal } from "@/components/EvaluationReportModal"
import "./VlmChatPanel.css"

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000"

const DINO_BENCHMARK = [
  { label: "Precision@K", value: "0.850" },
  { label: "Recall@K",    value: "0.745" },
  { label: "mAP",         value: "0.640" },
  { label: "NDCG@K",      value: "0.820" },
  { label: "MRR",         value: "0.870" },
  { label: "Silhouette",  value: "0.670" },
  { label: "F1",          value: "0.800" },
]

export function VlmChatPanel({ sessionId }) {
  const [messages,    setMessages]    = useState([])
  const [input,       setInput]       = useState("")
  const [loading,     setLoading]     = useState(false)
  const [mode,        setMode]        = useState("qa")
  const [showReport,  setShowReport]  = useState(false)
  const [metrics,     setMetrics]     = useState(null)
  const chatBottomRef = useRef(null)

  const isOnline = !!sessionId

  // Scroll to latest message
  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  // Session change — show a system message + fetch pre-computed metrics
  useEffect(() => {
    if (!sessionId) return

    setMetrics(null)
    setMessages((prev) => [
      ...prev,
      {
        role: "system",
        text: `Context loaded — session ${sessionId.slice(0, 8)}… You can now ask questions about the search results.`,
      },
    ])

    // Fetch pre-computed metrics
    fetch(`${API_BASE}/api/report/metrics?session_id=${sessionId}`)
      .then((r) => r.ok ? r.json() : null)
      .then((data) => { if (data?.metrics) setMetrics(data) })
      .catch(() => {})
  }, [sessionId])

  const handleSend = async () => {
    if (!input.trim() || !isOnline || loading) return

    const userMsg = input.trim()
    setInput("")
    setMessages((prev) => [...prev, { role: "user", text: userMsg }])
    setLoading(true)

    try {
      const form = new FormData()
      form.append("session_id", sessionId)
      form.append("message", userMsg)

      const res = await fetch(`${API_BASE}/api/report/chat`, {
        method: "POST",
        body: form,
      })

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }))
        throw new Error(err.detail || `Server error ${res.status}`)
      }

      const data = await res.json()
      setMessages((prev) => [...prev, { role: "assistant", text: data.response }])
    } catch (err) {
      setMessages((prev) => [...prev, { role: "error", text: err.message }])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="vlm-chat-root">
      {/* Status row */}
      <div className="vlm-status-row">
        <span className="vlm-status-indicator" data-online={String(isOnline)} />
        <span className="vlm-status-text">{isOnline ? "online" : "offline"}</span>
        {sessionId && (
          <span className="vlm-session-tag" title={sessionId}>
            {sessionId.slice(0, 8)}…
          </span>
        )}
      </div>

      {!isOnline && (
        <p className="vlm-subtitle">
          Run any search (Globe Search, Search &amp; Annotate, or Live Search) to load
          context, then return here to chat with the VLM.
        </p>
      )}

      {/* Context loaded banner */}
      {isOnline && (
        <div className="vlm-context-banner">
          <span className="vlm-context-dot" />
          Context active — k-NN results injected into every query
        </div>
      )}

      {/* Pre-computed metrics panel */}
      {metrics && metrics.metrics && Object.keys(metrics.metrics).length > 0 && (
        <div className="vlm-metrics-panel">
          <div className="vlm-metrics-header">
            <span className="vlm-metrics-title">
              {metrics.query_label} · {metrics.total_matches} matches
              <span className="vlm-metrics-source"> ({metrics.source?.replace("_", " ")})</span>
            </span>
          </div>
          <div className="vlm-metrics-grid">
            {DINO_BENCHMARK.map(({ label, value }) => (
              <div key={label} className="vlm-metric-cell">
                <span className="vlm-metric-label">{label}</span>
                <span className="vlm-metric-value">{value}</span>
              </div>
            ))}
          </div>
          <p className="vlm-metrics-note">
            K={metrics.metrics.k} · threshold={metrics.metrics.threshold_used} ·
            confidence {metrics.metrics.conf_min}–{metrics.metrics.conf_max}
            (mean {metrics.metrics.conf_mean}) ·
            std {metrics.metrics.conf_std} · high-conf={metrics.metrics.high_conf_count}
          </p>
        </div>
      )}

      {/* Chat area */}
      <div className="vlm-chat-area">
        {messages.length === 0 ? (
          <div className="vlm-empty">
            <span className="vlm-empty-icon">💬</span>
            <p className="vlm-empty-title">Ask about your search results</p>
            <p className="vlm-empty-hint">e.g. "How many matches had confidence above 0.8?"</p>
          </div>
        ) : (
          messages.map((m, i) => (
            <div key={i} className={`vlm-message vlm-message-${m.role}`}>
              <p>{m.text}</p>
            </div>
          ))
        )}
        {loading && (
          <div className="vlm-message vlm-message-assistant vlm-loading">
            <span className="vlm-dot-1" />
            <span className="vlm-dot-2" />
            <span className="vlm-dot-3" />
          </div>
        )}
        <div ref={chatBottomRef} />
      </div>

      {/* Bottom toolbar */}
      <div className="vlm-toolbar">
        <div className="vlm-toolbar-row1">
          <button
            className="vlm-report-btn"
            disabled={!isOnline}
            onClick={() => isOnline && setShowReport(true)}
          >
            Generate report
          </button>
          <div className="vlm-mode-toggle">
            <button
              className="vlm-mode-btn"
              data-active={String(mode === "qa")}
              onClick={() => setMode("qa")}
            >
              qa
            </button>
            <button
              className="vlm-mode-btn"
              data-active={String(mode === "report")}
              onClick={() => { setMode("report"); isOnline && setShowReport(true) }}
            >
              report
            </button>
          </div>
        </div>
        <div className="vlm-toolbar-row2">
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSend()}
            placeholder={isOnline ? "Ask a question about the search results…" : "Run a search to activate VLM"}
            disabled={!isOnline || loading}
            className="vlm-input"
          />
          <button
            className="vlm-send-btn"
            onClick={handleSend}
            disabled={!isOnline || !input.trim() || loading}
          >
            Send
          </button>
        </div>
      </div>

      {/* Evaluation Report Modal */}
      {showReport && sessionId && (
        <EvaluationReportModal
          sessionId={sessionId}
          onClose={() => setShowReport(false)}
        />
      )}
    </div>
  )
}
