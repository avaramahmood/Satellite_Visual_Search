import { useEffect, useRef, useState, useCallback } from "react"
import * as maptiler from "@maptiler/sdk"
import "@maptiler/sdk/dist/maptiler-sdk.css"
import { EvaluationReportModal } from "@/components/EvaluationReportModal"
import "./LiveMap.css"

maptiler.config.apiKey = import.meta.env.VITE_MAPTILER_KEY ?? ""

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000"

export function LiveMap({ location, detections, searchParams, onClose, onSessionId }) {
  const mapContainerRef  = useRef(null)
  const mapRef           = useRef(null)
  const [analyzing,      setAnalyzing]      = useState(false)
  const [annotatedUrl,   setAnnotatedUrl]   = useState(null)
  const [analyzeError,   setAnalyzeError]   = useState(null)
  const [matchCount,     setMatchCount]     = useState(null)
  const [analysis,       setAnalysis]       = useState(null)
  const [sessionId,      setSessionId]      = useState(null)
  const [showAnalysis,   setShowAnalysis]   = useState(false)
  const [showReport,     setShowReport]     = useState(false)

  // ── Initialize map ────────────────────────────────────────────────────────
  useEffect(() => {
    if (!mapContainerRef.current || !location) return
    if (mapRef.current) { mapRef.current.remove(); mapRef.current = null }

    const map = new maptiler.Map({
      container: mapContainerRef.current,
      style: maptiler.MapStyle.SATELLITE,
      center: [location.lng, location.lat],
      zoom: 14,
      attributionControl: false,
      preserveDrawingBuffer: true,
    })

    map.addControl(new maptiler.NavigationControl({ visualizePitch: false }), "top-right")

    map.on("load", () => {
      if (!detections?.length) return
      const features = detections.map((d) => ({
        type: "Feature",
        geometry: {
          type: "Polygon",
          coordinates: [[
            [d.lng_min, d.lat_max],
            [d.lng_max, d.lat_max],
            [d.lng_max, d.lat_min],
            [d.lng_min, d.lat_min],
            [d.lng_min, d.lat_max],
          ]],
        },
        properties: { label: d.label, conf: d.confidence },
      }))
      map.addSource("detections", { type: "geojson", data: { type: "FeatureCollection", features } })
      map.addLayer({ id: "detection-fill", type: "fill",   source: "detections", paint: { "fill-color": "#ff6428", "fill-opacity": 0.18 } })
      map.addLayer({ id: "detection-line", type: "line",   source: "detections", paint: { "line-color": "#ff6428", "line-width": 1.5 } })
    })

    mapRef.current = map
    return () => { map.remove(); mapRef.current = null }
  }, [location, detections])

  // ── Run DINO analysis ────────────────────────────────────────────────────
  const handleAnalyze = useCallback(() => {
    const map = mapRef.current
    if (!map || !searchParams?.chipFile) return

    setAnalyzing(true)
    setAnnotatedUrl(null)
    setAnalyzeError(null)
    setMatchCount(null)
    setAnalysis(null)
    setSessionId(null)
    setShowAnalysis(false)

    const doCapture = () => {
      const canvas = mapContainerRef.current?.querySelector("canvas.maplibregl-canvas")
      if (!canvas) {
        setAnalyzeError("Could not access map canvas.")
        setAnalyzing(false)
        return
      }

      canvas.toBlob(async (mapBlob) => {
        try {
          const form = new FormData()
          form.append("query_chip",           searchParams.chipFile)
          form.append("map_screenshot",       mapBlob, "live_map.jpg")
          form.append("chip_size",            String(searchParams.chipSize))
          form.append("similarity_threshold", String(searchParams.threshold))
          form.append("location_name",        location?.placeName ?? `${location?.lat}, ${location?.lng}`)
          form.append("target_label",         "target object")

          const res = await fetch(`${API_BASE}/api/live/search`, {
            method: "POST",
            body: form,
          })

          if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }))
            throw new Error(err.detail || `Server error ${res.status}`)
          }

          const data = await res.json()

          setMatchCount(data.match_count ?? null)
          setAnnotatedUrl(data.annotated_image ?? null)

          if (data.session_id) {
            setSessionId(data.session_id)
            onSessionId?.(data.session_id)
          }

          if (data.analysis) {
            setAnalysis(data.analysis)
            setShowAnalysis(true)
          }
        } catch (err) {
          setAnalyzeError(err.message)
        } finally {
          setAnalyzing(false)
        }
      }, "image/jpeg")
    }

    if (map.loaded()) {
      map.once("idle", doCapture)
      map.triggerRepaint()
    } else {
      map.once("idle", doCapture)
    }
  }, [searchParams, location, onSessionId])

  const clearAnnotation = () => {
    setAnnotatedUrl(null)
    setAnalyzeError(null)
    setMatchCount(null)
    setAnalysis(null)
    setShowAnalysis(false)
  }

  const canAnalyze = !!searchParams?.chipFile && !analyzing

  return (
    <div className="live-map-overlay">
      {/* Header */}
      <div className="live-map-header">
        <span className="live-map-title">
          Live View — {location?.placeName ?? `${location?.lat.toFixed(4)}, ${location?.lng.toFixed(4)}`}
        </span>

        <div className="live-map-header-actions">
          {/* Context loaded badge */}
          {sessionId && (
            <span className="live-context-badge" title={`Session: ${sessionId}`}>
              <span className="live-context-dot" />
              Context JSON loaded
            </span>
          )}

          {/* Analyze / result actions */}
          {canAnalyze && !annotatedUrl && (
            <button className="live-analyze-btn" onClick={handleAnalyze}>
              ◉ Run DINO Analysis
            </button>
          )}
          {analyzing && (
            <span className="live-analyze-status">Analyzing…</span>
          )}
          {annotatedUrl && (
            <>
              {matchCount !== null && (
                <span className="live-analyze-count">{matchCount} match{matchCount !== 1 ? "es" : ""}</span>
              )}
              {analysis && (
                <button
                  className="live-vlm-btn"
                  onClick={() => setShowAnalysis((v) => !v)}
                >
                  {showAnalysis ? "Hide Analysis" : "VLM Analysis"}
                </button>
              )}
              {sessionId && (
                <button
                  className="live-report-btn"
                  onClick={() => setShowReport(true)}
                >
                  📊 Report
                </button>
              )}
              <button className="live-reanalyze-btn" onClick={() => { clearAnnotation(); handleAnalyze() }}>
                Re-analyze
              </button>
              <button className="live-clear-annotation-btn" onClick={clearAnnotation}>
                Show Map
              </button>
            </>
          )}

          <button className="live-map-close" onClick={onClose}>✕</button>
        </div>
      </div>

      {/* Error bar */}
      {analyzeError && (
        <div className="live-analyze-error-bar">
          {analyzeError}
          <button onClick={() => setAnalyzeError(null)}>✕</button>
        </div>
      )}

      {/* Map + annotated image overlay */}
      <div className="live-map-body">
        <div ref={mapContainerRef} className="live-map-canvas" />

        {annotatedUrl && (
          <div className="live-annotated-overlay">
            <img src={annotatedUrl} alt="DINO analysis result" className="live-annotated-img" />
            <div className="live-annotated-badge">
              DINO · {searchParams?.chipSize}px · threshold {searchParams?.threshold?.toFixed(2)}
            </div>
          </div>
        )}
      </div>

      {/* VLM Analysis panel */}
      {showAnalysis && analysis && (
        <div className="live-analysis-panel">
          <div className="live-analysis-header">
            <span className="live-analysis-title">
              <span className="live-analysis-dot" />
              VLM Intelligence Analysis
            </span>
            <button className="live-analysis-close" onClick={() => setShowAnalysis(false)}>✕</button>
          </div>
          <div className="live-analysis-body">
            <pre className="live-analysis-text">{analysis}</pre>
          </div>
        </div>
      )}

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
