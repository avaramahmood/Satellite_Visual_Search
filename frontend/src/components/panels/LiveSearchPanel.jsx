import { useState, useRef, useCallback, useEffect } from "react"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import "./LiveSearchPanel.css"

const MAPTILER_KEY = import.meta.env.VITE_MAPTILER_KEY ?? ""
const API_BASE     = import.meta.env.VITE_API_BASE || "http://localhost:8000"

async function geocodePlace(query) {
  const res  = await fetch(`${API_BASE}/api/live/geocode?q=${encodeURIComponent(query)}`)
  const data = await res.json()
  if (!res.ok) throw new Error(data.detail ?? "Place not found")
  return { lat: data[0].lat, lng: data[0].lng, placeName: data[0].display_name }
}

async function fetchSuggestions(query) {
  const res  = await fetch(`${API_BASE}/api/live/geocode?q=${encodeURIComponent(query)}`)
  if (!res.ok) return []
  return res.json()  // [{display_name, short, lat, lng}]
}

async function fetchSatelliteImage(lat, lng, zoom = 15) {
  const url = `${API_BASE}/api/live/satellite?lat=${lat}&lng=${lng}&zoom=${zoom}&grid=4`
  const res = await fetch(url)
  if (!res.ok) throw new Error(`Satellite image fetch failed (${res.status})`)
  return res.blob()
}

export function LiveSearchPanel({ onClear, onFlyTo, onSession }) {
  const [location, setLocation]       = useState("")
  const [chip, setChip]               = useState(null)
  const [chipSize, setChipSize]       = useState("256")
  const [threshold, setThreshold]     = useState(0.60)
  const [status, setStatus]           = useState("idle") // idle | loading | done | error
  const [progress, setProgress]       = useState("")
  const [errorMsg, setErrorMsg]       = useState("")
  const [downloadUrl, setDownloadUrl] = useState(null)
  const [downloadName, setDownloadName] = useState("")
  const [dragOver, setDragOver]       = useState(false)
  const [suggestions, setSuggestions] = useState([])
  const [showSugg, setShowSugg]       = useState(false)
  const geoRef                        = useRef(null)  // cached geocode result
  const fileInputRef                  = useRef(null)

  // Debounced suggestions via backend proxy
  useEffect(() => {
    if (location.trim().length < 2) { setSuggestions([]); setShowSugg(false); return }
    const timer = setTimeout(async () => {
      const results = await fetchSuggestions(location.trim())
      if (results.length) {
        setSuggestions(results)
        setShowSugg(true)
      }
    }, 380)
    return () => clearTimeout(timer)
  }, [location])

  const selectSuggestion = (s) => {
    setLocation(s.short)
    geoRef.current = { lat: s.lat, lng: s.lng, placeName: s.label }
    setSuggestions([])
    setShowSugg(false)
    if (onFlyTo) onFlyTo({ lat: s.lat, lng: s.lng, zoom: 13 })
  }

  // Geocode + zoom globe on Enter (location field, no chip required)
  const handleLocate = useCallback(async () => {
    const q = location.trim()
    if (!q) return
    setErrorMsg("")
    try {
      const geo = await geocodePlace(q)
      geoRef.current = geo
      if (onFlyTo) onFlyTo({ lat: geo.lat, lng: geo.lng, zoom: 13 })
    } catch (err) {
      setErrorMsg(err.message)
    }
  }, [location, onFlyTo])

  const handleChipDrop = useCallback((e) => {
    e.preventDefault(); setDragOver(false)
    const f = e.dataTransfer.files[0]
    if (f?.type.startsWith("image/")) setChip(f)
  }, [])

  const handleSearch = async () => {
    if (!location.trim() || !chip) return
    setStatus("loading")
    setErrorMsg("")
    setDownloadUrl(null)
    setProgress("Geocoding location…")
    if (onClear) onClear()

    try {
      // 1. Geocode (use cached if available)
      const geo = geoRef.current ?? await geocodePlace(location.trim())
      geoRef.current = geo

      // Zoom globe to location with animation
      if (onFlyTo) onFlyTo({ lat: geo.lat, lng: geo.lng, zoom: 15 })

      // 2. Fetch satellite image from MapTiler static API
      setProgress("Fetching satellite imagery…")
      const satBlob = await fetchSatelliteImage(geo.lat, geo.lng, 15)

      // 3. POST to annotate endpoint (same as Search & Annotate)
      setProgress("Embedding chip & satellite patches…")
      const form = new FormData()
      form.append("label",                location.trim())
      form.append("chip_size",            chipSize)
      form.append("similarity_threshold", String(threshold))
      form.append("top_k",               "300")
      form.append("reference_chips",      chip)
      form.append("satellite_images",     satBlob, "live_satellite.jpg")

      setProgress("Searching for matches…")
      const resp = await fetch(`${API_BASE}/api/annotate/search`, {
        method: "POST",
        body: form,
      })

      if (!resp.ok) {
        const data = await resp.json().catch(() => ({ detail: resp.statusText }))
        throw new Error(data.detail || `Server error ${resp.status}`)
      }

      // Capture session_id for VLM chat before consuming body
      const sid = resp.headers.get("x-session-id")
      if (sid && onSession) onSession(sid)

      setProgress("Building YOLO dataset…")
      const blob = await resp.blob()
      const url  = URL.createObjectURL(blob)
      const cd   = resp.headers.get("content-disposition") ?? ""
      const name = cd.match(/filename="?([^"]+)"?/)?.[1] ?? `${location.trim()}_live_matches.zip`

      setDownloadUrl(url)
      setDownloadName(name)
      setStatus("done")
      setProgress("")
    } catch (err) {
      setErrorMsg(err.message)
      setStatus("error")
      setProgress("")
    }
  }

  const handleClear = () => {
    setStatus("idle"); setDownloadUrl(null); setProgress(""); setErrorMsg("")
    setLocation(""); setChip(null); geoRef.current = null
    if (onClear) onClear()
  }

  const canSearch = location.trim().length > 0 && chip !== null && status !== "loading"

  return (
    <div className="live-panel">
      {/* Location */}
      <div style={{ position: "relative" }}>
        <Label className="section-label">Location</Label>
        <Input
          value={location}
          onChange={(e) => { setLocation(e.target.value); geoRef.current = null }}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              if (showSugg && suggestions.length > 0) {
                selectSuggestion(suggestions[0])   // pick top result
              } else if (canSearch) {
                handleSearch()
              } else {
                handleLocate()
              }
            }
            if (e.key === "Escape") setShowSugg(false)
          }}
          onFocus={() => suggestions.length > 0 && setShowSugg(true)}
          onBlur={() => setTimeout(() => setShowSugg(false), 150)}
          placeholder="e.g. Baghdad, Iraq"
          autoComplete="off"
        />
        {showSugg && suggestions.length > 0 && (
          <ul className="live-suggestions">
            {suggestions.map((s, i) => (
              <li key={i} className="live-suggestion-item" onMouseDown={() => selectSuggestion(s)}>
                <span className="live-sugg-short">{s.short}</span>
                <span className="live-sugg-full">{s.label}</span>
              </li>
            ))}
          </ul>
        )}
        <p className="live-hint">Press Enter to zoom globe · add chip to search</p>
      </div>

      {/* Chip upload */}
      <div>
        <span className="section-label">Query Chip</span>
        {chip ? (
          <div className="live-chip-preview">
            <img src={URL.createObjectURL(chip)} alt="chip" className="live-chip-img" />
            <div className="live-chip-info">
              <span className="live-chip-name">{chip.name}</span>
              <button className="live-chip-remove" onClick={() => setChip(null)}>Remove</button>
            </div>
          </div>
        ) : (
          <div
            className="dropzone"
            data-over={String(dragOver)}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleChipDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <p>Drop a reference chip image,<br />or click to browse</p>
          </div>
        )}
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          style={{ display: "none" }}
          onChange={(e) => { if (e.target.files[0]) setChip(e.target.files[0]) }}
        />
      </div>

      {/* Chip size */}
      <div>
        <span className="section-label">Chip Size</span>
        <div className="size-group">
          {["128", "256", "512"].map((s) => (
            <button
              key={s}
              className="size-btn"
              data-active={String(chipSize === s)}
              onClick={() => setChipSize(s)}
            >{s}px</button>
          ))}
        </div>
      </div>

      {/* Similarity threshold */}
      <div>
        <div className="threshold-header">
          <span className="section-label" style={{ marginBottom: 0 }}>Similarity Threshold</span>
          <span className="threshold-value">{threshold.toFixed(2)}</span>
        </div>
        <Slider
          min={0.30}
          max={0.95}
          step={0.01}
          value={[threshold]}
          onValueChange={([v]) => setThreshold(v)}
        />
        <div className="threshold-labels">
          <span>0.30 (more)</span>
          <span>0.95 (precise)</span>
        </div>
      </div>

      {/* Progress */}
      {status === "loading" && progress && (
        <p className="annotate-progress">{progress}</p>
      )}

      {/* Error — shown for both locate-only failures and full search failures */}
      {errorMsg && <p className="live-error">{errorMsg}</p>}

      {/* Download result */}
      {status === "done" && downloadUrl && (
        <div className="search-result">
          <span className="search-result-count">Matches found</span>
          <a className="download-btn" href={downloadUrl} download={downloadName}>
            Download YOLO zip
          </a>
        </div>
      )}

      {/* Main button */}
      <button
        className="search-btn live-search-btn"
        disabled={!canSearch}
        onClick={handleSearch}
      >
        {status === "loading" ? "Processing…" : "Generate Live Matches"}
      </button>

      {status === "done" && (
        <button className="live-clear-btn" onClick={handleClear}>Clear &amp; Reset</button>
      )}
    </div>
  )
}
