import { useState, useRef, useCallback } from "react"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import "./AnnotatePanel.css"

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000"

const CHIP_SIZE_HINTS = {
  "128": "Vehicles, small objects",
  "256": "Helipads, depots",
  "512": "Airfields, large structures",
}

export function AnnotatePanel({ onSession }) {
  const [chips, setChips] = useState([])
  const [satImages, setSatImages] = useState([])
  const [label, setLabel] = useState("object")
  const [chipSize, setChipSize] = useState("256")
  const [threshold, setThreshold] = useState(0.6)
  const [chipDragOver, setChipDragOver] = useState(false)
  const [satDragOver, setSatDragOver] = useState(false)

  const [isSearching, setIsSearching] = useState(false)
  const [progress, setProgress]       = useState("")   // status message while processing
  const [error, setError] = useState(null)
  const [downloadUrl, setDownloadUrl] = useState(null)
  const [downloadName, setDownloadName] = useState("")
  const [matchCount, setMatchCount]   = useState(null)

  const chipInputRef = useRef(null)
  const satInputRef = useRef(null)

  const addChips = useCallback((files) => {
    if (!files) return
    const images = Array.from(files).filter((f) => f.type.startsWith("image/"))
    setChips((prev) => [...prev, ...images].slice(0, 5))
  }, [])

  const addSatImages = useCallback((files) => {
    if (!files) return
    const images = Array.from(files).filter(
      (f) => f.type.startsWith("image/") || /\.(tiff?|geotiff)$/i.test(f.name)
    )
    setSatImages((prev) => [...prev, ...images])
  }, [])

  const handleChipDrop = useCallback((e) => {
    e.preventDefault(); setChipDragOver(false); addChips(e.dataTransfer.files)
  }, [addChips])

  const handleSatDrop = useCallback((e) => {
    e.preventDefault(); setSatDragOver(false); addSatImages(e.dataTransfer.files)
  }, [addSatImages])

  const handleSearch = async () => {
    setError(null)
    setDownloadUrl(null)
    setMatchCount(null)
    setIsSearching(true)
    setProgress("Embedding reference chips…")

    try {
      const form = new FormData()
      form.append("label", label)
      form.append("chip_size", chipSize)
      form.append("similarity_threshold", String(threshold))
      form.append("top_k", "300")
      chips.forEach((chip) => form.append("reference_chips", chip))
      satImages.forEach((img) => form.append("satellite_images", img))

      setProgress(`Slicing ${satImages.length} image(s) & embedding patches…`)

      const resp = await fetch(`${API_BASE}/api/annotate/search`, {
        method: "POST",
        body: form,
      })

      if (!resp.ok) {
        const data = await resp.json().catch(() => ({ detail: resp.statusText }))
        throw new Error(data.detail || `Server error ${resp.status}`)
      }

      setProgress("Building YOLO dataset…")
      // Capture session_id before consuming the body
      const sid = resp.headers.get("x-session-id")
      if (sid && onSession) onSession(sid)

      const blob = await resp.blob()
      const url = URL.createObjectURL(blob)
      const cd = resp.headers.get("content-disposition") ?? ""
      const name = cd.match(/filename="?([^"]+)"?/)?.[1] ?? `${label}_annotated.zip`

      setDownloadUrl(url)
      setDownloadName(name)
      setMatchCount("complete")
      setProgress("")
    } catch (err) {
      setError(err.message)
      setProgress("")
    } finally {
      setIsSearching(false)
    }
  }

  const canSearch = chips.length > 0 && satImages.length > 0

  return (
    <div className="panel annotate-panel">
      {/* Reference chips */}
      <div>
        <span className="section-label">Reference Chips (1–5) ({chips.length}/5)</span>
        {chips.length > 0 ? (
          <div className="chip-grid">
            {chips.map((f, i) => (
              <div key={i} className="chip-thumb">
                <img src={URL.createObjectURL(f)} alt={f.name} />
                <button
                  className="chip-thumb-remove"
                  onClick={() => setChips((prev) => prev.filter((_, j) => j !== i))}
                >✕</button>
              </div>
            ))}
            {chips.length < 5 && (
              <button className="chip-add" onClick={() => chipInputRef.current?.click()}>+</button>
            )}
          </div>
        ) : (
          <div
            className="dropzone"
            data-over={String(chipDragOver)}
            onDragOver={(e) => { e.preventDefault(); setChipDragOver(true) }}
            onDragLeave={() => setChipDragOver(false)}
            onDrop={handleChipDrop}
            onClick={() => chipInputRef.current?.click()}
          >
            <p>Drag &amp; drop up to 5 chips,<br />or click to browse</p>
          </div>
        )}
        <input
          ref={chipInputRef}
          type="file"
          accept="image/*"
          multiple
          style={{ display: "none" }}
          onChange={(e) => addChips(e.target.files)}
        />
      </div>

      {/* Satellite images */}
      <div>
        <span className="section-label">
          Satellite Images ({satImages.length} loaded)
        </span>
        {satImages.length > 0 ? (
          <div className="sat-list">
            {satImages.map((f, i) => (
              <div key={i} className="sat-list-item">
                <span className="sat-list-name">{f.name}</span>
                <button
                  className="sat-list-remove"
                  onClick={() => setSatImages((prev) => prev.filter((_, j) => j !== i))}
                >✕</button>
              </div>
            ))}
            <button
              className="sat-add-btn"
              onClick={() => satInputRef.current?.click()}
            >+ Add more</button>
          </div>
        ) : (
          <div
            className="dropzone"
            data-over={String(satDragOver)}
            onDragOver={(e) => { e.preventDefault(); setSatDragOver(true) }}
            onDragLeave={() => setSatDragOver(false)}
            onDrop={handleSatDrop}
            onClick={() => satInputRef.current?.click()}
          >
            <p>Drop PNG / JPG / GeoTIFF<br />Multiple images supported</p>
          </div>
        )}
        <input
          ref={satInputRef}
          type="file"
          accept="image/*,.tiff,.tif"
          multiple
          style={{ display: "none" }}
          onChange={(e) => addSatImages(e.target.files)}
        />
      </div>

      {/* Label */}
      <div>
        <Label className="section-label">Label</Label>
        <Input value={label} onChange={(e) => setLabel(e.target.value)} placeholder="object" />
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
        <p className="size-hint">{CHIP_SIZE_HINTS[chipSize]}</p>
      </div>

      {/* Threshold */}
      <div>
        <div className="threshold-header">
          <span className="section-label" style={{ marginBottom: 0 }}>Similarity Threshold</span>
          <span className="threshold-value">{threshold.toFixed(2)}</span>
        </div>
        <Slider
          min={0.5}
          max={0.95}
          step={0.01}
          value={[threshold]}
          onValueChange={([v]) => setThreshold(v)}
        />
        <div className="threshold-labels">
          <span>0.50 (more results)</span>
          <span>0.95 (precise)</span>
        </div>
      </div>

      {/* Progress */}
      {isSearching && progress && (
        <p className="annotate-progress">{progress}</p>
      )}

      {/* Error */}
      {error && <p className="search-error">{error}</p>}

      {/* Download */}
      {downloadUrl && (
        <div className="search-result">
          <span className="search-result-count">Annotation complete</span>
          <a className="download-btn" href={downloadUrl} download={downloadName}>
            Download YOLO zip
          </a>
        </div>
      )}

      <button
        className="search-btn"
        disabled={!canSearch || isSearching}
        onClick={handleSearch}
      >
        {isSearching ? "Annotating…" : "Search & Annotate"}
      </button>
    </div>
  )
}
