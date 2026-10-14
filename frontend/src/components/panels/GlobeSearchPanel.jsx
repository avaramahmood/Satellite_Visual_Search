import { useState, useRef, useCallback } from "react"
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import "./GlobeSearchPanel.css"

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000"

const CHIP_SIZE_HINTS = {
  "128": "Vehicles, small objects",
  "256": "Helipads, depots",
  "512": "Airfields, large structures",
}

const DATASET_META = {
  xview: { label: "xView",  hint: "846 GeoTIFFs · globe hotspots" },
  dior:  { label: "DIOR",   hint: "Dataset generation · object classes" },
}

export function GlobeSearchPanel({ onSearch, onSession }) {
  const [dataset, setDataset]     = useState("xview")
  const [mode, setMode] = useState("image")
  const [chips, setChips] = useState([])
  const [label, setLabel] = useState("object")
  const [chipSize, setChipSize] = useState("256")
  const [threshold, setThreshold] = useState(0.6)
  const [topK, setTopK]           = useState(200)
  const [textQuery, setTextQuery] = useState("")
  const [dragOver, setDragOver]   = useState(false)

  const [isSearching, setIsSearching] = useState(false)
  const [error, setError] = useState(null)
  const [result, setResult] = useState(null)   // { count, points, download_token }

  const fileInputRef = useRef(null)

  const addChips = useCallback((files) => {
    if (!files) return
    const images = Array.from(files).filter((f) => f.type.startsWith("image/"))
    setChips((prev) => [...prev, ...images].slice(0, 5))
  }, [])

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault()
      setDragOver(false)
      addChips(e.dataTransfer.files)
    },
    [addChips]
  )

  const handleSearch = async () => {
    setError(null)
    setResult(null)
    setIsSearching(true)

    try {
      const form = new FormData()
      form.append("label", label)
      form.append("chip_size", chipSize)
      form.append("similarity_threshold", String(threshold))
      form.append("top_k", String(topK))

      let endpoint
      if (mode === "image") {
        endpoint = `${API_BASE}/api/search/globe/image`
        chips.forEach((chip) => form.append("chips", chip))
        form.append("dataset", dataset)   // dataset selector is image-only
      } else {
        endpoint = `${API_BASE}/api/search/globe/text`
        form.append("query", textQuery.trim())
      }

      const resp = await fetch(endpoint, { method: "POST", body: form })
      if (!resp.ok) {
        const detail = await resp.json().catch(() => ({ detail: resp.statusText }))
        throw new Error(detail.detail || `Server error ${resp.status}`)
      }

      const data = await resp.json()
      setResult(data)

      // Forward matched points to App so globe can plot them
      if (onSearch && data.points) {
        onSearch({ points: data.points.map((p) => ({ lat: p.lat, lon: p.lng })) })
      }
      // Forward session_id to App for VLM chat
      if (onSession && data.session_id) {
        onSession(data.session_id)
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setIsSearching(false)
    }
  }

  const handleDownload = () => {
    if (!result?.download_token) return
    window.open(`${API_BASE}/api/download/${result.download_token}`, "_blank")
  }

  const canSearch = mode === "image" ? chips.length > 0 : textQuery.trim().length > 0

  return (
    <div className="panel">
      {/* Mode tabs */}
      <Tabs value={mode} onValueChange={(v) => { setMode(v); setResult(null); setError(null) }}>
        <TabsList className="w-full">
          <TabsTrigger value="image" className="flex-1">Image chips</TabsTrigger>
          <TabsTrigger value="text" className="flex-1">Text query</TabsTrigger>
        </TabsList>

        {/* Image chips mode */}
        <TabsContent value="image" className="mt-4 flex flex-col gap-3">
          {/* Dataset selector — image mode only */}
          <div>
            <span className="section-label">Dataset</span>
            <div className="size-group">
              {Object.entries(DATASET_META).map(([key, meta]) => (
                <button
                  key={key}
                  className="size-btn"
                  data-active={String(dataset === key)}
                  onClick={() => { setDataset(key); setResult(null); setError(null) }}
                >
                  {meta.label}
                </button>
              ))}
            </div>
            <p className="size-hint">{DATASET_META[dataset].hint}</p>
          </div>

          <div>
            <span className="section-label">Reference Chips ({chips.length}/5)</span>

            {chips.length > 0 ? (
              <div className="chip-grid">
                {chips.map((f, i) => (
                  <div key={i} className="chip-thumb">
                    <img src={URL.createObjectURL(f)} alt={f.name} />
                    <button
                      className="chip-thumb-remove"
                      onClick={() => setChips((prev) => prev.filter((_, j) => j !== i))}
                    >
                      ✕
                    </button>
                  </div>
                ))}
                {chips.length < 5 && (
                  <button className="chip-add" onClick={() => fileInputRef.current?.click()}>
                    +
                  </button>
                )}
              </div>
            ) : (
              <div
                className="dropzone"
                data-over={String(dragOver)}
                onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
                onDragLeave={() => setDragOver(false)}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
              >
                <p>Drag & drop up to 5 chips,<br />or click to browse</p>
              </div>
            )}

            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              multiple
              style={{ display: "none" }}
              onChange={(e) => addChips(e.target.files)}
            />
          </div>

          <div>
            <Label className="section-label">Label</Label>
            <Input
              value={label}
              onChange={(e) => setLabel(e.target.value)}
              placeholder="object"
            />
          </div>
        </TabsContent>

        {/* Text query mode */}
        <TabsContent value="text" className="mt-4 flex flex-col gap-3">
          <div>
            <Label className="section-label">Describe what to find</Label>
            <Input
              value={textQuery}
              onChange={(e) => setTextQuery(e.target.value)}
              placeholder="e.g. military aircraft on runway"
              onKeyDown={(e) => e.key === "Enter" && canSearch && !isSearching && handleSearch()}
            />
          </div>
        </TabsContent>
      </Tabs>

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
            >
              {s}px
            </button>
          ))}
        </div>
        <p className="size-hint">{CHIP_SIZE_HINTS[chipSize]}</p>
      </div>

      {/* Similarity threshold */}
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

      {/* Max matches (k) */}
      <div>
        <div className="threshold-header">
          <span className="section-label" style={{ marginBottom: 0 }}>Max Matches (k)</span>
          <span className="threshold-value">{topK}</span>
        </div>
        <Slider
          min={1}
          max={500}
          step={1}
          value={[topK]}
          onValueChange={([v]) => setTopK(v)}
        />
        <div className="threshold-labels">
          <span>1 (top match)</span>
          <span>500 (all)</span>
        </div>
      </div>

      {/* Error */}
      {error && (
        <p className="search-error">{error}</p>
      )}

      {/* Result summary + download */}
      {result && (
        <div className="search-result">
          <span className="search-result-count">{result.count} matches found</span>
          {result.download_token && (
            <button className="download-btn" onClick={handleDownload}>
              Download YOLO zip
            </button>
          )}
        </div>
      )}

      {/* Search */}
      <button
        className="search-btn"
        disabled={!canSearch || isSearching}
        onClick={handleSearch}
      >
        {isSearching ? "Searching…" : "Search Globe"}
      </button>
    </div>
  )
}
