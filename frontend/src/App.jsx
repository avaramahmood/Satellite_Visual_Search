import { useState } from "react"
import { SpaceBackground } from "@/components/SpaceBackground"
import { GlobeHero } from "@/components/GlobeHero"
import { Navbar } from "@/components/Navbar"
import { LiveMap } from "@/components/LiveMap"
import { GlobeSearchPanel } from "@/components/panels/GlobeSearchPanel"
import { AnnotatePanel } from "@/components/panels/AnnotatePanel"
import { LiveSearchPanel } from "@/components/panels/LiveSearchPanel"
import { VlmChatPanel } from "@/components/panels/VlmChatPanel"
import "./App.css"

const PANEL_TITLES = {
  "globe-search": "Globe Search",
  "annotate":     "Search & Annotate",
  "live-search":  "Live Search",
  "vlm-chat":     "Intelligence Chat",
}

export function App() {
  const [activePanel,  setActivePanel]  = useState(null)
  const [matchedPoints, setMatchedPoints] = useState(undefined)
  const [flyTo,        setFlyTo]        = useState(null)
  const [liveResult,   setLiveResult]   = useState(null)
  // Global session ID — set when any search completes; used by VLM chat + reports
  const [sessionId,    setSessionId]    = useState(null)

  const handleNavClick = (id) => {
    setActivePanel((prev) => (prev === id ? null : id))
    if (id !== "live-search") { setLiveResult(null); setFlyTo(null) }
  }

  const handleSearch = ({ points } = {}) => {
    setMatchedPoints(points ?? [])
  }

  const handleLiveResult = (result) => {
    setFlyTo({ lat: result.lat, lng: result.lng, zoom: 13 })
    setLiveResult(result)
  }

  const handleLiveClear = () => {
    setLiveResult(null)
    setFlyTo(null)
  }

  const panelOpen = activePanel !== null

  return (
    <div className="app-root dark">
      {/* Layer 0 — full-screen rotating starfield */}
      <SpaceBackground />

      {/* Layer 1 — globe sphere on the right (transparent canvas) */}
      <GlobeHero
        matchedPoints={matchedPoints}
        flyTo={flyTo}
        dimmed={liveResult !== null}
        liveMode={activePanel === "live-search"}
      />

      {/* Left-side gradient vignette for hero readability */}
      <div className="hero-gradient" />

      {/* Navbar */}
      <Navbar activePanel={activePanel} onNavClick={handleNavClick} />

      {/* Hero text — fades out when a panel is open */}
      <section className="hero" data-hidden={String(panelOpen)}>
        <span className="hero-eyebrow">AI-Powered Satellite Intelligence</span>
        <h1 className="hero-title">
          Satellite<br />
          <span className="hero-title-accent">Visual</span>
          <br />Search
        </h1>
        <p className="hero-sub">
          Find any object across 1M+ indexed chips<br />
          using DINOv2 + CLIP — in seconds.
        </p>
        <div className="hero-cta">
          <button className="hero-btn-primary" onClick={() => setActivePanel("globe-search")}>
            Start Searching →
          </button>
          <button className="hero-btn-ghost" onClick={() => setActivePanel("live-search")}>
            ◉ Live Search
          </button>
        </div>
      </section>

      {/* Feature popup panel */}
      <div className="feature-popup" data-open={String(panelOpen)}>
        {activePanel && (
          <>
            <div className="popup-header">
              <h2 className="popup-title">{PANEL_TITLES[activePanel]}</h2>
              <button className="popup-close" onClick={() => { setActivePanel(null); handleLiveClear() }}>✕</button>
            </div>
            <div className="popup-body">
              {activePanel === "globe-search" && (
                <GlobeSearchPanel onSearch={handleSearch} onSession={setSessionId} />
              )}
              {activePanel === "annotate"     && (
                <AnnotatePanel onSession={setSessionId} />
              )}
              {activePanel === "live-search"  && (
                <LiveSearchPanel
                  onResult={handleLiveResult}
                  onClear={handleLiveClear}
                  onFlyTo={setFlyTo}
                  onSession={setSessionId}
                />
              )}
              {activePanel === "vlm-chat"     && (
                <VlmChatPanel sessionId={sessionId} />
              )}
            </div>
          </>
        )}
      </div>

      {/* MapTiler satellite map — appears on right when Live Search returns a location */}
      {liveResult && (
        <LiveMap
          location={liveResult}
          detections={liveResult.detections}
          searchParams={{
            chipFile:  liveResult.chipFile,
            chipSize:  liveResult.chipSize,
            threshold: liveResult.threshold,
          }}
          onClose={handleLiveClear}
          onSessionId={setSessionId}
        />
      )}
    </div>
  )
}

export default App
