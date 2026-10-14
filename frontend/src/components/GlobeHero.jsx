import { useEffect, useRef, useState, useCallback } from "react"
import Globe from "react-globe.gl"
import "./GlobeHero.css"

const MAPTILER_KEY = import.meta.env.VITE_MAPTILER_KEY ?? ""
const GLOBE_DIV_ID = "we-earth-globe"

// ── WebGL Earth 2 loader singleton ──────────────────────────────────────────
let _weState = "idle" // idle | loading | ready
const _weWaiters = []

function loadWebGLEarth(cb) {
  if (_weState === "ready") { cb(); return }
  _weWaiters.push(cb)
  if (_weState !== "idle") return
  _weState = "loading"

  // WE2 requires its CSS or the globe canvas is invisible
  const link = document.createElement("link")
  link.rel  = "stylesheet"
  link.href = "https://www.webglearth.com/v2/api.css"
  document.head.appendChild(link)

  const s = document.createElement("script")
  s.src = "https://www.webglearth.com/v2/api.js"
  s.onload  = () => { _weState = "ready"; _weWaiters.splice(0).forEach(fn => fn()) }
  s.onerror = () => { _weState = "idle" }
  document.head.appendChild(s)
}

// ── WebGL Earth 2 sub-component (Live Search only) ──────────────────────────
function SatelliteGlobe({ flyTo, dimmed }) {
  const earthRef    = useRef(null)
  const rotateRaf   = useRef(null)
  const flyRaf      = useRef(null)
  const flyingRef   = useRef(false)
  const landedRef   = useRef(false)  // true after first fly-to completes → stop auto-rotate
  const lngRef      = useRef(20)
  const pendingFly  = useRef(null)   // flyTo that arrived before globe was ready
  const [loaded, setLoaded] = useState(false)

  const startFlyTo = useCallback((target) => {
    const earth = earthRef.current
    if (!earth) return

    if (flyRaf.current) cancelAnimationFrame(flyRaf.current)
    flyingRef.current = true

    const startLat   = 20
    const startLng   = lngRef.current % 360
    const startZoom  = 2
    const spinOffset = 2.5 * 360
    const duration   = 3200
    const startTime  = Date.now()
    const ease = (t) => t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2

    const tick = () => {
      if (!earthRef.current) return  // stopped on unmount
      const t = Math.min((Date.now() - startTime) / duration, 1)
      const e = ease(t)
      const lat  = startLat + (target.lat - startLat) * e
      const lng  = startLng + spinOffset * (1 - e) + (target.lng - startLng) * e
      const zoom = startZoom + ((target.zoom ?? 13) - startZoom) * e
      earth.setView([lat, lng], zoom)
      lngRef.current = lng
      if (t < 1) flyRaf.current = requestAnimationFrame(tick)
      else { flyingRef.current = false; landedRef.current = true }
    }
    flyRaf.current = requestAnimationFrame(tick)
  }, [])

  useEffect(() => {
    let alive = true

    const init = () => {
      if (!alive) return
      const divEl = document.getElementById(GLOBE_DIV_ID)
      if (!divEl) return
      divEl.innerHTML = ""   // clear any stale canvas from previous mount

      try {
        const earth = new window.WE.map(GLOBE_DIV_ID, {
          center: [20, lngRef.current],
          zoom: 2,
          dragging: true,
          scrollWheelZoom: true,
          atmosphere: true,
          preserveDrawingBuffer: true,
        })
        window.WE.tileLayer(
          `https://api.maptiler.com/tiles/satellite-v2/{z}/{x}/{y}.jpg?key=${MAPTILER_KEY}`,
          { tileSize: 256, minZoom: 0, maxZoom: 18, attribution: "© MapTiler" }
        ).addTo(earth)

        earthRef.current = earth
        setLoaded(true)

        // Apply any flyTo that arrived before the globe was ready
        if (pendingFly.current) {
          startFlyTo(pendingFly.current)
          pendingFly.current = null
        }

        // Auto-rotation — runs only while no destination has been set
        const rotate = () => {
          if (!earthRef.current) return         // stops on unmount
          if (!landedRef.current && !flyingRef.current) {
            lngRef.current += 0.18
            earthRef.current.setView([20, lngRef.current % 360], 2)
          }
          rotateRaf.current = requestAnimationFrame(rotate)
        }
        rotateRaf.current = requestAnimationFrame(rotate)
      } catch (e) {
        console.error("[SatelliteGlobe] init failed:", e)
      }
    }

    loadWebGLEarth(init)

    return () => {
      alive = false
      landedRef.current = false
      if (rotateRaf.current) cancelAnimationFrame(rotateRaf.current)
      if (flyRaf.current)    cancelAnimationFrame(flyRaf.current)
      try { earthRef.current?.remove() } catch (_) {}
      earthRef.current = null
    }
  }, [startFlyTo])

  // ── Fly-to trigger ─────────────────────────────────────────────────────────
  useEffect(() => {
    if (!flyTo) return
    if (earthRef.current) {
      startFlyTo(flyTo)
    } else {
      pendingFly.current = flyTo  // globe not ready yet — apply once it is
    }
  }, [flyTo, startFlyTo])

  return (
    <div
      className="globe-root"
      data-loaded={String(loaded)}
      data-dimmed={String(!!dimmed)}
      data-live="true"
    >
      <div id={GLOBE_DIV_ID} style={{ width: "100%", height: "100%" }} />
    </div>
  )
}

// ── react-globe.gl sub-component (all other panels) ─────────────────────────
function ReactGlobe({ matchedPoints, flyTo, dimmed }) {
  const globeRef     = useRef(undefined)
  const containerRef = useRef(null)
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 })
  const [loaded, setLoaded]         = useState(false)

  const points =
    matchedPoints && matchedPoints.length > 0
      ? matchedPoints.map((p) => ({
          lat: p.lat,
          lng: p.lon,
          color: "rgb(255, 0, 0)",
        }))
      : []

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries)
        setDimensions({ width: entry.contentRect.width, height: entry.contentRect.height })
    })
    observer.observe(el)
    setDimensions({ width: el.clientWidth, height: el.clientHeight })
    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    if (!flyTo || !globeRef.current) return
    const globe = globeRef.current
    globe.controls().autoRotate = false
    globe.pointOfView(
      { lat: flyTo.lat, lng: flyTo.lng, altitude: flyTo.altitude ?? 0.08 },
      2200
    )
  }, [flyTo])

  const handleGlobeReady = useCallback(() => {
    const globe = globeRef.current
    if (!globe) return
    globe.renderer().setClearColor(0x000000, 0)
    globe.pointOfView({ lat: 20, lng: 20, altitude: 1.9 }, 0)
    const controls = globe.controls()
    controls.autoRotate      = true
    controls.autoRotateSpeed = 0.28
    controls.enableZoom      = true
    controls.enablePan       = false
    setLoaded(true)
  }, [])

  return (
    <div
      ref={containerRef}
      className="globe-root"
      data-loaded={String(loaded)}
      data-dimmed={String(!!dimmed)}
    >
      {dimensions.width > 0 && (
        <Globe
          ref={globeRef}
          width={dimensions.width}
          height={dimensions.height}
          globeImageUrl="//unpkg.com/three-globe/example/img/earth-blue-marble.jpg"
          bumpImageUrl="//unpkg.com/three-globe/example/img/earth-topology.png"
          atmosphereColor="rgba(30, 100, 255, 0.35)"
          atmosphereAltitude={0.25}
          pointsData={points}
          pointLat="lat"
          pointLng="lng"
          pointAltitude={0.008}
          pointRadius={0.28}
          pointColor="color"
          pointsMerge={true}
          onGlobeReady={handleGlobeReady}
        />
      )}
    </div>
  )
}

// ── Public component ─────────────────────────────────────────────────────────
export function GlobeHero({ matchedPoints, flyTo, dimmed, liveMode }) {
  if (liveMode) {
    return <SatelliteGlobe flyTo={flyTo} dimmed={dimmed} />
  }
  return <ReactGlobe matchedPoints={matchedPoints} flyTo={flyTo} dimmed={dimmed} />
}
