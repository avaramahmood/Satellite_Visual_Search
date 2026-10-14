import "./Navbar.css"

const NAV_ITEMS = [
  { id: "globe-search",  label: "Globe Search" },
  { id: "annotate",      label: "Annotate" },
  { id: "live-search",   label: "Live Search" },
  { id: "vlm-chat",      label: "VLM Chat" },
]

export function Navbar({ activePanel, onNavClick }) {
  return (
    <nav className="navbar">
      <a className="navbar-logo" href="#">
        <span className="navbar-logo-glyph">◈</span>
        <span className="navbar-logo-text">SpaceYard</span>
      </a>

      {/* spacer pushes nav to right */}
      <div className="navbar-spacer" />

      <div className="navbar-nav">
        {NAV_ITEMS.map((item) => (
          <button
            key={item.id}
            className="navbar-item"
            data-active={String(activePanel === item.id)}
            data-live={String(item.id === "live-search")}
            onClick={() => onNavClick(item.id)}
          >
            {item.id === "live-search" && <span className="navbar-item-dot" />}
            {item.label}
          </button>
        ))}
      </div>
    </nav>
  )
}
