"""
PS-3 Satellite Visual Search — FastAPI backend
Endpoints:
  POST /api/search/globe/image  — DINOv2 image embedding search
  POST /api/search/globe/text   — CLIP text embedding search
Both return:
  - JSON with matched chip metadata + lat/lon points (for globe plotting)
  - Triggers a YOLO zip download via a separate /api/download/<token> endpoint
"""
from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import List

import io
import math
import urllib.request

# Load .env so OPENROUTER_API_KEY is available without shell exports
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image as _PIL_Image

from config import (
    XVIEW_DINO_INDEX_PATHS,
    XVIEW_DINO_METADATA_DB,
    XVIEW_CLIP_INDEX_PATH,
    XVIEW_CLIP_METADATA_DB,
    DIOR_DINO_INDEX_PATHS,
    DIOR_DINO_METADATA_DB,
    DIOR_CLIP_INDEX_PATH,
    DIOR_CLIP_METADATA_DB,
    TEMP_DATASET_DIR,
)
from services.dataset_builder import build_yolo_zip
from services.ml_service import embed_images, embed_text
from services.vector_store import search
from services import pg_store
from services.context_manager import save_search_context
from routers.annotate_router import router as annotate_router
from routers.live_search_router import router as live_search_router
from routers.report_router import router as report_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Satellite Visual Search API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Match-Count", "X-Session-Id"],
)

app.include_router(annotate_router)
app.include_router(live_search_router)
app.include_router(report_router)

# In-memory token → zip path store (ephemeral, fine for demo)
_download_tokens: dict[str, Path] = {}


# ── helpers ───────────────────────────────────────────────────────────────────

def _run_search(query_vec, index_path, metadata_db, top_k, threshold, label, use_db=False):
    if use_db:
        # Cloud mode: Postgres + pgvector (DATABASE_URL set)
        matches = pg_store.search(query_vec=query_vec, top_k=top_k, threshold=threshold)
    else:
        matches = search(
            query_vec=query_vec,
            index_path=index_path,
            metadata_db=metadata_db,
            top_k=top_k,
            threshold=threshold,
        )
    zip_path = build_yolo_zip(matches, label, TEMP_DATASET_DIR)
    token = str(uuid.uuid4())
    _download_tokens[token] = zip_path

    session_id = save_search_context(
        matches, label, source="globe_search", threshold=threshold
    )

    points = [
        {"lat": m["lat"], "lng": m["lon"], "confidence": m["confidence"]}
        for m in matches
        if m.get("lat") is not None and m.get("lon") is not None
    ]
    return {
        "count": len(matches),
        "points": points,
        "download_token": token,
        "session_id": session_id,
    }


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


_MAPTILER_KEY = os.getenv("MAPTILER_KEY", "")

@app.get("/api/live/satellite")
def get_satellite_image(lat: float, lng: float, zoom: int = 15, grid: int = 4):
    """Stitch MapTiler satellite tiles into one JPEG server-side."""
    n    = 2 ** zoom
    cx   = int((lng + 180) / 360 * n)
    lr   = math.radians(lat)
    cy   = int((1 - math.log(math.tan(lr) + 1 / math.cos(lr)) / math.pi) / 2 * n)
    half = grid // 2
    px   = 256
    canvas = _PIL_Image.new("RGB", (px * grid, px * grid))
    for dy in range(grid):
        for dx in range(grid):
            url = (f"https://api.maptiler.com/tiles/satellite-v2"
                   f"/{zoom}/{cx-half+dx}/{cy-half+dy}.jpg?key={_MAPTILER_KEY}")
            try:
                with urllib.request.urlopen(url, timeout=10) as r:
                    canvas.paste(_PIL_Image.open(io.BytesIO(r.read())).convert("RGB"),
                                 (dx * px, dy * px))
            except Exception:
                pass
    buf = io.BytesIO()
    canvas.save(buf, format="JPEG", quality=92)
    return Response(content=buf.getvalue(), media_type="image/jpeg")


def _resolve_indexes(dataset: str, chip_size: int):
    """Return (dino_index_path, dino_metadata_db, clip_index_path, clip_metadata_db)
    for the requested dataset.  Falls back to xView if DIOR indexes aren't built yet."""
    if dataset == "dior":
        dino_paths = DIOR_DINO_INDEX_PATHS
        dino_db    = DIOR_DINO_METADATA_DB
        clip_path  = DIOR_CLIP_INDEX_PATH
        clip_db    = DIOR_CLIP_METADATA_DB
    else:
        dino_paths = XVIEW_DINO_INDEX_PATHS
        dino_db    = XVIEW_DINO_METADATA_DB
        clip_path  = XVIEW_CLIP_INDEX_PATH
        clip_db    = XVIEW_CLIP_METADATA_DB

    dino_index = dino_paths.get(chip_size)
    if dino_index is None:
        raise HTTPException(status_code=400, detail=f"chip_size must be one of {list(dino_paths)}")

    # If DIOR indexes aren't ready yet, return a clear error
    if dataset == "dior" and not dino_index.exists() and not clip_path.exists():
        raise HTTPException(
            status_code=503,
            detail="DIOR vector store not yet available. Please build the DIOR index first.",
        )

    return dino_index, dino_db, clip_path, clip_db


@app.post("/api/search/globe/image")
async def search_by_image(
    chips: List[UploadFile] = File(...),
    label: str = Form("object"),
    chip_size: int = Form(256),
    similarity_threshold: float = Form(0.6),
    top_k: int = Form(200),
    dataset: str = Form("xview"),
):
    if not chips:
        raise HTTPException(status_code=400, detail="At least one chip image is required")
    if not (0.0 <= similarity_threshold <= 1.0):
        raise HTTPException(status_code=400, detail="similarity_threshold must be between 0 and 1")

    use_db = pg_store.enabled()
    if use_db:
        if dataset != "xview":
            raise HTTPException(status_code=503, detail="Only the xView dataset is loaded in the cloud demo.")
        dino_index = dino_db = None
    else:
        dino_index, dino_db, _, _ = _resolve_indexes(dataset, chip_size)

    image_bytes = [await chip.read() for chip in chips]
    query_vec = embed_images(image_bytes)

    result = _run_search(query_vec, dino_index, dino_db, top_k, similarity_threshold, label, use_db=use_db)
    logger.info("Image search (dataset=%s, size=%d): %d matches", dataset, chip_size, result["count"])
    return JSONResponse(result)


@app.post("/api/search/globe/text")
async def search_by_text(
    query: str = Form(...),
    label: str = Form("object"),
    chip_size: int = Form(256),
    similarity_threshold: float = Form(0.6),
    top_k: int = Form(200),
    dataset: str = Form("xview"),
):
    if not query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")
    if not (0.0 <= similarity_threshold <= 1.0):
        raise HTTPException(status_code=400, detail="similarity_threshold must be between 0 and 1")

    if pg_store.enabled():
        raise HTTPException(status_code=503, detail="Text search is not available in the cloud demo (no CLIP vectors loaded).")
    _, _, clip_path, clip_db = _resolve_indexes(dataset, chip_size)

    query_vec = embed_text(query)

    result = _run_search(query_vec, clip_path, clip_db, top_k, similarity_threshold, label)
    logger.info("Text search (dataset=%s): %d matches for query='%s'", dataset, result["count"], query)
    return JSONResponse(result)


@app.get("/api/download/{token}")
def download_zip(token: str):
    zip_path = _download_tokens.get(token)
    if not zip_path or not zip_path.exists():
        raise HTTPException(status_code=404, detail="Download token expired or not found")
    return FileResponse(
        path=str(zip_path),
        media_type="application/zip",
        filename=zip_path.name,
    )
