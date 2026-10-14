"""
Live Globe Search router.
POST /api/live/search  — receives a WebGL map screenshot + a query chip,
runs sliding-window DINOv2 search in-memory, draws bounding boxes with
Pillow, saves context JSON for the VLM, runs visual analysis, and returns
a JSON payload with the annotated image (base64) + VLM analysis + session_id.
GET  /api/live/geocode?q=... — server-side Nominatim proxy (avoids CORS).
"""
from __future__ import annotations

import base64
import io
import logging
import math
import os
import json as _json
import urllib.parse
import urllib.request

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image

from services.ml_service import encode_image_chips
from services.annotate_ml import slice_and_encode_satellite_image
from services.annotate_vector import search_temp_annotation_db
from services.image_drawer import draw_bounding_boxes_on_image
from services.context_manager import save_search_context
from services.vlm_service import analyze_satellite_scene

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/live", tags=["Live Globe Search"])


MAPTILER_KEY = os.getenv("MAPTILER_KEY", "")

@router.get("/geocode")
def geocode_place(q: str = Query(..., min_length=1)):
    """
    MapTiler geocoding proxy — returns up to 5 results:
    [{display_name, short, lat, lng}, ...]
    """
    encoded = urllib.parse.quote(q)
    url     = f"https://api.maptiler.com/geocoding/{encoded}.json?key={MAPTILER_KEY}&limit=5"
    try:
        with urllib.request.urlopen(url, timeout=8) as resp:
            data = _json.loads(resp.read().decode())
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Geocoding error: {exc}")

    features = data.get("features", [])
    if not features:
        raise HTTPException(status_code=404, detail="Place not found")

    results = [
        {
            "display_name": f.get("place_name", q),
            "short":        f.get("text", f.get("place_name", q).split(",")[0].strip()),
            "lat":          f["geometry"]["coordinates"][1],
            "lng":          f["geometry"]["coordinates"][0],
        }
        for f in features
        if f.get("geometry", {}).get("coordinates")
    ]
    return JSONResponse(results)


def _lat_lng_to_tile(lat: float, lng: float, zoom: int):
    n   = 2 ** zoom
    x   = int((lng + 180) / 360 * n)
    lr  = math.radians(lat)
    y   = int((1 - math.log(math.tan(lr) + 1 / math.cos(lr)) / math.pi) / 2 * n)
    return x, y


@router.get("/satellite")
def get_satellite_image(lat: float, lng: float, zoom: int = 15, grid: int = 4):
    """
    Fetch a grid×grid block of MapTiler satellite tiles server-side,
    stitch them with PIL, and return a single JPEG.
    """
    cx, cy   = _lat_lng_to_tile(lat, lng, zoom)
    half     = grid // 2
    tile_px  = 256
    canvas   = Image.new("RGB", (tile_px * grid, tile_px * grid))

    for dy in range(grid):
        for dx in range(grid):
            tx  = cx - half + dx
            ty  = cy - half + dy
            url = (f"https://api.maptiler.com/tiles/satellite-v2"
                   f"/{zoom}/{tx}/{ty}.jpg?key={MAPTILER_KEY}")
            try:
                with urllib.request.urlopen(url, timeout=10) as r:
                    tile_img = Image.open(io.BytesIO(r.read())).convert("RGB")
                canvas.paste(tile_img, (dx * tile_px, dy * tile_px))
            except Exception:
                pass   # leave blank patch on failure

    buf = io.BytesIO()
    canvas.save(buf, format="JPEG", quality=92)
    return Response(content=buf.getvalue(), media_type="image/jpeg")


@router.post("/search")
async def live_globe_search(
    chip_size: int = Form(...),
    similarity_threshold: float = Form(0.60),
    top_k: int = Form(10),
    query_chip: UploadFile = File(...),
    map_screenshot: UploadFile = File(...),
    location_name: str = Form("the requested area"),
    target_label: str = Form("target object"),
):
    """
    1. Embed the reference chip with DINOv2.
    2. Slice the map screenshot with a sliding window and embed every patch.
    3. Search patches against the chip in a temporary FAISS index.
    4. Save k-NN results to a session context JSON file.
    5. Draw bounding boxes on the screenshot.
    6. Run VLM analysis using the annotated image + context file.
    7. Return JSON: {session_id, annotated_image (base64 data URI), analysis, match_count}.
    """
    if not (0.0 <= similarity_threshold <= 1.0):
        raise HTTPException(status_code=400, detail="similarity_threshold must be between 0 and 1.")
    if not (1 <= top_k <= 200):
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 200.")

    try:
        # 1. Embed the reference chip
        chip_bytes = await query_chip.read()
        chip_embeddings = await encode_image_chips([chip_bytes])
        logger.info("Encoded 1 query chip for live search")

        # 2. Sliding-window embed the map screenshot
        map_bytes = await map_screenshot.read()
        map_embeddings, patch_metadata = await slice_and_encode_satellite_image(
            map_bytes, chip_size, map_screenshot.filename or "live_map.jpg"
        )
        logger.info("Live search: %d patches extracted at chip_size=%d", len(map_embeddings), chip_size)

        # 3. Search with top_k cap
        results = await search_temp_annotation_db(
            map_embeddings,
            patch_metadata,
            chip_embeddings,
            similarity_threshold,
            top_k,
        )
        logger.info(
            "Live search: %d matches (threshold=%.2f, top_k=%d)",
            len(results), similarity_threshold, top_k,
        )

        # 4. Save k-NN results to context file — returns session_id
        session_id = save_search_context(results, query_label=target_label)
        logger.info("Context saved with session_id=%s", session_id)

        # 5. Draw bounding boxes
        if results:
            annotated_bytes = draw_bounding_boxes_on_image(map_bytes, results)
        else:
            annotated_bytes = map_bytes

        annotated_b64 = f"data:image/jpeg;base64,{base64.b64encode(annotated_bytes).decode()}"

        # 6. Run VLM analysis (best-effort — never block the response if it fails)
        analysis = ""
        try:
            analysis = await analyze_satellite_scene(
                annotated_map_bytes=annotated_bytes,
                reference_chip_bytes=chip_bytes,
                location_name=location_name,
                target_label=target_label,
                session_id=session_id,
            )
        except Exception as vlm_exc:
            logger.warning("VLM analysis skipped: %s", vlm_exc)
            analysis = ""

        # 7. Return JSON payload
        return JSONResponse(
            content={
                "session_id": session_id,
                "annotated_image": annotated_b64,
                "analysis": analysis,
                "match_count": len(results),
            },
            headers={"X-Match-Count": str(len(results))},
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Live search failed")
        raise HTTPException(status_code=500, detail=str(exc))
