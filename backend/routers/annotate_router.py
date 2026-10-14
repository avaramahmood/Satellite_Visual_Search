"""
Search & Annotate API router.
POST /api/annotate/search
"""
from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from services.ml_service import encode_image_chips
from services.annotate_ml import slice_and_encode_satellite_image
from services.annotate_vector import search_temp_annotation_db
from services.annotate_dataset import build_custom_yolo_dataset
from services.context_manager import save_search_context

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/annotate", tags=["Search & Annotate"])


@router.post("/search")
async def search_and_annotate(
    label: str = Form(...),
    chip_size: int = Form(256),
    similarity_threshold: float = Form(0.6),
    top_k: int = Form(200),
    reference_chips: List[UploadFile] = File(...),
    satellite_images: List[UploadFile] = File(...),
):
    """
    1. Embed reference chips with DINOv2 (raw vectors).
    2. Slice every satellite image with a 50%-overlap sliding window and embed patches.
    3. Normalise both, build a temp IndexFlatIP, search by cosine similarity.
    4. Draw bounding boxes and return a YOLO zip.
    """
    if not reference_chips:
        raise HTTPException(400, "At least one reference chip is required.")
    if not satellite_images:
        raise HTTPException(400, "At least one satellite image is required.")
    if not (0.0 <= similarity_threshold <= 1.0):
        raise HTTPException(400, "similarity_threshold must be between 0 and 1.")
    if chip_size not in (128, 256, 512):
        raise HTTPException(400, "chip_size must be 128, 256, or 512.")

    # 1. Embed reference chips (returns raw unnormalized DINOv2 vectors)
    chip_bytes_list = [await chip.read() for chip in reference_chips]
    chip_embeddings = await encode_image_chips(chip_bytes_list)
    logger.info("Encoded %d reference chip(s)", len(chip_embeddings))

    # 2. Sliding-window embed all satellite images
    all_embeddings: list[list[float]] = []
    all_metadata:   list[dict]        = []
    raw_uploaded:   dict[str, bytes]  = {}

    for sat_file in satellite_images:
        sat_bytes = await sat_file.read()
        raw_uploaded[sat_file.filename] = sat_bytes

        embeddings, metadata = await slice_and_encode_satellite_image(
            sat_bytes, chip_size, sat_file.filename
        )
        all_embeddings.extend(embeddings)
        all_metadata.extend(metadata)
        logger.info("  '%s': %d patches", sat_file.filename, len(embeddings))

    logger.info("Total patches across all images: %d", len(all_embeddings))

    if not all_embeddings:
        raise HTTPException(400, "No valid patches could be extracted from the satellite images.")

    # 3. Temp FAISS search (cosine similarity via IndexFlatIP)
    results = await search_temp_annotation_db(
        all_embeddings,
        all_metadata,
        chip_embeddings,
        similarity_threshold,
        top_k=top_k,
    )

    if not results:
        raise HTTPException(
            404,
            f"No matches found above threshold {similarity_threshold:.2f}. "
            "Try lowering the similarity threshold."
        )

    logger.info("%d matches found for label='%s'", len(results), label)

    # 4. Save k-NN context for VLM chat
    session_id = save_search_context(
        results, label, source="annotate_search", threshold=similarity_threshold
    )

    # 5. Build YOLO zip with annotated images
    zip_path = await build_custom_yolo_dataset(results, raw_uploaded, label)

    return FileResponse(
        path=str(zip_path),
        media_type="application/zip",
        filename=f"{label}_annotated.zip",
        headers={
            "Content-Disposition": f'attachment; filename="{label}_annotated.zip"',
            "X-Session-Id": session_id,
        },
    )
