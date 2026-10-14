"""
Sliding-window patch extractor and DINOv2 embedder for user-uploaded satellite images.
Uses 50% overlap to avoid missing objects at patch boundaries.
Returns L2-normalised patch vectors (norm=1) compatible with IndexFlatIP temp index.
"""
from __future__ import annotations

import io
import logging

import torch
import torch.nn.functional as F
from PIL import Image

from services.ml_service import get_dino

logger = logging.getLogger(__name__)


async def slice_and_encode_satellite_image(
    image_bytes: bytes,
    chip_size: int,
    filename: str,
    stride: int | None = None,
) -> tuple[list[list[float]], list[dict]]:
    """
    Slices a satellite image into overlapping patches with a sliding window,
    embeds each patch with DINOv2, and returns:
      - embeddings: list of per-patch L2-normalised vectors (384-d, dinov2-small)
      - metadata:   list of dicts with pixel-exact and YOLO-normalised bounding boxes

    Args:
        image_bytes: raw image bytes (PNG/JPG/TIFF)
        chip_size:   patch size in pixels (128, 256, or 512)
        filename:    original filename, stored in metadata for result grouping
        stride:      sliding window step — defaults to chip_size//2 (50% overlap)
    """
    processor, model, device = get_dino()

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_w, img_h = img.size

    if stride is None:
        stride = chip_size // 2   # 50% overlap by default

    patches: list[Image.Image] = []
    metadata: list[dict] = []

    for y in range(0, img_h, stride):
        for x in range(0, img_w, stride):
            x2 = min(x + chip_size, img_w)
            y2 = min(y + chip_size, img_h)

            # Skip patches smaller than 50% of target — avoids noisy edge fragments
            if (x2 - x) < chip_size * 0.5 or (y2 - y) < chip_size * 0.5:
                continue

            patch = img.crop((x, y, x2, y2))
            # Pad partial patches at image edges to full chip_size
            if patch.size != (chip_size, chip_size):
                padded = Image.new("RGB", (chip_size, chip_size), (0, 0, 0))
                padded.paste(patch, (0, 0))
                patch = padded

            patches.append(patch)
            metadata.append({
                "filename": filename,
                # Pixel coords (for drawing bboxes on the original image)
                "px": {"x1": x, "y1": y, "x2": x2, "y2": y2},
                # YOLO-normalised bbox (cx, cy, w, h)
                "bbox": [
                    ((x + x2) / 2.0) / img_w,
                    ((y + y2) / 2.0) / img_h,
                    (x2 - x) / img_w,
                    (y2 - y) / img_h,
                ],
            })

    logger.info("[annotate_ml] '%s': %d patches (chip=%d, stride=%d)", filename, len(patches), chip_size, stride)

    embeddings: list[list[float]] = []
    batch_size = 32

    for i in range(0, len(patches), batch_size):
        batch = patches[i: i + batch_size]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
            emb = out.last_hidden_state[:, 0, :]   # (B, 384)
            emb = F.normalize(emb, p=2, dim=1)     # unit-norm for cosine similarity
        embeddings.extend(emb.cpu().numpy().tolist())

    return embeddings, metadata
