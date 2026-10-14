"""
YOLO dataset builder — given search result metadata, produces a zip with:

  images/    <chip_id>.jpg   — raw extracted chip (for YOLO training)
  annotated/ <chip_id>.jpg   — context crop from the full source image with
                               the chip region boxed in orange
  labels/    <chip_id>.txt   — YOLO label  (class cx cy w h, full-chip bbox)
  classes.txt
  manifest.csv
"""
from __future__ import annotations

import io
import zipfile
import logging
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from config import XVIEW_IMAGES_DIR, DIOR_IMAGES_DIRS

logger = logging.getLogger(__name__)

# Annotation style
_BOX_COLOR   = (255, 100, 40)   # orange — matches globe points
_BOX_WIDTH   = 3                # px
_LABEL_COLOR = (255, 255, 255)

# Context padding: show this many chip-widths of surrounding area in annotated image
_CONTEXT_MULT = 1   # 1× padding each side → 3× chip size total context window


def _find_source_image(tile_id: str, source: str) -> Path | None:
    if source == "xview":
        p = XVIEW_IMAGES_DIR / tile_id
        return p if p.exists() else None
    if source == "dior":
        for d in DIOR_IMAGES_DIRS:
            p = d / tile_id
            if p.exists():
                return p
    return None


def _open_source(tile_id: str, source: str) -> Image.Image | None:
    src = _find_source_image(tile_id, source)
    if src is None:
        logger.warning("Source image not found: %s (%s)", tile_id, source)
        return None
    try:
        return Image.open(src).convert("RGB")
    except Exception as e:
        logger.warning("Failed to open %s: %s", src, e)
        return None


def _extract_chip(src_img: Image.Image, x: int, y: int, size: int) -> Image.Image:
    """Crop the exact chip region from an already-open source image."""
    iw, ih = src_img.size
    x2, y2 = min(x + size, iw), min(y + size, ih)
    chip = src_img.crop((x, y, x2, y2))
    if chip.size != (size, size):
        padded = Image.new("RGB", (size, size), (0, 0, 0))
        padded.paste(chip, (0, 0))
        return padded
    return chip


def _annotated_context(
    src_img: Image.Image,
    x: int, y: int, size: int,
    confidence: float, label: str,
) -> Image.Image:
    """
    Crop a context window from the full source image centred on the chip,
    then draw an orange bounding box around exactly the chip region.

    The context window is (1 + 2×_CONTEXT_MULT) chip-widths on each side,
    so surrounding terrain is visible and the box shows where the chip sits.
    """
    iw, ih = src_img.size
    pad = size * _CONTEXT_MULT

    # Context window bounds in source-image coordinates
    ctx_x0 = max(x - pad, 0)
    ctx_y0 = max(y - pad, 0)
    ctx_x1 = min(x + size + pad, iw)
    ctx_y1 = min(y + size + pad, ih)

    context = src_img.crop((ctx_x0, ctx_y0, ctx_x1, ctx_y1))

    # Chip position inside the context crop
    bx0 = x - ctx_x0
    by0 = y - ctx_y0
    bx1 = bx0 + size
    by1 = by0 + size

    draw = ImageDraw.Draw(context)

    # Bounding box
    for t in range(_BOX_WIDTH):
        draw.rectangle([bx0 + t, by0 + t, bx1 - 1 - t, by1 - 1 - t], outline=_BOX_COLOR)

    # Confidence label just above the box
    text = f"{label} {confidence:.2f}"
    tx = bx0 + 4
    ty = max(by0 - 18, 2)
    try:
        tb = draw.textbbox((tx, ty), text)
        draw.rectangle([tb[0] - 2, tb[1] - 2, tb[2] + 2, tb[3] + 2], fill=_BOX_COLOR)
        draw.text((tx, ty), text, fill=_LABEL_COLOR)
    except Exception:
        pass  # older Pillow

    return context


def _to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def _placeholder(size: int) -> Image.Image:
    arr = np.full((size, size, 3), 60, dtype=np.uint8)
    arr[::32, :] = 80
    arr[:, ::32] = 80
    return Image.fromarray(arr)


def _parse_tile_id(chip_id: str, source: str) -> str | None:
    try:
        if source == "xview" and chip_id.startswith("xview_"):
            rest = chip_id[len("xview_"):]
        elif source == "dior" and chip_id.startswith("dior_"):
            rest = chip_id[len("dior_"):]
        else:
            return None
        idx = rest.find("_s")
        return rest[:idx] if idx != -1 else None
    except Exception:
        return None


def _fetch_thumb(url: str) -> Image.Image | None:
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return Image.open(io.BytesIO(r.read())).convert("RGB")
    except Exception as e:
        logger.warning("Thumbnail fetch failed %s: %s", url, e)
        return None


def _prefetch_thumbnails(matches: list[dict]) -> dict[str, Image.Image | None]:
    """Download chip thumbnails (http chip_path) for matches whose source tile is missing."""
    todo = {
        m["chip_id"]: m["chip_path"]
        for m in matches
        if str(m.get("chip_path", "")).startswith("http")
        and _find_source_image(m.get("tile_id") or "", m.get("source", "")) is None
    }
    if not todo:
        return {}
    with ThreadPoolExecutor(16) as ex:
        return dict(zip(todo, ex.map(_fetch_thumb, todo.values())))


def build_yolo_zip(matches: list[dict], label: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / f"yolo_{label}.zip"

    extracted = placeholder = 0

    thumbs = _prefetch_thumbnails(matches)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("classes.txt", f"{label}\n")
        manifest_lines = ["chip_id,lat,lon,confidence,source,ground_truth_class"]

        for match in matches:
            chip_id  = match.get("chip_id", f"chip_{match.get('idx', 0):06d}")
            tile_id  = match.get("tile_id") or match.get("chip_path", "")
            x_off    = int(match.get("x_offset", 0))
            y_off    = int(match.get("y_offset", 0))
            scale    = int(match.get("scale", 256))
            source   = match.get("source", "unknown")
            conf     = float(match.get("confidence", 0.0))
            lat      = match.get("lat", "")
            lon      = match.get("lon", "")
            gt_class = match.get("ground_truth_class", "")

            if not tile_id or tile_id in ("memory_only", ""):
                tile_id = _parse_tile_id(chip_id, source)

            src_img = _open_source(tile_id, source) if tile_id else None

            if src_img is not None:
                chip      = _extract_chip(src_img, x_off, y_off, scale)
                annotated = _annotated_context(src_img, x_off, y_off, scale, conf, label)
                extracted += 1
            elif thumbs.get(chip_id) is not None:
                # Cloud mode: original tiles aren't on the server — use the stored thumbnail
                chip      = thumbs[chip_id].resize((scale, scale))
                annotated = chip.copy()
                extracted += 1
            else:
                chip      = _placeholder(scale)
                annotated = chip.copy()
                placeholder += 1

            safe_id = chip_id.replace("/", "_").replace("\\", "_")
            zf.writestr(f"images/{safe_id}.jpg",    _to_bytes(chip))
            zf.writestr(f"annotated/{safe_id}.jpg", _to_bytes(annotated))
            zf.writestr(f"labels/{safe_id}.txt",    "0 0.5 0.5 1.0 1.0\n")

            manifest_lines.append(
                f"{chip_id},{lat},{lon},{conf},{source},{gt_class}"
            )

        zf.writestr("manifest.csv", "\n".join(manifest_lines) + "\n")

    zip_path.write_bytes(buf.getvalue())
    logger.info("YOLO zip: %d extracted, %d placeholders → %s", extracted, placeholder, zip_path.name)
    return zip_path
