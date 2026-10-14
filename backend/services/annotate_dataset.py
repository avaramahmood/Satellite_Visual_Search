"""
YOLO dataset builder for the Search & Annotate feature.

Produces a zip with:
  images/    <filename>       — original satellite image (clean, for YOLO training)
  annotated/ <filename>.jpg   — same image with all matched bounding boxes drawn
  labels/    <basename>.txt   — YOLO format: class cx cy w h  (one line per match)
  classes.txt
  manifest.csv
"""
from __future__ import annotations

import io
import os
import shutil
import uuid
import zipfile
import logging
from pathlib import Path

from PIL import Image, ImageDraw

from config import TEMP_DATASET_DIR

logger = logging.getLogger(__name__)

_BOX_COLOR   = (255, 100, 40)    # orange
_BOX_WIDTH   = 2
_LABEL_COLOR = (255, 255, 255)


def _draw_boxes(img_bytes: bytes, matches: list[dict], label_name: str) -> bytes:
    """
    Open the original image and draw an orange bounding box for every matched
    patch, using pixel coordinates stored in match['px'].
    Returns JPEG bytes of the annotated image.
    """
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)

    for m in matches:
        px = m.get("px")
        if not px:
            continue
        x1, y1, x2, y2 = px["x1"], px["y1"], px["x2"], px["y2"]
        conf = m["confidence"]

        # Bounding box
        for t in range(_BOX_WIDTH):
            draw.rectangle([x1 + t, y1 + t, x2 - 1 - t, y2 - 1 - t], outline=_BOX_COLOR)

        # Label above the box
        text = f"{label_name} {conf:.2f}"
        tx, ty = x1 + 3, max(y1 - 16, 1)
        try:
            tb = draw.textbbox((tx, ty), text)
            draw.rectangle([tb[0] - 2, tb[1] - 2, tb[2] + 2, tb[3] + 2], fill=_BOX_COLOR)
            draw.text((tx, ty), text, fill=_LABEL_COLOR)
        except Exception:
            pass

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


async def build_custom_yolo_dataset(
    search_results: list[dict],
    uploaded_images: dict[str, bytes],
    label_name: str,
) -> Path:
    """
    Build a YOLO-format zip archive.

    Args:
        search_results:  list of {image_id, bbox, px, confidence}
        uploaded_images: {filename → raw bytes}
        label_name:      YOLO class name

    Returns:
        Path to the created zip file
    """
    dataset_id  = str(uuid.uuid4())[:8]
    dataset_dir = TEMP_DATASET_DIR / f"annotate_{dataset_id}"
    images_dir  = dataset_dir / "images"
    labels_dir  = dataset_dir / "labels"
    annotated_dir = dataset_dir / "annotated"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)

    # Group results by image
    by_image: dict[str, list[dict]] = {}
    for r in search_results:
        by_image.setdefault(r["image_id"], []).append(r)

    # Write each satellite image
    for filename, img_bytes in uploaded_images.items():
        base = os.path.splitext(filename)[0]

        # images/ — original clean image
        (images_dir / filename).write_bytes(img_bytes)

        # labels/ — YOLO bbox lines
        label_lines = []
        for m in by_image.get(filename, []):
            b = m["bbox"]
            label_lines.append(f"0 {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}")
        (labels_dir / f"{base}.txt").write_text("\n".join(label_lines) + "\n" if label_lines else "")

        # annotated/ — only images that have at least one match
        matches_for_img = by_image.get(filename, [])
        if matches_for_img:
            ann_bytes = _draw_boxes(img_bytes, matches_for_img, label_name)
            (annotated_dir / f"{base}.jpg").write_bytes(ann_bytes)

    # classes.txt
    (dataset_dir / "classes.txt").write_text(f"{label_name}\n")

    # manifest.csv
    lines = ["image_id,cx,cy,w,h,confidence"]
    for r in search_results:
        b = r["bbox"]
        lines.append(f"{r['image_id']},{b[0]:.6f},{b[1]:.6f},{b[2]:.6f},{b[3]:.6f},{r['confidence']}")
    (dataset_dir / "manifest.csv").write_text("\n".join(lines) + "\n")

    # Zip everything
    zip_name = f"{label_name.replace(' ', '_')}_annotated_{dataset_id}.zip"
    zip_path = TEMP_DATASET_DIR / zip_name
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in dataset_dir.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(dataset_dir))
    zip_path.write_bytes(buf.getvalue())

    shutil.rmtree(dataset_dir, ignore_errors=True)
    logger.info("[annotate_dataset] Built %s (%d matches across %d images)",
                zip_name, len(search_results), len(uploaded_images))
    return zip_path
