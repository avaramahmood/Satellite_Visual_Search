"""
Draws YOLO-format bounding boxes onto a map screenshot using Pillow.
Returns the modified image as JPEG bytes ready to stream to the frontend.
"""
from __future__ import annotations

import io
from PIL import Image, ImageDraw


def draw_bounding_boxes_on_image(image_bytes: bytes, search_results: list[dict]) -> bytes:
    """
    Takes the original map screenshot and a list of YOLO-format results,
    draws bounding boxes on the image, and returns the modified image bytes.

    Args:
        image_bytes:    Raw bytes of the map screenshot (any PIL-readable format).
        search_results: List of dicts with keys:
                          bbox       — [cx, cy, w, h] normalised 0.0–1.0
                          confidence — float 0.0–1.0

    Returns:
        JPEG bytes of the annotated image.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)
    img_w, img_h = img.size

    for result in search_results:
        cx, cy, w, h = result["bbox"]

        abs_cx = cx * img_w
        abs_cy = cy * img_h
        abs_w  = w  * img_w
        abs_h  = h  * img_h

        x1 = abs_cx - abs_w / 2
        y1 = abs_cy - abs_h / 2
        x2 = abs_cx + abs_w / 2
        y2 = abs_cy + abs_h / 2

        # Bright red box, 3 px thick
        draw.rectangle([x1, y1, x2, y2], outline="#ff0044", width=3)

        # Confidence label in the top-left corner of the box
        conf_text = f"{result['confidence']:.2f}"
        draw.text((x1 + 4, y1 + 2), conf_text, fill="#ff0044")

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()
