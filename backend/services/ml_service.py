"""
ML inference service — DINOv2 (image) and CLIP (text) embeddings.
Both return L2-normalised float32 numpy vectors.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
import io
import logging
import torch

logger = logging.getLogger(__name__)

def _pick_device() -> str:
    """DEVICE env override, else CUDA only if this torch build supports the GPU's architecture."""
    import os
    forced = os.getenv("DEVICE")
    if forced:
        return forced
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        if f"sm_{major}{minor}" in torch.cuda.get_arch_list():
            return "cuda"
    return "cpu"


DEVICE: str = _pick_device()

# ── lazy singletons ──────────────────────────────────────────────────────────
_dino_processor = None
_dino_model      = None
_clip_processor  = None
_clip_model      = None


def _load_dino():
    global _dino_processor, _dino_model
    if _dino_model is None:
        from transformers import AutoImageProcessor, AutoModel
        logger.info("Loading DINOv2…")
        _dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        _dino_model     = AutoModel.from_pretrained("facebook/dinov2-small").to(DEVICE)
        _dino_model.eval()
        logger.info("DINOv2-small ready (device=%s, d=384)", DEVICE)
    return _dino_processor, _dino_model


def get_dino():
    """Return (processor, model, device) — used by annotate_ml."""
    processor, model = _load_dino()
    return processor, model, DEVICE


def _load_clip():
    global _clip_processor, _clip_model
    if _clip_model is None:
        from transformers import CLIPProcessor, CLIPModel
        logger.info("Loading CLIP…")
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model.eval()
        logger.info("CLIP ready")
    return _clip_processor, _clip_model


def embed_images(image_bytes_list: list[bytes]) -> np.ndarray:
    """
    Encode a list of raw image bytes with DINOv2.
    Returns mean-pooled, L2-normalised CLS embeddings → shape (768,).
    Used by the global globe-search endpoint.
    """
    processor, model = _load_dino()

    images = [Image.open(io.BytesIO(b)).convert("RGB") for b in image_bytes_list]
    inputs = processor(images=images, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]   # (N, 768)

    # Mean over all provided chips — keep raw (unnormalized) features
    # The DINO FAISS index stores unnormalized vectors (norm ~47); we must
    # keep the same scale so IP similarity works as cosine similarity.
    vec = embeddings.mean(dim=0).cpu().numpy().astype(np.float32)
    return vec


async def encode_image_chips(image_bytes_list: list[bytes]) -> list[list[float]]:
    """
    Encode each chip independently with DINOv2.
    Returns raw (unnormalized) CLS embeddings — shape (N, 384).
    Used by the annotate endpoint (each chip stays separate for multi-query search).
    The DINO FAISS index stores unnormalized vectors so we must match that scale.
    """
    processor, model = _load_dino()

    images = [Image.open(io.BytesIO(b)).convert("RGB") for b in image_bytes_list]
    inputs = processor(images=images, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]   # (N, 384)

    return embeddings.cpu().numpy().astype(np.float32).tolist()


def embed_text(query: str) -> np.ndarray:
    """
    Encode a text query with CLIP.
    Returns L2-normalised text features → shape (512,).
    """
    processor, model = _load_clip()

    inputs = processor(text=[query], return_tensors="pt", padding=True)
    with torch.no_grad():
        features = model.get_text_features(**inputs)   # (1, 512)

    vec = features[0].cpu().numpy().astype(np.float32)
    vec /= np.linalg.norm(vec) + 1e-9
    return vec
