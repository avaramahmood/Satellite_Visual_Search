"""
Temporary in-memory FAISS index for the Search & Annotate feature.
Uses IndexFlatIP with L2-normalised vectors so the inner product = cosine similarity.
Both satellite patches (from annotate_ml) and chip queries are normalised to unit norm.
"""
from __future__ import annotations

import logging

import faiss
import numpy as np

logger = logging.getLogger(__name__)

DINO_DIM = 384   # dinov2-small CLS token dimension


async def search_temp_annotation_db(
    satellite_embeddings: list[list[float]],
    patch_metadata: list[dict],
    chip_embeddings: list[list[float]],
    threshold: float,
    top_k: int = 200,
) -> list[dict]:
    """
    Build a temporary IndexFlatIP FAISS index from satellite patch embeddings,
    query it with each reference chip, filter by cosine-similarity threshold,
    de-duplicate, and return matching patch metadata.

    Both satellite patches and chip queries are L2-normalised here so that
    IP score = cosine similarity ∈ [-1, 1].  Confidence = IP score.

    Args:
        satellite_embeddings: (M, 384) patch vectors (already unit-norm from annotate_ml)
        patch_metadata:       parallel list of {filename, px, bbox} dicts
        chip_embeddings:      (N, 384) reference chip vectors (raw from encode_image_chips)
        threshold:            minimum cosine-similarity confidence to keep
        top_k:                max total matches returned across all queries

    Returns:
        List of {image_id, bbox, px, confidence} sorted by confidence desc
    """
    if not satellite_embeddings:
        return []

    # Build temp IndexFlatIP
    temp_index = faiss.IndexFlatIP(DINO_DIM)

    db_array = np.array(satellite_embeddings, dtype=np.float32)
    # Satellite patches arrive pre-normalised from annotate_ml; renormalise just in case
    norms = np.linalg.norm(db_array, axis=1, keepdims=True)
    db_array = db_array / (norms + 1e-9)
    temp_index.add(db_array)

    # Normalise chip queries (encode_image_chips returns raw unnormalised vectors)
    query_array = np.array(chip_embeddings, dtype=np.float32)
    qnorms = np.linalg.norm(query_array, axis=1, keepdims=True)
    query_array = query_array / (qnorms + 1e-9)

    k = min(top_k * 2, len(satellite_embeddings))   # over-fetch, then filter
    scores, indices = temp_index.search(query_array, k)   # IP = cosine similarity

    logger.info(
        "[annotate_vector] %d patches indexed, %d chip queries, top-5 scores: %s",
        len(satellite_embeddings),
        len(chip_embeddings),
        [round(float(x), 3) for x in scores[0, :5]],
    )

    results: list[dict] = []
    seen: set[int] = set()

    for q_idx in range(len(chip_embeddings)):
        for j in range(k):
            db_id = int(indices[q_idx, j])
            if db_id < 0 or db_id in seen:
                continue
            confidence = float(scores[q_idx, j])   # already cosine similarity
            if confidence < threshold:
                break   # scores are sorted desc, so we can stop early
            seen.add(db_id)
            meta = patch_metadata[db_id]
            results.append({
                "image_id":   meta["filename"],
                "bbox":       meta["bbox"],
                "px":         meta["px"],
                "confidence": round(confidence, 4),
            })

    results.sort(key=lambda r: r["confidence"], reverse=True)
    return results[:top_k]
