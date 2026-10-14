"""
FAISS vector store — loads index + SQLite metadata, runs kNN search,
returns chip metadata rows with a confidence score.

Supports both IndexFlatIP (all current indexes) and IndexFlatL2.

Confidence mapping:
  - DINO (IndexFlatIP, unnormalized vectors, norm ~47):
      confidence = clip(IP_score / query_norm^2, 0, 1)
      This equals cosine similarity when query and db vectors have similar norms.
  - CLIP (IndexFlatIP, normalized vectors, norm ~1):
      Raw IP = cosine similarity in [-1, 1], but text-satellite scores are ~0.10-0.25.
      Rescale to [0, 1]: confidence = clip((score - CLIP_MIN) / CLIP_RANGE, 0, 1)
  - L2 (fallback):
      confidence = clip(1 - distance / 4.0, 0, 1)
"""
from __future__ import annotations

import sqlite3
import logging
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger(__name__)

# CLIP practical score range on satellite imagery
# Raw cosine similarity for text-vs-satellite-chip is typically 0.10–0.22.
# We rescale so that score=0.05 → confidence=0, score=0.20 → confidence=1.0
_CLIP_MIN   = 0.05   # noise floor
_CLIP_RANGE = 0.15   # 0.05 + 0.15 = 0.20 maps to confidence=1.0

# Lazy index cache
_indexes: dict[str, faiss.Index] = {}


def _load_index(index_path: Path, dim: int) -> faiss.Index:
    key = str(index_path)
    if key not in _indexes:
        if index_path.exists():
            logger.info("Loading FAISS index from %s", index_path)
            _indexes[key] = faiss.read_index(str(index_path))
        else:
            logger.warning("FAISS index not found at %s — using dummy index", index_path)
            idx = faiss.IndexFlatL2(dim)
            rng = np.random.default_rng(42)
            vecs = rng.random((1000, dim), dtype=np.float32)
            idx.add(vecs)
            _indexes[key] = idx
    return _indexes[key]


def _scores_to_confidence(
    scores: np.ndarray,
    query_vec: np.ndarray,
    index: faiss.Index,
) -> np.ndarray:
    """
    Convert raw FAISS scores to [0, 1] confidence values.

    For IndexFlatIP with unnormalized DINO vectors (norm ~47):
        confidence = score / ||query||^2   (≈ cosine similarity)

    For IndexFlatIP with normalized CLIP vectors (norm ~1):
        confidence = (score - CLIP_MIN) / CLIP_RANGE  (rescale to practical range)

    For IndexFlatL2:
        confidence = 1 - distance / 4.0
    """
    if isinstance(index, faiss.IndexFlatIP):
        query_norm_sq = float(np.dot(query_vec, query_vec))
        if query_norm_sq > 2.0:
            # DINO-style: unnormalized vectors, use cosine-similarity approximation
            return np.clip(scores / query_norm_sq, 0.0, 1.0)
        else:
            # CLIP-style: normalized vectors, rescale practical range to [0, 1]
            return np.clip((scores - _CLIP_MIN) / _CLIP_RANGE, 0.0, 1.0)
    else:
        # Fallback: treat as L2 distance
        return np.clip(1.0 - scores / 4.0, 0.0, 1.0)


def search(
    query_vec: np.ndarray,
    index_path: Path,
    metadata_db: Path,
    top_k: int = 200,
    threshold: float = 0.6,
) -> list[dict]:
    """
    Search the FAISS index and return matching chip metadata.

    Args:
        query_vec:    Raw (unnormalized for DINO) or L2-normalised (CLIP) query embedding
        index_path:   Path to FAISS index file
        metadata_db:  Path to SQLite metadata db (DINO or CLIP)
        top_k:        Max candidates to retrieve from FAISS before threshold filter
        threshold:    Minimum confidence to include in results

    Returns:
        List of dicts with keys: idx, chip_id, lat, lon, scale, source,
                                  chip_path (optional), ground_truth_class (optional),
                                  confidence
    """
    dim = query_vec.shape[0]
    index = _load_index(index_path, dim)

    q = query_vec.reshape(1, -1).astype(np.float32)
    scores, indices = index.search(q, top_k)   # (1, k) — IP scores or L2 distances

    confidences = _scores_to_confidence(scores[0], query_vec, index)
    raw_indices = indices[0]

    logger.info(
        "Search: top-5 raw scores=%s  confidences=%s  threshold=%.2f",
        [round(float(x), 2) for x in scores[0][:5]],
        [round(float(x), 3) for x in confidences[:5]],
        threshold,
    )

    # Filter by threshold
    mask = confidences >= threshold
    matched_idx = raw_indices[mask].tolist()
    matched_conf = confidences[mask].tolist()

    if not matched_idx:
        logger.info("No matches above threshold %.2f", threshold)
        return []

    # Fetch metadata from SQLite
    results: list[dict] = []
    if metadata_db.exists():
        conn = sqlite3.connect(str(metadata_db))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        # Discover which optional columns are present
        cur.execute("PRAGMA table_info(chips)")
        available_cols = {row[1] for row in cur.fetchall()}

        base_cols = "idx, chip_id, lat, lon, scale, source"
        extra_cols = []
        for col in ("tile_id", "x_offset", "y_offset", "chip_path", "ground_truth_class"):
            if col in available_cols:
                extra_cols.append(col)

        select_clause = base_cols
        if extra_cols:
            select_clause += ", " + ", ".join(extra_cols)

        placeholders = ",".join("?" * len(matched_idx))
        cur.execute(
            f"SELECT {select_clause} FROM chips WHERE idx IN ({placeholders})",
            matched_idx,
        )
        rows = {row["idx"]: dict(row) for row in cur.fetchall()}
        conn.close()

        for db_idx, conf in zip(matched_idx, matched_conf):
            if db_idx in rows:
                row = rows[db_idx]
                row.setdefault("chip_path", "")
                row.setdefault("ground_truth_class", "unknown")
                row["confidence"] = round(float(conf), 4)
                results.append(row)
    else:
        for db_idx, conf in zip(matched_idx, matched_conf):
            results.append({
                "idx": db_idx,
                "chip_id": f"chip_{db_idx:06d}",
                "lat": float(np.random.uniform(-60, 60)),
                "lon": float(np.random.uniform(-180, 180)),
                "scale": 256,
                "source": "dummy",
                "chip_path": "",
                "ground_truth_class": "unknown",
                "confidence": round(float(conf), 4),
            })

    logger.info("Returning %d matches", len(results))
    return results
