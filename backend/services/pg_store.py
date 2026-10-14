"""
Postgres + pgvector vector store (cloud demo) — drop-in replacement for the FAISS
`vector_store.search` when DATABASE_URL is set.

Table `chips` is created by notebooks/build_demo_dataset.ipynb:
  idx, chip_id, lat, lon, scale, source, tile_id, x_offset, y_offset,
  chip_path, ground_truth_class, embedding vector(384)

Confidence = cosine similarity (1 - cosine distance), clipped to [0, 1]. The old FAISS
path used IP / |q|^2 on unnormalized vectors, which is the same quantity when norms match.
"""
from __future__ import annotations

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

_pool = None

_COLS = ("idx, chip_id, lat, lon, scale, source, tile_id, x_offset, y_offset, "
         "chip_path, ground_truth_class")


def enabled() -> bool:
    return bool(os.getenv("DATABASE_URL"))


def demo_chip_size() -> int:
    """The demo dataset holds a single chip size; requested sizes are mapped onto it."""
    return int(os.getenv("DEMO_CHIP_SIZE", "128"))


def _get_pool():
    global _pool
    if _pool is None:
        from pgvector.psycopg import register_vector
        from psycopg_pool import ConnectionPool

        _pool = ConnectionPool(
            os.environ["DATABASE_URL"],
            min_size=1,
            max_size=4,
            configure=register_vector,
            kwargs={"prepare_threshold": None},   # safe behind Supabase's poolers
            open=True,
        )
    return _pool


def _thumb_url(chip_id: str) -> str:
    base = os.getenv("SUPABASE_URL", "").rstrip("/")
    bucket = os.getenv("STORAGE_BUCKET", "thumbs")
    return f"{base}/storage/v1/object/public/{bucket}/{chip_id}.webp" if base else ""


def search(
    query_vec: np.ndarray,
    scale: int | None = None,
    top_k: int = 200,
    threshold: float = 0.6,
    source: str = "xview",
) -> list[dict]:
    from psycopg.rows import dict_row

    q = np.asarray(query_vec, dtype=np.float32)
    scale = scale or demo_chip_size()
    ef = min(max(top_k, 40), 1000)

    with _get_pool().connection() as conn:
        with conn.transaction():
            conn.execute(f"SET LOCAL hnsw.ef_search = {int(ef)}")
            cur = conn.cursor(row_factory=dict_row)
            rows = cur.execute(
                f"SELECT {_COLS}, 1 - (embedding <=> %s) AS confidence "
                "FROM chips WHERE source = %s AND scale = %s "
                "ORDER BY embedding <=> %s LIMIT %s",
                (q, source, scale, q, top_k),
            ).fetchall()

    results = []
    for r in rows:
        conf = float(np.clip(r["confidence"], 0.0, 1.0))
        if conf < threshold:
            continue
        r["confidence"] = round(conf, 4)
        r["chip_path"] = _thumb_url(r["chip_id"]) or r["chip_path"] or ""
        r["ground_truth_class"] = r["ground_truth_class"] or "unknown"
        results.append(r)

    logger.info("pg search: %d candidates, %d above threshold %.2f", len(rows), len(results), threshold)
    return results
