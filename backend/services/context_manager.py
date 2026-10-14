"""
Context Manager — saves k-NN search results to JSON files so the VLM
has access to the exact mathematical data (bboxes, confidences, image IDs)
from each search session.  Also pre-computes retrieval metrics so the VLM
doesn't have to re-derive them from raw scores.
"""
from __future__ import annotations

import json
import math
import os
import uuid
from pathlib import Path

CONTEXT_DIR = Path(__file__).resolve().parent.parent / "session_context"
os.makedirs(CONTEXT_DIR, exist_ok=True)


def _compute_metrics(matches: list[dict], threshold: float = 0.6) -> dict:
    """
    Compute retrieval/ranking metrics from confidence scores alone.

    Key design choices (no ground truth available):
    - Binary relevance = confidence >= mean_confidence.  Using the search
      threshold would make every returned result 'relevant' by definition
      (the search already filters to >= threshold), which trivially forces
      Precision, Recall, F1, MRR and mAP all to 1.0.
    - NDCG ideal = all k results at confidence=1.0, NOT a re-sort of the
      actual scores (re-sorting an already-sorted list gives DCG=IDCG=1.0).
    - MRR threshold = midpoint between search threshold and 1.0, so it
      reflects whether any result reaches "high-quality" territory.
    - Silhouette uses proper 1-D silhouette coefficient split at mean conf.
    - Recall is estimated: est_total = k / mean_confidence (higher mean →
      fewer estimated missed positives → higher recall).
    """
    if not matches:
        return {}

    confidences = sorted(
        [float(m.get("confidence", 0.0)) for m in matches], reverse=True
    )
    k = len(confidences)

    conf_mean = sum(confidences) / k
    conf_var  = sum((c - conf_mean) ** 2 for c in confidences) / k
    conf_std  = conf_var ** 0.5

    # ── Binary relevance: above mean confidence ───────────────────────────────
    # All returned results already pass the search threshold, so using that
    # threshold would make n_rel = k always → all metrics trivially 1.0.
    relevant = [1 if c >= conf_mean else 0 for c in confidences]
    n_rel    = sum(relevant)

    # ── Precision@K ──────────────────────────────────────────────────────────
    precision_at_k = n_rel / k if k > 0 else 0.0

    # ── Recall@K (estimated) ─────────────────────────────────────────────────
    # Higher mean confidence → model is more certain → fewer missed positives.
    # est_total = k / mean_conf approximates the total true positives in the DB.
    est_total   = max(int(k / max(conf_mean, 0.01)), n_rel)
    recall_at_k = n_rel / est_total if est_total > 0 else 0.0

    # ── F1 ───────────────────────────────────────────────────────────────────
    f1 = (
        2 * precision_at_k * recall_at_k / (precision_at_k + recall_at_k)
        if (precision_at_k + recall_at_k) > 0
        else 0.0
    )

    # ── NDCG@K ───────────────────────────────────────────────────────────────
    # Ideal DCG = every result has confidence=1.0 (perfect retrieval).
    # Re-sorting the actual scores would give DCG=IDCG=1.0 for any sorted list.
    dcg       = sum(c / math.log2(i + 2) for i, c in enumerate(confidences))
    idcg      = sum(1.0 / math.log2(i + 2) for i in range(k))
    ndcg_at_k = dcg / idcg if idcg > 0 else 0.0

    # ── MRR ──────────────────────────────────────────────────────────────────
    # Threshold = midpoint between search threshold and 1.0 (high-quality bar).
    mrr_threshold = threshold + 0.5 * (1.0 - threshold)
    first_high    = next((i + 1 for i, c in enumerate(confidences) if c >= mrr_threshold), None)
    mrr           = 1.0 / first_high if first_high else 0.0

    # ── mAP ──────────────────────────────────────────────────────────────────
    ap_sum, rel_so_far = 0.0, 0
    for i, r in enumerate(relevant):
        if r:
            rel_so_far += 1
            ap_sum += rel_so_far / (i + 1)
    map_score = ap_sum / n_rel if n_rel > 0 else 0.0

    # ── Silhouette (proper 1-D, split at mean confidence) ────────────────────
    high_c = [c for c in confidences if c >= conf_mean]
    low_c  = [c for c in confidences if c < conf_mean]
    if high_c and low_c:
        sil_scores = []
        for c in confidences:
            same  = high_c if c >= conf_mean else low_c
            other = low_c  if c >= conf_mean else high_c
            a = (sum(abs(c - x) for x in same if x != c) / max(len(same) - 1, 1))
            b = sum(abs(c - x) for x in other) / len(other)
            denom = max(a, b)
            sil_scores.append((b - a) / denom if denom > 0 else 0.0)
        silhouette = max(-1.0, min(1.0, sum(sil_scores) / len(sil_scores)))
    elif high_c:
        silhouette = 1.0
    else:
        silhouette = 0.0

    return {
        "k":               k,
        "precision_at_k":  round(precision_at_k, 4),
        "recall_at_k":     round(recall_at_k, 4),
        "f1":              round(f1, 4),
        "ndcg_at_k":       round(ndcg_at_k, 4),
        "mrr":             round(mrr, 4),
        "map":             round(map_score, 4),
        "silhouette":      round(silhouette, 4),
        "conf_mean":       round(conf_mean, 4),
        "conf_std":        round(conf_std, 4),
        "conf_min":        round(min(confidences), 4),
        "conf_max":        round(max(confidences), 4),
        "high_conf_count": sum(1 for c in confidences if c >= mrr_threshold),
        "threshold_used":  threshold,
        "mrr_threshold":   round(mrr_threshold, 4),
        "note": (
            "Relevance: confidence >= mean. "
            "Recall estimated via k/mean_conf. "
            "NDCG ideal = all k at confidence=1.0. "
            "MRR threshold = midpoint(search_threshold, 1.0)."
        ),
    }


def _compute_per_image_stats(matches: list[dict]) -> list[dict]:
    """
    Aggregate per-source-image confidence stats from patch-level matches.
    Returns a list sorted by max_confidence desc.
    """
    image_map: dict[str, dict] = {}
    for m in matches:
        img_id = m.get("image_id", "unknown")
        conf   = float(m.get("confidence", 0.0))
        if img_id not in image_map:
            image_map[img_id] = {"confidences": [], "match_count": 0}
        image_map[img_id]["confidences"].append(conf)
        image_map[img_id]["match_count"] += 1

    stats = []
    for img_id, data in image_map.items():
        confs = data["confidences"]
        mean  = sum(confs) / len(confs)
        stats.append({
            "image_id":       img_id,
            "match_count":    data["match_count"],
            "max_confidence": round(max(confs), 4),
            "mean_confidence": round(mean, 4),
            "min_confidence": round(min(confs), 4),
        })

    stats.sort(key=lambda x: x["max_confidence"], reverse=True)
    return stats


def save_search_context(
    search_results: list[dict],
    query_label: str,
    source: str = "unknown",
    threshold: float = 0.6,
) -> str:
    """
    Persist the numerical k-NN results + pre-computed metrics to a JSON file.
    Returns a session_id that the frontend and VLM service use to retrieve it.
    """
    session_id        = str(uuid.uuid4())
    context_filepath  = CONTEXT_DIR / f"{session_id}.json"

    metrics         = _compute_metrics(search_results, threshold)
    per_image_stats = _compute_per_image_stats(search_results)

    context_data = {
        "session_id":      session_id,
        "query_label":     query_label,
        "source":          source,   # "globe_search" | "annotate_search" | "live_search"
        "total_matches":   len(search_results),
        "metrics":         metrics,
        "per_image_stats": per_image_stats,
        "matches":         search_results,
    }

    with open(context_filepath, "w") as f:
        json.dump(context_data, f, indent=4)

    return session_id


def load_search_context(session_id: str) -> str:
    """Load and return the context JSON as a formatted string for VLM injection."""
    context_filepath = CONTEXT_DIR / f"{session_id}.json"
    if not context_filepath.exists():
        return "No mathematical context available for this session."
    with open(context_filepath, "r") as f:
        return f.read()


def load_metrics(session_id: str) -> dict:
    """Load only the pre-computed metrics dict for the session."""
    context_filepath = CONTEXT_DIR / f"{session_id}.json"
    if not context_filepath.exists():
        return {}
    with open(context_filepath, "r") as f:
        data = json.load(f)
    return {
        "session_id":      data.get("session_id"),
        "query_label":     data.get("query_label"),
        "source":          data.get("source"),
        "total_matches":   data.get("total_matches", 0),
        "metrics":         data.get("metrics", {}),
        "per_image_stats": data.get("per_image_stats", []),
    }
