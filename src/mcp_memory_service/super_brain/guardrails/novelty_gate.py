"""Novelty gate — blocks redundant memory writes.

Before any chunk is written, compute cosine similarity to the top-k nearest
existing chunks. If max similarity exceeds NOVELTY_THRESHOLD the write is
skipped — only genuine surprises propagate into storage.

Implements Hinton's prediction-error principle: high-surprise signals (low
similarity to existing memories) are information-rich and worth storing;
near-duplicates are not.

Usage (pre-write hook pattern):
    from .novelty_gate import gate_write

    result = await gate_write(server, content, tags)
    if not result["allow"]:
        return {"status": "skipped", **result}
    await storage.store(memory)
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default cosine-similarity threshold. At 0.92 the gate stops writes that are
# ≥92% semantically identical to an existing chunk, letting genuine variations
# (rewording, updated facts) through while blocking pure duplicates.
NOVELTY_THRESHOLD = 0.92

# Number of nearest neighbours to check.
TOP_K = 5


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

async def _get_embedding(server: Any, text: str) -> Optional[List[float]]:
    """Obtain a vector embedding for *text* via the server's embedding model.

    Tries common API shapes in order:
    1. ``server.embeddings.embed(text)``        — sentence-transformer wrapper
    2. ``server.embed(text)``                   — direct method
    3. ``server.storage.embed(text)``           — storage-attached embedder
    4. ``server.embedding_model.encode(text)``  — raw model (sync, run in executor)
    """
    # Shape 1 — dedicated embeddings object
    emb_obj = getattr(server, "embeddings", None)
    if emb_obj is not None:
        for method_name in ("embed", "encode", "get_embedding"):
            fn = getattr(emb_obj, method_name, None)
            if fn is not None:
                try:
                    result = fn(text)
                    # Some encoders return a coroutine
                    if hasattr(result, "__await__"):
                        result = await result
                    # Could be a 2-D array (batch of 1)
                    if hasattr(result, "tolist"):
                        result = result.tolist()
                    if result and isinstance(result[0], (list, tuple)):
                        result = result[0]
                    return list(result)
                except Exception as exc:
                    logger.debug("novelty_gate: embed via embeddings.%s failed: %s", method_name, exc)

    # Shape 2 — server.embed()
    fn = getattr(server, "embed", None)
    if fn is not None:
        try:
            result = fn(text)
            if hasattr(result, "__await__"):
                result = await result
            if hasattr(result, "tolist"):
                result = result.tolist()
            return list(result)
        except Exception as exc:
            logger.debug("novelty_gate: embed via server.embed failed: %s", exc)

    # Shape 3 — storage-attached embedder
    storage = getattr(server, "storage", None)
    if storage is not None:
        fn = getattr(storage, "embed", None)
        if fn is not None:
            try:
                result = fn(text)
                if hasattr(result, "__await__"):
                    result = await result
                if hasattr(result, "tolist"):
                    result = result.tolist()
                return list(result)
            except Exception as exc:
                logger.debug("novelty_gate: embed via storage.embed failed: %s", exc)

    # Shape 4 — raw model (sync)
    model = getattr(server, "embedding_model", None)
    if model is not None:
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, model.encode, text)
            if hasattr(result, "tolist"):
                result = result.tolist()
            if result and isinstance(result[0], (list, tuple)):
                result = result[0]
            return list(result)
        except Exception as exc:
            logger.debug("novelty_gate: embed via embedding_model.encode failed: %s", exc)

    logger.warning("novelty_gate: no embedding method found on server — treating as novel")
    return None


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Return the cosine similarity between two equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Nearest-neighbour query
# ---------------------------------------------------------------------------

async def _query_nearest(
    server: Any,
    embedding: List[float],
    top_k: int,
) -> List[Dict[str, Any]]:
    """Retrieve up to *top_k* nearest stored memories using the given embedding.

    Returns a list of dicts: {"memory_id": str, "similarity": float}

    Falls back to an empty list if storage does not support vector queries.
    """
    storage = getattr(server, "storage", None)
    if storage is None:
        return []

    # Prefer a direct vector search method that accepts a raw embedding.
    for method_name in ("search_by_vector", "nearest_by_vector", "query_by_embedding"):
        fn = getattr(storage, method_name, None)
        if fn is not None:
            try:
                results = await fn(embedding, n_results=top_k)
                return [
                    {
                        "memory_id": getattr(r, "content_hash", None)
                                     or getattr(r, "id", None)
                                     or (r.get("memory_id") if isinstance(r, dict) else None),
                        "similarity": getattr(r, "relevance", None)
                                      or (r.get("similarity") if isinstance(r, dict) else None)
                                      or 0.0,
                    }
                    for r in (results or [])
                ]
            except Exception as exc:
                logger.debug("novelty_gate: vector query via %s failed: %s", method_name, exc)

    # Fallback: use the standard retrieve() and compute cosine manually using
    # whatever token the content gives us.  We cannot access the stored vectors
    # directly in this path, so similarity will be 0 — conservative (allows
    # write).  This is the safe fallback: better to over-write than over-block.
    logger.debug("novelty_gate: no vector-search method on storage; fallback returns no candidates")
    return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def is_novel(
    server: Any,
    content: str,
    tags: List[str],
    threshold: float = NOVELTY_THRESHOLD,
) -> Tuple[bool, float]:
    """Return ``(is_novel, max_similarity)`` for *content* against existing storage.

    Parameters
    ----------
    server:
        The MCP server instance (needs ``.storage`` and an embedding method).
    content:
        Raw text to be written.
    tags:
        Tags that would be attached to the new memory (unused in similarity
        check but kept for API symmetry with gate_write).
    threshold:
        Cosine similarity above which the content is considered a duplicate.

    Returns
    -------
    (True, max_sim)  — content should be written
    (False, max_sim) — content is a near-duplicate; skip the write
    """
    storage = getattr(server, "storage", None)

    # If storage is empty or unavailable, everything is novel.
    if storage is None:
        logger.debug("novelty_gate: no storage — marking as novel")
        return True, 0.0

    embedding = await _get_embedding(server, content)
    if embedding is None:
        # Cannot embed → cannot gate → allow write (fail-open).
        return True, 0.0

    candidates = await _query_nearest(server, embedding, top_k=TOP_K)
    if not candidates:
        logger.debug("novelty_gate: no existing candidates — marking as novel")
        return True, 0.0

    max_sim = max((c.get("similarity") or 0.0) for c in candidates)
    novel = max_sim <= threshold

    if not novel:
        logger.debug(
            "novelty_gate: write blocked — max_similarity=%.4f >= threshold=%.4f",
            max_sim,
            threshold,
        )
    return novel, float(max_sim)


async def gate_write(
    server: Any,
    content: str,
    tags: List[str],
    threshold: float = NOVELTY_THRESHOLD,
) -> Dict[str, Any]:
    """Evaluate whether a write should proceed.

    Returns a dict with:

    - ``allow``          — bool; True → proceed with write
    - ``reason``         — human-readable explanation
    - ``max_similarity`` — highest cosine similarity found (0.0 if storage empty)
    - ``nearest_id``     — content_hash of the most-similar memory, or None
    """
    storage = getattr(server, "storage", None)

    # Empty / unavailable storage — always novel.
    if storage is None:
        return {
            "allow": True,
            "reason": "storage_unavailable",
            "max_similarity": 0.0,
            "nearest_id": None,
        }

    embedding = await _get_embedding(server, content)
    if embedding is None:
        # Fail-open: if we cannot embed, we let the write through rather than
        # silently dropping data we cannot assess.
        return {
            "allow": True,
            "reason": "embedding_unavailable",
            "max_similarity": 0.0,
            "nearest_id": None,
        }

    candidates = await _query_nearest(server, embedding, top_k=TOP_K)
    if not candidates:
        return {
            "allow": True,
            "reason": "storage_empty",
            "max_similarity": 0.0,
            "nearest_id": None,
        }

    # Find the single highest-similarity candidate.
    best = max(candidates, key=lambda c: c.get("similarity") or 0.0)
    max_sim = float(best.get("similarity") or 0.0)
    nearest_id: Optional[str] = best.get("memory_id")

    if max_sim > threshold:
        logger.debug(
            "novelty_gate: SKIP write — content is %.1f%% similar to %s",
            max_sim * 100,
            nearest_id,
        )
        return {
            "allow": False,
            "reason": f"near_duplicate: max_similarity={max_sim:.4f} > threshold={threshold:.4f}",
            "max_similarity": max_sim,
            "nearest_id": nearest_id,
        }

    return {
        "allow": True,
        "reason": f"novel: max_similarity={max_sim:.4f} <= threshold={threshold:.4f}",
        "max_similarity": max_sim,
        "nearest_id": nearest_id,
    }
