"""Episodic → procedural distillation (Hinton distillation principle).

Given a set of successful episodic runs, compose a single procedural artifact
that captures the reusable "how" — the know-how distilled from concrete
examples. Seeds outcome-scoring with a neutral prior (alpha=1, beta=1) so
the new procedure competes fairly via Thompson sampling.
"""
from __future__ import annotations

import hashlib
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


async def _fetch_memory_by_hash(server: Any, content_hash: str) -> Optional[Any]:
    """Best-effort memory lookup by content_hash. Falls back to tag search."""
    storage = getattr(server, "storage", None)
    if storage is None:
        return None
    for method_name in ("get_by_hash", "get_memory", "fetch_by_id"):
        if hasattr(storage, method_name):
            try:
                return await getattr(storage, method_name)(content_hash)
            except Exception as exc:
                logger.debug("memory fetch via %s failed: %s", method_name, exc)
    try:
        results = await storage.retrieve(content_hash, n_results=1, tags=None, min_confidence=0.0)
        if results:
            return results[0].memory
    except Exception as exc:
        logger.debug("retrieve fallback failed: %s", exc)
    return None


async def distill_episodic_to_procedural(
    server: Any,
    source_ids: List[str],
    tldr: str,
    body: str = "",
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compose a procedural memory from a set of episodic source memories.

    - Pulls each source memory and aggregates their citations into the new
      artifact's cross_refs (product-of-experts provenance).
    - Writes the new procedure with memory_type='procedural'.
    - Seeds sb_procedure_scores with alpha=1, beta=1 (uniform prior).
    """
    # This module lives at mcp_memory_service.super_brain.learning.distill, so
    # three dots climb back to mcp_memory_service.models.memory.
    from ...models.memory import Memory

    if not source_ids:
        raise ValueError("distill requires at least one source_id")

    sources = []
    for sid in source_ids:
        mem = await _fetch_memory_by_hash(server, sid)
        if mem is not None:
            sources.append(mem)

    # Aggregate cross_refs and citations from sources.
    cross_refs: List[str] = []
    for mem in sources:
        md = getattr(mem, "metadata", {}) or {}
        src = md.get("source")
        if src and src not in cross_refs:
            cross_refs.append(src)
        for cr in md.get("cross_refs", []) or []:
            if cr not in cross_refs:
                cross_refs.append(cr)

    final_tags = list(tags or [])
    if "entity:procedures" not in final_tags:
        final_tags.append("entity:procedures")

    content_parts = [tldr, ""]
    if body:
        content_parts.extend([body, ""])
    content_parts.append(f"Distilled from {len(sources)} source run(s):")
    for mem in sources:
        src_tldr = (getattr(mem, "metadata", {}) or {}).get("tldr") or (mem.content[:140] if getattr(mem, "content", None) else "")
        content_parts.append(f"  - {mem.content_hash[:12]}: {src_tldr}")
    content = "\n".join(content_parts)
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    metadata: Dict[str, Any] = {
        "tldr": tldr,
        "cross_refs": cross_refs,
        "distilled_from": source_ids,
        "schema_version": "1",
        "alpha": 1.0,
        "beta": 1.0,
        "source_type": "ai_assisted",
    }

    procedure = Memory(
        content=content,
        content_hash=content_hash,
        tags=final_tags,
        memory_type="procedural",
        metadata=metadata,
    )
    success, msg = await server.storage.store(procedure)
    if not success:
        logger.warning("distill: procedure store returned %s: %s", success, msg)

    # Seed Beta score.
    try:
        from .outcome import update_beta_score
        # "Seed" = store zero outcomes; use a zero-weight no-op approach:
        # insert the row with defaults by calling update with weight=0.
        # The update function handles upsert; we use direct SQL for clarity.
        from .outcome import _exec
        await _exec(
            server,
            """
            INSERT INTO sb_procedure_scores (memory_id, alpha, beta, n_runs, last_used_at, updated_at)
            VALUES (?, 1.0, 1.0, 0, NULL, ?)
            ON CONFLICT(memory_id) DO NOTHING
            """,
            [content_hash, now_iso],
        )
    except Exception as exc:
        logger.debug("distill: Beta seed failed (non-fatal): %s", exc)

    return {
        "memory_id": content_hash,
        "source_count": len(sources),
        "status": "ok",
    }
