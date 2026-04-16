"""Episodic → procedural distillation (Hinton distillation principle).

Given a set of successful episodic runs, compose a single procedural artifact
that captures the reusable "how" — the know-how distilled from concrete
examples. Seeds outcome-scoring with a neutral prior (alpha=1, beta=1) so
the new procedure competes fairly via Thompson sampling.

Also exposes ``distill_with_soft_distribution`` which implements Hinton's
"dark knowledge" principle: rather than storing only the winning label, we
preserve the full probability landscape over outcomes. This soft distribution
carries far more information than a hard label and lets future agents decide
whether to trust the cached answer (low entropy → high confidence) or re-run
inference (high entropy → uncertainty).
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
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


# ---------------------------------------------------------------------------
# Dark-knowledge distillation (soft distributions)
# ---------------------------------------------------------------------------

def _compute_entropy(distribution: Dict[str, float]) -> float:
    """Shannon entropy (bits) of a probability distribution.

    H = -sum(p * log2(p))  (zero-probability outcomes are skipped)
    """
    total = sum(distribution.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for p_raw in distribution.values():
        p = p_raw / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def _normalize(distribution: Dict[str, float]) -> Dict[str, float]:
    """Return a copy of *distribution* normalized so all values sum to 1.0."""
    total = sum(distribution.values())
    if total <= 0:
        raise ValueError("soft_distribution values must sum to a positive number")
    return {k: v / total for k, v in distribution.items()}


# Entropy thresholds for confidence classification.
ENTROPY_HIGH = 2.0   # >= 2.0 bits: uncertain, flag for future verification
ENTROPY_LOW  = 0.5   # <  0.5 bits: high-confidence, safe as cached answer


async def distill_with_soft_distribution(
    server: Any,
    source_ids: List[str],
    tldr: str,
    body: str = "",
    tags: Optional[List[str]] = None,
    soft_distribution: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Distill with full probability landscape preserved (dark knowledge).

    Extends ``distill_episodic_to_procedural`` by attaching a soft probability
    distribution over outcomes to the procedure's metadata. The distribution
    captures the teacher model's full confidence landscape rather than just the
    argmax answer.

    Parameters
    ----------
    source_ids:
        Content hashes of the episodic memories to distill.
    tldr:
        280-character summary of the distilled procedure.
    body:
        Optional extended description.
    tags:
        Optional list of tags; ``entity:procedures`` added automatically.
    soft_distribution:
        Mapping of ``outcome_label -> probability_weight`` (weights are
        normalized internally so they need not sum to 1).

        Examples::
            {"stripe": 0.7, "paypal": 0.2, "crypto": 0.1}
            {"success": 0.85, "partial": 0.10, "failure": 0.05}

        Entropy-based confidence classification is stored in metadata:
        - entropy >= 2.0 → ``confidence_class = "uncertain"``
          (flag for future verification)
        - entropy <  0.5 → ``confidence_class = "high"``
          (safe as cached answer)
        - otherwise       → ``confidence_class = "moderate"``

    Returns
    -------
    dict with ``memory_id``, ``source_count``, ``entropy``,
    ``confidence_class``, and ``status``.
    """
    # Run the core distillation first.
    base_result = await distill_episodic_to_procedural(
        server, source_ids, tldr=tldr, body=body, tags=tags
    )

    if soft_distribution is None:
        # No soft distribution provided — return the base result unchanged.
        return base_result

    # Normalize and compute Shannon entropy.
    try:
        normalized = _normalize(soft_distribution)
    except ValueError as exc:
        logger.warning("distill_soft: invalid soft_distribution (%s) — skipping dark-knowledge attachment", exc)
        return {**base_result, "entropy": None, "confidence_class": "unknown"}

    entropy = _compute_entropy(normalized)

    if entropy >= ENTROPY_HIGH:
        confidence_class = "uncertain"
        logger.debug(
            "distill_soft: HIGH entropy=%.3f — flagging procedure %s for future verification",
            entropy,
            base_result.get("memory_id"),
        )
    elif entropy < ENTROPY_LOW:
        confidence_class = "high"
        logger.debug(
            "distill_soft: LOW entropy=%.3f — procedure %s is high-confidence cached answer",
            entropy,
            base_result.get("memory_id"),
        )
    else:
        confidence_class = "moderate"

    # Patch the stored memory's metadata to attach the soft distribution.
    # We do a best-effort update; failure is non-fatal (the procedure still
    # exists — it just lacks the dark-knowledge annotation).
    memory_id = base_result.get("memory_id")
    soft_patch = {
        "soft_distribution_json": json.dumps(normalized),
        "entropy": round(entropy, 6),
        "confidence_class": confidence_class,
        "needs_verification": confidence_class == "uncertain",
    }
    try:
        storage = getattr(server, "storage", None)
        if storage is not None:
            for method_name in ("update_metadata", "patch_metadata", "update_memory_metadata"):
                fn = getattr(storage, method_name, None)
                if fn is not None:
                    await fn(memory_id, soft_patch)
                    logger.debug("distill_soft: dark-knowledge metadata written via %s", method_name)
                    break
            else:
                logger.debug(
                    "distill_soft: no metadata-patch method on storage; soft_distribution not persisted"
                )
    except Exception as exc:
        logger.warning("distill_soft: metadata patch failed (non-fatal): %s", exc)

    return {
        **base_result,
        "entropy": round(entropy, 6),
        "confidence_class": confidence_class,
        "needs_verification": confidence_class == "uncertain",
    }
