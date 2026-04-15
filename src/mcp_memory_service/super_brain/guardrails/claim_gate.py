"""Provenance gate for episodic → semantic promotion.

A memory cannot be promoted to ``semantic`` (durable, retrievable as fact)
unless it carries enough provenance to be verifiable. This module checks
the minimum bar and delegates the evidence test to :mod:`crossref`.

The policy here is intentionally conservative — Hinton's product-of-experts
only works if each "expert" is grounded, so an ungrounded claim must not
be allowed to pollute the semantic tier.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from . import crossref

logger = logging.getLogger(__name__)

# Minimum required metadata fields before a promotion can even be evaluated.
REQUIRED_FIELDS = ("tldr",)
REQUIRED_PROVENANCE_ONE_OF = ("source", "cross_refs")

# Thresholds — conservative on purpose.
CONFIDENCE_FLOOR = 0.6
GROUNDEDNESS_FLOOR = 0.5


def _meta(memory: Any) -> Dict[str, Any]:
    return getattr(memory, "metadata", {}) or {}


def _missing_required(memory: Any) -> List[str]:
    md = _meta(memory)
    missing = [f for f in REQUIRED_FIELDS if not md.get(f)]
    if not any(md.get(f) for f in REQUIRED_PROVENANCE_ONE_OF):
        missing.append("source|cross_refs")
    return missing


async def evaluate_claim(server: Any, memory: Any) -> Dict[str, Any]:
    """Decide whether this memory may be promoted to the semantic tier.

    Returns a dict with:
      - allow_promotion: bool
      - reason: short human-readable string
      - confidence: float (from cross-ref verifier)
      - groundedness: float (from cross-ref verifier)
      - missing: list of missing required fields
    """
    md = _meta(memory)
    missing = _missing_required(memory)
    if missing:
        return {
            "allow_promotion": False,
            "reason": f"missing_required_fields: {missing}",
            "confidence": 0.0,
            "groundedness": 0.0,
            "missing": missing,
        }

    claim_text = md.get("tldr") or (getattr(memory, "content", "") or "")[:280]
    verdict = await crossref.verify(server, claim_text)
    confidence = float(verdict.get("confidence", 0.0))
    groundedness = float(verdict.get("groundedness", 0.0))
    supported = verdict.get("verdict") == "supported"

    if not supported:
        return {
            "allow_promotion": False,
            "reason": f"crossref_verdict={verdict.get('verdict')}",
            "confidence": confidence,
            "groundedness": groundedness,
            "missing": [],
            "evidence": verdict.get("evidence", []),
        }

    if confidence < CONFIDENCE_FLOOR or groundedness < GROUNDEDNESS_FLOOR:
        return {
            "allow_promotion": False,
            "reason": (
                f"below_floor: confidence={confidence:.2f} "
                f"(>= {CONFIDENCE_FLOOR}) groundedness={groundedness:.2f} "
                f"(>= {GROUNDEDNESS_FLOOR})"
            ),
            "confidence": confidence,
            "groundedness": groundedness,
            "missing": [],
            "evidence": verdict.get("evidence", []),
        }

    return {
        "allow_promotion": True,
        "reason": "supported_with_provenance",
        "confidence": confidence,
        "groundedness": groundedness,
        "missing": [],
        "evidence": verdict.get("evidence", []),
    }


async def bulk_evaluate(server: Any, memory_ids: List[str]) -> List[Dict[str, Any]]:
    """Evaluate many memories; skips any that cannot be fetched."""
    from ..learning.distill import _fetch_memory_by_hash  # reuse tolerant lookup
    out: List[Dict[str, Any]] = []
    for mid in memory_ids:
        mem = await _fetch_memory_by_hash(server, mid)
        if mem is None:
            out.append({
                "memory_id": mid,
                "allow_promotion": False,
                "reason": "not_found",
                "confidence": 0.0,
                "groundedness": 0.0,
                "missing": ["memory"],
            })
            continue
        res = await evaluate_claim(server, mem)
        res["memory_id"] = mid
        out.append(res)
    return out
