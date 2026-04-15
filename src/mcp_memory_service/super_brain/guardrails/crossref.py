"""Product-of-experts cross-reference verifier.

Gates episodic → semantic promotion. Given a claim, retrieves relevant
memories and runs a simple multi-expert voting panel using lexical-overlap
and citation-presence heuristics. Returns a verdict + confidence +
groundedness so callers can gate promotion.

The "experts" here are intentionally simple, local, and fast. Swapping in
LLM-backed experts is a future upgrade; the interface stays stable.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set:
    return set(_TOKEN_RE.findall((text or "").lower()))


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def _has_citations(memory: Any) -> bool:
    md = getattr(memory, "metadata", {}) or {}
    if md.get("source"):
        return True
    if md.get("cross_refs"):
        return True
    return False


def _expert_lexical_overlap(claim_tokens: set, memory: Any) -> float:
    content_tokens = _tokens(getattr(memory, "content", "") or "")
    return _jaccard(claim_tokens, content_tokens)


def _expert_tldr_overlap(claim_tokens: set, memory: Any) -> float:
    md = getattr(memory, "metadata", {}) or {}
    tldr_tokens = _tokens(md.get("tldr") or "")
    return _jaccard(claim_tokens, tldr_tokens) if tldr_tokens else 0.0


def _expert_citation_weight(memory: Any) -> float:
    md = getattr(memory, "metadata", {}) or {}
    trust = md.get("trust_tier") or md.get("source_type")
    weight = {"primary_doc": 1.0, "paper": 0.9, "vendor": 0.6, "guru": 0.4}.get(trust, 0.5)
    has_source = 1.0 if _has_citations(memory) else 0.2
    return weight * has_source


EXPERT_FNS = [
    ("lexical_overlap", _expert_lexical_overlap),
    ("tldr_overlap", _expert_tldr_overlap),
]


async def verify(
    server: Any,
    claim: str,
    context_memory_ids: Optional[List[str]] = None,
    n_experts: int = 3,
) -> Dict[str, Any]:
    """Verify a claim against the memory corpus.

    Returns:
        dict with keys: verdict ('supported'|'refuted'|'inconclusive'),
        confidence (0-1), evidence (list), n_expert_agree (int),
        groundedness (0-1).
    """
    storage = getattr(server, "storage", None)
    if storage is None:
        return {"verdict": "inconclusive", "confidence": 0.0,
                "evidence": [], "n_expert_agree": 0, "groundedness": 0.0,
                "reason": "no storage"}

    claim_tokens = _tokens(claim)
    try:
        results = await storage.retrieve(claim, n_results=max(10, n_experts * 3),
                                          tags=None, min_confidence=0.0)
    except Exception as exc:
        logger.warning("crossref: retrieval failed: %s", exc)
        return {"verdict": "inconclusive", "confidence": 0.0,
                "evidence": [], "n_expert_agree": 0, "groundedness": 0.0,
                "reason": f"retrieval_error: {exc}"}

    if context_memory_ids:
        # Prioritize the caller-provided context memories; filter in.
        results = [r for r in results if r.memory.content_hash in context_memory_ids] or results

    if not results:
        return {"verdict": "inconclusive", "confidence": 0.0,
                "evidence": [], "n_expert_agree": 0, "groundedness": 0.0,
                "reason": "no relevant memories"}

    # Run experts against top-n memories; a "supporting vote" = overlap >= 0.2
    # and memory has citations.
    votes: List[Dict[str, Any]] = []
    evidence: List[Dict[str, Any]] = []
    for r in results[:n_experts * 2]:
        mem = r.memory
        expert_scores = {name: fn(claim_tokens, mem) for name, fn in EXPERT_FNS}
        citation_weight = _expert_citation_weight(mem)
        composite = (
            0.4 * expert_scores["lexical_overlap"]
            + 0.3 * expert_scores["tldr_overlap"]
            + 0.3 * citation_weight
        )
        supports = composite >= 0.35 and _has_citations(mem)
        votes.append({"memory_id": mem.content_hash, "composite": composite,
                      "supports": supports, "expert_scores": expert_scores,
                      "citation_weight": citation_weight})
        evidence.append({
            "memory_id": mem.content_hash,
            "tldr": (getattr(mem, "metadata", {}) or {}).get("tldr", (mem.content or "")[:140]),
            "retrieval_score": r.relevance_score,
            "composite": composite,
            "supports": supports,
        })

    agree = sum(1 for v in votes if v["supports"])
    disagree = sum(1 for v in votes if not v["supports"])
    total = max(1, len(votes))

    if agree >= max(2, n_experts):
        verdict = "supported"
    elif agree == 0 and total >= 2:
        verdict = "refuted"
    else:
        verdict = "inconclusive"

    confidence = agree / total
    # Groundedness: fraction of claim tokens appearing in supporting memories' content.
    supporting_tokens = set()
    for v in votes:
        if v["supports"]:
            for r in results:
                if r.memory.content_hash == v["memory_id"]:
                    supporting_tokens |= _tokens(r.memory.content)
                    break
    groundedness = len(claim_tokens & supporting_tokens) / max(1, len(claim_tokens))

    return {
        "verdict": verdict,
        "confidence": confidence,
        "evidence": evidence,
        "n_expert_agree": agree,
        "n_expert_disagree": disagree,
        "groundedness": groundedness,
    }
