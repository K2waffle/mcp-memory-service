"""Uncertainty scoring (SelfCheckGPT-NLI-style, local-only).

The upstream SelfCheckGPT approach samples the generator N times and
checks consistency — a high-variance output is a hallucination signal.
We can't rerun the caller's model here, but we expose the shape of the
scorer so callers that *do* have samples can flag high-uncertainty
memories before they land in the durable tier.

Three public entry points:
  * ``sample_variance_score(samples)`` — fraction of pairwise disagreement.
  * ``entropy_from_logprobs(logprobs)`` — Shannon entropy in nats.
  * ``flag_if_uncertain(server, memory, threshold)`` — tags the memory
    with ``sb:uncertain`` when the stored uncertainty exceeds threshold.
"""
from __future__ import annotations

import logging
import math
import re
from typing import Any, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set:
    return set(_TOKEN_RE.findall((text or "").lower()))


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def sample_variance_score(samples: Sequence[str]) -> float:
    """Return disagreement in [0, 1]. 0 = all samples agree; 1 = all differ.

    Uses pairwise Jaccard distance over token sets. Cheap, local, good enough
    as a first-pass hallucination flag until we wire a real NLI model.
    """
    if not samples or len(samples) < 2:
        return 0.0
    token_sets = [_tokens(s) for s in samples]
    n = len(token_sets)
    total = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1.0 - _jaccard(token_sets[i], token_sets[j])
            pairs += 1
    return total / max(1, pairs)


def entropy_from_logprobs(logprobs: Iterable[float]) -> float:
    """Shannon entropy (nats) of a distribution given as log-probabilities.

    Accepts natural-log probabilities (what most LLM APIs return).
    Higher entropy = more uncertain.
    """
    lps = list(logprobs)
    if not lps:
        return 0.0
    # Convert to probabilities, renormalize in case they don't sum to 1.
    probs = [math.exp(lp) for lp in lps]
    z = sum(probs)
    if z <= 0:
        return 0.0
    probs = [p / z for p in probs]
    return -sum(p * math.log(p) for p in probs if p > 0)


async def flag_if_uncertain(
    server: Any,
    memory: Any,
    threshold: float = 0.4,
) -> bool:
    """Tag the memory with ``sb:uncertain`` if its stored uncertainty exceeds
    ``threshold``. Reads ``metadata.uncertainty`` (preferred) or computes from
    ``metadata.samples`` if present. Returns True iff the flag was applied.
    """
    md = getattr(memory, "metadata", {}) or {}
    score: Optional[float] = None
    if isinstance(md.get("uncertainty"), (int, float)):
        score = float(md["uncertainty"])
    elif md.get("samples") and isinstance(md["samples"], (list, tuple)):
        score = sample_variance_score(md["samples"])

    if score is None or score < threshold:
        return False

    tags = list(getattr(memory, "tags", []) or [])
    if "sb:uncertain" in tags:
        return False

    content_hash = getattr(memory, "content_hash", None)
    if not content_hash:
        return False

    # Best-effort: append tag via SQL; storage-layer updates vary, keep it safe.
    storage = getattr(server, "storage", None)
    if storage is None:
        return False
    try:
        from ..learning.outcome import _exec
        await _exec(
            server,
            "UPDATE memories SET tags_str = tags_str || ',sb:uncertain' "
            "WHERE content_hash = ? AND tags_str NOT LIKE '%sb:uncertain%'",
            [content_hash],
        )
        return True
    except Exception as exc:
        logger.debug("flag_if_uncertain: tag update skipped: %s", exc)
        return False
