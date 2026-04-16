"""MC Dropout Uncertainty Router — stochastic query routing for the MCP memory service.

# Hinton Connection

Geoffrey Hinton's MC Dropout (Gal & Ghahramani, 2016, building on Hinton's dropout idea)
repurposes training-time dropout as a Bayesian approximation at inference time: run the
same forward pass N times with different dropout masks, and treat the variance across
outputs as a calibrated uncertainty estimate.

In standard neural-net inference the idea is:
    uncertainty ≈ variance of predictions over N stochastic forward passes

We adapt that directly to vector retrieval:
- "Forward pass" → one nearest-neighbour lookup in embedding space
- "Dropout" → small Gaussian noise on the query embedding (simulates the geometric effect
  of randomly zeroing dimensions, without requiring an actual dropout layer)
- "Prediction" → the top-1 memory ID returned by that noisy query

High variance across the N passes (different top-1 IDs each time) ≡ high epistemic
uncertainty: the query sits near a decision boundary between several candidate memories
and the answer is unreliable.

Low variance (same top-1 ID every time) ≡ high confidence: the query lands firmly in
one memory's Voronoi cell, noise can't dislodge it.

# Routing

Uncertainty maps to compute budget:

    agreement >= 0.8  →  "cache"      skip re-inference; top result is stable
    agreement >= 0.5  →  "standard"   normal retrieval; mild uncertainty
    agreement <  0.5  →  "expensive"  flag for human review or re-rank with heavier model

This is prediction uncertainty → routing, not just filtering.  The novelty_gate asks
"is this new?" — the MC router asks "do we know the answer well enough?".
"""
from __future__ import annotations

import logging
import math
import random
from collections import Counter
from typing import Any, Dict, List, Optional

# Import the embedding helper and nearest-neighbour query from novelty_gate so we
# don't duplicate the multi-shape server introspection logic.
from ..guardrails.novelty_gate import _get_embedding, _query_nearest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Routing thresholds
# ---------------------------------------------------------------------------

THRESHOLD_CACHE: float = 0.8     # agreement >= this → "cache"
THRESHOLD_STANDARD: float = 0.5  # agreement >= this → "standard"; else "expensive"

# Default number of noisy passes (Hinton recommends ~15; 10 balances speed vs accuracy).
DEFAULT_N_PASSES: int = 10

# Default standard deviation of Gaussian noise injected onto the embedding.
DEFAULT_NOISE_SCALE: float = 0.02

# How many nearest neighbours to fetch per pass (we read only the top-1 ID, but
# fetching a small k gives us a richer frequency distribution for entropy).
_TOP_K: int = 5


# ---------------------------------------------------------------------------
# Shannon entropy helper
# ---------------------------------------------------------------------------

def _shannon_entropy(counts: Counter) -> float:
    """Compute Shannon entropy (nats) of a frequency distribution.

    Parameters
    ----------
    counts:
        Counter mapping ID → frequency.

    Returns
    -------
    H  (float) — entropy in nats; 0 if only one distinct outcome.
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log(p)
    return entropy


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def estimate_uncertainty(
    server: Any,
    query: str,
    n_passes: int = DEFAULT_N_PASSES,
    noise_scale: float = DEFAULT_NOISE_SCALE,
) -> Dict[str, Any]:
    """Run *n_passes* noisy retrievals and return an uncertainty dict.

    Each pass:
    1. Obtains the query embedding (cached after first pass — same base vector).
    2. Adds independent Gaussian noise: ``perturbed[i] = base[i] + N(0, noise_scale)``.
    3. Runs a nearest-neighbour query via ``_query_nearest``.
    4. Records the top-1 memory ID returned.

    Aggregation:
    - ``agreement``  = fraction of passes that returned the same top-1 ID as the
                       mode (most frequent result).  Range [1/n_passes … 1.0].
    - ``entropy``    = Shannon entropy (nats) over the ID frequency distribution.
                       0 = perfect agreement, higher = more uncertain.
    - ``confidence`` = "high" | "moderate" | "low" derived from ``agreement``.
    - ``top_ids``    = ordered list of most-frequent IDs (most frequent first).
    - ``route``      = "cache" | "standard" | "expensive".

    If embedding is unavailable the function returns a safe default dict with
    ``confidence="unknown"`` and ``route="standard"``.

    Parameters
    ----------
    server:
        MCP server instance.  Must expose an embedding method and ``storage``
        (same contract as ``novelty_gate.py``).
    query:
        Raw text of the retrieval query.
    n_passes:
        Number of stochastic forward passes.  Higher = more accurate uncertainty
        estimate but proportionally more storage round-trips.
    noise_scale:
        Standard deviation of the Gaussian noise injected per dimension.  0.02
        perturbs embeddings by ~2% of a unit-scale dimension — enough to probe
        the local neighbourhood without moving far from the query point.

    Returns
    -------
    Dict with keys:
        "entropy"    — float, mean Shannon entropy across passes
        "agreement"  — float in [0, 1]
        "confidence" — "high" | "moderate" | "low"
        "top_ids"    — list[str], most-frequent top-k IDs (mode first)
        "route"      — "cache" | "standard" | "expensive"
    """
    # --- Step 1: obtain base embedding ---
    base_embedding: Optional[List[float]] = await _get_embedding(server, query)

    if base_embedding is None:
        logger.warning("mc_router: embedding unavailable for query — returning safe default")
        return {
            "entropy": 0.0,
            "agreement": 0.0,
            "confidence": "unknown",
            "top_ids": [],
            "route": "standard",
        }

    dim = len(base_embedding)

    # --- Step 2: run n_passes noisy queries ---
    top1_ids: List[Optional[str]] = []

    for _ in range(n_passes):
        # Perturb: add independent Gaussian noise to each dimension.
        perturbed: List[float] = [
            v + random.gauss(0, noise_scale) for v in base_embedding
        ]

        results = await _query_nearest(server, perturbed, top_k=_TOP_K)

        if results:
            top_id = results[0].get("memory_id")
        else:
            top_id = None

        top1_ids.append(top_id)

    # --- Step 3: aggregate statistics ---
    # Filter out None entries (storage misses) for frequency analysis.
    valid_ids = [mid for mid in top1_ids if mid is not None]

    if not valid_ids:
        # Storage is empty or consistently returning nothing.
        logger.debug("mc_router: all passes returned no results — treating as low confidence")
        return {
            "entropy": 0.0,
            "agreement": 0.0,
            "confidence": "low",
            "top_ids": [],
            "route": "expensive",
        }

    id_counts: Counter = Counter(valid_ids)

    # Agreement: fraction of (all) passes that match the most common ID.
    most_common_id, most_common_count = id_counts.most_common(1)[0]
    agreement: float = most_common_count / n_passes

    # Shannon entropy over the observed distribution.
    entropy: float = _shannon_entropy(id_counts)

    # Ordered unique IDs by frequency (descending).
    top_ids: List[str] = [mid for mid, _ in id_counts.most_common()]

    # Confidence label.
    if agreement >= THRESHOLD_CACHE:
        confidence = "high"
    elif agreement >= THRESHOLD_STANDARD:
        confidence = "moderate"
    else:
        confidence = "low"

    # Route.
    route = _agreement_to_route(agreement)

    logger.debug(
        "mc_router: query=%r passes=%d agreement=%.3f entropy=%.4f route=%s",
        query[:60],
        n_passes,
        agreement,
        entropy,
        route,
    )

    return {
        "entropy": entropy,
        "agreement": agreement,
        "confidence": confidence,
        "top_ids": top_ids,
        "route": route,
    }


async def route_query(server: Any, query: str) -> str:
    """Convenience wrapper — return the routing label for *query*.

    Returns
    -------
    "cache" | "standard" | "expensive"

    Uses default ``n_passes`` and ``noise_scale``.  Prefer ``estimate_uncertainty``
    when you also need the entropy / agreement / top_ids breakdown.
    """
    result = await estimate_uncertainty(server, query)
    return result["route"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _agreement_to_route(agreement: float) -> str:
    """Map a [0, 1] agreement score to a routing string."""
    if agreement >= THRESHOLD_CACHE:
        return "cache"
    if agreement >= THRESHOLD_STANDARD:
        return "standard"
    return "expensive"
