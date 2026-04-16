"""Nightly CLS consolidation cron.

Runs every 24 hours. Fetches episodic memories written in the last 24 h,
groups them by co-occurring tag clusters, then calls
``distill_episodic_to_procedural`` for every cluster that has 3+ members.

Design principles:
- Fire-and-forget: all exceptions are caught and logged; the main server
  is never impacted.
- Fail-open: partial failures within a cluster are non-fatal.
- No new deps: uses only asyncio + standard library + existing project code.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

logger = logging.getLogger(__name__)

# 24-hour interval in seconds
_INTERVAL_SECONDS = 24 * 60 * 60

# Minimum cluster size before we bother distilling
_MIN_CLUSTER_SIZE = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unix_24h_ago() -> float:
    """Return Unix timestamp for 24 hours ago."""
    return time.time() - _INTERVAL_SECONDS


async def _fetch_recent_episodic(server: Any) -> List[Any]:
    """Return episodic Memory objects created in the last 24 h.

    Uses ``storage.get_all_memories(memory_type='episodic')`` which is
    available on all storage backends, then filters by created_at.
    Falls back to an empty list on any failure.
    """
    storage = getattr(server, "storage", None)
    if storage is None:
        logger.warning("consolidation_cron: server.storage not found")
        return []

    cutoff = _unix_24h_ago()

    try:
        # get_all_memories supports memory_type filter on all backends
        memories = await storage.get_all_memories(memory_type="episodic")
    except Exception as exc:
        logger.warning("consolidation_cron: get_all_memories failed: %s", exc)
        return []

    # Filter to last 24 h
    recent = [
        m for m in (memories or [])
        if (getattr(m, "created_at", None) or 0.0) >= cutoff
    ]
    return recent


def _build_tag_clusters(memories: List[Any]) -> Dict[Tuple[str, ...], List[Any]]:
    """Group memories by their frozenset of tags (co-occurrence cluster key).

    We represent each cluster by a *sorted tuple* of its tags so that two
    memories sharing the exact same tag set land in the same bucket.
    Memories with no tags get a special ``("__untagged__",)`` key and are
    excluded from distillation (we require at least one tag for a meaningful
    cluster label).
    """
    clusters: Dict[Tuple[str, ...], List[Any]] = defaultdict(list)
    for mem in memories:
        tags: List[str] = getattr(mem, "tags", None) or []
        if not tags:
            continue
        key = tuple(sorted(set(tags)))
        clusters[key].append(mem)
    return dict(clusters)


async def _distill_cluster(
    server: Any,
    tag_key: Tuple[str, ...],
    members: List[Any],
) -> bool:
    """Distill one cluster. Returns True on success, False on failure."""
    from .distill import distill_episodic_to_procedural

    source_ids = [
        getattr(m, "content_hash", None) or ""
        for m in members
        if getattr(m, "content_hash", None)
    ]
    if len(source_ids) < _MIN_CLUSTER_SIZE:
        return False

    tag_label = ", ".join(tag_key[:4])  # keep tldr short
    tldr = f"Nightly CLS distillation — cluster [{tag_label}] ({len(source_ids)} episodes)"

    try:
        result = await distill_episodic_to_procedural(
            server,
            source_ids=source_ids,
            tldr=tldr,
            tags=list(tag_key) + ["cls-consolidation"],
        )
        logger.info(
            "consolidation_cron: distilled cluster [%s] -> procedural %s (sources=%d)",
            tag_label,
            result.get("memory_id", "?")[:12],
            result.get("source_count", 0),
        )
        return True
    except Exception as exc:
        logger.warning(
            "consolidation_cron: distillation failed for cluster [%s]: %s",
            tag_label,
            exc,
        )
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def run_nightly_consolidation(server: Any) -> Dict[str, Any]:
    """Fetch last-24 h episodic memories, cluster by tags, distill clusters.

    Returns a summary dict (useful for testing / manual invocation):
        {
            "episodic_fetched": int,
            "clusters_evaluated": int,
            "consolidations_done": int,
        }
    """
    logger.info("consolidation_cron: starting nightly consolidation pass")

    memories = await _fetch_recent_episodic(server)
    logger.info("consolidation_cron: found %d episodic memories in last 24 h", len(memories))

    if not memories:
        return {"episodic_fetched": 0, "clusters_evaluated": 0, "consolidations_done": 0}

    clusters = _build_tag_clusters(memories)
    qualifying = {k: v for k, v in clusters.items() if len(v) >= _MIN_CLUSTER_SIZE}

    logger.info(
        "consolidation_cron: %d tag clusters found, %d qualify (size >= %d)",
        len(clusters),
        len(qualifying),
        _MIN_CLUSTER_SIZE,
    )

    done = 0
    for tag_key, members in qualifying.items():
        success = await _distill_cluster(server, tag_key, members)
        if success:
            done += 1

    logger.info(
        "consolidation_cron: complete — %d episodic->procedural consolidations performed",
        done,
    )
    return {
        "episodic_fetched": len(memories),
        "clusters_evaluated": len(qualifying),
        "consolidations_done": done,
    }


async def schedule_cron(server: Any) -> None:
    """Run ``run_nightly_consolidation`` every 24 hours, indefinitely.

    Fire-and-forget: exceptions are caught and logged; the loop continues.
    Intended to be wrapped in ``asyncio.ensure_future()``.
    """
    logger.info("consolidation_cron: scheduler started (interval=24 h)")
    while True:
        await asyncio.sleep(_INTERVAL_SECONDS)
        try:
            await run_nightly_consolidation(server)
        except asyncio.CancelledError:
            logger.info("consolidation_cron: scheduler cancelled")
            return
        except Exception as exc:
            logger.exception("consolidation_cron: unhandled error in consolidation pass: %s", exc)
