"""Self-upgrade research pipeline.

Runs every 6 hours. Implements a fully automated loop:

  1. Discover  — find ``research_source`` memories tagged "ai-memory" or
                 "hinton" that have NOT been tagged "implemented".
  2. Score     — compute implementation_value = (novelty_score *
                 revenue_impact) / implementation_hours from metadata.
  3. Select    — pick the highest-scored pending item.
  4. Plan      — inspect which Hinton modules are already live on disk;
                 generate a plain-text implementation plan.
  5. Record    — write a ``procedural`` memory containing the plan.
  6. Tag       — mark the source research memory "queued-for-implementation".
  7. Return    — dict with selected title, score, plan_written flag.

Design principles:
- Fail-open: every step wraps exceptions; partial success is acceptable.
- No new deps: asyncio + standard library + existing project primitives.
- Async-safe: no blocking I/O.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 6-hour interval in seconds
_INTERVAL_SECONDS = 6 * 60 * 60

# Tags that identify a pending research item
_RESEARCH_TAGS = ["ai-memory", "hinton"]
_SKIP_TAG = "implemented"
_QUEUE_TAG = "queued-for-implementation"

# Metadata score defaults
_DEFAULT_NOVELTY = 0.7
_DEFAULT_REVENUE = 1.0
_DEFAULT_HOURS = 2.0

# Hinton modules we check for "already live"
_HINTON_MODULE_PATHS = {
    "novelty_gate": "guardrails/novelty_gate.py",
    "distill": "learning/distill.py",
    "mc_router": "learning/mc_router.py",
}


# ---------------------------------------------------------------------------
# Step 1 — Discover
# ---------------------------------------------------------------------------

async def _discover_pending(server: Any) -> List[Any]:
    """Return research_source memories tagged ai-memory or hinton, not implemented."""
    storage = getattr(server, "storage", None)
    if storage is None:
        logger.warning("research_pipeline: server.storage not found")
        return []

    candidates: List[Any] = []
    for tag in _RESEARCH_TAGS:
        try:
            batch = await storage.search_by_tag([tag])
        except Exception as exc:
            logger.warning("research_pipeline: search_by_tag(%s) failed: %s", tag, exc)
            continue

        for mem in (batch or []):
            # Filter to research_source type (loose check)
            mtype = getattr(mem, "memory_type", None) or ""
            if "research" not in mtype.lower() and "research_source" not in (
                getattr(mem, "tags", None) or []
            ):
                continue
            # Skip already-implemented
            tags: List[str] = getattr(mem, "tags", None) or []
            if _SKIP_TAG in tags:
                continue
            candidates.append(mem)

    # Deduplicate by content_hash
    seen: set = set()
    unique = []
    for mem in candidates:
        h = getattr(mem, "content_hash", None)
        if h and h not in seen:
            seen.add(h)
            unique.append(mem)

    logger.debug("research_pipeline: discovered %d pending research items", len(unique))
    return unique


# ---------------------------------------------------------------------------
# Step 2 — Score
# ---------------------------------------------------------------------------

def _score(mem: Any) -> float:
    """Compute implementation_value from memory metadata."""
    md: Dict[str, Any] = getattr(mem, "metadata", None) or {}
    novelty = float(md.get("novelty_score", _DEFAULT_NOVELTY))
    revenue = float(md.get("revenue_impact", _DEFAULT_REVENUE))
    hours = float(md.get("implementation_hours", _DEFAULT_HOURS))
    if hours <= 0:
        hours = _DEFAULT_HOURS
    return (novelty * revenue) / hours


# ---------------------------------------------------------------------------
# Step 4 — Plan
# ---------------------------------------------------------------------------

def _live_modules() -> Dict[str, bool]:
    """Check which Hinton modules exist on disk, relative to this file's package root."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # super_brain/
    return {
        name: os.path.isfile(os.path.join(base, rel))
        for name, rel in _HINTON_MODULE_PATHS.items()
    }


def _generate_plan(mem: Any, live: Dict[str, bool]) -> str:
    """Generate a plain-text implementation plan for the selected research item."""
    title = _get_title(mem)
    content_preview = (getattr(mem, "content", None) or "")[:400]

    live_summary = "; ".join(
        f"{name}: {'live' if present else 'missing'}"
        for name, present in live.items()
    )

    plan_lines = [
        f"Auto-generated implementation plan for: {title}",
        "",
        "Research item summary:",
        content_preview,
        "",
        "Existing Hinton module status:",
        f"  {live_summary}",
        "",
        "Suggested implementation steps:",
        "  1. Review research item content and attached metadata for specific guidance.",
        "  2. Identify which Hinton modules (above) need to be created or extended.",
        "  3. Write unit tests covering the new behaviour before coding.",
        "  4. Implement the feature; keep cyclomatic complexity <= 8.",
        "  5. Run the eval harness (super_brain/eval/) to confirm quality delta.",
        "  6. Tag the source research memory 'implemented' once shipped.",
        "",
        f"Source memory: {getattr(mem, 'content_hash', 'unknown')}",
        f"Score (novelty*revenue/hours): {_score(mem):.4f}",
        f"Generated at: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
    ]
    return "\n".join(plan_lines)


def _get_title(mem: Any) -> str:
    """Best-effort title extraction from a memory object."""
    md: Dict[str, Any] = getattr(mem, "metadata", None) or {}
    title = md.get("title") or md.get("tldr") or md.get("name")
    if title:
        return str(title)[:120]
    content = getattr(mem, "content", None) or ""
    return content[:80].replace("\n", " ")


# ---------------------------------------------------------------------------
# Step 5a — Write plan to sb_task_packets (no embedding needed)
# ---------------------------------------------------------------------------

async def _write_task_packet(server: Any, title: str, plan_text: str, source_hash: str) -> bool:
    """Insert an action item into sb_task_packets via direct D1 HTTP call.

    Uses storage._retry_request + storage.d1_url so no embedding is required.
    Falls back silently if storage attributes are missing.
    """
    import json as _json
    import uuid as _uuid

    storage = getattr(server, "storage", None)
    if storage is None:
        return False

    # Support HybridStorage: prefer the cloud sub-storage if present
    for attr in ("_cloud", "_cloudflare", "cloud", "cloudflare"):
        sub = getattr(storage, attr, None)
        if sub is not None and hasattr(sub, "d1_url"):
            storage = sub
            break

    d1_url = getattr(storage, "d1_url", None)
    retry_fn = getattr(storage, "_retry_request", None)
    if not d1_url or retry_fn is None:
        logger.warning("research_pipeline: task_packet write skipped — no d1_url on storage")
        return False

    packet_id = str(_uuid.uuid4())
    sql = (
        "INSERT OR IGNORE INTO sb_task_packets "
        "(id, memory_id, goal, inputs_json) "
        "VALUES (?, ?, ?, ?)"
    )
    params = [
        packet_id,
        source_hash or packet_id,
        title[:500],
        _json.dumps({"plan": plan_text}),
    ]

    try:
        resp = await retry_fn("POST", f"{d1_url}/query", json={"sql": sql, "params": params})
        data = resp.json() if hasattr(resp, "json") else {}
        if isinstance(data, dict) and data.get("success") is False:
            logger.warning("research_pipeline: task_packet D1 insert returned: %s", data)
            return False
        logger.info("research_pipeline: task_packet written id=%s", packet_id)
        return True
    except Exception as exc:
        logger.warning("research_pipeline: task_packet write failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Step 5b — Record plan as procedural memory
# ---------------------------------------------------------------------------

async def _write_plan_memory(server: Any, plan_text: str, source_hash: str) -> Optional[str]:
    """Persist the plan as a procedural memory. Returns content_hash or None."""
    from ...models.memory import Memory

    storage = getattr(server, "storage", None)
    if storage is None:
        return None

    content_hash = hashlib.sha256(plan_text.encode("utf-8")).hexdigest()
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    mem = Memory(
        content=plan_text,
        content_hash=content_hash,
        tags=["research-pipeline", "pending-implementation", "auto-generated"],
        memory_type="procedural",
        metadata={
            "tldr": plan_text.splitlines()[0][:280],
            "source_research_hash": source_hash,
            "schema_version": "1",
            "source_type": "ai_assisted",
            "generated_at": now_iso,
        },
    )

    try:
        success, msg = await storage.store(mem)
        if success:
            return content_hash
        logger.warning("research_pipeline: plan store returned: %s", msg)
        return None
    except Exception as exc:
        logger.warning("research_pipeline: plan store failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Step 6 — Tag source memory as queued
# ---------------------------------------------------------------------------

async def _tag_source_queued(server: Any, mem: Any) -> bool:
    """Add 'queued-for-implementation' tag to the source research memory."""
    storage = getattr(server, "storage", None)
    if storage is None:
        return False

    content_hash = getattr(mem, "content_hash", None)
    if not content_hash:
        return False

    existing_tags: List[str] = list(getattr(mem, "tags", None) or [])
    if _QUEUE_TAG in existing_tags:
        return True  # already tagged

    new_tags = existing_tags + [_QUEUE_TAG]

    try:
        success, msg = await storage.update_memory_metadata(
            content_hash,
            {"tags": new_tags},
            preserve_timestamps=True,
        )
        if not success:
            logger.warning("research_pipeline: tag update returned: %s", msg)
        return bool(success)
    except Exception as exc:
        logger.warning("research_pipeline: tag update failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def run_research_cycle(server: Any) -> Dict[str, Any]:
    """Execute one full research → plan → record cycle.

    Returns::

        {
            "selected": str,      # title of selected research item, or "" if none
            "score": float,
            "plan_written": bool,
        }
    """
    logger.info("research_pipeline: starting research cycle")

    # 1. Discover
    pending = await _discover_pending(server)
    if not pending:
        logger.info("research_pipeline: no pending research items found")
        return {"selected": "", "score": 0.0, "plan_written": False}

    # 2. Score
    scored = sorted(pending, key=_score, reverse=True)

    # 3. Select
    top = scored[0]
    top_score = _score(top)
    title = _get_title(top)
    logger.info(
        "research_pipeline: selected '%s' (score=%.4f) from %d candidates",
        title,
        top_score,
        len(scored),
    )

    # 4. Plan
    live = _live_modules()
    plan_text = _generate_plan(top, live)

    # 5. Record — write to sb_task_packets (no embedding) AND procedural memory
    source_hash = getattr(top, "content_hash", "") or ""

    # 5a. Write actionable task packet (bypasses embedding — always attempted first)
    packet_ok = await _write_task_packet(server, title, plan_text, source_hash)
    if packet_ok:
        logger.info("research_pipeline: task packet written for '%s'", title[:60])
    else:
        logger.warning("research_pipeline: task packet write failed for '%s'", title[:60])

    # 5b. Also try procedural memory (may fail if embedding 401 — non-fatal)
    plan_hash = await _write_plan_memory(server, plan_text, source_hash)
    plan_written = plan_hash is not None or packet_ok
    if plan_hash:
        logger.info("research_pipeline: plan memory written (%s)", plan_hash[:12])

    # 6. Tag source
    tagged = await _tag_source_queued(server, top)
    if not tagged:
        logger.warning("research_pipeline: could not tag source memory as queued")

    return {
        "selected": title,
        "score": top_score,
        "plan_written": plan_written,
        "packet_written": packet_ok,
    }


async def schedule_research(server: Any) -> None:
    """Run ``run_research_cycle`` every 6 hours, indefinitely.

    Fire-and-forget: exceptions are caught and logged; the loop continues.
    Intended to be wrapped in ``asyncio.ensure_future()``.
    """
    logger.info("research_pipeline: scheduler started (interval=6 h)")
    while True:
        await asyncio.sleep(_INTERVAL_SECONDS)
        try:
            await run_research_cycle(server)
        except asyncio.CancelledError:
            logger.info("research_pipeline: scheduler cancelled")
            return
        except Exception as exc:
            logger.exception("research_pipeline: unhandled error in research cycle: %s", exc)
