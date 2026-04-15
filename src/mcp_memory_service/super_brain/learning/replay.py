"""Reflexion-style failure replay.

Before a new task begins, surface lessons from prior related failures so
the same pitfall doesn't repeat ("neural education" — Hinton's term for
showing the model its own past mistakes so it can correct them next time).
"""
from __future__ import annotations

import hashlib
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


async def replay_relevant_failures(server: Any, current_goal: str, k: int = 3) -> List[Dict[str, Any]]:
    """Retrieve up to k prior failures/incidents relevant to the current goal."""
    storage = getattr(server, "storage", None)
    if storage is None:
        return []

    tags_to_try = [["entity:incidents"], ["incident"], ["sb:failure"]]
    results = []
    for tag_set in tags_to_try:
        try:
            res = await storage.retrieve(current_goal, n_results=k, tags=tag_set, min_confidence=0.0)
            if res:
                results = res
                break
        except Exception as exc:
            logger.debug("replay: retrieve with tags=%s failed: %s", tag_set, exc)

    if not results:
        try:
            results = await storage.retrieve(current_goal, n_results=k, tags=None, min_confidence=0.0)
            # Filter post-hoc for failure-ish memories.
            results = [
                r for r in results
                if any(t in (r.memory.tags or []) for t in ("entity:incidents", "incident", "sb:failure"))
            ][:k]
        except Exception as exc:
            logger.debug("replay: fallback retrieve failed: %s", exc)
            return []

    out: List[Dict[str, Any]] = []
    for r in results:
        md = getattr(r.memory, "metadata", {}) or {}
        out.append({
            "memory_id": r.memory.content_hash,
            "tldr": md.get("tldr") or (r.memory.content or "")[:140],
            "lesson": md.get("lesson") or md.get("tldr") or "",
            "relevance_score": r.relevance_score,
        })
    return out


async def record_failure_lesson(server: Any, task_packet_id: str, lesson: str) -> str:
    """Store an incident memory capturing the lesson from a failed run."""
    from ...models.memory import Memory

    content = f"Lesson: {lesson}\n\nFrom task_packet: {task_packet_id}"
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    memory = Memory(
        content=content,
        content_hash=content_hash,
        tags=["entity:incidents", "sb:failure"],
        memory_type="episodic",
        metadata={
            "tldr": lesson[:280],
            "lesson": lesson,
            "supersedes_id": task_packet_id,
            "created_at_iso": now_iso,
            "schema_version": "1",
        },
    )
    success, msg = await server.storage.store(memory)
    if not success:
        logger.warning("replay: incident store returned %s: %s", success, msg)
    return content_hash
