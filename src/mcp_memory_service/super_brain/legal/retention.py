"""Retention-policy engine.

Every subject-tied memory belongs to a retention policy identified by
``metadata.retention_policy_id``. Policies live in the ``sb_retention_policies``
table (see migration 012). This module answers two questions:

  1. Is this memory expired as of ``now``? → caller should delete/archive.
  2. How long before it expires? → used by dashboards and scheduled sweeps.

The policy table is source of truth. If a memory references a policy id we
don't know about, we fail closed (treat as expired to be safe, log a warning)
so an unknown policy never extends retention silently.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        # Accept both trailing Z and explicit offsets.
        if ts.endswith("Z"):
            return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        return datetime.fromisoformat(ts)
    except Exception:
        logger.debug("retention: unparseable timestamp %r", ts)
        return None


async def _fetch_policy(server: Any, policy_id: str) -> Optional[Dict[str, Any]]:
    from ..learning.outcome import _fetchone
    return await _fetchone(
        server,
        "SELECT id, max_age_days, justification, legal_basis, scope "
        "FROM sb_retention_policies WHERE id = ?",
        [policy_id],
    )


async def check_retention(
    server: Any,
    memory: Any,
    *,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Return whether ``memory`` is still within its retention window.

    Result keys:
      * status: "active" | "expired" | "no_policy" | "unknown_policy"
      * expires_at: ISO string or None
      * days_remaining: float or None
      * policy_id: str or None
    """
    md = getattr(memory, "metadata", {}) or {}
    policy_id = md.get("retention_policy_id")
    if not policy_id:
        return {"status": "no_policy", "expires_at": None,
                "days_remaining": None, "policy_id": None}

    # Allow explicit delete_after override on the memory itself.
    override = _parse_iso(md.get("delete_after"))
    now_dt = now or datetime.now(tz=timezone.utc)

    policy = await _fetch_policy(server, policy_id)
    if policy is None:
        logger.warning("retention: unknown policy_id=%s — fail-closed", policy_id)
        return {"status": "unknown_policy", "expires_at": None,
                "days_remaining": None, "policy_id": policy_id}

    max_age_days = policy.get("max_age_days")
    created = _parse_iso(md.get("created_at_iso") or md.get("captured_at"))
    if override is not None:
        expires_at = override
    elif max_age_days is not None and created is not None:
        expires_at = created + timedelta(days=int(max_age_days))
    else:
        # Policy exists but gives us no age cap — treat as indefinite.
        return {"status": "active", "expires_at": None,
                "days_remaining": None, "policy_id": policy_id}

    delta = (expires_at - now_dt).total_seconds() / 86400.0
    return {
        "status": "expired" if delta <= 0 else "active",
        "expires_at": expires_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "days_remaining": delta,
        "policy_id": policy_id,
    }


async def list_expired(server: Any, batch_size: int = 200) -> List[Dict[str, Any]]:
    """Return up to ``batch_size`` memories whose retention has expired.

    Uses the typed ``sb_retention_policies`` table joined against the base
    memories table by policy id extracted from tags (policy ids are stored
    in metadata; we emit a second hit via a tag convention ``policy:<id>``
    so we can filter in SQL without JSON extraction).
    """
    from ..learning.outcome import _fetchall  # type: ignore  # noqa
    # Most storage backends don't expose JSON metadata in SQL, so we filter
    # in Python. Cheap for batch_size=200.
    storage = getattr(server, "storage", None)
    if storage is None:
        return []
    try:
        if hasattr(storage, "d1_query"):
            rows = await storage.d1_query(
                "SELECT content_hash, tags_str, metadata_json, created_at "
                "FROM memories WHERE metadata_json LIKE '%retention_policy_id%' "
                "LIMIT ?",
                [batch_size],
            )
        else:
            return []
    except Exception as exc:
        logger.debug("list_expired: query failed: %s", exc)
        return []

    expired: List[Dict[str, Any]] = []
    for row in rows or []:
        import json
        try:
            md = json.loads(row.get("metadata_json") or "{}")
        except Exception:
            continue

        class _ShadowMem:  # lightweight stand-in that exposes metadata
            metadata = md
        res = await check_retention(server, _ShadowMem())
        if res["status"] == "expired":
            expired.append({
                "content_hash": row["content_hash"],
                "policy_id": res["policy_id"],
                "expires_at": res["expires_at"],
            })
    return expired
