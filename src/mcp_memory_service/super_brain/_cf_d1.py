"""Cloudflare D1 raw-SQL helpers for super_brain handlers.

The sb handlers need to write to typed tables (sb_decisions, sb_opportunities,
sb_revenue_events, sb_procedure_scores, ...). The Cloudflare storage backend
doesn't expose convenience methods like ``d1_execute`` / ``d1_query``, so
these helpers speak directly to the D1 REST API via the backend's own
``_retry_request`` + ``d1_url``.

If the storage object isn't Cloudflare-shaped, callers fall back to their
existing sqlite paths.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def is_cloudflare_like(storage: Any) -> bool:
    """True if the storage object looks like the Cloudflare backend."""
    return storage is not None and hasattr(storage, "d1_url") and hasattr(storage, "_retry_request")


async def d1_query(storage: Any, sql: str, params: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
    """Run a single D1 query; return the first result set's rows.

    Raises on transport error or D1-reported failure so callers can surface.
    """
    payload = {"sql": sql, "params": list(params or [])}
    resp = await storage._retry_request("POST", f"{storage.d1_url}/query", json=payload)
    data = resp.json()
    if not data.get("success"):
        raise RuntimeError(f"D1 query failed: {data.get('errors') or data}")
    results = data.get("result") or []
    if not results:
        return []
    return results[0].get("results", []) or []


async def d1_execute(storage: Any, sql: str, params: Optional[List[Any]] = None) -> None:
    """Fire-and-forget write; raises on failure."""
    await d1_query(storage, sql, params)


async def d1_fetchone(storage: Any, sql: str, params: Optional[List[Any]] = None) -> Optional[Dict[str, Any]]:
    rows = await d1_query(storage, sql, params)
    return rows[0] if rows else None


async def d1_fetchall(storage: Any, sql: str, params: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
    return await d1_query(storage, sql, params)
