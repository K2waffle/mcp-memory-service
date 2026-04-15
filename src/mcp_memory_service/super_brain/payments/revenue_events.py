"""Payment-rail-agnostic revenue event recording.

Idempotent via ``revenue_event_id`` primary key. Writes to
``sb_revenue_events`` and tags the referenced memory with
``revenue:recorded`` so retrieval queries can surface monetized artifacts.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


async def _exec(server: Any, sql: str, params: Optional[List[Any]] = None) -> Any:
    from ..learning.outcome import _exec as _ex
    return await _ex(server, sql, params)


async def _fetchone(server: Any, sql: str, params: Optional[List[Any]] = None) -> Optional[Dict[str, Any]]:
    from ..learning.outcome import _fetchone as _fo
    return await _fo(server, sql, params)


async def _fetchall(server: Any, sql: str, params: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
    storage = getattr(server, "storage", None)
    if storage is None:
        return []
    params = params or []
    if hasattr(storage, "d1_query"):
        return await storage.d1_query(sql, params)
    for attr in ("conn", "_conn", "db", "_db"):
        conn = getattr(storage, attr, None)
        if conn is not None:
            try:
                cursor = conn.execute(sql, params)
                rows = cursor.fetchall()
                if not rows:
                    return []
                try:
                    return [{k: row[k] for k in row.keys()} for row in rows]
                except Exception:
                    cols = [d[0] for d in cursor.description]
                    return [dict(zip(cols, r)) for r in rows]
            except Exception as exc:
                logger.debug("fetchall via %s failed: %s", attr, exc)
    return []


async def record_event(server: Any, args: Dict[str, Any]) -> Dict[str, Any]:
    """Idempotently record a revenue event.

    If ``revenue_event_id`` already exists, returns ``already_recorded``
    without double-counting.
    """
    revenue_event_id = args["revenue_event_id"]
    existing = await _fetchone(
        server,
        "SELECT id FROM sb_revenue_events WHERE id = ?",
        [revenue_event_id],
    )
    if existing:
        total = await sum_revenue_for_memory(server, args.get("memory_id")) if args.get("memory_id") else None
        return {
            "revenue_event_id": revenue_event_id,
            "status": "already_recorded",
            "total_attributed_to_memory": total.get("total_cents") if total else None,
        }

    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    await _exec(
        server,
        """
        INSERT INTO sb_revenue_events
          (id, memory_id, counterparty_ref, counterparty_rail,
           amount_cents, currency, margin, recognized_at, pci_scope, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            revenue_event_id,
            args.get("memory_id"),
            args.get("counterparty_ref"),
            args["counterparty_rail"],
            int(args["amount_cents"]),
            args.get("currency", "USD"),
            args.get("margin"),
            args["recognized_at"],
            args.get("pci_scope"),
            now_iso,
        ],
    )

    # Best-effort tag the referenced memory.
    if args.get("memory_id"):
        try:
            await _exec(
                server,
                "UPDATE memories SET tags_str = tags_str || ',revenue:recorded' WHERE content_hash = ? AND tags_str NOT LIKE '%revenue:recorded%'",
                [args["memory_id"]],
            )
        except Exception as exc:
            logger.debug("record_event: tag update skipped: %s", exc)

    total = await sum_revenue_for_memory(server, args.get("memory_id")) if args.get("memory_id") else None
    return {
        "revenue_event_id": revenue_event_id,
        "status": "recorded",
        "total_attributed_to_memory": total.get("total_cents") if total else None,
    }


async def sum_revenue_for_memory(server: Any, memory_id: str) -> Dict[str, Any]:
    rows = await _fetchall(
        server,
        """
        SELECT amount_cents, currency FROM sb_revenue_events
        WHERE memory_id = ?
        """,
        [memory_id],
    )
    total_cents = sum(int(r["amount_cents"]) for r in rows)
    currencies = sorted({r.get("currency", "USD") for r in rows})
    return {"total_cents": total_cents, "count": len(rows), "currencies": currencies}


async def list_recent_events(server: Any, limit: int = 50) -> List[Dict[str, Any]]:
    return await _fetchall(
        server,
        """
        SELECT id, memory_id, counterparty_rail, amount_cents, currency, recognized_at
        FROM sb_revenue_events
        ORDER BY recognized_at DESC
        LIMIT ?
        """,
        [int(limit)],
    )
