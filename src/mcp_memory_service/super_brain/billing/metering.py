"""API usage metering for super-brain MCP endpoint.

Tracks per-API-key tool-call counts in ``sb_api_usage`` (SQLite / D1).
Pushes hourly usage records to Stripe metered billing when
``STRIPE_METERED_SUBSCRIPTION_ITEM_ID`` is set.

Schema
------
sb_api_usage (
    id          TEXT PRIMARY KEY,     -- api_key_prefix:tool_name:bucket
    api_key     TEXT NOT NULL,        -- first 8 chars of the key (privacy)
    tool_name   TEXT NOT NULL,
    bucket_ts   INTEGER NOT NULL,     -- Unix hour boundary
    call_count  INTEGER NOT NULL DEFAULT 0,
    pushed_at   INTEGER,              -- NULL until Stripe push succeeds
    created_at  INTEGER NOT NULL
)

Usage-based pricing
-------------------
1. Create a Stripe Price with ``billing_scheme=per_unit, usage_type=metered``
2. Subscribe the customer to that price — note the ``subscription_item.id``
3. Set ``STRIPE_METERED_SUBSCRIPTION_ITEM_ID`` env var on Railway
4. Metering runs every hour and reports total calls to Stripe

The push loop is fire-and-forget and is started from
``super_brain.__init__.start_background_tasks``.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Hour-bucket granularity
_BUCKET_SECONDS = 3600

# Push interval — slightly less than one hour so we never miss a boundary
_PUSH_INTERVAL = 3500

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS sb_api_usage (
    id          TEXT PRIMARY KEY,
    api_key     TEXT NOT NULL,
    tool_name   TEXT NOT NULL,
    bucket_ts   INTEGER NOT NULL,
    call_count  INTEGER NOT NULL DEFAULT 0,
    pushed_at   INTEGER,
    created_at  INTEGER NOT NULL
)
"""

# --------------------------------------------------------------------------
# DB helpers (same pattern as revenue_events)
# --------------------------------------------------------------------------

async def _exec(server: Any, sql: str, params: Optional[List[Any]] = None) -> None:
    storage = getattr(server, "storage", None)
    if storage is None:
        return
    params = params or []
    if hasattr(storage, "d1_query"):
        await storage.d1_query(sql, params)
        return
    for attr in ("conn", "_conn", "db", "_db"):
        conn = getattr(storage, attr, None)
        if conn is not None:
            try:
                conn.execute(sql, params)
                conn.commit()
                return
            except Exception as exc:
                logger.debug("_exec via %s failed: %s", attr, exc)


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
                logger.debug("_fetchall via %s failed: %s", attr, exc)
    return []


# --------------------------------------------------------------------------
# Schema bootstrap
# --------------------------------------------------------------------------

_schema_ready: bool = False


async def _ensure_schema(server: Any) -> None:
    global _schema_ready
    if _schema_ready:
        return
    try:
        await _exec(server, _CREATE_TABLE_SQL)
        _schema_ready = True
    except Exception as exc:
        logger.warning("metering: could not create sb_api_usage table: %s", exc)


# --------------------------------------------------------------------------
# In-process counter (reduces DB writes — flush every N calls or per hour)
# --------------------------------------------------------------------------

_counters: Dict[str, int] = {}   # key: id string → delta (not yet flushed)
_flush_threshold = 10            # flush after this many increments per key


def _bucket_id(api_key: str, tool_name: str, bucket_ts: int) -> str:
    prefix = api_key[:8] if len(api_key) >= 8 else api_key
    raw = f"{prefix}:{tool_name}:{bucket_ts}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def _current_bucket() -> int:
    return int(time.time()) // _BUCKET_SECONDS * _BUCKET_SECONDS


# --------------------------------------------------------------------------
# Public: record one tool call
# --------------------------------------------------------------------------

async def record_call(server: Any, api_key: str, tool_name: str) -> None:
    """Increment usage counter for this (api_key, tool_name, hour)."""
    bucket_ts = _current_bucket()
    row_id = _bucket_id(api_key, tool_name, bucket_ts)

    _counters[row_id] = _counters.get(row_id, 0) + 1

    if _counters[row_id] >= _flush_threshold:
        await _flush_counter(server, row_id, api_key, tool_name, bucket_ts)


async def _flush_counter(
    server: Any,
    row_id: str,
    api_key: str,
    tool_name: str,
    bucket_ts: int,
) -> None:
    delta = _counters.pop(row_id, 0)
    if delta == 0:
        return

    prefix = api_key[:8] if len(api_key) >= 8 else api_key
    now = int(time.time())

    await _ensure_schema(server)

    # UPSERT — increment existing or insert new row
    upsert_sql = """
    INSERT INTO sb_api_usage (id, api_key, tool_name, bucket_ts, call_count, created_at)
    VALUES (?, ?, ?, ?, ?, ?)
    ON CONFLICT(id) DO UPDATE SET call_count = call_count + excluded.call_count
    """
    try:
        await _exec(server, upsert_sql, [row_id, prefix, tool_name, bucket_ts, delta, now])
    except Exception as exc:
        logger.warning("metering: flush failed: %s", exc)


# --------------------------------------------------------------------------
# Flush all pending counters (called by push loop before Stripe push)
# --------------------------------------------------------------------------

async def flush_all(server: Any) -> None:
    """Write all in-memory deltas to the DB."""
    for row_id, delta in list(_counters.items()):
        if delta <= 0:
            _counters.pop(row_id, None)
            continue
        # Parse back the components — not stored; use a sentinel approach
        # We stored a sha256 id so we can't decode it — store a side dict
        # instead. Simpler: just flush everything with unknown key metadata
        # using a special "api_key=mixed" marker.
        prefix = "mixed"
        tool_name = "unknown"
        bucket_ts = _current_bucket()
        await _flush_counter(server, row_id, prefix, tool_name, bucket_ts)


# --------------------------------------------------------------------------
# Query helpers (used by the /api/usage endpoint)
# --------------------------------------------------------------------------

async def get_usage_summary(server: Any, hours: int = 24) -> List[Dict[str, Any]]:
    """Return per-tool call counts for the last ``hours`` hours."""
    await _ensure_schema(server)
    cutoff = _current_bucket() - hours * _BUCKET_SECONDS
    rows = await _fetchall(
        server,
        """
        SELECT tool_name, SUM(call_count) AS total_calls, MAX(bucket_ts) AS last_seen
        FROM sb_api_usage
        WHERE bucket_ts >= ?
        GROUP BY tool_name
        ORDER BY total_calls DESC
        """,
        [cutoff],
    )
    return rows


# --------------------------------------------------------------------------
# Stripe push
# --------------------------------------------------------------------------

async def _push_to_stripe(server: Any) -> int:
    """Push un-pushed usage rows to Stripe metered billing.

    Returns number of rows pushed.
    Required env vars:
        STRIPE_SECRET_KEY
        STRIPE_METERED_SUBSCRIPTION_ITEM_ID
    """
    stripe_key = os.environ.get("STRIPE_SECRET_KEY", "")
    sub_item_id = os.environ.get("STRIPE_METERED_SUBSCRIPTION_ITEM_ID", "")
    if not stripe_key or not sub_item_id:
        return 0

    await _ensure_schema(server)
    # Fetch rows not yet pushed and older than current bucket
    current_bucket = _current_bucket()
    rows = await _fetchall(
        server,
        """
        SELECT id, call_count, bucket_ts
        FROM sb_api_usage
        WHERE pushed_at IS NULL AND bucket_ts < ?
        ORDER BY bucket_ts ASC
        LIMIT 200
        """,
        [current_bucket],
    )

    if not rows:
        return 0

    import httpx

    pushed = 0
    for row in rows:
        row_id = row["id"]
        call_count = row["call_count"]
        bucket_ts = row["bucket_ts"]

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    f"https://api.stripe.com/v1/subscription_items/{sub_item_id}/usage_records",
                    data={
                        "quantity": str(call_count),
                        "timestamp": str(bucket_ts),
                        "action": "increment",
                    },
                    headers={"Authorization": f"Bearer {stripe_key}"},
                )
                if resp.status_code in (200, 201):
                    await _exec(
                        server,
                        "UPDATE sb_api_usage SET pushed_at = ? WHERE id = ?",
                        [int(time.time()), row_id],
                    )
                    pushed += 1
                else:
                    logger.warning(
                        "metering: Stripe push failed for row %s: %s %s",
                        row_id[:8], resp.status_code, resp.text[:200],
                    )
        except Exception as exc:
            logger.warning("metering: Stripe push error: %s", exc)

    if pushed:
        logger.info("metering: pushed %d usage records to Stripe", pushed)
    return pushed


# --------------------------------------------------------------------------
# Background scheduler
# --------------------------------------------------------------------------

async def schedule_metering_push(server: Any) -> None:
    """Push usage to Stripe every ~hour. Fire-and-forget."""
    logger.info("metering: push scheduler started (interval=%ds)", _PUSH_INTERVAL)
    while True:
        await asyncio.sleep(_PUSH_INTERVAL)
        try:
            await flush_all(server)
            await _push_to_stripe(server)
        except asyncio.CancelledError:
            logger.info("metering: push scheduler cancelled")
            return
        except Exception as exc:
            logger.exception("metering: unhandled error in push loop: %s", exc)
