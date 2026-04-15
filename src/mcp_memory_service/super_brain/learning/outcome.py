"""Beta(alpha, beta) procedure outcome scoring.

Every procedural memory has a posterior over "will this work next time?"
represented by a Beta distribution. Runs update alpha (success pseudo-count)
or beta (failure pseudo-count); selection uses Thompson sampling so
successful procedures get more opportunities to prove themselves without
starving exploration of new ones.

Why Beta: closed-form conjugate prior for Bernoulli outcomes, trivially
updatable, and well-understood. Thompson sampling is known to be
near-optimal for multi-armed bandits without hand-tuned epsilon.
"""
from __future__ import annotations

import logging
import random
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


async def _exec(server: Any, sql: str, params: Optional[List[Any]] = None) -> Any:
    """Execute SQL against whichever backend is attached.

    Mirrors the dispatch pattern in ``super_brain.tools._insert_typed``.
    """
    storage = getattr(server, "storage", None)
    if storage is None:
        raise RuntimeError("server.storage not available")
    params = params or []
    if hasattr(storage, "d1_execute"):
        return await storage.d1_execute(sql, params)
    from .._cf_d1 import is_cloudflare_like, d1_execute as _cf_exec
    if is_cloudflare_like(storage):
        await _cf_exec(storage, sql, params)
        return None
    for attr in ("conn", "_conn", "db", "_db"):
        conn = getattr(storage, attr, None)
        if conn is not None:
            try:
                cursor = conn.execute(sql, params)
                conn.commit()
                return cursor
            except Exception as exc:
                logger.debug("exec via %s failed: %s", attr, exc)
    raise RuntimeError("no supported storage connection for raw SQL")


async def _fetchone(server: Any, sql: str, params: Optional[List[Any]] = None) -> Optional[Dict[str, Any]]:
    storage = getattr(server, "storage", None)
    if storage is None:
        return None
    params = params or []
    if hasattr(storage, "d1_query_one"):
        return await storage.d1_query_one(sql, params)
    from .._cf_d1 import is_cloudflare_like, d1_fetchone as _cf_fetchone
    if is_cloudflare_like(storage):
        return await _cf_fetchone(storage, sql, params)
    for attr in ("conn", "_conn", "db", "_db"):
        conn = getattr(storage, attr, None)
        if conn is not None:
            try:
                cursor = conn.execute(sql, params)
                row = cursor.fetchone()
                if row is None:
                    return None
                # sqlite3.Row → dict when row_factory set; fall back to keys()
                try:
                    return {k: row[k] for k in row.keys()}
                except Exception:
                    cols = [d[0] for d in cursor.description]
                    return dict(zip(cols, row))
            except Exception as exc:
                logger.debug("fetchone via %s failed: %s", attr, exc)
    return None


async def update_beta_score(server: Any, procedure_memory_id: str, success: bool,
                            weight: float = 1.0) -> Dict[str, Any]:
    """Upsert alpha/beta for a procedure after an observed outcome.

    Returns the posterior stats so callers can decide whether to keep using
    this procedure.
    """
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    # Try to fetch current state; if missing, seed with uniform prior (1, 1).
    row = await _fetchone(
        server,
        "SELECT alpha, beta, n_runs FROM sb_procedure_scores WHERE memory_id = ?",
        [procedure_memory_id],
    )
    alpha = float(row["alpha"]) if row else 1.0
    beta = float(row["beta"]) if row else 1.0
    n_runs = int(row["n_runs"]) if row else 0

    if success:
        alpha += weight
    else:
        beta += weight
    n_runs += 1

    await _exec(
        server,
        """
        INSERT INTO sb_procedure_scores (memory_id, alpha, beta, n_runs, last_used_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(memory_id) DO UPDATE SET
            alpha = excluded.alpha,
            beta = excluded.beta,
            n_runs = excluded.n_runs,
            last_used_at = excluded.last_used_at,
            updated_at = excluded.updated_at
        """,
        [procedure_memory_id, alpha, beta, n_runs, now, now],
    )

    return {
        "procedure_memory_id": procedure_memory_id,
        "alpha": alpha,
        "beta": beta,
        "n_runs": n_runs,
        "posterior_mean": alpha / (alpha + beta),
    }


async def get_score(server: Any, procedure_memory_id: str) -> Dict[str, Any]:
    row = await _fetchone(
        server,
        "SELECT alpha, beta, n_runs, last_used_at FROM sb_procedure_scores WHERE memory_id = ?",
        [procedure_memory_id],
    )
    if not row:
        return {"alpha": 1.0, "beta": 1.0, "n_runs": 0,
                "posterior_mean": 0.5, "seen": False}
    alpha = float(row["alpha"])
    beta = float(row["beta"])
    return {
        "alpha": alpha,
        "beta": beta,
        "n_runs": int(row["n_runs"]),
        "last_used_at": row.get("last_used_at"),
        "posterior_mean": alpha / (alpha + beta),
        "seen": True,
    }


async def thompson_sample(server: Any, candidate_procedure_ids: List[str]) -> str:
    """Draw a sample from each candidate's Beta posterior; return the argmax.

    Exploration/exploitation tradeoff is automatic: new procedures (alpha=beta=1)
    have uniform draws, so they win sometimes even against procs with track records.
    """
    if not candidate_procedure_ids:
        raise ValueError("candidate_procedure_ids must be non-empty")

    best_id = candidate_procedure_ids[0]
    best_sample = -1.0
    for proc_id in candidate_procedure_ids:
        score = await get_score(server, proc_id)
        sample = random.betavariate(score["alpha"], score["beta"])
        if sample > best_sample:
            best_sample = sample
            best_id = proc_id
    return best_id
