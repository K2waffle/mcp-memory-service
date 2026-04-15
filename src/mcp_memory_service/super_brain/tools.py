"""MCP tool registration for super-brain entity-specific tools.

Adds these tools on top of upstream's 12 unified tools:
- ``decision_record`` — write a decision with rationale, supersedes link, validity window
- ``opportunity_rank`` — write an opportunity with revenue-ranking fields
- ``research_source_with_trust`` — write a research_sources entity with trust tier
- ``task_packet_write`` — write a structured task packet
- ``distill`` — promote episodic runs to a procedural artifact (Hinton distillation)
- ``verify_claim`` — product-of-experts cross-reference verifier
- ``procedure_score_update`` — update Beta(alpha, beta) outcome counts
- ``revenue_event_record`` — idempotently record a revenue-recognition event

Each tool is a thin wrapper around the existing MemoryStorage.store() plus
a typed-table write. Fork-surface principle: no upstream file edits; we
register through the server's extension point.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# --- Tool schemas (JSON-schema-lite; server_impl picks these up via register_tools) ---

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "name": "decision_record",
        "description": (
            "Record a decision artifact with rationale, supersedes link, and validity "
            "window. Writes to sb_decisions typed table and also lands as a memory "
            "with memory_type='decisions' for vector search."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["tldr", "rationale", "valid_from"],
            "properties": {
                "tldr": {"type": "string", "maxLength": 280},
                "rationale": {"type": "string"},
                "supersedes_id": {"type": "string"},
                "valid_from": {"type": "string", "description": "ISO 8601 UTC"},
                "valid_to": {"type": "string", "description": "ISO 8601 UTC"},
                "scope": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "source": {"type": "string", "description": "URL or citation"},
                "cross_refs": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
    {
        "name": "opportunity_rank",
        "description": (
            "Record a revenue opportunity with the 7 ranking fields "
            "(speed_to_first_dollar, earning_ceiling, automation_leverage, "
            "repeatability, distribution, retention, compliance_risk) and a "
            "monetization_route. Used by the revenue-ignition workflow."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["tldr", "speed_to_first_dollar", "earning_ceiling"],
            "properties": {
                "tldr": {"type": "string", "maxLength": 280},
                "body": {"type": "string"},
                "speed_to_first_dollar": {"type": "integer", "minimum": 1, "maximum": 5},
                "earning_ceiling": {"type": "number", "description": "USD estimate"},
                "automation_leverage": {"type": "integer", "minimum": 1, "maximum": 5},
                "repeatability": {"type": "integer", "minimum": 1, "maximum": 5},
                "distribution": {"type": "integer", "minimum": 1, "maximum": 5},
                "retention": {"type": "integer", "minimum": 1, "maximum": 5},
                "compliance_risk": {"type": "integer", "minimum": 1, "maximum": 5},
                "monetization_route": {
                    "type": "string",
                    "enum": ["direct_sale", "gated_api", "derived_signal", "agent_service", "other"],
                },
                "counterparty_ref": {"type": "string"},
            },
        },
    },
    {
        "name": "research_source_with_trust",
        "description": (
            "Record a research source with explicit trust tier. Trust tiers: "
            "primary_doc > paper > vendor > guru. Semantic-tier promotion "
            "requires >=2 independent cross-references."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["url", "trust_tier", "tldr"],
            "properties": {
                "url": {"type": "string", "format": "uri"},
                "trust_tier": {"type": "string", "enum": ["primary_doc", "paper", "vendor", "guru"]},
                "tldr": {"type": "string", "maxLength": 280},
                "body": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "license": {"type": "string"},
                "attribution_text": {"type": "string"},
                "is_ai_generated": {"type": "boolean"},
            },
        },
    },
    {
        "name": "task_packet_write",
        "description": (
            "Write a structured task packet (goal, inputs, outputs, citations, cost, "
            "outcome). Born episodic; promoted to procedural when repeated "
            "successfully (>=3 similar successful runs)."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["goal", "outcome"],
            "properties": {
                "goal": {"type": "string"},
                "inputs": {"type": "array"},
                "outputs": {"type": "array"},
                "citations": {"type": "array"},
                "tokens_in": {"type": "integer"},
                "tokens_out": {"type": "integer"},
                "dollars": {"type": "number"},
                "wall_ms": {"type": "integer"},
                "tool_calls": {"type": "integer"},
                "retries": {"type": "integer"},
                "outcome": {"type": "string", "enum": ["success", "partial", "failure", "abandoned"]},
                "verifier_id": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
    {
        "name": "distill",
        "description": (
            "Distill a set of episodic memory ids into a single procedural artifact. "
            "Implements Hinton's distillation principle: compress many runs into "
            "reusable know-how. Promoted procedure gets alpha/beta init from inputs."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["source_ids", "tldr"],
            "properties": {
                "source_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "tldr": {"type": "string", "maxLength": 280},
                "body": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
    {
        "name": "verify_claim",
        "description": (
            "Product-of-experts verifier. Given a claim, runs the configured "
            "verifier panel (multi-model consensus + retrieval-grounded check) "
            "and returns a confidence score plus groundedness evidence. Gates "
            "episodic->semantic promotion."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["claim"],
            "properties": {
                "claim": {"type": "string"},
                "context_memory_ids": {"type": "array", "items": {"type": "string"}},
                "n_experts": {"type": "integer", "minimum": 1, "maximum": 7, "default": 3},
            },
        },
    },
    {
        "name": "procedure_score_update",
        "description": (
            "Update Beta(alpha, beta) outcome counts for a procedure after a run. "
            "Used for Thompson-sampling procedure selection."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["procedure_memory_id", "success"],
            "properties": {
                "procedure_memory_id": {"type": "string"},
                "success": {"type": "boolean"},
                "weight": {"type": "number", "default": 1.0},
            },
        },
    },
    {
        "name": "revenue_event_record",
        "description": (
            "Idempotently record a revenue-recognition event. Requires "
            "revenue_event_id (idempotency key). Payment-rail agnostic."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["revenue_event_id", "amount_cents", "counterparty_rail", "recognized_at"],
            "properties": {
                "revenue_event_id": {"type": "string"},
                "memory_id": {"type": "string", "description": "Opportunity or task_packet id"},
                "counterparty_ref": {"type": "string"},
                "counterparty_rail": {
                    "type": "string",
                    "enum": ["stripe", "crypto", "ach", "wire", "paypal", "platform_credit", "other"],
                },
                "amount_cents": {"type": "integer", "minimum": 0},
                "currency": {"type": "string", "default": "USD"},
                "margin": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "recognized_at": {"type": "string", "description": "ISO 8601 UTC"},
                "pci_scope": {
                    "type": "string",
                    "enum": ["saq_a", "saq_a_ep", "saq_d", "out_of_scope"],
                },
            },
        },
    },
]


# --- Handler implementations ---

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _ulid() -> str:
    # Use uuid4 as ULID substitute; good enough for unique IDs and avoids extra dep.
    return uuid.uuid4().hex


async def _store_memory(server: Any, *, content: str, memory_type: str, tags: List[str],
                        metadata: Optional[Dict[str, Any]] = None) -> str:
    """Store a memory via upstream pipeline; returns content_hash."""
    from ..models.memory import Memory

    import hashlib
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    memory = Memory(
        content=content,
        content_hash=content_hash,
        tags=tags or ["untagged"],
        memory_type=memory_type,
        metadata=metadata or {},
    )
    storage = getattr(server, "storage", None)
    if storage is None:
        raise RuntimeError("server.storage not available")
    await storage.store(memory)
    return content_hash


async def handle_decision_record(server: Any, args: Dict[str, Any]) -> Dict[str, Any]:
    tldr = args["tldr"]
    rationale = args["rationale"]
    tags = list(args.get("tags", []))
    if "entity:decisions" not in tags:
        tags.append("entity:decisions")
    meta = {
        "tldr": tldr,
        "source": args.get("source"),
        "cross_refs": args.get("cross_refs", []),
        "supersedes_id": args.get("supersedes_id"),
        "valid_from": args["valid_from"],
        "valid_to": args.get("valid_to"),
        "scope": args.get("scope"),
    }
    content = f"{tldr}\n\n{rationale}"
    memory_id = await _store_memory(
        server, content=content, memory_type="semantic", tags=tags, metadata=meta
    )
    # Typed-table insert — wired by caller into storage-specific impl.
    await _insert_typed(server, "sb_decisions", {
        "id": _ulid(),
        "memory_id": memory_id,
        "rationale": rationale,
        "supersedes_id": args.get("supersedes_id"),
        "valid_from": args["valid_from"],
        "valid_to": args.get("valid_to"),
        "scope": args.get("scope"),
    })
    return {"memory_id": memory_id, "status": "ok"}


async def handle_opportunity_rank(server: Any, args: Dict[str, Any]) -> Dict[str, Any]:
    tldr = args["tldr"]
    body = args.get("body", "")
    tags = ["entity:opportunities"]
    mr = args.get("monetization_route")
    if mr:
        tags.append(f"revenue:{mr}")
    meta = {
        "tldr": tldr,
        "speed_to_first_dollar": args["speed_to_first_dollar"],
        "earning_ceiling": args["earning_ceiling"],
        "automation_leverage": args.get("automation_leverage"),
        "repeatability": args.get("repeatability"),
        "distribution": args.get("distribution"),
        "retention": args.get("retention"),
        "compliance_risk": args.get("compliance_risk"),
        "monetization_route": mr,
        "counterparty_ref": args.get("counterparty_ref"),
    }
    content = f"{tldr}\n\n{body}"
    memory_id = await _store_memory(
        server, content=content, memory_type="semantic", tags=tags, metadata=meta
    )
    return {"memory_id": memory_id, "status": "ok"}


async def handle_research_source_with_trust(server: Any, args: Dict[str, Any]) -> Dict[str, Any]:
    tldr = args["tldr"]
    url = args["url"]
    trust = args["trust_tier"]
    tags = list(args.get("tags", []))
    tags.extend(["entity:research_sources", f"source:{_domain_of(url)}"])
    meta = {
        "tldr": tldr,
        "source": url,
        "trust_tier": trust,
        "license": args.get("license"),
        "attribution_text": args.get("attribution_text"),
        "is_ai_generated": args.get("is_ai_generated"),
    }
    body = args.get("body", "")
    content = f"{tldr}\n\nSource: {url}\n\n{body}"
    memory_id = await _store_memory(
        server, content=content, memory_type="semantic", tags=tags, metadata=meta
    )
    return {"memory_id": memory_id, "status": "ok"}


async def handle_task_packet_write(server: Any, args: Dict[str, Any]) -> Dict[str, Any]:
    goal = args["goal"]
    tags = ["entity:task_packets"]
    meta = {
        "tldr": goal[:280],
        "tokens_in": args.get("tokens_in"),
        "tokens_out": args.get("tokens_out"),
        "wall_ms": args.get("wall_ms"),
        "tool_calls": args.get("tool_calls"),
        "retries": args.get("retries"),
        "outcome": args["outcome"],
        "verifier_id": args.get("verifier_id"),
    }
    content = f"Goal: {goal}\nOutcome: {args['outcome']}"
    memory_id = await _store_memory(
        server, content=content, memory_type="episodic", tags=tags, metadata=meta
    )
    await _insert_typed(server, "sb_task_packets", {
        "id": _ulid(),
        "memory_id": memory_id,
        "goal": goal,
        "inputs_json": json.dumps(args.get("inputs", [])),
        "outputs_json": json.dumps(args.get("outputs", [])),
        "citations_json": json.dumps(args.get("citations", [])),
        "cost_tokens_in": args.get("tokens_in"),
        "cost_tokens_out": args.get("tokens_out"),
        "cost_dollars": args.get("dollars"),
        "wall_ms": args.get("wall_ms"),
        "tool_calls": args.get("tool_calls"),
        "retries": args.get("retries"),
        "outcome": args["outcome"],
        "verifier_id": args.get("verifier_id"),
    })
    return {"memory_id": memory_id, "status": "ok"}


async def handle_distill(server: Any, args: Dict[str, Any]) -> Dict[str, Any]:
    from .learning.distill import distill_episodic_to_procedural
    return await distill_episodic_to_procedural(server, args["source_ids"],
                                                 tldr=args["tldr"],
                                                 body=args.get("body", ""),
                                                 tags=args.get("tags", []))


async def handle_verify_claim(server: Any, args: Dict[str, Any]) -> Dict[str, Any]:
    from .guardrails.crossref import verify
    return await verify(server, args["claim"],
                        context_memory_ids=args.get("context_memory_ids"),
                        n_experts=args.get("n_experts", 3))


async def handle_procedure_score_update(server: Any, args: Dict[str, Any]) -> Dict[str, Any]:
    from .learning.outcome import update_beta_score
    return await update_beta_score(server, args["procedure_memory_id"],
                                   success=args["success"],
                                   weight=args.get("weight", 1.0))


async def handle_revenue_event_record(server: Any, args: Dict[str, Any]) -> Dict[str, Any]:
    from .payments.revenue_events import record_event
    return await record_event(server, args)


# --- Typed-table insert helper ---

async def _insert_typed(server: Any, table: str, row: Dict[str, Any]) -> None:
    """Insert into a sb_* typed table via the underlying storage connection.

    Tries multiple integration hooks depending on backend:
    - SQLite-vec local: ``storage._conn`` or ``storage.conn``
    - Cloudflare D1: ``storage.d1_execute(sql, params)``
    - Hybrid: routes to the local sqlite_vec half.
    """
    storage = getattr(server, "storage", None)
    if storage is None:
        raise RuntimeError("storage unavailable")

    keys = [k for k, v in row.items() if v is not None]
    vals = [row[k] for k in keys]
    placeholders = ", ".join(["?"] * len(keys))
    sql = f"INSERT OR REPLACE INTO {table} ({', '.join(keys)}) VALUES ({placeholders})"

    if hasattr(storage, "d1_execute"):
        await storage.d1_execute(sql, vals)
        return

    # Cloudflare backend — use raw D1 REST via storage's own retry client.
    from ._cf_d1 import is_cloudflare_like, d1_execute as _cf_exec
    if is_cloudflare_like(storage):
        await _cf_exec(storage, sql, vals)
        return

    # SQLite fallback
    for attr in ("conn", "_conn", "db", "_db"):
        conn = getattr(storage, attr, None)
        if conn is not None:
            try:
                cursor = conn.execute(sql, vals)
                conn.commit()
                return
            except Exception as exc:
                logger.debug("typed insert via %s failed: %s", attr, exc)

    logger.warning("typed insert: no supported connection attr on storage (table=%s)", table)


def _domain_of(url: str) -> str:
    from urllib.parse import urlparse
    try:
        return urlparse(url).netloc or "unknown"
    except Exception:
        return "unknown"


# --- Registration entry point ---

HANDLER_MAP = {
    "decision_record": handle_decision_record,
    "opportunity_rank": handle_opportunity_rank,
    "research_source_with_trust": handle_research_source_with_trust,
    "task_packet_write": handle_task_packet_write,
    "distill": handle_distill,
    "verify_claim": handle_verify_claim,
    "procedure_score_update": handle_procedure_score_update,
    "revenue_event_record": handle_revenue_event_record,
}


def register_tools(server: Any) -> None:
    """Install our tool handlers onto the server.

    Supports three shapes of server extension API:
    - ``server.register_tool(name, handler, schema)``
    - ``server.add_tool(...)``
    - Fallback: attaches ``server.super_brain_tools`` and
      ``server.super_brain_handlers`` dicts for the caller (server_impl
      shim) to merge into its dispatcher.
    """
    schemas_by_name = {t["name"]: t for t in TOOL_SCHEMAS}

    if hasattr(server, "register_tool"):
        for name, handler in HANDLER_MAP.items():
            server.register_tool(name, handler, schemas_by_name[name])
        return

    if hasattr(server, "add_tool"):
        for name, handler in HANDLER_MAP.items():
            server.add_tool(name=name, handler=handler, schema=schemas_by_name[name])
        return

    # Fallback: attach dicts. server_impl's own shim can pick these up.
    server.super_brain_tools = schemas_by_name
    server.super_brain_handlers = dict(HANDLER_MAP)
    logger.info("super_brain: tool dicts attached (shim-mode registration)")
