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

# --- CLS consolidation counter (Hinton chain-reaction / catastrophic-forgetting fix) ---
# After CONSOLIDATION_THRESHOLD episodic writes in a session, automatically trigger
# a background distillation pass to compress episodic → procedural gists.
# This mirrors the hippocampus→neocortex replay that prevents fast learning
# from overwriting previously consolidated structure.
_CONSOLIDATION_THRESHOLD = 8
_write_counter: Dict[int, int] = {}  # keyed by id(server)


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
    {
        "name": "analogical_search",
        "description": (
            "Zero-inference creative retrieval via vector arithmetic. "
            "Computes v(new_problem) - v(known_problem) + v(known_solution) "
            "in embedding space. Implements Hinton thought-vector analogy "
            "principle: creativity as offset navigation through concept-space."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["new_problem", "template_problem", "template_solution_id"],
            "properties": {
                "new_problem": {
                    "type": "string",
                    "description": "The novel problem you want to solve.",
                },
                "template_problem": {
                    "type": "string",
                    "description": "A known problem whose solution is already stored.",
                },
                "template_solution_id": {
                    "type": "string",
                    "description": "memory_id (content_hash) of the known solution.",
                },
                "top_k": {
                    "type": "integer",
                    "default": 3,
                    "description": "Number of analogical candidates to return.",
                },
            },
        },
    },
    {
        "name": "distill_soft",
        "description": (
            "Distill with soft probability distribution over outcomes (dark "
            "knowledge). Stores the full confidence landscape, not just the "
            "winning answer. Future agents query the distribution entropy to "
            "decide whether to trust the cached answer (low entropy) or "
            "re-run inference (high entropy). Implements Hinton's dark-"
            "knowledge principle."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["source_ids", "tldr", "soft_distribution"],
            "properties": {
                "source_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "content_hash IDs of the episodic source memories.",
                },
                "tldr": {"type": "string", "maxLength": 280},
                "body": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "soft_distribution": {
                    "type": "object",
                    "description": (
                        "outcome -> probability weight mapping. "
                        "Values are normalized internally so they need not "
                        "sum to 1. Example: {\"success\": 0.8, \"partial\": 0.15, "
                        "\"failure\": 0.05}"
                    ),
                    "additionalProperties": {"type": "number"},
                },
            },
        },
    },
    {
        "name": "revenue_plan",
        "description": (
            "Run an Economic Monte Carlo Tree Search (MCTS) over the known "
            "revenue-action library to find the highest-expected-value move "
            "sequence for the next 30 days.  Returns a ranked action plan "
            "with projected revenue, confidence, and which steps can run "
            "fully automatically without human input."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "target_monthly": {
                    "type": "number",
                    "description": "Monthly revenue target in USD.",
                    "default": 5000,
                },
                "n_simulations": {
                    "type": "integer",
                    "description": "MCTS rollout iterations (higher = more thorough).",
                    "default": 50,
                    "minimum": 1,
                    "maximum": 500,
                },
                "current_assets": {
                    "type": "object",
                    "description": (
                        "Optional description of already-deployed assets.  "
                        "Pass {\"deployed\": [\"deploy_api_metering\"]} to mark "
                        "a move as already completed so its dependents unlock."
                    ),
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


async def _maybe_consolidate(server: Any, memory_type: str) -> None:
    """Fire-and-forget consolidation after burst episodic writes.

    Counts episodic writes per server instance. When the count hits
    _CONSOLIDATION_THRESHOLD, retrieves recent episodic memories and
    distills any tag-group with ≥3 members into a procedural gist.
    Runs async so it never blocks the caller.
    """
    if memory_type != "episodic":
        return
    server_key = id(server)
    _write_counter[server_key] = _write_counter.get(server_key, 0) + 1
    if _write_counter[server_key] < _CONSOLIDATION_THRESHOLD:
        return
    _write_counter[server_key] = 0  # reset before async work to avoid double-fire

    try:
        storage = getattr(server, "storage", None)
        if storage is None:
            return
        # Pull recent episodic memories.
        recent = await storage.retrieve("recent episodic", n_results=20, tags=None, min_confidence=0.0)
        if not recent:
            return
        # Group by primary entity tag.
        from collections import defaultdict
        groups: Dict[str, List[Any]] = defaultdict(list)
        for r in recent:
            mem = getattr(r, "memory", r)
            mem_type = getattr(mem, "memory_type", "") or ""
            if mem_type != "episodic":
                continue
            tags = getattr(mem, "tags", []) or []
            entity_tag = next((t for t in tags if t.startswith("entity:")), "entity:general")
            groups[entity_tag].append(getattr(mem, "content_hash", None))

        from .learning.distill import distill_episodic_to_procedural
        for entity_tag, ids in groups.items():
            ids = [i for i in ids if i]
            if len(ids) < 3:
                continue
            try:
                result = await distill_episodic_to_procedural(
                    server,
                    source_ids=ids[:10],
                    tldr=f"Auto-consolidated {entity_tag} procedures ({len(ids)} episodes)",
                    tags=[entity_tag, "auto_consolidated"],
                )
                logger.info(
                    "_maybe_consolidate: distilled %d episodes → %s (id=%s)",
                    len(ids), entity_tag, result.get("memory_id", "?")[:12],
                )
            except Exception as exc:
                logger.debug("_maybe_consolidate: distill failed for %s: %s", entity_tag, exc)
    except Exception as exc:
        logger.debug("_maybe_consolidate: consolidation pass failed (non-fatal): %s", exc)


def _extract_gist(content: str, max_len: int = 280) -> str:
    """Hinton CLS principle: extract the gist, discard the episode.

    Returns the first complete sentence(s) up to max_len chars. If the content
    is already short, returns it unchanged. This is the neocortical compression
    step — store the highly-semanticized gist, not raw episodic verbatim.
    """
    if len(content) <= max_len:
        return content
    # Try to cut at a sentence boundary within the limit.
    for sep in (". ", ".\n", "! ", "? "):
        idx = content.rfind(sep, 0, max_len)
        if idx > max_len // 2:  # must be at least halfway in
            return content[:idx + 1].strip()
    # No sentence boundary found — hard truncate at word boundary.
    idx = content.rfind(" ", 0, max_len)
    return content[:idx].strip() + "…" if idx > 0 else content[:max_len] + "…"


async def _store_memory(server: Any, *, content: str, memory_type: str, tags: List[str],
                        metadata: Optional[Dict[str, Any]] = None,
                        skip_novelty_gate: bool = False) -> str:
    """Store a memory via upstream pipeline; returns content_hash.

    Applies the novelty gate (Hinton prediction-error principle) before writing:
    if the content is ≥92% cosine-similar to an existing chunk the write is
    skipped and the nearest existing ID is returned instead.
    """
    from ..models.memory import Memory

    import hashlib
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

    # --- Novelty gate (prediction-error filter) ---
    if not skip_novelty_gate:
        try:
            from .guardrails.novelty_gate import gate_write
            gate = await gate_write(server, content, tags)
            if not gate["allow"]:
                logger.info(
                    "_store_memory: novelty gate blocked write — %s (nearest=%s)",
                    gate["reason"],
                    gate.get("nearest_id"),
                )
                return gate["nearest_id"] or content_hash
        except Exception as exc:  # gate errors must never break the write
            logger.warning("_store_memory: novelty gate error (skipping gate): %s", exc)

    # --- CLS gist extraction (Hinton forgetting-as-intelligence principle) ---
    # Auto-populate tldr with compressed gist when not already set.
    # The gist is stored as metadata so retrieval returns the high-signal
    # summary rather than the full episodic blob.
    meta = dict(metadata or {})
    if not meta.get("tldr") and len(content) > 280:
        meta["tldr"] = _extract_gist(content)

    memory = Memory(
        content=content,
        content_hash=content_hash,
        tags=tags or ["untagged"],
        memory_type=memory_type,
        metadata=meta,
    )
    storage = getattr(server, "storage", None)
    if storage is None:
        raise RuntimeError("server.storage not available")
    await storage.store(memory)

    # --- CLS consolidation gate (fire-and-forget) ---
    import asyncio
    asyncio.ensure_future(_maybe_consolidate(server, memory_type))

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
    """Verify a claim using product-of-experts cross-reference.

    MC dropout pre-routing (Hinton uncertainty principle):
    - route="cache"     → high agreement across retrieval passes; return best existing evidence
    - route="standard"  → normal verification (default)
    - route="expensive" → low agreement; flag claim as high-uncertainty for human review
    """
    claim = args["claim"]

    # --- MC Dropout pre-route ---
    route = "standard"
    uncertainty_meta: Dict[str, Any] = {}
    try:
        from .learning.mc_router import estimate_uncertainty
        u = await estimate_uncertainty(server, claim, n_passes=10)
        route = u.get("route", "standard")
        uncertainty_meta = {
            "mc_route": route,
            "mc_agreement": u.get("agreement"),
            "mc_confidence": u.get("confidence"),
        }
        logger.debug("verify_claim: MC route=%s agreement=%.2f", route, u.get("agreement", 0))
    except Exception as exc:
        logger.debug("verify_claim: MC dropout pre-route failed (non-fatal): %s", exc)

    if route == "cache":
        # High confidence — skip expensive cross-reference; return top retrieved evidence.
        try:
            storage = getattr(server, "storage", None)
            if storage is not None:
                results = await storage.retrieve(claim, n_results=3, tags=None, min_confidence=0.0)
                top = [
                    {"memory_id": getattr(getattr(r, "memory", r), "content_hash", None),
                     "similarity": getattr(r, "relevance", None)}
                    for r in (results or [])
                ]
                return {
                    "verdict": "cached_high_confidence",
                    "evidence": top,
                    "route": "cache",
                    **uncertainty_meta,
                }
        except Exception:
            pass  # fall through to standard verification

    from .guardrails.crossref import verify
    result = await verify(server, claim,
                          context_memory_ids=args.get("context_memory_ids"),
                          n_experts=args.get("n_experts", 3))

    result.update(uncertainty_meta)
    if route == "expensive":
        result["needs_human_review"] = True
        result["review_reason"] = "mc_dropout_low_agreement"

    return result


async def handle_procedure_score_update(server: Any, args: Dict[str, Any]) -> Dict[str, Any]:
    from .learning.outcome import update_beta_score
    return await update_beta_score(server, args["procedure_memory_id"],
                                   success=args["success"],
                                   weight=args.get("weight", 1.0))


async def handle_revenue_event_record(server: Any, args: Dict[str, Any]) -> Dict[str, Any]:
    from .payments.revenue_events import record_event
    return await record_event(server, args)


async def handle_analogical_search(server: Any, args: Dict[str, Any]) -> Dict[str, Any]:
    """Thought-vector analogy: v(new_problem) - v(template_problem) + v(template_solution).

    Hinton principle: creativity is navigation through vector space. A solution
    to a known problem, offset by the delta between the new and known problem
    descriptions, should land near a good solution to the new problem.
    """
    import numpy as np
    from .guardrails.novelty_gate import _get_embedding, _query_nearest

    new_problem        = args["new_problem"]
    template_problem   = args["template_problem"]
    template_sol_id    = args["template_solution_id"]
    top_k              = int(args.get("top_k") or 3)

    # Step 1 — embed new problem (A) and template problem (B).
    vec_A = await _get_embedding(server, new_problem)
    vec_B = await _get_embedding(server, template_problem)

    # Step 2 — fetch the template solution memory and get its embedding (C).
    vec_C: Optional[List[float]] = None
    from .learning.distill import _fetch_memory_by_hash

    solution_mem = await _fetch_memory_by_hash(server, template_sol_id)
    if solution_mem is not None:
        # Try to get the stored embedding if available; else re-embed content.
        stored_vec = getattr(solution_mem, "embedding", None)
        if stored_vec is not None:
            if hasattr(stored_vec, "tolist"):
                stored_vec = stored_vec.tolist()
            vec_C = list(stored_vec)
        if vec_C is None:
            content = getattr(solution_mem, "content", "") or ""
            vec_C = await _get_embedding(server, content)

    # Step 3 — check if vector arithmetic is possible.
    if vec_A is None or vec_B is None or vec_C is None:
        # Fallback: semantic search on the new problem alone.
        logger.debug(
            "analogical_search: vector arithmetic unavailable (A=%s B=%s C=%s) — "
            "falling back to semantic search on new_problem",
            vec_A is not None, vec_B is not None, vec_C is not None,
        )
        storage = getattr(server, "storage", None)
        if storage is None:
            return {"results": [], "method": "fallback_no_storage"}
        try:
            results = await storage.retrieve(new_problem, n_results=top_k, tags=None, min_confidence=0.0)
            return {
                "results": [
                    {
                        "memory_id": getattr(r.memory, "content_hash", None),
                        "tldr": (getattr(r.memory, "metadata", {}) or {}).get("tldr")
                                or (r.memory.content[:140] if getattr(r.memory, "content", None) else ""),
                        "similarity": float(getattr(r, "relevance", 0.0) or 0.0),
                    }
                    for r in (results or [])
                ],
                "method": "semantic_fallback",
            }
        except Exception as exc:
            logger.debug("analogical_search: semantic fallback failed: %s", exc)
            return {"results": [], "method": "fallback_error", "error": str(exc)}

    # Step 4 — vector arithmetic: result = A - B + C
    a = np.array(vec_A, dtype=float)
    b = np.array(vec_B, dtype=float)
    c = np.array(vec_C, dtype=float)
    result_vector = a - b + c

    # Step 5 — query nearest neighbours to the result vector.
    candidates = await _query_nearest(server, result_vector.tolist(), top_k=top_k)

    # Enrich each candidate with tldr if we can fetch the memory.
    enriched = []
    for cand in candidates:
        mid = cand.get("memory_id")
        tldr_str = None
        if mid:
            mem = await _fetch_memory_by_hash(server, mid)
            if mem is not None:
                tldr_str = (getattr(mem, "metadata", {}) or {}).get("tldr") or (
                    mem.content[:140] if getattr(mem, "content", None) else None
                )
        enriched.append({
            "memory_id": mid,
            "tldr": tldr_str,
            "similarity": float(cand.get("similarity") or 0.0),
        })

    return {"results": enriched, "method": "thought_vector_arithmetic"}


async def handle_distill_soft(server: Any, args: Dict[str, Any]) -> Dict[str, Any]:
    """Dark-knowledge distillation: preserve full soft probability landscape."""
    from .learning.distill import distill_with_soft_distribution
    return await distill_with_soft_distribution(
        server,
        source_ids=args["source_ids"],
        tldr=args["tldr"],
        body=args.get("body", ""),
        tags=args.get("tags", []),
        soft_distribution=args.get("soft_distribution"),
    )


async def handle_revenue_plan(server: Any, args: Dict[str, Any]) -> Dict[str, Any]:
    """Economic MCTS revenue planner.

    Runs Monte Carlo Tree Search over the known revenue-action library and
    returns the highest-expected-value move sequence for the next 30 days.
    """
    from .learning.revenue_planner import plan_revenue_path
    target_monthly = float(args.get("target_monthly") or 5000.0)
    n_simulations = int(args.get("n_simulations") or 50)
    current_assets = args.get("current_assets") or {}
    return await plan_revenue_path(
        server,
        current_assets=current_assets,
        target_monthly=target_monthly,
        n_simulations=n_simulations,
    )


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
    # Hinton thought-vector analogy: v(new) - v(template_problem) + v(template_solution)
    "analogical_search": handle_analogical_search,
    # Hinton dark knowledge: distill with full soft probability distribution
    "distill_soft": handle_distill_soft,
    # Economic MCTS revenue planning
    "revenue_plan": handle_revenue_plan,
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
