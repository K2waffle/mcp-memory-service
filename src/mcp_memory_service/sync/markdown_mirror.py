"""Markdown mirror — write-only post-store hook.

Every successful memory write also lands as a YAML-front-matter markdown file
at ``artifacts/{entity_type}/{slug}.md`` under the repo root. Substrate stays
canonical for retrieval; markdown is the always-readable portability mirror.

Design notes:
- **Non-blocking path**: mirror failure never prevents the primary store from
  succeeding. Errors are logged + counted, not raised.
- **Fire-and-forget async**: runs as a background task via ``asyncio.create_task``
  so store-path latency is unaffected.
- **Content-addressed slug**: first 16 chars of content_hash + slug-ified first
  line. Stable across re-writes; human-readable in the filesystem.
- **Idempotent**: writing the same artifact twice overwrites; no duplicates.
- **Root override**: ``MCP_SUPER_BRAIN_ARTIFACT_ROOT`` env var; defaults to
  ``../../../../artifacts`` relative to this file (i.e. the repo root's
  ``artifacts/`` dir).

Security: writes restricted to the configured root. Paths are normalized and
validated against directory traversal (an entity_type like ``../../etc``
would be rejected by the allow-list below).
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

logger = logging.getLogger(__name__)

# Counter for observability; read by health endpoints.
_stats = {"writes": 0, "failures": 0, "skips": 0}

# Entity types we know about — anything else is bucketed under 'misc/'. This
# also acts as a directory-traversal allow-list.
KNOWN_ENTITIES = frozenset(
    {
        "projects", "sites", "integrations", "tasks", "task_packets",
        "memory_artifacts", "memory_tags", "memory_links", "decisions",
        "procedures", "incidents", "research_sources", "opportunities",
        "object_artifacts", "facts", "experiments",
        "site_profiles", "integration_profiles",
        # 4 tiers may appear as memory_type as well
        "working", "episodic", "semantic", "procedural",
    }
)

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def get_stats() -> Dict[str, int]:
    """Return mirror stats (copy; safe to expose via health endpoint)."""
    return dict(_stats)


def _artifact_root() -> Path:
    env_root = os.environ.get("MCP_SUPER_BRAIN_ARTIFACT_ROOT")
    if env_root:
        return Path(env_root).resolve()
    # Default: repo-root/artifacts (four levels up from this file).
    here = Path(__file__).resolve()
    return (here.parent.parent.parent.parent.parent / "artifacts").resolve()


def _slug(text: str, max_len: int = 64) -> str:
    lowered = (text or "").strip().lower()
    slugged = _SLUG_RE.sub("-", lowered).strip("-")
    return (slugged[:max_len] or "artifact").rstrip("-")


def _entity_dir(memory: Any) -> str:
    """Decide the top-level directory for an artifact."""
    # Entity tag wins (e.g. 'entity:decisions'); otherwise fall back to
    # memory_type; otherwise 'memory_artifacts'.
    tags: Iterable[str] = getattr(memory, "tags", []) or []
    for t in tags:
        if t.startswith("entity:"):
            name = t.split(":", 1)[1]
            if name in KNOWN_ENTITIES:
                return name
    mem_type = getattr(memory, "memory_type", None)
    if mem_type and mem_type in KNOWN_ENTITIES:
        return mem_type
    return "memory_artifacts"


def _format_front_matter(memory: Any) -> str:
    """Render memory dataclass → YAML front-matter. Lightweight, no PyYAML dep."""
    lines = ["---"]

    def _emit(key: str, value: Any) -> None:
        if value is None:
            return
        if isinstance(value, bool):
            lines.append(f"{key}: {'true' if value else 'false'}")
        elif isinstance(value, (int, float)):
            lines.append(f"{key}: {value}")
        elif isinstance(value, (list, tuple)):
            items = [str(v).replace('"', '\\"') for v in value if v is not None]
            if items:
                lines.append(f"{key}: [{', '.join(chr(34) + i + chr(34) for i in items)}]")
        elif isinstance(value, dict):
            # One-line JSON-like; avoids nested YAML complexity.
            lines.append(f"{key}: {value}")
        else:
            s = str(value).replace("\n", " ").replace('"', '\\"')
            lines.append(f'{key}: "{s}"')

    _emit("id", getattr(memory, "content_hash", None))
    _emit("memory_type", getattr(memory, "memory_type", None))
    _emit("tags", list(getattr(memory, "tags", []) or []))
    _emit("created_at", getattr(memory, "created_at_iso", None))
    _emit("updated_at", getattr(memory, "updated_at_iso", None))

    # Selected metadata fields — super-brain + SHODH-aligned.
    md: Dict[str, Any] = getattr(memory, "metadata", {}) or {}
    for key in (
        "source", "source_type", "credibility", "tldr",
        "confidence", "groundedness", "abstain",
        "atomic_fact_coverage", "citation_f1",
        "lawful_basis", "contains_special_category", "contains_spi_cpra",
        "license", "attribution_required", "attribution_text",
        "tdm_opt_out_observed", "is_ai_generated", "ai_content_marking",
        "alpha", "beta", "llm_summary", "uncertainty",
        "rubric_score", "tokens_in", "tokens_out", "wall_ms",
        "tool_calls", "retries", "verifier_id", "outcome",
        "jurisdiction_of_subject", "jurisdiction_of_capture",
        "erasure_reachable", "per_subject_key_id",
        "retention_policy_id", "delete_after",
        "counterparty_ref", "counterparty_rail", "pci_scope",
        "revenue_timestamp", "revenue_attributed", "margin",
        "embedding_model_id", "embedding_model_version", "schema_version",
        "skill_shim_path",
        # Opportunity-entity revenue ranking
        "speed_to_first_dollar", "earning_ceiling", "automation_leverage",
        "repeatability", "distribution", "retention", "compliance_risk",
        "monetization_route",
    ):
        if key in md:
            _emit(key, md[key])

    lines.append("---")
    return "\n".join(lines)


def _render_markdown(memory: Any) -> str:
    front = _format_front_matter(memory)
    body = getattr(memory, "content", "") or ""
    # Pull a tldr into body header if present; keeps file skimmable.
    md: Dict[str, Any] = getattr(memory, "metadata", {}) or {}
    tldr = md.get("tldr")
    header = f"# {tldr}\n\n" if tldr else ""
    return f"{front}\n\n{header}{body}\n"


def _resolve_path(memory: Any) -> Optional[Path]:
    root = _artifact_root()
    entity = _entity_dir(memory)
    if entity not in KNOWN_ENTITIES and entity != "memory_artifacts":
        entity = "memory_artifacts"
    content_hash = getattr(memory, "content_hash", None) or ""
    if not content_hash:
        return None
    short = content_hash[:16]
    first_line = (getattr(memory, "content", "") or "").splitlines()[0] if getattr(memory, "content", None) else ""
    slug = _slug(first_line)
    filename = f"{short}-{slug}.md"
    target = (root / entity / filename).resolve()
    # Directory traversal guard: the resolved path must be under root.
    try:
        target.relative_to(root)
    except ValueError:
        logger.error("markdown_mirror: path escape rejected for %s", target)
        return None
    return target


async def _write_async(memory: Any) -> None:
    try:
        path = _resolve_path(memory)
        if path is None:
            _stats["skips"] += 1
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        content = _render_markdown(memory)
        # Atomic write via tmp + rename; safe under concurrent writers.
        tmp = path.with_suffix(path.suffix + ".tmp")
        await asyncio.to_thread(tmp.write_text, content, encoding="utf-8")
        await asyncio.to_thread(os.replace, str(tmp), str(path))
        _stats["writes"] += 1
    except Exception as exc:
        _stats["failures"] += 1
        logger.warning("markdown_mirror: write failed: %s", exc)


def mirror_write(memory: Any) -> None:
    """Fire-and-forget mirror write. Safe to call from sync or async contexts."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No loop — run synchronously in a thread-local loop.
        asyncio.run(_write_async(memory))
        return
    loop.create_task(_write_async(memory))


def install_hook(server: Any) -> None:
    """Attach the mirror to the given server instance.

    Tries several integration points depending on what's available upstream:
    - ``server.storage.add_post_store_hook(fn)`` if supported
    - Otherwise monkey-patches ``storage.store`` to invoke the mirror after
      successful writes.
    """
    storage = getattr(server, "storage", None)
    if storage is None:
        logger.warning("markdown_mirror: server has no .storage — hook not installed")
        return

    if hasattr(storage, "add_post_store_hook"):
        storage.add_post_store_hook(mirror_write)
        logger.info("markdown_mirror: installed via add_post_store_hook")
        return

    # Fallback: wrap the store() method.
    original_store = getattr(storage, "store", None)
    if original_store is None:
        logger.warning("markdown_mirror: storage has no store() method")
        return

    async def wrapped_store(memory, *args, **kwargs):
        result = await original_store(memory, *args, **kwargs)
        try:
            success = bool(result[0]) if isinstance(result, tuple) else True
            if success:
                mirror_write(memory)
        except Exception as exc:  # pragma: no cover
            logger.debug("markdown_mirror: post-store hook exception: %s", exc)
        return result

    storage.store = wrapped_store  # type: ignore[assignment]
    logger.info("markdown_mirror: installed via store() wrapper")
