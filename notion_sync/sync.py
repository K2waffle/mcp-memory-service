"""Two-way Notion mirror for ``decisions``, ``procedures``, ``opportunities``.

Why these three: they're the artifacts humans actually browse / edit — the
rest of the 14 entity types are internal substrate concerns. Notion is a
UI mirror, never the system of record.

Conflict resolution rule (locked in Layer-1 plan): **substrate wins**.
We push substrate → Notion on every run, and pull Notion → substrate only
for pages that have no matching substrate hash. That way, manual edits in
Notion become *new* substrate memories (tagged ``source:notion``) rather
than overwriting existing ones.

Config via env:
  * ``NOTION_API_KEY`` — integration token
  * ``NOTION_DB_DECISIONS`` — database id for decisions
  * ``NOTION_DB_PROCEDURES`` — database id for procedures
  * ``NOTION_DB_OPPORTUNITIES`` — database id for opportunities
  * ``NOTION_SYNC_DRY_RUN=1`` — log-only, don't write

Run manually:
  ``python notion_sync/sync.py --entity decisions``

Or on a schedule via your preferred scheduler (Codex Automations, cron,
APScheduler). We stay dumb and stateless — the substrate holds the truth.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger("super_brain.notion_sync")

SUPPORTED_ENTITIES = ("decisions", "procedures", "opportunities")

ENV_API_KEY = "NOTION_API_KEY"
ENV_DB_IDS = {
    "decisions": "NOTION_DB_DECISIONS",
    "procedures": "NOTION_DB_PROCEDURES",
    "opportunities": "NOTION_DB_OPPORTUNITIES",
}
ENV_DRY_RUN = "NOTION_SYNC_DRY_RUN"

# Where we import the substrate server from. Fall back to a hybrid-only
# storage path if the full server is too heavy to bring up from a sync job.
SERVER_FACTORY_ENV = "SUPER_BRAIN_SERVER_FACTORY"


def _dry_run() -> bool:
    return os.environ.get(ENV_DRY_RUN, "").lower() in ("1", "true", "yes")


def _notion_client():
    """Lazy import so the package isn't a hard dep on the server."""
    try:
        from notion_client import Client  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "notion_sync requires `notion-client` (pip install notion-client)"
        ) from exc
    api_key = os.environ.get(ENV_API_KEY)
    if not api_key:
        raise RuntimeError(f"missing {ENV_API_KEY}")
    return Client(auth=api_key)


@dataclass
class SyncResult:
    entity: str
    pushed: int = 0
    pulled: int = 0
    skipped: int = 0
    errors: int = 0


# --- Field mapping -----------------------------------------------------------

def _rich_text(value: Optional[str]) -> List[Dict[str, Any]]:
    if not value:
        return []
    # Notion caps each rich_text item at 2000 chars.
    chunks = [value[i:i + 2000] for i in range(0, len(value), 2000)]
    return [{"type": "text", "text": {"content": c}} for c in chunks]


def _title(value: str) -> List[Dict[str, Any]]:
    return _rich_text(value[:200] or "(untitled)")


def _memory_to_notion_props(entity: str, memory: Any) -> Dict[str, Any]:
    md = getattr(memory, "metadata", {}) or {}
    title = md.get("tldr") or (getattr(memory, "content", "") or "")[:120] or "(untitled)"
    content_hash = getattr(memory, "content_hash", "")

    base: Dict[str, Any] = {
        "Title": {"title": _title(title)},
        "Hash": {"rich_text": _rich_text(content_hash)},
        "TLDR": {"rich_text": _rich_text(md.get("tldr") or "")},
    }

    if entity == "decisions":
        base.update({
            "Rationale": {"rich_text": _rich_text(md.get("rationale") or "")},
            "Supersedes": {"rich_text": _rich_text(md.get("supersedes_id") or "")},
            "Valid From": (
                {"date": {"start": md["valid_from"]}}
                if md.get("valid_from") else {"date": None}
            ),
            "Valid To": (
                {"date": {"start": md["valid_to"]}}
                if md.get("valid_to") else {"date": None}
            ),
            "Source": (
                {"url": md["source"]} if md.get("source") else {"url": None}
            ),
        })
    elif entity == "procedures":
        base.update({
            "Steps": {"rich_text": _rich_text(md.get("body") or getattr(memory, "content", "") or "")},
            "Alpha": {"number": float(md.get("alpha") or 1.0)},
            "Beta": {"number": float(md.get("beta") or 1.0)},
            "N Runs": {"number": int(md.get("n_runs") or 0)},
        })
    elif entity == "opportunities":
        score = float(md.get("score") or 0.0)
        base.update({
            "Speed to First Dollar": (
                {"select": {"name": md["speed_to_first_dollar"]}}
                if md.get("speed_to_first_dollar") else {"select": None}
            ),
            "Automation Leverage": (
                {"select": {"name": md["automation_leverage"]}}
                if md.get("automation_leverage") else {"select": None}
            ),
            "Score": {"number": score},
        })
    return base


def _notion_page_to_memory_fields(entity: str, page: Dict[str, Any]) -> Dict[str, Any]:
    props = page.get("properties", {}) or {}

    def _txt(field: str) -> str:
        block = props.get(field) or {}
        rt = block.get("rich_text") or block.get("title") or []
        return "".join(r.get("plain_text", "") for r in rt)

    def _num(field: str) -> Optional[float]:
        block = props.get(field) or {}
        return block.get("number")

    def _sel(field: str) -> Optional[str]:
        block = props.get(field) or {}
        sel = block.get("select") or {}
        return sel.get("name") if sel else None

    def _date(field: str) -> Optional[str]:
        block = props.get(field) or {}
        date = block.get("date") or {}
        return date.get("start") if date else None

    def _url(field: str) -> Optional[str]:
        block = props.get(field) or {}
        return block.get("url")

    title = _txt("Title") or "(untitled)"
    tldr = _txt("TLDR") or title
    base: Dict[str, Any] = {
        "title": title,
        "tldr": tldr,
        "entity": entity,
        "notion_page_id": page.get("id"),
    }
    if entity == "decisions":
        base.update({
            "rationale": _txt("Rationale"),
            "supersedes_id": _txt("Supersedes") or None,
            "valid_from": _date("Valid From"),
            "valid_to": _date("Valid To"),
            "source": _url("Source"),
        })
    elif entity == "procedures":
        base.update({
            "body": _txt("Steps"),
            "alpha": _num("Alpha"),
            "beta": _num("Beta"),
            "n_runs": int(_num("N Runs") or 0),
        })
    elif entity == "opportunities":
        base.update({
            "speed_to_first_dollar": _sel("Speed to First Dollar"),
            "automation_leverage": _sel("Automation Leverage"),
            "score": _num("Score"),
        })
    return base


# --- Push: substrate → Notion ------------------------------------------------

async def _query_substrate_memories(server: Any, entity: str) -> List[Any]:
    """Retrieve memories tagged ``entity:<entity>`` from the substrate."""
    storage = getattr(server, "storage", None)
    if storage is None:
        return []
    try:
        res = await storage.retrieve(entity, n_results=500,
                                     tags=[f"entity:{entity}"],
                                     min_confidence=0.0)
    except Exception as exc:
        logger.warning("substrate query failed for %s: %s", entity, exc)
        return []
    return [r.memory for r in res if getattr(r, "memory", None) is not None]


def _find_notion_page_by_hash(
    notion: Any, db_id: str, content_hash: str
) -> Optional[Dict[str, Any]]:
    try:
        resp = notion.databases.query(
            database_id=db_id,
            filter={"property": "Hash", "rich_text": {"equals": content_hash}},
            page_size=1,
        )
    except Exception as exc:
        logger.debug("notion query by hash failed: %s", exc)
        return None
    results = resp.get("results") or []
    return results[0] if results else None


async def push_substrate_to_notion(server: Any, entity: str) -> SyncResult:
    result = SyncResult(entity=entity)
    if entity not in SUPPORTED_ENTITIES:
        raise ValueError(f"unsupported entity {entity!r}")
    db_id = os.environ.get(ENV_DB_IDS[entity])
    if not db_id:
        raise RuntimeError(f"missing {ENV_DB_IDS[entity]}")

    memories = await _query_substrate_memories(server, entity)
    if not memories:
        logger.info("push %s: no substrate memories", entity)
        return result

    if _dry_run():
        logger.info("[dry-run] would push %d %s to Notion", len(memories), entity)
        return result

    notion = _notion_client()
    for mem in memories:
        content_hash = getattr(mem, "content_hash", "")
        if not content_hash:
            result.skipped += 1
            continue
        props = _memory_to_notion_props(entity, mem)
        try:
            existing = _find_notion_page_by_hash(notion, db_id, content_hash)
            if existing:
                notion.pages.update(page_id=existing["id"], properties=props)
            else:
                notion.pages.create(parent={"database_id": db_id}, properties=props)
            result.pushed += 1
        except Exception as exc:
            logger.warning("push %s hash=%s failed: %s", entity, content_hash, exc)
            result.errors += 1
    return result


# --- Pull: Notion → substrate (for pages with no matching hash) --------------

async def pull_notion_to_substrate(server: Any, entity: str) -> SyncResult:
    result = SyncResult(entity=entity)
    if entity not in SUPPORTED_ENTITIES:
        raise ValueError(f"unsupported entity {entity!r}")
    db_id = os.environ.get(ENV_DB_IDS[entity])
    if not db_id:
        raise RuntimeError(f"missing {ENV_DB_IDS[entity]}")

    if _dry_run():
        logger.info("[dry-run] would pull Notion %s → substrate", entity)
        return result

    notion = _notion_client()
    cursor: Optional[str] = None
    while True:
        query_args: Dict[str, Any] = {"database_id": db_id, "page_size": 100}
        if cursor:
            query_args["start_cursor"] = cursor
        try:
            resp = notion.databases.query(**query_args)
        except Exception as exc:
            logger.warning("notion pull %s failed: %s", entity, exc)
            result.errors += 1
            break

        for page in resp.get("results", []):
            fields = _notion_page_to_memory_fields(entity, page)
            # If the Hash column already has a value, assume substrate owns it.
            props = page.get("properties", {}) or {}
            existing_hash_block = props.get("Hash") or {}
            existing_rt = existing_hash_block.get("rich_text") or []
            existing_hash = "".join(r.get("plain_text", "") for r in existing_rt)
            if existing_hash:
                result.skipped += 1
                continue

            ok = await _write_memory_from_notion(server, entity, fields)
            if ok:
                result.pulled += 1
            else:
                result.errors += 1

        if not resp.get("has_more"):
            break
        cursor = resp.get("next_cursor")
    return result


async def _write_memory_from_notion(
    server: Any, entity: str, fields: Dict[str, Any]
) -> bool:
    """Materialize a Notion page as a new substrate memory."""
    from mcp_memory_service.models.memory import Memory  # type: ignore
    content_parts = [fields.get("tldr") or fields.get("title") or ""]
    if entity == "decisions":
        content_parts.append(f"\nRationale: {fields.get('rationale') or ''}")
    elif entity == "procedures":
        content_parts.append(f"\n{fields.get('body') or ''}")
    content = "\n".join(p for p in content_parts if p).strip() or "(empty)"
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    metadata: Dict[str, Any] = {
        "tldr": fields.get("tldr") or fields.get("title"),
        "source_type": "notion",
        "notion_page_id": fields.get("notion_page_id"),
        "created_at_iso": now_iso,
        "schema_version": "1",
    }
    # Pass-through entity-specific fields.
    for key in (
        "rationale", "supersedes_id", "valid_from", "valid_to", "source",
        "body", "alpha", "beta", "n_runs",
        "speed_to_first_dollar", "automation_leverage", "score",
    ):
        if fields.get(key) is not None:
            metadata[key] = fields[key]

    mem = Memory(
        content=content,
        content_hash=content_hash,
        tags=[f"entity:{entity}", "source:notion"],
        memory_type="semantic" if entity != "procedures" else "procedural",
        metadata=metadata,
    )
    try:
        ok, _ = await server.storage.store(mem)
        return bool(ok)
    except Exception as exc:
        logger.warning("pull %s: store failed: %s", entity, exc)
        return False


# --- CLI entry point ---------------------------------------------------------

async def run_once(
    server_factory: Callable[[], Awaitable[Any]],
    entities: Iterable[str] = SUPPORTED_ENTITIES,
) -> List[SyncResult]:
    server = await server_factory()
    out: List[SyncResult] = []
    for entity in entities:
        try:
            push = await push_substrate_to_notion(server, entity)
            pull = await pull_notion_to_substrate(server, entity)
            # Merge counts into one row per entity for the CLI report.
            push.pulled = pull.pulled
            push.skipped += pull.skipped
            push.errors += pull.errors
            out.append(push)
        except Exception as exc:
            logger.error("run_once(%s): %s", entity, exc)
            out.append(SyncResult(entity=entity, errors=1))
    return out


async def _default_server_factory() -> Any:
    """Build a minimal server with just storage attached, using env config."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server" / "src"))
    from mcp_memory_service.storage.factory import create_storage_instance  # type: ignore

    class _Shell:
        pass
    shell = _Shell()
    shell.storage = await create_storage_instance()  # type: ignore[attr-defined]
    return shell


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sync super-brain artifacts to Notion.")
    p.add_argument("--entity", choices=list(SUPPORTED_ENTITIES) + ["all"],
                   default="all")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    entities = SUPPORTED_ENTITIES if args.entity == "all" else (args.entity,)
    results = asyncio.run(run_once(_default_server_factory, entities))
    for r in results:
        print(f"{r.entity:13s}  pushed={r.pushed:3d}  pulled={r.pulled:3d}  "
              f"skipped={r.skipped:3d}  errors={r.errors:3d}")
    return 0 if all(r.errors == 0 for r in results) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
