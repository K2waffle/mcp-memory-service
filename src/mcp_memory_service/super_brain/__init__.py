"""Super-Brain extension package.

All additions to the upstream mcp-memory-service live here. The goal is a
single namespace for our fork so upstream pulls rarely collide.

Wire-in points:
- ``super_brain.register(server)`` is called once from ``server_impl.py`` at
  startup. It installs MCP tool handlers, post-write hooks (markdown mirror),
  ontology extensions (4 memory tiers + 14 entities), guardrails, learning
  hooks, and the skill-shim emitter.
- Everything is feature-flag gated on ``MCP_SUPER_BRAIN_ENABLED`` (default on
  when the env var is present, off otherwise). Upstream continues to work
  unchanged when the flag is unset.

Import discipline: lazy. Submodules may depend on optional third-party libs
(stripe, rapidfuzz, etc.) — import them only when the feature is actually
invoked, so a missing optional dep never breaks server startup.
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

ENABLED_ENV = "MCP_SUPER_BRAIN_ENABLED"
SCHEMA_VERSION = "1"


def is_enabled() -> bool:
    """True when the super-brain extension layer should be active."""
    return os.environ.get(ENABLED_ENV, "").lower() in {"1", "true", "yes", "on"}


def register(server: Any) -> None:
    """Entry point called from server_impl at init.

    Gracefully no-ops when disabled, so upstream behavior is unchanged.
    """
    if not is_enabled():
        logger.info("super_brain: disabled (%s not set)", ENABLED_ENV)
        return

    logger.info("super_brain: enabling extension layer")

    # Ontology extension — teaches the Memory model about our 4 tiers and 14
    # entity types without modifying upstream files.
    try:
        from . import ontology_extensions  # noqa: F401 — side-effect import
        logger.info("super_brain: ontology extensions registered")
    except Exception as exc:  # pragma: no cover — defensive
        logger.exception("super_brain: ontology extension failed: %s", exc)

    # Markdown mirror — post-write hook that serializes artifacts to disk.
    try:
        from ..sync.markdown_mirror import install_hook
        install_hook(server)
        logger.info("super_brain: markdown mirror hook installed")
    except Exception as exc:  # pragma: no cover
        logger.exception("super_brain: markdown mirror hook failed: %s", exc)

    # Novelty gate — Hinton prediction-error principle.
    # NOTE: novelty_gate.gate_write() is a PRE-WRITE HOOK. Call it before
    # storage.store() on any user-initiated write path to block redundant
    # chunks. Example pattern in a write handler:
    #
    #     from .guardrails.novelty_gate import gate_write
    #     gate = await gate_write(server, content, tags)
    #     if not gate["allow"]:
    #         return {"status": "skipped", **gate}
    #     await storage.store(memory)
    #
    # The module is imported here to surface import errors at startup rather
    # than silently at first write.
    try:
        from .guardrails import novelty_gate as _novelty_gate  # noqa: F401 — validates import
        logger.info("super_brain: novelty gate loaded (pre-write hook available)")
    except Exception as exc:
        logger.exception("super_brain: novelty gate import failed: %s", exc)

    # Custom MCP tools (decision_record, opportunity_rank, research_source_with_trust,
    # analogical_search, distill_soft, ...)
    try:
        from . import tools as sb_tools
        sb_tools.register_tools(server)
        logger.info("super_brain: custom MCP tools registered")
    except Exception as exc:
        logger.exception("super_brain: tool registration failed: %s", exc)
