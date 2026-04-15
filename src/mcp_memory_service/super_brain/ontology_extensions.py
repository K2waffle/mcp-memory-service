"""Extend the upstream memory-type ontology with our 4 tiers + 14 entities.

Side-effect: importing this module registers the additions on
``MemoryTypeOntology`` and ``TagTaxonomy``. Safe to import multiple times —
registration is idempotent.

Rationale: the super-brain models memory as a 4-tier hierarchy
(working / episodic / semantic / procedural) with 14 artifact entities.
Upstream's ontology defaults to ``observation`` and rejects unknown types.
We widen it additively so our writes pass validation without edits to
upstream files.
"""
from __future__ import annotations

import logging
from typing import Iterable

logger = logging.getLogger(__name__)

# 4-tier memory hierarchy (see MEMORY_SCHEMA.md)
TIERS = ("working", "episodic", "semantic", "procedural")

# 14 artifact entity types. These become either memory_type values directly
# (for tag-based entities) or entity-specific metadata on a typed memory.
ENTITIES = (
    "projects",
    "sites",
    "integrations",
    "tasks",
    "task_packets",
    "memory_artifacts",
    "memory_tags",
    "memory_links",
    "decisions",
    "procedures",
    "incidents",
    "research_sources",
    "opportunities",
    "object_artifacts",
)

# Tag namespaces we introduce on top of upstream's (sys:, q:, proj:, topic:, t:, user:).
SUPER_BRAIN_NAMESPACES = (
    "sb:",            # generic super-brain-scoped tag
    "entity:",        # entity:{name}, e.g. entity:decisions
    "tier:",          # tier:{working|episodic|semantic|procedural}
    "opportunity:",   # opportunity:{slug}
    "source:",        # source:{domain} for research_sources
    "site:",          # site:{domain} for sites entity
    "integration:",   # integration:{vendor}
    "incident:",      # incident:{slug}
    "revenue:",       # revenue:{route}
)


def _register_types(ontology_cls, names: Iterable[str]) -> int:
    """Best-effort registration; returns count actually added."""
    added = 0
    for name in names:
        try:
            if hasattr(ontology_cls, "register_type"):
                ontology_cls.register_type(name)
                added += 1
            elif hasattr(ontology_cls, "VALID_TYPES"):
                if name not in ontology_cls.VALID_TYPES:
                    ontology_cls.VALID_TYPES.add(name)
                    added += 1
            elif hasattr(ontology_cls, "_valid_types"):
                ontology_cls._valid_types.add(name)
                added += 1
        except Exception as exc:
            logger.debug("ontology register failed for %s: %s", name, exc)
    return added


def _register_namespaces(taxonomy_cls, namespaces: Iterable[str]) -> int:
    added = 0
    for ns in namespaces:
        try:
            if hasattr(taxonomy_cls, "register_namespace"):
                taxonomy_cls.register_namespace(ns)
                added += 1
            elif hasattr(taxonomy_cls, "VALID_NAMESPACES"):
                if ns not in taxonomy_cls.VALID_NAMESPACES:
                    taxonomy_cls.VALID_NAMESPACES.add(ns)
                    added += 1
        except Exception as exc:
            logger.debug("taxonomy register failed for %s: %s", ns, exc)
    return added


try:
    from ..models.ontology import MemoryTypeOntology
    added = _register_types(MemoryTypeOntology, TIERS) + _register_types(
        MemoryTypeOntology, ENTITIES
    )
    logger.info("super_brain ontology: +%d memory types registered", added)
except Exception as exc:  # pragma: no cover
    logger.warning("super_brain ontology: could not extend MemoryTypeOntology: %s", exc)


try:
    from ..models.tag_taxonomy import TagTaxonomy
    added = _register_namespaces(TagTaxonomy, SUPER_BRAIN_NAMESPACES)
    logger.info("super_brain ontology: +%d tag namespaces registered", added)
except Exception as exc:  # pragma: no cover
    logger.warning("super_brain ontology: could not extend TagTaxonomy: %s", exc)
