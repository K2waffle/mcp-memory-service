"""Codex skill-shim generator.

Codex reads per-project skill files from ``.agents/skills/*.md`` and exposes
each as a user-invocable shortcut. Our custom MCP tools (decision_record,
opportunity_rank, etc.) get a one-to-one shim file so Codex surfaces them
without the operator having to remember tool IDs.

We derive the shim from ``super_brain.tools.TOOL_SCHEMAS`` so the shim
stays in sync with the schema — add a tool → regenerate → shim appears.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = Path("C:/Claude Super Brain/.agents/skills")

_TEMPLATE = """---
name: {name}
description: {description}
mcp_server: super_brain
mcp_tool: {tool_name}
schema_version: 1
---

# {title}

{long_description}

## When to use

{when_to_use}

## Input arguments

{args_doc}

## Example

```json
{example_json}
```

## How it reaches the super-brain

This shim forwards the above JSON payload to the ``super_brain`` MCP server
tool ``{tool_name}``. The server validates the payload, writes a typed
artifact (where applicable), and returns a structured result you can
reference in the next turn.
"""


def _humanize(name: str) -> str:
    return name.replace("_", " ").strip().capitalize()


def _args_doc(input_schema: Dict[str, Any]) -> str:
    """Render an input schema as a bulleted markdown list."""
    props = (input_schema or {}).get("properties", {}) or {}
    required = set((input_schema or {}).get("required", []) or [])
    if not props:
        return "_(no arguments)_"
    lines: List[str] = []
    for key, spec in props.items():
        req = "**required**" if key in required else "optional"
        typ = spec.get("type", "any")
        desc = spec.get("description", "").strip()
        line = f"- `{key}` ({typ}, {req})"
        if desc:
            line += f" — {desc}"
        lines.append(line)
    return "\n".join(lines)


def _example_payload(input_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Build a minimal example payload from schema defaults / types."""
    props = (input_schema or {}).get("properties", {}) or {}
    required = (input_schema or {}).get("required", []) or []
    out: Dict[str, Any] = {}
    for key in required:
        spec = props.get(key, {})
        typ = spec.get("type")
        if "default" in spec:
            out[key] = spec["default"]
        elif spec.get("enum"):
            out[key] = spec["enum"][0]
        elif typ == "string":
            out[key] = f"<{key}>"
        elif typ == "integer":
            out[key] = 0
        elif typ == "number":
            out[key] = 0.0
        elif typ == "boolean":
            out[key] = False
        elif typ == "array":
            out[key] = []
        elif typ == "object":
            out[key] = {}
        else:
            out[key] = None
    return out


def _when_to_use(tool_name: str, description: str) -> str:
    base = description.strip() or f"Invoke the {tool_name} MCP tool."
    return base + (
        "\n\nInvoke this skill whenever the corresponding super-brain artifact "
        "needs to be created, updated, or queried. Prefer this shim over "
        "crafting the raw MCP call manually — the shim enforces the schema."
    )


def render_shim(tool_schema: Dict[str, Any]) -> str:
    """Render a single skill shim from a tool schema dict."""
    name = tool_schema.get("name") or "unnamed_tool"
    description = (tool_schema.get("description") or "").strip()
    input_schema = tool_schema.get("inputSchema") or tool_schema.get("input_schema") or {}
    example = _example_payload(input_schema)
    return _TEMPLATE.format(
        name=f"sb_{name}",
        description=description.replace("\n", " ")[:200],
        title=_humanize(name),
        long_description=description or f"Invokes the {name} super-brain tool.",
        when_to_use=_when_to_use(name, description),
        args_doc=_args_doc(input_schema),
        example_json=json.dumps(example, indent=2),
        tool_name=name,
    )


def generate_all(
    output_dir: Optional[os.PathLike] = None,
    tool_schemas: Optional[Iterable[Dict[str, Any]]] = None,
) -> List[str]:
    """Generate one shim per tool schema. Returns the list of file paths written."""
    out = Path(output_dir) if output_dir else DEFAULT_OUTPUT
    out.mkdir(parents=True, exist_ok=True)

    if tool_schemas is None:
        from ..tools import TOOL_SCHEMAS
        tool_schemas = TOOL_SCHEMAS

    written: List[str] = []
    for schema in tool_schemas:
        name = schema.get("name")
        if not name:
            continue
        path = out / f"sb_{name}.md"
        path.write_text(render_shim(schema), encoding="utf-8")
        written.append(str(path))
    logger.info("skill_shim_generator: wrote %d shims to %s", len(written), out)
    return written


def cli() -> int:
    """Allow ``python -m ...codex.skill_shim_generator`` to regenerate shims."""
    import argparse
    parser = argparse.ArgumentParser(description="Regenerate Codex skill shims.")
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT),
                        help="Output directory (default: %(default)s)")
    args = parser.parse_args()
    paths = generate_all(output_dir=args.out)
    for p in paths:
        print(p)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli())
