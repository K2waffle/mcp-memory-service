"""Tool description poisoning scanner (ATPA-style defenses).

Upstream research calls this "Tool Poisoning" or "Agent-Targeted Prompt
Attacks": a malicious MCP server publishes a tool whose *description*
contains hidden instructions aimed at the model, not the human — zero-width
unicode, base-64 blobs, role-override imperatives, exfiltration URLs.

This scanner runs over the descriptions we publish AND the descriptions
we receive from downstream MCP servers (see Codex integration).
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

# Unicode categories that should never appear in a legitimate tool description.
# Zero-width joiners, zero-width non-joiners, bidi overrides, format controls.
_SUSPICIOUS_CHARS = {
    "\u200b",  # zero-width space
    "\u200c",  # zero-width non-joiner
    "\u200d",  # zero-width joiner
    "\u2060",  # word joiner
    "\ufeff",  # BOM / zero-width no-break space
    "\u202a", "\u202b", "\u202c", "\u202d", "\u202e",  # bidi overrides
    "\u2066", "\u2067", "\u2068", "\u2069",  # bidi isolates
}

# Direct role-override phrases we've seen in the wild.
_ROLE_OVERRIDE = [
    re.compile(r"\bignore\s+(the\s+)?(previous|prior|above)\b", re.IGNORECASE),
    re.compile(r"\bdisregard\s+(the\s+)?(previous|prior|above)\b", re.IGNORECASE),
    re.compile(r"\byou\s+are\s+now\s+", re.IGNORECASE),
    re.compile(r"\bnew\s+system\s+(message|prompt|instruction)", re.IGNORECASE),
    re.compile(r"</?system>", re.IGNORECASE),
    re.compile(r"<\s*\|?\s*(assistant|user|system)\s*\|?\s*>", re.IGNORECASE),
]

# Exfiltration markers: markdown images / links pointing at suspicious hosts,
# base64 blobs larger than 128 chars.
_MARKDOWN_IMG = re.compile(r"!\[[^\]]*\]\((?P<url>[^)]+)\)")
_URL_CREDENTIALS = re.compile(r"\b(token|key|secret|password|bearer)=", re.IGNORECASE)
_BASE64_BLOB = re.compile(r"[A-Za-z0-9+/]{128,}={0,2}")


def _find_suspicious_chars(text: str) -> List[str]:
    found = [c for c in text if c in _SUSPICIOUS_CHARS]
    # Dedupe while preserving encounter order so the caller can see what ran.
    seen = set()
    out: List[str] = []
    for c in found:
        if c not in seen:
            seen.add(c)
            out.append(f"U+{ord(c):04X}")
    return out


def scan_tool_description(description: str) -> Dict[str, Any]:
    """Scan a single tool description; return a structured finding report.

    Keys:
      * clean: bool — True iff no findings triggered.
      * findings: list of {type, detail}
      * severity: "clean" | "warn" | "block"
    """
    findings: List[Dict[str, str]] = []
    text = description or ""

    susp_chars = _find_suspicious_chars(text)
    if susp_chars:
        findings.append({
            "type": "hidden_unicode",
            "detail": ",".join(susp_chars[:10]),
        })

    for pattern in _ROLE_OVERRIDE:
        m = pattern.search(text)
        if m:
            findings.append({
                "type": "role_override_phrase",
                "detail": m.group(0)[:80],
            })

    for m in _MARKDOWN_IMG.finditer(text):
        url = m.group("url") or ""
        if _URL_CREDENTIALS.search(url):
            findings.append({
                "type": "credential_exfil_url",
                "detail": url[:120],
            })

    if _BASE64_BLOB.search(text):
        findings.append({
            "type": "base64_blob",
            "detail": "base64_len>=128",
        })

    # Severity: any role-override or exfil URL is block-worthy; hidden unicode
    # or base64 is warn-worthy.
    severity = "clean"
    block_types = {"role_override_phrase", "credential_exfil_url"}
    warn_types = {"hidden_unicode", "base64_blob"}
    types = {f["type"] for f in findings}
    if types & block_types:
        severity = "block"
    elif types & warn_types:
        severity = "warn"

    return {
        "clean": not findings,
        "findings": findings,
        "severity": severity,
    }


def scan_tool_catalog(tools: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Scan a full tool catalog (list of {name, description, ...}).

    Returns:
      {
        "clean": bool,
        "by_tool": {name: scan_tool_description-result, ...},
        "blocked": [name, ...],
      }
    """
    by_tool: Dict[str, Dict[str, Any]] = {}
    blocked: List[str] = []
    any_dirty = False
    for t in tools:
        name = t.get("name", "<unnamed>")
        desc = t.get("description", "") or t.get("summary", "")
        result = scan_tool_description(desc)
        by_tool[name] = result
        if not result["clean"]:
            any_dirty = True
        if result["severity"] == "block":
            blocked.append(name)
    return {
        "clean": not any_dirty,
        "by_tool": by_tool,
        "blocked": blocked,
    }
