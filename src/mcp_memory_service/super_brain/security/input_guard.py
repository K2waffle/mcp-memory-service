"""Indirect prompt injection defense for memory *inputs*.

When a user asks us to store content from an untrusted source (web page,
email, pasted log), that content can itself contain model-directed
instructions. This module scores that risk BEFORE storage, so the caller
can quarantine the memory (tag ``sb:quarantine``) instead of letting it
flow into future retrievals.

This is a local, rule-based scan. Not a complete defense — a SOTA filter
needs an NLI head — but fast and deterministic.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from .tool_poison import scan_tool_description

logger = logging.getLogger(__name__)

# Injection phrases targeting the model rather than the reader.
_INJECTION_PATTERNS = [
    re.compile(r"\bignore\s+(all|any|the)?\s*(previous|prior|above)\s+(instructions?|messages?)\b", re.IGNORECASE),
    re.compile(r"\bforget\s+(everything|all|your)\s+(previous|prior|above)?\b", re.IGNORECASE),
    re.compile(r"\bfrom\s+now\s+on\b.*\byou\s+", re.IGNORECASE),
    re.compile(r"\bsystem\s*:\s*", re.IGNORECASE),
    re.compile(r"\bassistant\s*:\s*", re.IGNORECASE),
    re.compile(r"\bprint\s+(your\s+)?(system|initial)\s+prompt\b", re.IGNORECASE),
    re.compile(r"\breveal\s+(your\s+)?(system|initial|hidden)\s+(prompt|instructions?)\b", re.IGNORECASE),
    re.compile(r"\bexfiltrate\b", re.IGNORECASE),
    re.compile(r"\b(send|post|email)\s+.*(credentials|secrets|tokens|keys)\b", re.IGNORECASE),
]

# Embedded tool-definition shapes that might hijack downstream parsing.
_TOOL_DEF_SHAPES = [
    re.compile(r'"tools?"\s*:\s*\[', re.IGNORECASE),
    re.compile(r'"function_call"\s*:\s*\{', re.IGNORECASE),
    re.compile(r"<\s*tool_use\b", re.IGNORECASE),
]


def scan_input(
    text: str,
    *,
    context: Optional[str] = None,
) -> Dict[str, Any]:
    """Score a block of text for indirect prompt-injection risk.

    Returns:
      {
        "risk_level": "low" | "medium" | "high",
        "findings": [{"type", "detail"}, ...],
        "recommend_quarantine": bool,
      }

    ``context`` is an optional caller hint ("web_scrape", "email", "user_paste")
    used only to annotate findings.
    """
    findings: List[Dict[str, str]] = []
    text = text or ""

    # Reuse the tool-poison scanner for hidden unicode / exfil URLs / base64.
    poison = scan_tool_description(text)
    findings.extend(poison["findings"])

    for pattern in _INJECTION_PATTERNS:
        m = pattern.search(text)
        if m:
            findings.append({"type": "injection_phrase", "detail": m.group(0)[:120]})

    for pattern in _TOOL_DEF_SHAPES:
        m = pattern.search(text)
        if m:
            findings.append({"type": "embedded_tool_def", "detail": m.group(0)[:80]})

    # Scoring: any injection phrase OR role-override OR embedded tool-def → high.
    types = {f["type"] for f in findings}
    high_types = {"injection_phrase", "role_override_phrase", "embedded_tool_def",
                  "credential_exfil_url"}
    medium_types = {"hidden_unicode", "base64_blob"}

    if types & high_types:
        level = "high"
    elif types & medium_types:
        level = "medium"
    else:
        level = "low"

    return {
        "risk_level": level,
        "findings": findings,
        "recommend_quarantine": level == "high",
        "context": context,
    }


async def quarantine_memory(server: Any, content_hash: str) -> bool:
    """Tag a memory with ``sb:quarantine`` so retrieval can exclude it.

    Returns True iff the tag was applied.
    """
    storage = getattr(server, "storage", None)
    if storage is None or not content_hash:
        return False
    try:
        from ..learning.outcome import _exec
        await _exec(
            server,
            "UPDATE memories SET tags_str = tags_str || ',sb:quarantine' "
            "WHERE content_hash = ? AND tags_str NOT LIKE '%sb:quarantine%'",
            [content_hash],
        )
        return True
    except Exception as exc:
        logger.debug("quarantine_memory: tag update skipped: %s", exc)
        return False
