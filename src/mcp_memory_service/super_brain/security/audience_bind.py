"""RFC 8707 audience binding.

Prevents the "confused deputy" attack: an MCP client that has a legitimate
token for server A cannot present that token to server B. The check here
is simple — verify that the ``aud`` claim on the bearer token matches this
server's configured audience — but getting it right at every tool boundary
is how we neutralize CVE-2025-6514-style abuses.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterable, Optional, Tuple

logger = logging.getLogger(__name__)

EXPECTED_AUDIENCE_ENV = "MCP_SUPER_BRAIN_EXPECTED_AUDIENCE"


def _expected_audience() -> Optional[str]:
    return os.environ.get(EXPECTED_AUDIENCE_ENV) or None


def _as_iterable(aud: Any) -> Tuple[str, ...]:
    if aud is None:
        return ()
    if isinstance(aud, str):
        return (aud,)
    if isinstance(aud, (list, tuple, set)):
        return tuple(str(a) for a in aud)
    return (str(aud),)


def verify_audience(
    token_claims: Dict[str, Any],
    expected: Optional[str] = None,
) -> Dict[str, Any]:
    """Verify the ``aud`` claim matches this server's identity.

    Returns:
        dict with ``ok`` (bool), ``reason`` (str), ``token_audiences`` (tuple),
        and ``expected`` (str|None). Fail-closed: missing expected audience
        configuration returns ``ok=False`` with a diagnostic reason.
    """
    expected = expected or _expected_audience()
    if not expected:
        return {
            "ok": False,
            "reason": f"no_expected_audience_configured ({EXPECTED_AUDIENCE_ENV} unset)",
            "token_audiences": (),
            "expected": None,
        }

    auds = _as_iterable((token_claims or {}).get("aud"))
    if not auds:
        return {
            "ok": False,
            "reason": "token_missing_aud_claim",
            "token_audiences": (),
            "expected": expected,
        }

    if expected not in auds:
        return {
            "ok": False,
            "reason": "audience_mismatch",
            "token_audiences": auds,
            "expected": expected,
        }

    return {
        "ok": True,
        "reason": "ok",
        "token_audiences": auds,
        "expected": expected,
    }


def assert_audience(
    token_claims: Dict[str, Any],
    expected: Optional[str] = None,
) -> None:
    """Raise :class:`PermissionError` if audience verification fails."""
    result = verify_audience(token_claims, expected)
    if not result["ok"]:
        raise PermissionError(
            f"audience_bind: {result['reason']} "
            f"(expected={result['expected']!r}, got={result['token_audiences']!r})"
        )
