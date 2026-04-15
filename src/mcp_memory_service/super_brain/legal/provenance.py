"""Provenance schema validation.

Layer-1 legal research locked these as the minimum metadata every memory
must carry once it's tagged into a regulated bucket: GDPR Art. 6 lawful
basis, CPRA SPI flag, licensing (for ingestion), C2PA / ai_content_marking
for generated artifacts, attribution ledger.

This module does the validation at write-time and annotates any gaps.
It does NOT decide retention — see :mod:`retention` for that.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Allowed values by field. These mirror MEMORY_SCHEMA.md → "Optional fields".
LAWFUL_BASIS_VALUES = {
    "consent", "contract", "legal_obligation", "vital_interests",
    "public_task", "legitimate_interests",
}
SOURCE_TYPE_VALUES = {
    "primary_doc", "paper", "vendor", "guru", "user_input", "ai_assisted",
    "benchmark", "scrape", "email", "chat", "code_repo",
}
LICENSE_VALUES = {
    "public_domain", "cc0", "cc_by", "cc_by_sa", "cc_by_nc", "cc_by_nd",
    "mit", "apache_2_0", "gpl_3_0", "proprietary", "unknown",
    # Explicit signal that the publisher opted out of TDM (DSM Art. 4(3)).
    "tdm_opt_out",
}


def _as_bool(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ("1", "true", "yes")
    return None


def validate_provenance(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a memory's metadata provenance block.

    Returns:
      {
        "ok": bool,
        "errors": [str, ...],        # schema violations (block writes in strict mode)
        "warnings": [str, ...],      # soft issues (surface to user, don't block)
        "normalized": {...}          # cleaned fields the caller may use
      }
    """
    md = metadata or {}
    errors: List[str] = []
    warnings: List[str] = []
    normalized: Dict[str, Any] = {}

    lb = md.get("lawful_basis")
    if lb is not None:
        if lb not in LAWFUL_BASIS_VALUES:
            errors.append(f"lawful_basis={lb!r} not in {sorted(LAWFUL_BASIS_VALUES)}")
        else:
            normalized["lawful_basis"] = lb

    st = md.get("source_type")
    if st is not None:
        if st not in SOURCE_TYPE_VALUES:
            warnings.append(f"source_type={st!r} not in known set; storing as-is")
        normalized["source_type"] = st

    lic = md.get("license")
    if lic is not None:
        if lic not in LICENSE_VALUES:
            warnings.append(f"license={lic!r} not in known set; storing as-is")
        normalized["license"] = lic

    # Attribution requirement — if required, attribution_text must be non-empty.
    attr_req = _as_bool(md.get("attribution_required"))
    if attr_req:
        if not md.get("attribution_text"):
            errors.append("attribution_required=true but attribution_text is empty")
        else:
            normalized["attribution_text"] = md["attribution_text"]
    if attr_req is not None:
        normalized["attribution_required"] = bool(attr_req)

    # TDM opt-out observed: if license == 'tdm_opt_out', the flag must agree.
    tdm = _as_bool(md.get("tdm_opt_out_observed"))
    if lic == "tdm_opt_out" and tdm is False:
        errors.append("license=tdm_opt_out but tdm_opt_out_observed is false")
    if tdm is not None:
        normalized["tdm_opt_out_observed"] = bool(tdm)

    # SPI / special-category data require explicit acknowledgement.
    spi = _as_bool(md.get("contains_spi_cpra"))
    sc = _as_bool(md.get("contains_special_category"))
    if (spi or sc) and not md.get("lawful_basis"):
        errors.append(
            "contains_spi_cpra or contains_special_category is true, "
            "but lawful_basis is not set"
        )
    if spi is not None:
        normalized["contains_spi_cpra"] = bool(spi)
    if sc is not None:
        normalized["contains_special_category"] = bool(sc)

    # AI-generated content marking (EU AI Act Art. 50).
    is_ai = _as_bool(md.get("is_ai_generated"))
    if is_ai and not md.get("ai_content_marking"):
        errors.append(
            "is_ai_generated=true but ai_content_marking not set "
            "(e.g. 'c2pa:manifest-ref:<id>' or 'watermark:<scheme>')"
        )
    if is_ai is not None:
        normalized["is_ai_generated"] = bool(is_ai)
    if md.get("ai_content_marking"):
        normalized["ai_content_marking"] = md["ai_content_marking"]

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "normalized": normalized,
    }


def assert_valid(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Strict version: raises :class:`ValueError` if provenance fails.

    Returns the normalized metadata on success.
    """
    result = validate_provenance(metadata)
    if not result["ok"]:
        raise ValueError(
            "provenance validation failed: " + "; ".join(result["errors"])
        )
    return result["normalized"]
