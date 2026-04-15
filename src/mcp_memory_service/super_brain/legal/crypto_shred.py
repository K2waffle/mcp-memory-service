"""Per-subject key registry + crypto-shred erasure.

GDPR Art. 17 requires us to honor erasure even when vector indices and
object stores make physical deletion slow or impossible (embeddings get
baked into index shards; R2 objects replicate). The answer is
*crypto-shredding*: encrypt per-subject data with a per-subject key,
store only ciphertext in the durable layers, and throw the key away on
erasure. After shredding, the ciphertext remains but is unreadable.

This module owns the key registry. Fernet (AES-128-CBC + HMAC-SHA256) is
used for its opinionated defaults — not a bespoke crypto construction.
Keys live in ``sb_subject_keys``; erasure flips a column AND zeroes the
key material so even a DB dump post-shred can't recover it.
"""
from __future__ import annotations

import base64
import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _ensure_fernet():
    try:
        from cryptography.fernet import Fernet  # type: ignore
        return Fernet
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "crypto_shred requires `cryptography` (pip install cryptography)"
        ) from exc


def _new_key_material() -> bytes:
    """Generate a fresh 256-bit key, URL-safe base64 encoded (Fernet format)."""
    return base64.urlsafe_b64encode(os.urandom(32))


async def ensure_subject_key(server: Any, subject_ref: str) -> Dict[str, Any]:
    """Fetch or create the key for a subject. Idempotent.

    Returns ``{"subject_ref", "key_id", "material_b64", "status"}`` where
    ``status`` is one of ``created`` | ``existing`` | ``shredded``.
    """
    from ..learning.outcome import _exec, _fetchone

    row = await _fetchone(
        server,
        "SELECT id, key_material, shredded_at FROM sb_subject_keys "
        "WHERE subject_ref = ?",
        [subject_ref],
    )
    if row:
        if row.get("shredded_at"):
            return {
                "subject_ref": subject_ref,
                "key_id": row["id"],
                "material_b64": None,
                "status": "shredded",
            }
        return {
            "subject_ref": subject_ref,
            "key_id": row["id"],
            "material_b64": row["key_material"],
            "status": "existing",
        }

    key_id = f"sk_{subject_ref}_{int(time.time())}"
    material = _new_key_material().decode("ascii")
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    await _exec(
        server,
        "INSERT INTO sb_subject_keys (id, subject_ref, key_material, created_at) "
        "VALUES (?, ?, ?, ?)",
        [key_id, subject_ref, material, now_iso],
    )
    return {
        "subject_ref": subject_ref,
        "key_id": key_id,
        "material_b64": material,
        "status": "created",
    }


async def encrypt_for_subject(server: Any, subject_ref: str, plaintext: str) -> str:
    """Encrypt plaintext under the subject's key. Returns a token safe for storage.

    The token is Fernet-formatted (self-identifying), so ``decrypt_for_subject``
    can reject tokens created under a different key without additional metadata.
    """
    Fernet = _ensure_fernet()
    key = await ensure_subject_key(server, subject_ref)
    if key["status"] == "shredded" or not key["material_b64"]:
        raise PermissionError(f"subject {subject_ref!r} is shredded; cannot encrypt")
    f = Fernet(key["material_b64"].encode("ascii"))
    return f.encrypt(plaintext.encode("utf-8")).decode("ascii")


async def decrypt_for_subject(server: Any, subject_ref: str, token: str) -> Optional[str]:
    """Decrypt a token for ``subject_ref``. Returns ``None`` if shredded/invalid."""
    Fernet = _ensure_fernet()
    key = await ensure_subject_key(server, subject_ref)
    if key["status"] == "shredded" or not key["material_b64"]:
        return None
    try:
        f = Fernet(key["material_b64"].encode("ascii"))
        return f.decrypt(token.encode("ascii")).decode("utf-8")
    except Exception as exc:
        logger.debug("decrypt_for_subject failed: %s", exc)
        return None


async def shred_subject(server: Any, subject_ref: str) -> Dict[str, Any]:
    """Honor erasure: zero out the key and flip ``shredded_at``.

    Ciphertext stays put; without the key, it's unreadable. Any memories
    tagged ``subject:<subject_ref>`` should ALSO be tagged ``sb:shredded``
    so retrieval skips them fast.
    """
    from ..learning.outcome import _exec, _fetchone

    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    row = await _fetchone(
        server,
        "SELECT id FROM sb_subject_keys WHERE subject_ref = ?",
        [subject_ref],
    )
    if not row:
        return {"subject_ref": subject_ref, "status": "never_existed"}

    await _exec(
        server,
        "UPDATE sb_subject_keys SET key_material = NULL, shredded_at = ? "
        "WHERE subject_ref = ?",
        [now_iso, subject_ref],
    )
    # Mark all memories under this subject as shredded.
    try:
        await _exec(
            server,
            "UPDATE memories SET tags_str = tags_str || ',sb:shredded' "
            "WHERE tags_str LIKE ? AND tags_str NOT LIKE '%sb:shredded%'",
            [f"%subject:{subject_ref}%"],
        )
    except Exception as exc:
        logger.debug("shred_subject: tag sweep skipped: %s", exc)

    return {
        "subject_ref": subject_ref,
        "key_id": row["id"],
        "status": "shredded",
        "shredded_at": now_iso,
    }


async def list_shredded(server: Any) -> List[Dict[str, Any]]:
    from ..learning.outcome import _exec  # noqa
    storage = getattr(server, "storage", None)
    if storage is None:
        return []
    try:
        if hasattr(storage, "d1_query"):
            return await storage.d1_query(
                "SELECT id, subject_ref, shredded_at FROM sb_subject_keys "
                "WHERE shredded_at IS NOT NULL",
                [],
            )
    except Exception as exc:
        logger.debug("list_shredded: query failed: %s", exc)
    return []
