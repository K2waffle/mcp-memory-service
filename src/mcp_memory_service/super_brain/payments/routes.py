"""FastAPI routes for the super-brain payments surface.

Currently exposes a single endpoint:

    POST /api/stripe/webhook

Stripe posts events here. We verify the ``Stripe-Signature`` header using
``STRIPE_WEBHOOK_SECRET`` (fail closed if absent), then translate qualifying
events into ``sb_revenue_events`` rows via
``super_brain.payments.revenue_events.record_event``.

This router is only mounted when ``super_brain.is_enabled()`` returns True,
so upstream behavior is unchanged when the feature flag is off.
"""
from __future__ import annotations

import json
import logging
from types import SimpleNamespace

from fastapi import APIRouter, Depends, HTTPException, Request

from ...web.dependencies import get_storage
from .stripe_client import parse_event_for_revenue, verify_webhook_signature
from .revenue_events import record_event

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/stripe/webhook")
async def stripe_webhook(request: Request, storage=Depends(get_storage)):
    """Receive a Stripe event, verify signature, record revenue.

    Idempotent: Stripe retries the same ``event.id`` on non-2xx, and
    ``record_event`` deduplicates on the same id.
    """
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")

    ok, reason = verify_webhook_signature(payload, sig_header)
    if not ok:
        # Don't leak signature-verification internals in the response body.
        logger.warning("stripe webhook rejected: %s", reason)
        raise HTTPException(status_code=400, detail="invalid_signature")

    try:
        event = json.loads(payload.decode("utf-8"))
    except Exception as exc:
        logger.warning("stripe webhook: bad json: %s", exc)
        raise HTTPException(status_code=400, detail="bad_json")

    revenue_args = parse_event_for_revenue(event)
    if revenue_args is None:
        # Not a revenue event we care about — 200 so Stripe doesn't retry.
        return {"status": "ignored", "type": event.get("type", "unknown")}

    # revenue_events.record_event expects a server-like object with `.storage`
    server_shim = SimpleNamespace(storage=storage)
    result = await record_event(server_shim, revenue_args)
    return {"status": "ok", **result}
