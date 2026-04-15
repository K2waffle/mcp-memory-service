"""Thin Stripe wrapper used by the payments / revenue tools.

We keep this wrapper deliberately small. The goal is to:

  * lazily import ``stripe`` so the server still imports without the SDK;
  * normalize product/price/payment-link creation into one call;
  * verify webhook signatures before ``revenue_events.record_event``.

Any sophisticated billing logic (subscriptions, tax, proration) should go
through the Stripe SDK directly at the call site — this file is the
minimum surface to earn revenue and close the loop into the memory system.

PCI scope: we never see full card numbers. Customers pay on Stripe-hosted
payment links / checkout pages, which keeps us under SAQ-A.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

STRIPE_SECRET_ENV = "STRIPE_SECRET_KEY"
STRIPE_WEBHOOK_SECRET_ENV = "STRIPE_WEBHOOK_SECRET"


class StripeNotConfigured(RuntimeError):
    """Raised when a Stripe call is attempted without credentials."""


def _secret_key() -> str:
    key = os.environ.get(STRIPE_SECRET_ENV)
    if not key:
        raise StripeNotConfigured(
            f"missing {STRIPE_SECRET_ENV} — set it before creating products/links"
        )
    return key


def _stripe():
    """Lazy import so the package imports cleanly without stripe installed."""
    try:
        import stripe  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise StripeNotConfigured(
            "stripe SDK not installed (pip install stripe)"
        ) from exc
    stripe.api_key = _secret_key()
    return stripe


def create_product_and_price(
    name: str,
    amount_cents: int,
    currency: str = "usd",
    description: Optional[str] = None,
    metadata: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Create a product + one-time price. Returns ``{product_id, price_id}``."""
    stripe = _stripe()
    product = stripe.Product.create(
        name=name,
        description=description or name,
        metadata=metadata or {},
    )
    price = stripe.Price.create(
        product=product.id,
        unit_amount=int(amount_cents),
        currency=currency,
    )
    return {"product_id": product.id, "price_id": price.id}


def create_payment_link(
    price_id: str,
    quantity: int = 1,
    metadata: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Create a Stripe Payment Link. Returns ``{link_id, url}``.

    ``metadata.memory_id`` (if set) flows through the webhook so we can
    attribute revenue to the originating memory without a database join.
    """
    stripe = _stripe()
    link = stripe.PaymentLink.create(
        line_items=[{"price": price_id, "quantity": quantity}],
        metadata=metadata or {},
    )
    return {"link_id": link.id, "url": link.url}


def create_one_shot(
    name: str,
    amount_cents: int,
    currency: str = "usd",
    memory_id: Optional[str] = None,
) -> Dict[str, str]:
    """Convenience: create product + price + payment link in one call.

    Returns ``{product_id, price_id, link_id, url}``.
    """
    ids = create_product_and_price(name=name, amount_cents=amount_cents,
                                   currency=currency)
    meta: Dict[str, str] = {"created_at": str(int(time.time()))}
    if memory_id:
        meta["memory_id"] = memory_id
    link = create_payment_link(ids["price_id"], metadata=meta)
    return {**ids, **link}


def verify_webhook_signature(
    payload: bytes,
    sig_header: str,
    secret: Optional[str] = None,
    tolerance_seconds: int = 300,
) -> Tuple[bool, str]:
    """Verify the ``Stripe-Signature`` header.

    Returns ``(ok, reason)``. Implements the Stripe v1 scheme directly so the
    server can reject spoofed webhook posts even if the stripe SDK is absent.
    """
    secret = secret or os.environ.get(STRIPE_WEBHOOK_SECRET_ENV)
    if not secret:
        return False, "no_webhook_secret"
    if not sig_header:
        return False, "missing_signature_header"

    parts = {}
    for p in sig_header.split(","):
        if "=" in p:
            k, v = p.split("=", 1)
            parts.setdefault(k.strip(), []).append(v.strip())

    try:
        ts = int(parts.get("t", ["0"])[0])
    except ValueError:
        return False, "bad_timestamp"

    if tolerance_seconds and abs(time.time() - ts) > tolerance_seconds:
        return False, "timestamp_outside_tolerance"

    v1s = parts.get("v1", [])
    if not v1s:
        return False, "no_v1_signature"

    signed = f"{ts}.".encode("utf-8") + payload
    expected = hmac.new(secret.encode("utf-8"), signed, hashlib.sha256).hexdigest()
    for sig in v1s:
        if hmac.compare_digest(expected, sig):
            return True, "ok"
    return False, "signature_mismatch"


def parse_event_for_revenue(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Pull the revenue-relevant fields out of a Stripe event.

    Returns a dict ready to hand to ``revenue_events.record_event``, or
    ``None`` if the event isn't a completed payment.
    """
    event_type = event.get("type", "")
    if event_type not in (
        "checkout.session.completed",
        "payment_intent.succeeded",
        "charge.succeeded",
    ):
        return None

    obj = (event.get("data") or {}).get("object") or {}
    amount = obj.get("amount_total") or obj.get("amount_received") or obj.get("amount")
    currency = (obj.get("currency") or "usd").upper()
    metadata = obj.get("metadata") or {}
    memory_id = metadata.get("memory_id")
    counterparty_ref = obj.get("customer") or obj.get("id")

    if amount is None:
        return None

    return {
        "revenue_event_id": event.get("id") or obj.get("id"),
        "memory_id": memory_id,
        "counterparty_rail": "stripe",
        "counterparty_ref": counterparty_ref,
        "amount_cents": int(amount),
        "currency": currency,
        "recognized_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pci_scope": "SAQ-A",
    }
