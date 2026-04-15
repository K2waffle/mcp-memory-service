"""Create the Stripe webhook endpoint pointing at the Super-Brain Railway URL.

Usage:
    STRIPE_SECRET_KEY=sk_... python create_stripe_webhook.py --url https://<domain>/api/stripe/webhook

Idempotent: if a webhook endpoint with the same URL already exists, updates
it (enabling all required events) and returns the existing id + secret.
If the endpoint was created previously and we've lost the secret, the Stripe
API does not expose the secret after creation — you'd need to roll it via
the dashboard. This script therefore creates a fresh endpoint if necessary.

Prints a JSON blob: {"id": "we_...", "secret": "whsec_..."}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
import urllib.error


EVENTS = [
    "checkout.session.completed",
    "payment_intent.succeeded",
    "charge.succeeded",
]


def stripe_call(secret: str, method: str, path: str, form: dict | None = None):
    url = f"https://api.stripe.com/v1{path}"
    data = None
    if form is not None:
        # Stripe accepts repeated keys for list-valued params
        parts = []
        for k, v in form.items():
            if isinstance(v, list):
                for item in v:
                    parts.append((f"{k}[]", item))
            else:
                parts.append((k, v))
        data = urllib.parse.urlencode(parts).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {secret}",
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "super-brain-deploy/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Stripe {method} {path} HTTP {e.code}: {e.read().decode('utf-8')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="full webhook URL, e.g. https://x.up.railway.app/api/stripe/webhook")
    ap.add_argument("--secret", default=os.environ.get("STRIPE_SECRET_KEY"))
    args = ap.parse_args()
    if not args.secret:
        print("ERROR: --secret or $STRIPE_SECRET_KEY required", file=sys.stderr)
        return 2

    existing = stripe_call(args.secret, "GET", "/webhook_endpoints?limit=100")
    for ep in existing.get("data", []):
        if ep.get("url") == args.url:
            # Update the event list on the existing endpoint; secret is not
            # retrievable, so caller may need to delete + recreate if they
            # don't already have it.
            upd = stripe_call(
                args.secret,
                "POST",
                f"/webhook_endpoints/{ep['id']}",
                {"enabled_events": EVENTS, "disabled": "false"},
            )
            print(json.dumps({"id": upd["id"], "secret": None, "note": "existing endpoint updated; secret not retrievable"}))
            return 0

    created = stripe_call(
        args.secret,
        "POST",
        "/webhook_endpoints",
        {
            "url": args.url,
            "enabled_events": EVENTS,
            "description": "Super-Brain revenue event webhook",
            "api_version": "2023-10-16",
        },
    )
    print(json.dumps({"id": created["id"], "secret": created["secret"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
