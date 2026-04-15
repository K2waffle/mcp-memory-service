"""End-to-end smoke tests for the deployed Super-Brain MCP server.

Covers docs/handoff.md Step 9, fully automated:

  1. sb_decision_record via MCP JSON-RPC
  2. D1 row check via Cloudflare REST API
  3. sb_opportunity_rank via MCP JSON-RPC
  4. notion_sync/sync.py --entity opportunities (local run, pulls from D1)
  5. Simulated Stripe webhook hit (correctly signed) -> sb_revenue_events row

All env is read from the sibling .env (same loader as deploy_railway.py).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import ssl
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import urllib.error
import uuid


ROOT = os.path.dirname(os.path.abspath(__file__))


def load_env():
    env = {}
    path = os.path.join(ROOT, ".env")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip().strip('"').strip("'")
    env.update({k: v for k, v in os.environ.items() if v})
    return env


def _ssl_ctx():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _http(method, url, headers=None, data=None, timeout=30):
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_ssl_ctx()) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def mcp_call(base, api_key, method, params):
    url = base.rstrip("/") + "/mcp"
    body = json.dumps({
        "jsonrpc": "2.0",
        "id": int(time.time() * 1000) % 100000,
        "method": method,
        "params": params,
    }).encode("utf-8")
    status, raw = _http(
        "POST",
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        },
        data=body,
    )
    # Streamable HTTP may return text/event-stream SSE frames; parse either.
    text = raw.decode("utf-8", errors="replace")
    if text.lstrip().startswith("{"):
        return status, json.loads(text)
    # SSE: extract the last "data: {...}" line
    events = [l[5:].strip() for l in text.splitlines() if l.startswith("data:")]
    if not events:
        return status, {"raw": text}
    return status, json.loads(events[-1])


def d1_query(account, d1_id, token, sql, params=None):
    url = f"https://api.cloudflare.com/client/v4/accounts/{account}/d1/database/{d1_id}/query"
    # If params were supplied, splice them into the SQL as safely-quoted
    # literals — CF D1 REST's prepared-statement shape varies, and a smoke
    # test only ever runs trusted values we generated ourselves.
    if params:
        out = []
        i = 0
        for ch in sql:
            if ch == "?":
                v = params[i]; i += 1
                if isinstance(v, (int, float)):
                    out.append(str(v))
                else:
                    s = str(v).replace("'", "''")
                    out.append(f"'{s}'")
            else:
                out.append(ch)
        sql = "".join(out)
    body = json.dumps({"sql": sql}).encode("utf-8")
    status, raw = _http(
        "POST",
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        data=body,
    )
    try:
        return status, json.loads(raw)
    except Exception:
        return status, {"raw": raw.decode("utf-8", errors="replace")}


def stripe_webhook_hit(base, whsec, event):
    url = base.rstrip("/") + "/api/stripe/webhook"
    payload = json.dumps(event).encode("utf-8")
    ts = int(time.time())
    signed = f"{ts}.".encode("utf-8") + payload
    sig = hmac.new(whsec.encode("utf-8"), signed, hashlib.sha256).hexdigest()
    header = f"t={ts},v1={sig}"
    status, raw = _http(
        "POST",
        url,
        headers={
            "Stripe-Signature": header,
            "Content-Type": "application/json",
        },
        data=payload,
    )
    return status, raw.decode("utf-8", errors="replace")


def main():
    env = load_env()
    base = "https://super-brain-production.up.railway.app"
    api_key = env["MCP_API_KEY"]
    cf_account = env["CLOUDFLARE_ACCOUNT_ID"]
    cf_token = env["CLOUDFLARE_API_TOKEN"]
    cf_d1 = env["CLOUDFLARE_D1_DATABASE_ID"]
    whsec = env.get("STRIPE_WEBHOOK_SECRET", "")

    results: dict = {"steps": []}

    # 0. tools/list sanity
    print("[0] tools/list", flush=True)
    status, resp = mcp_call(base, api_key, "tools/list", {})
    tools = [t["name"] for t in resp.get("result", {}).get("tools", [])]
    has_sb = any(n.startswith(("decision_record", "opportunity_rank", "revenue_event_record")) for n in tools)
    results["steps"].append({"step": "tools/list", "status": status, "tool_count": len(tools), "sb_visible": has_sb, "tools": tools})
    print(f"    -> {len(tools)} tools; sb visible={has_sb}")

    # 1. decision_record
    print("[1] decision_record", flush=True)
    dec_tldr = f"smoke-decision-{uuid.uuid4().hex[:8]}"
    status, resp = mcp_call(base, api_key, "tools/call", {
        "name": "decision_record",
        "arguments": {
            "tldr": dec_tldr,
            "rationale": "End-to-end smoke test of deployed Super-Brain.",
            "valid_from": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "scope": "smoke-test",
            "tags": ["smoke-test"],
        },
    })
    results["steps"].append({"step": "decision_record", "status": status, "response": resp})
    print(f"    -> status={status}")

    # 2. D1 check
    print("[2] D1 sb_decisions row check", flush=True)
    time.sleep(1)
    status, resp = d1_query(
        cf_account, cf_d1, cf_token,
        "SELECT id, tldr FROM sb_decisions WHERE tldr = ? ORDER BY rowid DESC LIMIT 1",
        [dec_tldr],
    )
    rows = (resp.get("result") or [{}])[0].get("results") or []
    results["steps"].append({"step": "d1_sb_decisions", "status": status, "found": bool(rows), "row": rows[0] if rows else None})
    print(f"    -> found={bool(rows)} row={rows[0] if rows else None}")

    # 3. opportunity_rank
    print("[3] opportunity_rank", flush=True)
    op_tldr = f"smoke-opp-{uuid.uuid4().hex[:8]}"
    status, resp = mcp_call(base, api_key, "tools/call", {
        "name": "opportunity_rank",
        "arguments": {
            "tldr": op_tldr,
            "speed_to_first_dollar": 1,
            "earning_ceiling": 10000,
            "automation_leverage": 5,
            "repeatability": 4,
            "distribution": 3,
            "retention": 3,
            "compliance_risk": 2,
            "monetization_route": "direct_sale",
        },
    })
    results["steps"].append({"step": "opportunity_rank", "status": status, "response": resp})
    print(f"    -> status={status}")

    # 4. notion_sync (dry run + real on opportunities)
    print("[4] notion_sync/sync.py --entity opportunities", flush=True)
    sync_env = os.environ.copy()
    for k in ("NOTION_API_KEY", "NOTION_DB_DECISIONS", "NOTION_DB_PROCEDURES",
              "NOTION_DB_OPPORTUNITIES", "CLOUDFLARE_API_TOKEN",
              "CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_D1_DATABASE_ID"):
        if env.get(k):
            sync_env[k] = env[k]
    try:
        proc = subprocess.run(
            [sys.executable, "notion_sync/sync.py", "--entity", "opportunities"],
            cwd=ROOT, env=sync_env, capture_output=True, text=True, timeout=120,
        )
        results["steps"].append({
            "step": "notion_sync",
            "returncode": proc.returncode,
            "stdout": proc.stdout[-2000:],
            "stderr": proc.stderr[-2000:],
        })
        print(f"    -> rc={proc.returncode}")
    except Exception as e:
        results["steps"].append({"step": "notion_sync", "error": str(e)})
        print(f"    -> error={e}")

    # 5. stripe webhook simulation
    print("[5] stripe webhook (simulated, signed)", flush=True)
    fake_event_id = f"evt_test_{uuid.uuid4().hex[:16]}"
    event = {
        "id": fake_event_id,
        "type": "checkout.session.completed",
        "data": {
            "object": {
                "id": f"cs_test_{uuid.uuid4().hex[:16]}",
                "amount_total": 100,  # $1.00 smoke
                "currency": "usd",
                "customer": "cus_smoke_test",
                "metadata": {"source": "smoke-test"},
            }
        },
    }
    if not whsec:
        results["steps"].append({"step": "stripe_webhook", "skipped": "no STRIPE_WEBHOOK_SECRET in .env"})
        print("    -> skipped (no whsec)")
    else:
        status, body = stripe_webhook_hit(base, whsec, event)
        results["steps"].append({"step": "stripe_webhook_post", "status": status, "body": body})
        print(f"    -> POST status={status} body={body[:200]}")

        # 5b. verify sb_revenue_events row
        time.sleep(2)
        status, resp = d1_query(
            cf_account, cf_d1, cf_token,
            "SELECT id, amount_cents, counterparty_rail FROM sb_revenue_events WHERE id = ?",
            [fake_event_id],
        )
        rows = (resp.get("result") or [{}])[0].get("results") or []
        results["steps"].append({"step": "sb_revenue_events", "status": status, "found": bool(rows), "row": rows[0] if rows else None})
        print(f"    -> sb_revenue_events found={bool(rows)} row={rows[0] if rows else None}")

    print()
    print(json.dumps(results, indent=2, default=str))
    # Exit non-zero if any core step failed
    core_ok = all(
        (s.get("status") in (None, 200) and s.get("returncode", 0) == 0)
        for s in results["steps"]
    )
    return 0 if core_ok else 1


if __name__ == "__main__":
    sys.exit(main())
