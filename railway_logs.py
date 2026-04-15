"""Fetch logs for a Railway deployment. Prints plain text to stdout.

Usage:
    RAILWAY_TOKEN=... python railway_logs.py --deployment-id <id> [--kind deploy|build]

kind=deploy returns runtime (container) logs — use this when status is DEPLOYING/FAILED.
kind=build returns build-phase logs — use this when the build itself failed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
import urllib.error


GRAPHQL_URL = "https://backboard.railway.com/graphql/v2"


def gql(token, query, variables=None):
    body = {"query": query}
    if variables:
        body["variables"] = variables
    req = urllib.request.Request(
        GRAPHQL_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "super-brain-deploy/1.0 (+https://github.com/K2waffle/mcp-memory-service)",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode('utf-8')}")
    if data.get("errors"):
        raise RuntimeError(json.dumps(data["errors"]))
    return data["data"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", default=os.environ.get("RAILWAY_TOKEN"))
    ap.add_argument("--deployment-id", required=True)
    ap.add_argument("--kind", choices=["deploy", "build"], default="deploy")
    ap.add_argument("--limit", type=int, default=1000)
    args = ap.parse_args()

    if args.kind == "deploy":
        query = """query Q($id: String!, $limit: Int) {
            deploymentLogs(deploymentId: $id, limit: $limit) {
                message
                timestamp
                severity
            }
        }"""
        key = "deploymentLogs"
    else:
        query = """query Q($id: String!, $limit: Int) {
            buildLogs(deploymentId: $id, limit: $limit) {
                message
                timestamp
                severity
            }
        }"""
        key = "buildLogs"

    data = gql(args.token, query, {"id": args.deployment_id, "limit": args.limit})
    logs = data.get(key) or []
    for entry in logs:
        ts = entry.get("timestamp", "")
        sev = entry.get("severity", "") or ""
        msg = entry.get("message", "")
        if sev:
            print(f"[{ts}] [{sev}] {msg}")
        else:
            print(f"[{ts}] {msg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
