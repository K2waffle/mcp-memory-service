"""Poll a Railway deployment until it completes. Exits 0 on SUCCESS,
1 on FAILED/CRASHED/REMOVED/timeout. Prints status transitions and a
final JSON blob including the domain if one already exists.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error


GRAPHQL_URL = "https://backboard.railway.com/graphql/v2"
TERMINAL = {"SUCCESS", "FAILED", "CRASHED", "REMOVED", "SKIPPED"}


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
            "User-Agent": "super-brain-deploy/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if data.get("errors"):
        raise RuntimeError(json.dumps(data["errors"]))
    return data["data"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", default=os.environ.get("RAILWAY_TOKEN"))
    ap.add_argument("--deployment-id", required=True)
    ap.add_argument("--service-id", required=True)
    ap.add_argument("--environment-id", required=True)
    ap.add_argument("--timeout-s", type=int, default=1500)
    args = ap.parse_args()

    start = time.time()
    last = ""
    while time.time() - start < args.timeout_s:
        data = gql(
            args.token,
            "query Q($id: String!) { deployment(id: $id) { status } }",
            {"id": args.deployment_id},
        )
        status = (data.get("deployment") or {}).get("status") or "?"
        if status != last:
            print(f"[{time.strftime('%H:%M:%S')}] {status}", flush=True)
            last = status
        if status in TERMINAL:
            # Fetch the domain if one exists
            try:
                doms = gql(
                    args.token,
                    """query Q($serviceId: String!, $environmentId: String!) {
                        domains(serviceId: $serviceId, environmentId: $environmentId) {
                            serviceDomains { domain }
                        }
                    }""",
                    {"serviceId": args.service_id, "environmentId": args.environment_id},
                )
                sd = (doms.get("domains") or {}).get("serviceDomains") or []
                domain = sd[0]["domain"] if sd else None
            except Exception:
                domain = None
            print(json.dumps({
                "deployment_id": args.deployment_id,
                "status": status,
                "domain": domain,
                "elapsed_s": int(time.time() - start),
            }))
            return 0 if status == "SUCCESS" else 1
        time.sleep(20)
    print(json.dumps({"deployment_id": args.deployment_id, "status": "TIMEOUT"}))
    return 1


if __name__ == "__main__":
    sys.exit(main())
