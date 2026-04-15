"""Patch specific Railway env vars and trigger a redeploy.

Usage:
    RAILWAY_TOKEN=... python railway_patch_and_redeploy.py \
        --project-id <pid> --environment-id <eid> --service-id <sid> \
        KEY1=VALUE1 KEY2=VALUE2

Returns the latest deployment id as JSON after triggering the redeploy.
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
        raise RuntimeError(json.dumps(data["errors"], indent=2))
    return data["data"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", default=os.environ.get("RAILWAY_TOKEN"))
    ap.add_argument("--project-id", required=True)
    ap.add_argument("--environment-id", required=True)
    ap.add_argument("--service-id", required=True)
    ap.add_argument("pairs", nargs="+", help="KEY=VALUE pairs to upsert")
    args = ap.parse_args()

    vars_ = {}
    for p in args.pairs:
        if "=" not in p:
            raise SystemExit(f"Bad pair (missing '='): {p}")
        k, _, v = p.partition("=")
        vars_[k.strip()] = v
    print(f"[var] upserting {len(vars_)} variables: {list(vars_)}", flush=True)

    gql(
        args.token,
        """mutation Upsert($input: VariableCollectionUpsertInput!) {
            variableCollectionUpsert(input: $input)
        }""",
        {
            "input": {
                "projectId": args.project_id,
                "environmentId": args.environment_id,
                "serviceId": args.service_id,
                "variables": vars_,
            }
        },
    )
    print("[var] ok", flush=True)

    print("[dep] triggering redeploy", flush=True)
    gql(
        args.token,
        """mutation($serviceId: String!, $environmentId: String!) {
            serviceInstanceRedeploy(serviceId: $serviceId, environmentId: $environmentId)
        }""",
        {"serviceId": args.service_id, "environmentId": args.environment_id},
    )

    # Fetch the newest deployment id (may take a second to appear)
    deploy_id = None
    for _ in range(10):
        data = gql(
            args.token,
            """query Deps($serviceId: String!, $environmentId: String!) {
                deployments(first: 1, input: { serviceId: $serviceId, environmentId: $environmentId }) {
                    edges { node { id status createdAt } }
                }
            }""",
            {"serviceId": args.service_id, "environmentId": args.environment_id},
        )
        edges = data["deployments"]["edges"]
        if edges:
            deploy_id = edges[0]["node"]["id"]
            status = edges[0]["node"]["status"]
            print(f"[dep] latest deployment id={deploy_id} status={status}", flush=True)
            break
        time.sleep(2)

    print(json.dumps({"deployment_id": deploy_id}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
