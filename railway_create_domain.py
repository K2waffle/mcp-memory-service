"""Create a public service domain on Railway. Idempotent: returns existing
domain if one is already attached to the service.

Usage:
    RAILWAY_TOKEN=... python railway_create_domain.py \
        --service-id <sid> --environment-id <eid>
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
            "User-Agent": "super-brain-deploy/1.0 (+https://github.com/K2waffle/coderabbit)",
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
    ap.add_argument("--service-id", required=True)
    ap.add_argument("--environment-id", required=True)
    ap.add_argument("--project-id", required=True)
    args = ap.parse_args()

    # Check for an existing domain first
    data = gql(
        args.token,
        """query Q($projectId: String!, $serviceId: String!, $environmentId: String!) {
            domains(projectId: $projectId, serviceId: $serviceId, environmentId: $environmentId) {
                serviceDomains { id domain }
            }
        }""",
        {"projectId": args.project_id, "serviceId": args.service_id, "environmentId": args.environment_id},
    )
    sd = (data.get("domains") or {}).get("serviceDomains") or []
    if sd:
        print(json.dumps({"domain": sd[0]["domain"], "created": False, "id": sd[0].get("id")}))
        return 0

    data = gql(
        args.token,
        """mutation($input: ServiceDomainCreateInput!) {
            serviceDomainCreate(input: $input) { id domain }
        }""",
        {"input": {"serviceId": args.service_id, "environmentId": args.environment_id}},
    )
    out = data["serviceDomainCreate"]
    print(json.dumps({"domain": out["domain"], "created": True, "id": out["id"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
