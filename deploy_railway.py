"""Automated Railway deploy for super-brain.

Usage:
    python deploy_railway.py [--token TOKEN] [--project-id ID]

Defaults:
    token: $RAILWAY_TOKEN
    project-id: auto-discovered from first project the token can see

Idempotent: safe to re-run. Creates the service if missing, upserts variables,
triggers a deploy, and polls until live.

Flow:
1. Create service from K2waffle/mcp-memory-service on super-brain branch
2. Upsert all env vars
3. Trigger redeploy
4. Poll deployment status until SUCCESS or FAILED
5. Generate public domain
6. Update MCP_OAUTH_ISSUER with real domain, redeploy
7. Print domain + service/deployment IDs
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import urllib.request
import urllib.error


GRAPHQL_URL = "https://backboard.railway.com/graphql/v2"

REPO = "K2waffle/mcp-memory-service"
BRANCH = "super-brain"
SERVICE_NAME = "super-brain"

# Non-secret defaults baked in. Secrets (anything in REQUIRED_SECRETS) are
# loaded from the caller's environment — NEVER hardcode them in this file,
# it's committed to a public repo and would trip GitHub's secret scanning.
NON_SECRET_DEFAULTS: dict[str, str] = {
    "MCP_SUPER_BRAIN_ENABLED": "true",
    "MCP_HTTP_ENABLED": "true",
    "MCP_HTTP_HOST": "0.0.0.0",
    "MCP_HTTP_PORT": "8080",
    "PORT": "8080",
    "MCP_MEMORY_STORAGE_BACKEND": "cloudflare",
    "CLOUDFLARE_VECTORIZE_INDEX": "mcp-memory-index",
    "CLOUDFLARE_R2_BUCKET": "super-brain-artifacts",
    "MCP_OAUTH_STORAGE_BACKEND": "sqlite",
    "MCP_OAUTH_SQLITE_PATH": "/app/sqlite_db/oauth.db",
    "MCP_OAUTH_ISSUER": "https://placeholder.up.railway.app",
    "MCP_SUPER_BRAIN_ARTIFACT_ROOT": "/app/artifacts",
    "MCP_QUALITY_SYSTEM_ENABLED": "true",
    "MCP_CONSOLIDATION_ENABLED": "true",
    "MCP_MEMORY_SQLITE_PRAGMAS": "journal_mode=WAL,busy_timeout=15000,cache_size=20000",
    "STRIPE_WEBHOOK_SECRET": "",
    "LOG_LEVEL": "INFO",
}

# Secrets that must be provided via the caller's environment.
REQUIRED_SECRETS: tuple[str, ...] = (
    "CLOUDFLARE_API_TOKEN",
    "CLOUDFLARE_ACCOUNT_ID",
    "CLOUDFLARE_D1_DATABASE_ID",
    "MCP_API_KEY",
    "STRIPE_SECRET_KEY",
)


def build_env_vars() -> dict[str, str]:
    """Compose the full env var set. Secrets pulled from the local environment
    or from a sibling `.env` file if present."""
    env = dict(NON_SECRET_DEFAULTS)

    # Load .env if present (simple KEY=VALUE parser, ignores comments/blanks).
    env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_file):
        with open(env_file, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and v:
                    env.setdefault(k, v)

    # Environment wins over .env file
    for key in REQUIRED_SECRETS:
        v = os.environ.get(key)
        if v:
            env[key] = v

    missing = [k for k in REQUIRED_SECRETS if not env.get(k)]
    if missing:
        raise RuntimeError(
            f"Missing required secrets (set via env or .env): {', '.join(missing)}"
        )
    return env


def gql(token: str, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
    """Execute a GraphQL query. Raises on errors."""
    body = {"query": query}
    if variables:
        body["variables"] = variables
    req = urllib.request.Request(
        GRAPHQL_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            # Cloudflare WAF rejects default Python-urllib UA with error 1010
            "User-Agent": "super-brain-deploy/1.0 (+https://github.com/K2waffle/mcp-memory-service)",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Railway API HTTP {e.code}: {e.read().decode('utf-8')}")
    if payload.get("errors"):
        raise RuntimeError(f"Railway API error: {json.dumps(payload['errors'], indent=2)}")
    return payload["data"]


def find_project(token: str, project_id: str | None) -> tuple[str, str]:
    """Return (project_id, production_environment_id)."""
    if project_id:
        data = gql(
            token,
            """
            query P($id: String!) {
              project(id: $id) {
                id
                environments { edges { node { id name } } }
              }
            }
            """,
            {"id": project_id},
        )
        p = data["project"]
    else:
        data = gql(token, "query { projects { edges { node { id name environments { edges { node { id name } } } } } } }")
        edges = data["projects"]["edges"]
        if not edges:
            raise RuntimeError("No Railway projects found for this token")
        p = edges[0]["node"]

    prod = next(
        e["node"]["id"] for e in p["environments"]["edges"] if e["node"]["name"] == "production"
    )
    return p["id"], prod


def find_or_create_service(token: str, project_id: str) -> str:
    """Return service id. Creates the service if it doesn't already exist."""
    data = gql(
        token,
        """
        query P($id: String!) {
          project(id: $id) {
            services { edges { node { id name } } }
          }
        }
        """,
        {"id": project_id},
    )
    for edge in data["project"]["services"]["edges"]:
        if edge["node"]["name"] == SERVICE_NAME:
            print(f"[svc] found existing service {edge['node']['id']}")
            return edge["node"]["id"]

    print(f"[svc] creating service {SERVICE_NAME} from {REPO}#{BRANCH}")
    data = gql(
        token,
        """
        mutation($input: ServiceCreateInput!) {
          serviceCreate(input: $input) { id name }
        }
        """,
        {
            "input": {
                "projectId": project_id,
                "name": SERVICE_NAME,
                "source": {"repo": REPO},
                "branch": BRANCH,
            }
        },
    )
    svc_id = data["serviceCreate"]["id"]
    print(f"[svc] created service {svc_id}")
    return svc_id


def upsert_variables(token: str, project_id: str, env_id: str, svc_id: str, vars_: dict[str, str]) -> None:
    """Bulk upsert variables using variableCollectionUpsert."""
    print(f"[var] upserting {len(vars_)} variables")
    gql(
        token,
        """
        mutation Upsert($input: VariableCollectionUpsertInput!) {
          variableCollectionUpsert(input: $input)
        }
        """,
        {
            "input": {
                "projectId": project_id,
                "environmentId": env_id,
                "serviceId": svc_id,
                "variables": vars_,
            }
        },
    )
    print(f"[var] variables upserted")


def trigger_deploy(token: str, svc_id: str, env_id: str) -> str:
    """Trigger a redeploy. Returns deployment id."""
    print(f"[dep] triggering deploy")
    data = gql(
        token,
        """
        mutation($serviceId: String!, $environmentId: String!) {
          serviceInstanceRedeploy(serviceId: $serviceId, environmentId: $environmentId)
        }
        """,
        {"serviceId": svc_id, "environmentId": env_id},
    )
    # serviceInstanceRedeploy returns a boolean; deployment id is fetched separately
    print(f"[dep] redeploy triggered; response={data}")
    return ""


def wait_for_deployment(token: str, svc_id: str, env_id: str, timeout_s: int = 900) -> str:
    """Poll deployment status until SUCCESS/FAILED. Returns final status."""
    start = time.time()
    last_status = ""
    while time.time() - start < timeout_s:
        data = gql(
            token,
            """
            query Deps($serviceId: String!, $environmentId: String!) {
              deployments(first: 1, input: { serviceId: $serviceId, environmentId: $environmentId }) {
                edges { node { id status createdAt } }
              }
            }
            """,
            {"serviceId": svc_id, "environmentId": env_id},
        )
        edges = data["deployments"]["edges"]
        if edges:
            node = edges[0]["node"]
            status = node["status"]
            if status != last_status:
                print(f"[dep] status={status} id={node['id']}")
                last_status = status
            if status == "SUCCESS":
                return status
            if status in ("FAILED", "CRASHED", "REMOVED"):
                return status
        time.sleep(10)
    raise TimeoutError(f"Deployment did not complete within {timeout_s}s; last={last_status}")


def generate_domain(token: str, svc_id: str, env_id: str) -> str:
    """Generate a public domain and return its hostname."""
    print(f"[dom] generating service domain")
    data = gql(
        token,
        """
        mutation($serviceId: String!, $environmentId: String!) {
          serviceDomainCreate(input: { serviceId: $serviceId, environmentId: $environmentId }) {
            domain
          }
        }
        """,
        {"serviceId": svc_id, "environmentId": env_id},
    )
    domain = data["serviceDomainCreate"]["domain"]
    print(f"[dom] domain={domain}")
    return domain


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", default=os.environ.get("RAILWAY_TOKEN"))
    ap.add_argument("--project-id", default=os.environ.get("RAILWAY_PROJECT_ID"))
    args = ap.parse_args()
    if not args.token:
        print("ERROR: --token or $RAILWAY_TOKEN required", file=sys.stderr)
        return 2

    env_vars = build_env_vars()

    project_id, env_id = find_project(args.token, args.project_id)
    print(f"[prj] project={project_id} env={env_id}")

    svc_id = find_or_create_service(args.token, project_id)

    upsert_variables(args.token, project_id, env_id, svc_id, env_vars)

    trigger_deploy(args.token, svc_id, env_id)

    # Initial build may take 4-8 min (Docker + ML model bake)
    status = wait_for_deployment(args.token, svc_id, env_id, timeout_s=900)
    if status != "SUCCESS":
        print(f"[dep] deploy ended with status={status}", file=sys.stderr)
        return 1

    domain = generate_domain(args.token, svc_id, env_id)

    # Re-inject MCP_OAUTH_ISSUER with real domain; redeploy
    final_vars = {**env_vars, "MCP_OAUTH_ISSUER": f"https://{domain}"}
    upsert_variables(args.token, project_id, env_id, svc_id, final_vars)
    trigger_deploy(args.token, svc_id, env_id)
    status2 = wait_for_deployment(args.token, svc_id, env_id, timeout_s=900)
    if status2 != "SUCCESS":
        print(f"[dep] post-issuer redeploy ended with status={status2}", file=sys.stderr)
        return 1

    print(json.dumps({"project_id": project_id, "environment_id": env_id, "service_id": svc_id, "domain": domain}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
