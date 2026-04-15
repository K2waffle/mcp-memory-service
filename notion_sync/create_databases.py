"""Create the 3 Super-Brain Notion databases under a shared parent page.

Usage:
    python create_databases.py --token <ntn_...> --parent-page-id <uuid>

Idempotent by title: if a database with the same title already exists under
the parent page, the existing database id is returned instead of creating a
new one.

Prints a JSON blob with:
    {"decisions": "<db_id>", "procedures": "<db_id>", "opportunities": "<db_id>"}

Schemas match what notion_sync/sync.py reads/writes:
    - decisions: Title, Hash, TLDR, Rationale, Supersedes, Valid From, Valid To, Source
    - procedures: Title, Hash, TLDR, Steps, Alpha, Beta, N Runs
    - opportunities: Title, Hash, TLDR, Speed to First Dollar, Automation Leverage, Score
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import urllib.error
from typing import Any


API_ROOT = "https://api.notion.com/v1"
API_VERSION = "2022-06-28"


SCHEMAS: dict[str, dict[str, Any]] = {
    "decisions": {
        "Title": {"title": {}},
        "Hash": {"rich_text": {}},
        "TLDR": {"rich_text": {}},
        "Rationale": {"rich_text": {}},
        "Supersedes": {"rich_text": {}},
        "Valid From": {"date": {}},
        "Valid To": {"date": {}},
        "Source": {"url": {}},
    },
    "procedures": {
        "Title": {"title": {}},
        "Hash": {"rich_text": {}},
        "TLDR": {"rich_text": {}},
        "Steps": {"rich_text": {}},
        "Alpha": {"number": {"format": "number"}},
        "Beta": {"number": {"format": "number"}},
        "N Runs": {"number": {"format": "number"}},
    },
    "opportunities": {
        "Title": {"title": {}},
        "Hash": {"rich_text": {}},
        "TLDR": {"rich_text": {}},
        "Speed to First Dollar": {
            "select": {
                "options": [
                    {"name": "days", "color": "green"},
                    {"name": "weeks", "color": "yellow"},
                    {"name": "months", "color": "orange"},
                    {"name": "quarters", "color": "red"},
                ]
            }
        },
        "Automation Leverage": {
            "select": {
                "options": [
                    {"name": "high", "color": "green"},
                    {"name": "medium", "color": "yellow"},
                    {"name": "low", "color": "red"},
                ]
            }
        },
        "Score": {"number": {"format": "number"}},
    },
}

TITLES = {
    "decisions": "Super-Brain / Decisions",
    "procedures": "Super-Brain / Procedures",
    "opportunities": "Super-Brain / Opportunities",
}


def call(token: str, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {token}",
        "Notion-Version": API_VERSION,
        "Content-Type": "application/json",
        "User-Agent": "super-brain-deploy/1.0",
    }
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(f"{API_ROOT}{path}", data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Notion API {method} {path} HTTP {e.code}: {e.read().decode('utf-8')}")


def find_existing(token: str, parent_page_id: str, title: str) -> str | None:
    """Search for an existing database with the given title under the parent page."""
    body = {
        "query": title,
        "filter": {"value": "database", "property": "object"},
        "page_size": 20,
    }
    data = call(token, "POST", "/search", body)
    for r in data.get("results", []):
        if r.get("object") != "database":
            continue
        title_rt = r.get("title") or []
        plain = "".join(t.get("plain_text", "") for t in title_rt)
        if plain.strip() == title.strip():
            parent = r.get("parent") or {}
            # only match if parent page matches (Notion page ids can be hyphen or dashless)
            found_parent = (parent.get("page_id") or "").replace("-", "")
            want = parent_page_id.replace("-", "")
            if found_parent == want:
                return r["id"]
    return None


def create_database(token: str, parent_page_id: str, title: str, properties: dict[str, Any]) -> str:
    body = {
        "parent": {"type": "page_id", "page_id": parent_page_id},
        "title": [{"type": "text", "text": {"content": title}}],
        "properties": properties,
    }
    data = call(token, "POST", "/databases", body)
    return data["id"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", required=True)
    ap.add_argument("--parent-page-id", required=True)
    args = ap.parse_args()

    ids: dict[str, str] = {}
    for entity, schema in SCHEMAS.items():
        title = TITLES[entity]
        existing = find_existing(args.token, args.parent_page_id, title)
        if existing:
            print(f"[notion] {entity}: reusing existing db {existing}", file=sys.stderr)
            ids[entity] = existing
        else:
            db_id = create_database(args.token, args.parent_page_id, title, schema)
            print(f"[notion] {entity}: created db {db_id}", file=sys.stderr)
            ids[entity] = db_id

    print(json.dumps(ids))
    return 0


if __name__ == "__main__":
    sys.exit(main())
