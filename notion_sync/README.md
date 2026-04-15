# Notion sync

Two-way mirror for the human-facing subset of super-brain artifacts.

| Entity          | Mirrored to Notion? | Notion DB env var          |
|-----------------|---------------------|----------------------------|
| decisions       | yes                 | `NOTION_DB_DECISIONS`      |
| procedures      | yes                 | `NOTION_DB_PROCEDURES`     |
| opportunities   | yes                 | `NOTION_DB_OPPORTUNITIES`  |
| all 11 others   | **no**              | —                          |

The substrate (Cloudflare D1 / Vectorize / R2) is always the system of record.
Notion is a UI mirror. Conflict rule: **substrate wins**. Manual edits in
Notion become *new* substrate memories (tagged `source:notion`) on the next
pull — they do not overwrite existing memories.

## One-time Notion setup

1. Create an internal integration at `https://www.notion.so/my-integrations`.
   Copy the "Internal Integration Secret" — that's `NOTION_API_KEY`.
2. Create three databases (or reuse existing ones) with the columns below.
3. Share each database with the integration you just created.
4. Grab each database id from the URL (32-char hex after the workspace slug)
   and put them in your environment.

### Column schema

#### `decisions`
| Column       | Type              |
|--------------|-------------------|
| Title        | Title             |
| Hash         | Text              |
| TLDR         | Rich text         |
| Rationale    | Rich text         |
| Supersedes   | Text              |
| Valid From   | Date              |
| Valid To     | Date              |
| Source       | URL               |

#### `procedures`
| Column  | Type         |
|---------|--------------|
| Title   | Title        |
| Hash    | Text         |
| TLDR    | Rich text    |
| Steps   | Rich text    |
| Alpha   | Number       |
| Beta    | Number       |
| N Runs  | Number       |

#### `opportunities`
| Column                   | Type                                        |
|--------------------------|---------------------------------------------|
| Title                    | Title                                       |
| Hash                     | Text                                        |
| TLDR                     | Rich text                                   |
| Speed to First Dollar    | Select (options: `now`, `days`, `weeks`, `months`) |
| Automation Leverage      | Select (options: `low`, `medium`, `high`)   |
| Score                    | Number                                      |

The sync script creates new pages but does NOT create databases or add
missing columns — do that once manually so the schema is your commitment,
not the script's assumption.

## Environment

```bash
export NOTION_API_KEY="secret_…"
export NOTION_DB_DECISIONS="<db id>"
export NOTION_DB_PROCEDURES="<db id>"
export NOTION_DB_OPPORTUNITIES="<db id>"

# Optional
export NOTION_SYNC_DRY_RUN=1    # log-only, no writes
```

The script reads the same storage env as the server:
`MCP_MEMORY_STORAGE_BACKEND`, `CLOUDFLARE_*`, etc. Any config that works
for `python -m mcp_memory_service.server` works here.

## Running the sync

```bash
# All three entities, push + pull
python notion_sync/sync.py --entity all

# Just decisions
python notion_sync/sync.py --entity decisions

# Verbose
python notion_sync/sync.py --entity procedures -v
```

Output looks like:

```
decisions     pushed=  7  pulled=  0  skipped=  2  errors=  0
procedures    pushed=  3  pulled=  1  skipped=  0  errors=  0
opportunities pushed= 12  pulled=  0  skipped=  4  errors=  0
```

- **pushed** — substrate → Notion (new or updated by Hash match)
- **pulled** — Notion → substrate (pages with no Hash yet; materialized as
  new memories tagged `source:notion`)
- **skipped** — already in sync, nothing to do
- **errors** — logged; check stderr for details

## Scheduling

The script is safe to run repeatedly (idempotent by `content_hash`). Pick
whichever scheduler fits:

- **Codex Automation** — create a scheduled Automation in the Codex desktop
  app that runs the command above nightly.
- **GitHub Action** — commit a workflow that pulls secrets from the repo
  and runs `python notion_sync/sync.py` on a cron.
- **Railway cron** — the same Dockerfile we use for the server can run a
  cron service that executes this script.

## Failure modes

| Symptom                                  | Cause                                            | Fix                                                        |
|------------------------------------------|--------------------------------------------------|------------------------------------------------------------|
| `missing NOTION_API_KEY`                 | env not loaded                                    | source `.env` or export                                    |
| `notion_client` import error             | SDK not installed                                 | `pip install notion-client`                                |
| `property Hash does not exist`           | DB schema missing the Hash column                 | add a Text column named exactly `Hash`                     |
| Pages multiply on every run              | Hash column not populated on push                 | check the integration has write perms                      |
| Pulls create duplicate memories          | Notion page edited, then pushed again             | expected — substrate wins, new page is a new memory        |

## Design notes

- Substrate query uses the `entity:<name>` tag, so the script does not need
  direct access to D1/Vectorize internals — just the storage interface.
- Long rich-text is chunked to Notion's 2000-char-per-item cap.
- Pull is conservative: if a Notion page already has a `Hash` value we
  assume the substrate created it and skip. Only fresh pages get pulled.
- No API is exposed back into Notion beyond the three databases — that's a
  deliberate containment. If you want broader mirroring, write another
  script; don't generalize this one.
