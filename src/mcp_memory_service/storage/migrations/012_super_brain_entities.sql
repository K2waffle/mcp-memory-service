-- 012_super_brain_entities.sql
-- Super-Brain extension: typed tables for decisions, task_packets, and
-- object_artifacts. All other 11 entity types live in the base memories
-- table with memory_type + tag conventions (thin schema). Tables added
-- here use the sb_ prefix to namespace cleanly from upstream.
--
-- Compatible with both sqlite-vec (local) and Cloudflare D1 (cloud).
-- All migrations are additive — no existing columns/tables modified.

-- ---------------------------------------------------------------------------
-- decisions — supersede-linked decision records with temporal validity
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sb_decisions (
    id             TEXT PRIMARY KEY,                 -- ULID, matches memory content_hash when 1:1
    memory_id      TEXT NOT NULL,                    -- link to memories.content_hash
    rationale      TEXT NOT NULL,
    supersedes_id  TEXT,                             -- prior decision this replaces (nullable)
    valid_from     TEXT NOT NULL,                    -- ISO 8601 UTC
    valid_to       TEXT,                             -- ISO 8601 UTC, null = open-ended
    scope          TEXT,                             -- free-form scope marker
    created_at     TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE INDEX IF NOT EXISTS idx_sb_decisions_memory
    ON sb_decisions(memory_id);
CREATE INDEX IF NOT EXISTS idx_sb_decisions_validity
    ON sb_decisions(valid_from, valid_to);
CREATE INDEX IF NOT EXISTS idx_sb_decisions_supersedes
    ON sb_decisions(supersedes_id);

-- ---------------------------------------------------------------------------
-- task_packets — structured goal/input/output/citation/cost records
-- born episodic; promoted via consolidation
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sb_task_packets (
    id               TEXT PRIMARY KEY,
    memory_id        TEXT NOT NULL,
    goal             TEXT NOT NULL,
    inputs_json      TEXT,                           -- JSON array
    outputs_json     TEXT,                           -- JSON array
    citations_json   TEXT,                           -- JSON array of {url, trust_tier}
    cost_tokens_in   INTEGER,
    cost_tokens_out  INTEGER,
    cost_dollars     REAL,
    wall_ms          INTEGER,
    tool_calls       INTEGER,
    retries          INTEGER,
    outcome          TEXT CHECK(outcome IN ('success','partial','failure','abandoned')),
    verifier_id      TEXT,                           -- id of cross-ref verifier run
    created_at       TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE INDEX IF NOT EXISTS idx_sb_task_packets_memory
    ON sb_task_packets(memory_id);
CREATE INDEX IF NOT EXISTS idx_sb_task_packets_outcome
    ON sb_task_packets(outcome);
CREATE INDEX IF NOT EXISTS idx_sb_task_packets_created
    ON sb_task_packets(created_at);

-- ---------------------------------------------------------------------------
-- object_artifacts — binary/URL pointer with R2 object key, mime, checksum
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sb_object_artifacts (
    id               TEXT PRIMARY KEY,
    memory_id        TEXT NOT NULL,
    r2_bucket        TEXT,
    r2_object_key    TEXT,
    url              TEXT,
    mime             TEXT,
    size_bytes       INTEGER,
    checksum_sha256  TEXT,
    created_at       TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE INDEX IF NOT EXISTS idx_sb_object_artifacts_memory
    ON sb_object_artifacts(memory_id);
CREATE INDEX IF NOT EXISTS idx_sb_object_artifacts_checksum
    ON sb_object_artifacts(checksum_sha256);

-- ---------------------------------------------------------------------------
-- outcome scoring — Beta(alpha, beta) distribution per procedure
-- supports learning/ module procedure-ranking
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sb_procedure_scores (
    memory_id        TEXT PRIMARY KEY,               -- procedure memory.content_hash
    alpha            REAL NOT NULL DEFAULT 1.0,      -- success pseudo-count
    beta             REAL NOT NULL DEFAULT 1.0,      -- failure pseudo-count
    last_used_at     TEXT,
    n_runs           INTEGER NOT NULL DEFAULT 0,
    updated_at       TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

-- ---------------------------------------------------------------------------
-- revenue events — idempotent revenue-recognition records
-- payment-rail agnostic; counterparty_ref is opaque
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sb_revenue_events (
    id                   TEXT PRIMARY KEY,           -- revenue_event_id (idempotency key)
    memory_id            TEXT,                       -- opportunity / task_packet that generated it
    counterparty_ref     TEXT,                       -- Stripe customer id / wallet addr / etc.
    counterparty_rail    TEXT CHECK(counterparty_rail IN
                           ('stripe','crypto','ach','wire','paypal','platform_credit','other')),
    amount_cents         INTEGER NOT NULL,
    currency             TEXT NOT NULL DEFAULT 'USD',
    margin               REAL,                       -- 0.0-1.0 gross margin estimate
    recognized_at        TEXT NOT NULL,              -- ISO 8601 UTC
    pci_scope            TEXT CHECK(pci_scope IN ('saq_a','saq_a_ep','saq_d','out_of_scope')),
    created_at           TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE INDEX IF NOT EXISTS idx_sb_revenue_events_memory
    ON sb_revenue_events(memory_id);
CREATE INDEX IF NOT EXISTS idx_sb_revenue_events_counterparty
    ON sb_revenue_events(counterparty_ref);
CREATE INDEX IF NOT EXISTS idx_sb_revenue_events_recognized
    ON sb_revenue_events(recognized_at);

-- ---------------------------------------------------------------------------
-- retention policies — named retention windows for legal/compliance
-- referenced by memory.metadata.retention_policy_id
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sb_retention_policies (
    id                 TEXT PRIMARY KEY,
    name               TEXT NOT NULL UNIQUE,
    min_days           INTEGER,                      -- must retain at least this many days
    max_days           INTEGER,                      -- must purge after this many days
    jurisdiction       TEXT,                         -- ISO 3166 or 'global'
    lawful_basis       TEXT,
    description        TEXT,
    created_at         TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

-- ---------------------------------------------------------------------------
-- per-subject encryption key registry for crypto-shredding (GDPR Art. 17)
-- the actual key material is NOT stored here — this is a registry of IDs and
-- their status. Key material lives in the secret manager (e.g. CF Secrets).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sb_subject_keys (
    id                 TEXT PRIMARY KEY,             -- key id; referenced by per_subject_key_id
    subject_ref        TEXT NOT NULL,                -- opaque subject identifier (hashed)
    jurisdiction       TEXT,
    created_at         TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    revoked_at         TEXT,                         -- set to erase: destroying the secret
                                                     -- is the actual shred; this is the audit trail
    revocation_reason  TEXT
);

CREATE INDEX IF NOT EXISTS idx_sb_subject_keys_subject
    ON sb_subject_keys(subject_ref);
CREATE INDEX IF NOT EXISTS idx_sb_subject_keys_revoked
    ON sb_subject_keys(revoked_at);
