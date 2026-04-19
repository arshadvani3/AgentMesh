-- Migration 001: agents table
-- Stores the manifest and runtime state for every registered agent.

CREATE TABLE IF NOT EXISTS agents (
    agent_id        TEXT        PRIMARY KEY,
    name            TEXT        NOT NULL,
    manifest        JSONB       NOT NULL,
    trust_score     REAL        NOT NULL DEFAULT 0.5,
    status          TEXT        NOT NULL DEFAULT 'healthy',
    registered_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_heartbeat  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tasks_completed INTEGER     NOT NULL DEFAULT 0,
    tasks_failed    INTEGER     NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_agents_status ON agents (status);
CREATE INDEX IF NOT EXISTS idx_agents_trust  ON agents (trust_score DESC);
