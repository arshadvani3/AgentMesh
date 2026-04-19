-- Migration 003: trust_scores table
-- Append-only log of every trust update event for audit and convergence analysis.

CREATE TABLE IF NOT EXISTS trust_scores (
    id              BIGSERIAL   PRIMARY KEY,
    agent_id        TEXT        NOT NULL REFERENCES agents(agent_id) ON DELETE CASCADE,
    task_id         TEXT        NOT NULL,
    reviewer_id     TEXT        NOT NULL,
    success         BOOLEAN     NOT NULL,
    quality_score   REAL        NOT NULL,
    latency_ms      INTEGER     NOT NULL,
    old_trust       REAL        NOT NULL,
    new_trust       REAL        NOT NULL,
    recorded_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trust_agent_id   ON trust_scores (agent_id);
CREATE INDEX IF NOT EXISTS idx_trust_recorded   ON trust_scores (recorded_at DESC);
