-- Migration 004: trace_events table
-- Full observability log of every cross-agent interaction.

CREATE TABLE IF NOT EXISTS trace_events (
    trace_id    TEXT        PRIMARY KEY,
    task_id     TEXT        NOT NULL,
    event_type  TEXT        NOT NULL,
    from_agent  TEXT        NOT NULL,
    to_agent    TEXT        NOT NULL,
    payload     JSONB       NOT NULL DEFAULT '{}',
    timestamp   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_traces_task_id   ON trace_events (task_id);
CREATE INDEX IF NOT EXISTS idx_traces_from      ON trace_events (from_agent);
CREATE INDEX IF NOT EXISTS idx_traces_to        ON trace_events (to_agent);
CREATE INDEX IF NOT EXISTS idx_traces_timestamp ON trace_events (timestamp DESC);
