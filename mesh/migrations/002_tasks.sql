-- Migration 002: tasks table
-- Records every task request and its final result.

CREATE TABLE IF NOT EXISTS tasks (
    task_id         TEXT        PRIMARY KEY,
    capability      TEXT        NOT NULL,
    requester_id    TEXT        NOT NULL,
    executor_id     TEXT,
    input_data      JSONB       NOT NULL DEFAULT '{}',
    output_data     JSONB,
    status          TEXT        NOT NULL DEFAULT 'pending',
    error           TEXT,
    tokens_used     INTEGER     NOT NULL DEFAULT 0,
    execution_time_ms INTEGER,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at    TIMESTAMPTZ,
    deadline_ms     INTEGER     NOT NULL DEFAULT 30000,
    priority        SMALLINT    NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_tasks_requester ON tasks (requester_id);
CREATE INDEX IF NOT EXISTS idx_tasks_executor  ON tasks (executor_id);
CREATE INDEX IF NOT EXISTS idx_tasks_status    ON tasks (status);
CREATE INDEX IF NOT EXISTS idx_tasks_created   ON tasks (created_at DESC);
