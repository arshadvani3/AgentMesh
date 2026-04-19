"""Database layer for AgentMesh registry.

Provides asyncpg connection pooling and CRUD operations for agents, tasks,
trust scores, and trace events. Falls back to the in-memory store when
DATABASE_URL is not set (useful for development without Docker).

Usage:
    from mesh.db import get_db
    db = await get_db()
    await db.save_agent(record)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any

from .models import AgentManifest, AgentRecord, TraceEvent

logger = logging.getLogger("agentmesh.db")

_pool = None  # asyncpg pool singleton


async def _ensure_pool():
    """Create the asyncpg connection pool if it does not exist."""
    global _pool
    if _pool is not None:
        return _pool

    try:
        import asyncpg  # noqa: PLC0415
    except ImportError:
        logger.warning("asyncpg not installed -- using in-memory fallback")
        return None

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        return None

    try:
        _pool = await asyncpg.create_pool(database_url, min_size=2, max_size=10)
        logger.info("Connected to PostgreSQL")
        return _pool
    except Exception as e:
        logger.warning(f"PostgreSQL connection failed ({e}) -- using in-memory fallback")
        return None


# ---------------------------------------------------------------------------
# Database abstraction
# ---------------------------------------------------------------------------

class Database:
    """Thin async wrapper over asyncpg for AgentMesh persistence.

    Falls back to a local in-memory dict store when no pool is available.
    """

    def __init__(self, pool):
        """Initialize with an asyncpg pool (may be None for in-memory mode).

        Args:
            pool: asyncpg Pool or None.
        """
        self._pool = pool
        # In-memory fallback stores
        self._agents: dict[str, AgentRecord] = {}
        self._traces: list[TraceEvent] = []
        self._task_store: dict[str, dict[str, Any]] = {}

    @property
    def is_postgres(self) -> bool:
        """Return True when backed by PostgreSQL."""
        return self._pool is not None

    # ---------------------------------------------------------------------------
    # Agent CRUD
    # ---------------------------------------------------------------------------

    async def save_agent(self, record: AgentRecord) -> None:
        """Persist an agent record (insert or replace).

        Args:
            record: The AgentRecord to save.
        """
        if not self._pool:
            self._agents[record.manifest.agent_id] = record
            return

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO agents (agent_id, name, manifest, trust_score, status,
                                    registered_at, last_heartbeat, tasks_completed, tasks_failed)
                VALUES ($1, $2, $3::jsonb, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (agent_id) DO UPDATE SET
                    manifest        = EXCLUDED.manifest,
                    trust_score     = EXCLUDED.trust_score,
                    status          = EXCLUDED.status,
                    last_heartbeat  = EXCLUDED.last_heartbeat,
                    tasks_completed = EXCLUDED.tasks_completed,
                    tasks_failed    = EXCLUDED.tasks_failed
                """,
                record.manifest.agent_id,
                record.manifest.name,
                json.dumps(record.manifest.model_dump(mode="json")),
                record.trust_score,
                record.status,
                record.registered_at,
                record.last_heartbeat,
                record.tasks_completed,
                record.tasks_failed,
            )

    async def get_agent(self, agent_id: str) -> AgentRecord | None:
        """Fetch a single agent record by ID.

        Args:
            agent_id: The agent_id to look up.

        Returns:
            AgentRecord if found, else None.
        """
        if not self._pool:
            return self._agents.get(agent_id)

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM agents WHERE agent_id = $1", agent_id
            )
            return _row_to_record(row) if row else None

    async def list_agents(self) -> list[AgentRecord]:
        """Return all registered agents.

        Returns:
            List of AgentRecord objects.
        """
        if not self._pool:
            return list(self._agents.values())

        async with self._pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM agents")
            return [_row_to_record(r) for r in rows]

    async def delete_agent(self, agent_id: str) -> bool:
        """Remove an agent from the store.

        Args:
            agent_id: The agent to delete.

        Returns:
            True if the agent existed and was removed.
        """
        if not self._pool:
            if agent_id in self._agents:
                del self._agents[agent_id]
                return True
            return False

        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM agents WHERE agent_id = $1", agent_id
            )
            return result.endswith("1")

    async def update_heartbeat(self, agent_id: str, status: str = "healthy") -> None:
        """Update the last_heartbeat timestamp and status for an agent.

        Args:
            agent_id: The agent to update.
            status: New status string.
        """
        if not self._pool:
            if agent_id in self._agents:
                self._agents[agent_id].last_heartbeat = datetime.utcnow()
                self._agents[agent_id].status = status
            return

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE agents SET last_heartbeat = NOW(), status = $2
                WHERE agent_id = $1
                """,
                agent_id,
                status,
            )

    async def update_trust(
        self,
        agent_id: str,
        trust_score: float,
        tasks_completed: int,
        tasks_failed: int,
    ) -> None:
        """Update trust score and task counters for an agent.

        Args:
            agent_id: The agent to update.
            trust_score: New computed trust score.
            tasks_completed: Updated completed count.
            tasks_failed: Updated failed count.
        """
        if not self._pool:
            if agent_id in self._agents:
                rec = self._agents[agent_id]
                rec.trust_score = trust_score
                rec.tasks_completed = tasks_completed
                rec.tasks_failed = tasks_failed
            return

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE agents
                SET trust_score = $2, tasks_completed = $3, tasks_failed = $4
                WHERE agent_id = $1
                """,
                agent_id,
                trust_score,
                tasks_completed,
                tasks_failed,
            )

    # ---------------------------------------------------------------------------
    # Trace events
    # ---------------------------------------------------------------------------

    async def save_trace(self, event: TraceEvent) -> None:
        """Append a trace event to the store.

        Args:
            event: The TraceEvent to persist.
        """
        if not self._pool:
            self._traces.append(event)
            return

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO trace_events
                    (trace_id, task_id, event_type, from_agent, to_agent, payload, timestamp)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7)
                """,
                event.trace_id,
                event.task_id,
                event.event_type,
                event.from_agent,
                event.to_agent,
                json.dumps(event.payload),
                event.timestamp,
            )

    async def list_traces(
        self, task_id: str | None = None, limit: int = 100
    ) -> list[TraceEvent]:
        """Query trace events, optionally filtered by task_id.

        Args:
            task_id: Optional task ID filter.
            limit: Maximum events to return.

        Returns:
            List of TraceEvent objects.
        """
        if not self._pool:
            events = self._traces
            if task_id:
                events = [e for e in events if e.task_id == task_id]
            return events[-limit:]

        async with self._pool.acquire() as conn:
            if task_id:
                rows = await conn.fetch(
                    "SELECT * FROM trace_events WHERE task_id = $1 ORDER BY timestamp DESC LIMIT $2",
                    task_id,
                    limit,
                )
            else:
                rows = await conn.fetch(
                    "SELECT * FROM trace_events ORDER BY timestamp DESC LIMIT $1", limit
                )
            return [_row_to_trace(r) for r in rows]


# ---------------------------------------------------------------------------
# Row converters
# ---------------------------------------------------------------------------

def _row_to_record(row) -> AgentRecord:
    """Convert an asyncpg Row to an AgentRecord.

    Args:
        row: asyncpg Row from the agents table.

    Returns:
        Parsed AgentRecord.
    """
    manifest_dict = json.loads(row["manifest"]) if isinstance(row["manifest"], str) else row["manifest"]
    return AgentRecord(
        manifest=AgentManifest(**manifest_dict),
        trust_score=row["trust_score"],
        status=row["status"],
        registered_at=row["registered_at"],
        last_heartbeat=row["last_heartbeat"],
        tasks_completed=row["tasks_completed"],
        tasks_failed=row["tasks_failed"],
    )


def _row_to_trace(row) -> TraceEvent:
    """Convert an asyncpg Row to a TraceEvent.

    Args:
        row: asyncpg Row from the trace_events table.

    Returns:
        Parsed TraceEvent.
    """
    payload = json.loads(row["payload"]) if isinstance(row["payload"], str) else row["payload"]
    return TraceEvent(
        trace_id=row["trace_id"],
        task_id=row["task_id"],
        event_type=row["event_type"],
        from_agent=row["from_agent"],
        to_agent=row["to_agent"],
        payload=payload or {},
        timestamp=row["timestamp"],
    )


# ---------------------------------------------------------------------------
# Module-level singleton accessor
# ---------------------------------------------------------------------------

_db_instance: Database | None = None


async def get_db() -> Database:
    """Return the singleton Database instance, initializing it if needed.

    Returns:
        Initialized Database (PostgreSQL or in-memory fallback).
    """
    global _db_instance
    if _db_instance is None:
        pool = await _ensure_pool()
        _db_instance = Database(pool)
    return _db_instance
