"""AgentMemory — cross-agent session state backed by Redis.

Agents use this to persist intermediate results across the nodes of a
multi-step workflow so that:
  - partial results survive a mid-workflow crash (retry picks them up)
  - follow-up queries can reference prior task outputs
  - the registry /memory endpoint gives operators visibility into live sessions

Redis is used when REDIS_URL is set; otherwise falls back to an in-memory
dict (same pattern as the DB layer). TTL defaults to 1 hour.

Usage:
    from mesh.memory import AgentMemory

    memory = AgentMemory()
    await memory.set("session-abc", "plan", {"subtasks": [...]})
    plan = await memory.get("session-abc", "plan")
    all_keys = await memory.get_session("session-abc")
    await memory.clear("session-abc")
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any

logger = logging.getLogger("agentmesh.memory")

DEFAULT_TTL = 3600  # 1 hour


class AgentMemory:
    """Key-value session store backed by Redis with an in-memory fallback.

    Keys are namespaced as  <session_id>:<key>  inside Redis so multiple
    sessions never collide.
    """

    def __init__(self, redis_url: str | None = None, ttl: int = DEFAULT_TTL):
        self._redis_url = redis_url or os.environ.get("REDIS_URL")
        self._ttl = ttl
        self._redis: Any = None  # redis.asyncio.Redis, lazily initialised
        # In-memory fallback: session_id -> {key -> (value, expires_at)}
        self._store: dict[str, dict[str, tuple[Any, float]]] = {}
        self._use_redis = bool(self._redis_url)

    # ---------------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------------

    async def _get_redis(self):
        """Lazily connect to Redis, falling back to in-memory on failure."""
        if not self._use_redis:
            return None
        if self._redis is not None:
            return self._redis
        try:
            import redis.asyncio as aioredis  # noqa: PLC0415
            self._redis = aioredis.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=2,
            )
            await self._redis.ping()
            logger.info("AgentMemory: connected to Redis")
        except Exception as e:
            logger.warning(f"AgentMemory: Redis unavailable ({e}), using in-memory fallback")
            self._use_redis = False
            self._redis = None
        return self._redis

    def _ns(self, session_id: str, key: str) -> str:
        return f"agentmesh:session:{session_id}:{key}"

    def _session_pattern(self, session_id: str) -> str:
        return f"agentmesh:session:{session_id}:*"

    # Redis set that tracks all live session IDs — avoids KEYS * scans
    _SESSION_INDEX = "agentmesh:session_index"

    # ---------------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------------

    async def set(self, session_id: str, key: str, value: Any, ttl: int | None = None) -> None:
        """Store a JSON-serialisable value under session_id / key.

        Args:
            session_id: Workflow session identifier (e.g. task_id).
            key: Logical name for this piece of state (e.g. "plan", "result_analyze_csv").
            value: Any JSON-serialisable object.
            ttl: Override default TTL in seconds.
        """
        ttl = ttl if ttl is not None else self._ttl
        serialised = json.dumps(value, default=str)

        r = await self._get_redis()
        if r:
            try:
                await r.setex(self._ns(session_id, key), ttl, serialised)
                await r.sadd(self._SESSION_INDEX, session_id)
                return
            except Exception as e:
                logger.warning(f"AgentMemory.set Redis error: {e}, falling back to memory")

        # In-memory fallback
        import time  # noqa: PLC0415
        expires_at = time.time() + ttl
        if session_id not in self._store:
            self._store[session_id] = {}
        self._store[session_id][key] = (value, expires_at)

    async def get(self, session_id: str, key: str) -> Any | None:
        """Retrieve a value, or None if missing / expired.

        Args:
            session_id: Workflow session identifier.
            key: Logical key name.

        Returns:
            Stored value, or None.
        """
        r = await self._get_redis()
        if r:
            try:
                raw = await r.get(self._ns(session_id, key))
                return json.loads(raw) if raw else None
            except Exception as e:
                logger.warning(f"AgentMemory.get Redis error: {e}")

        # In-memory fallback
        import time  # noqa: PLC0415
        session = self._store.get(session_id, {})
        if key not in session:
            return None
        value, expires_at = session[key]
        if time.time() > expires_at:
            del session[key]
            return None
        return value

    async def get_session(self, session_id: str) -> dict[str, Any]:
        """Return all stored key-value pairs for a session.

        Args:
            session_id: Workflow session identifier.

        Returns:
            Dict of all non-expired key-value pairs.
        """
        r = await self._get_redis()
        if r:
            try:
                pattern = self._session_pattern(session_id)
                keys = await r.keys(pattern)
                if not keys:
                    return {}
                prefix = f"agentmesh:session:{session_id}:"
                result = {}
                for full_key in keys:
                    short_key = full_key[len(prefix):]
                    raw = await r.get(full_key)
                    if raw:
                        result[short_key] = json.loads(raw)
                return result
            except Exception as e:
                logger.warning(f"AgentMemory.get_session Redis error: {e}")

        # In-memory fallback
        import time  # noqa: PLC0415
        now = time.time()
        session = self._store.get(session_id, {})
        return {k: v for k, (v, exp) in session.items() if now <= exp}

    async def clear(self, session_id: str) -> None:
        """Delete all keys for a session.

        Args:
            session_id: Workflow session identifier.
        """
        r = await self._get_redis()
        if r:
            try:
                pattern = self._session_pattern(session_id)
                keys = await r.keys(pattern)
                if keys:
                    await r.delete(*keys)
                await r.srem(self._SESSION_INDEX, session_id)
                return
            except Exception as e:
                logger.warning(f"AgentMemory.clear Redis error: {e}")

        self._store.pop(session_id, None)

    async def list_sessions(self) -> list[str]:
        """Return all session IDs that have live keys (for the registry endpoint).

        Returns:
            List of session_id strings.
        """
        r = await self._get_redis()
        if r:
            try:
                members = await r.smembers(self._SESSION_INDEX)
                return sorted(members)
            except Exception as e:
                logger.warning(f"AgentMemory.list_sessions Redis error: {e}")

        import time  # noqa: PLC0415
        now = time.time()
        return [sid for sid, keys in self._store.items() if any(exp > now for _, exp in keys.values())]


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_memory_instance: AgentMemory | None = None
_memory_lock = threading.Lock()


def get_memory() -> AgentMemory:
    """Return the shared AgentMemory singleton."""
    global _memory_instance
    with _memory_lock:
        if _memory_instance is None:
            _memory_instance = AgentMemory()
    return _memory_instance
