"""Tests for AgentMemory — Redis-backed session state with in-memory fallback."""

from __future__ import annotations

import time

import pytest

from mesh.memory import AgentMemory


@pytest.fixture
def memory() -> AgentMemory:
    """Fresh AgentMemory instance with in-memory fallback."""
    return AgentMemory(redis_url=None)


class TestSetAndGet:
    @pytest.mark.asyncio
    async def test_set_and_get_dict(self, memory: AgentMemory):
        await memory.set("s1", "plan", {"subtasks": ["a", "b"]})
        result = await memory.get("s1", "plan")
        assert result == {"subtasks": ["a", "b"]}

    @pytest.mark.asyncio
    async def test_set_and_get_list(self, memory: AgentMemory):
        await memory.set("s2", "items", [1, 2, 3])
        result = await memory.get("s2", "items")
        assert result == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_get_missing_key_returns_none(self, memory: AgentMemory):
        result = await memory.get("no-session", "no-key")
        assert result is None

    @pytest.mark.asyncio
    async def test_overwrite_existing_key(self, memory: AgentMemory):
        await memory.set("s3", "k", "first")
        await memory.set("s3", "k", "second")
        result = await memory.get("s3", "k")
        assert result == "second"

    @pytest.mark.asyncio
    async def test_different_sessions_are_isolated(self, memory: AgentMemory):
        await memory.set("sess-A", "key", "val-A")
        await memory.set("sess-B", "key", "val-B")
        a = await memory.get("sess-A", "key")
        b = await memory.get("sess-B", "key")
        assert a == "val-A"
        assert b == "val-B"


class TestGetSession:
    @pytest.mark.asyncio
    async def test_get_session_returns_all_keys(self, memory: AgentMemory):
        await memory.set("sx", "plan", {"q": "hello"})
        await memory.set("sx", "result_analyze_csv", {"rows": 100})
        session = await memory.get_session("sx")
        assert set(session.keys()) == {"plan", "result_analyze_csv"}
        assert session["plan"] == {"q": "hello"}

    @pytest.mark.asyncio
    async def test_get_session_empty_when_no_keys(self, memory: AgentMemory):
        result = await memory.get_session("ghost")
        assert result == {}


class TestClear:
    @pytest.mark.asyncio
    async def test_clear_removes_all_keys(self, memory: AgentMemory):
        await memory.set("sc", "a", 1)
        await memory.set("sc", "b", 2)
        await memory.clear("sc")
        session = await memory.get_session("sc")
        assert session == {}

    @pytest.mark.asyncio
    async def test_clear_nonexistent_session_is_noop(self, memory: AgentMemory):
        await memory.clear("does-not-exist")  # should not raise


class TestTTL:
    @pytest.mark.asyncio
    async def test_expired_key_returns_none(self, memory: AgentMemory):
        await memory.set("ttl-sess", "expired", "gone", ttl=60)
        # Back-date the stored timestamp to simulate expiry
        memory._store["ttl-sess"]["expired"] = (
            memory._store["ttl-sess"]["expired"][0],
            time.time() - 1,
        )
        result = await memory.get("ttl-sess", "expired")
        assert result is None

    @pytest.mark.asyncio
    async def test_non_expired_key_is_returned(self, memory: AgentMemory):
        await memory.set("ttl-live", "alive", "yes", ttl=3600)
        result = await memory.get("ttl-live", "alive")
        assert result == "yes"


class TestListSessions:
    @pytest.mark.asyncio
    async def test_list_sessions_includes_active(self, memory: AgentMemory):
        await memory.set("live-sess", "k", "v")
        sessions = await memory.list_sessions()
        assert "live-sess" in sessions

    @pytest.mark.asyncio
    async def test_list_sessions_excludes_cleared(self, memory: AgentMemory):
        await memory.set("tmp-sess", "k", "v")
        await memory.clear("tmp-sess")
        sessions = await memory.list_sessions()
        assert "tmp-sess" not in sessions


class TestRegistryEndpoints:
    @pytest.mark.asyncio
    async def test_memory_endpoint_returns_session_data(self):
        from fastapi.testclient import TestClient

        import mesh.memory as mem_module
        from mesh.registry import app

        test_memory = AgentMemory(redis_url=None)
        await test_memory.set("reg-sess", "plan", {"ok": True})
        original = mem_module._memory_instance
        mem_module._memory_instance = test_memory

        try:
            with TestClient(app) as client:
                resp = client.get("/memory/reg-sess")
                assert resp.status_code == 200
                data = resp.json()
                assert data["session_id"] == "reg-sess"
                assert "plan" in data["keys"]
                assert data["data"]["plan"] == {"ok": True}
        finally:
            mem_module._memory_instance = original

    @pytest.mark.asyncio
    async def test_memory_list_endpoint(self):
        from fastapi.testclient import TestClient

        import mesh.memory as mem_module
        from mesh.registry import app

        test_memory = AgentMemory(redis_url=None)
        await test_memory.set("list-sess", "k", "v")
        original = mem_module._memory_instance
        mem_module._memory_instance = test_memory

        try:
            with TestClient(app) as client:
                resp = client.get("/memory")
                assert resp.status_code == 200
                assert "list-sess" in resp.json()["sessions"]
        finally:
            mem_module._memory_instance = original
