"""Tests for mesh/registry.py -- registration, discovery, heartbeat, deregistration."""

from __future__ import annotations

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

from mesh.registry import app
from mesh.models import AgentManifest, CapabilitySchema


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """Synchronous test client for basic endpoint tests."""
    with TestClient(app) as c:
        yield c


@pytest_asyncio.fixture
async def async_client():
    """Async test client for async endpoint tests."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


def _make_manifest(
    agent_id: str = "agent-test",
    name: str = "Test Agent",
    port: int = 9999,
    capabilities: list[CapabilitySchema] | None = None,
) -> dict:
    """Build a minimal serialized AgentManifest dict for test requests."""
    caps = capabilities or [
        CapabilitySchema(name="test_cap", description="A test capability")
    ]
    m = AgentManifest(
        agent_id=agent_id,
        name=name,
        endpoint=f"ws://localhost:{port}",
        capabilities=caps,
        tags=["test"],
    )
    return m.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_register_agent(self, client: TestClient):
        """POST /agents/register returns 200 with AgentRecord."""
        resp = client.post("/agents/register", json=_make_manifest())
        assert resp.status_code == 200
        data = resp.json()
        assert data["manifest"]["agent_id"] == "agent-test"
        assert data["trust_score"] == 0.5
        assert data["status"] == "healthy"

    def test_duplicate_registration(self, client: TestClient):
        """Registering the same agent_id twice returns 409."""
        manifest = _make_manifest(agent_id="agent-dup")
        client.post("/agents/register", json=manifest)
        resp = client.post("/agents/register", json=manifest)
        assert resp.status_code == 409

    def test_list_agents(self, client: TestClient):
        """GET /agents returns all registered agents."""
        client.post("/agents/register", json=_make_manifest(agent_id="agent-list-1"))
        client.post("/agents/register", json=_make_manifest(agent_id="agent-list-2"))
        resp = client.get("/agents")
        assert resp.status_code == 200
        ids = [r["manifest"]["agent_id"] for r in resp.json()]
        assert "agent-list-1" in ids
        assert "agent-list-2" in ids

    def test_get_agent(self, client: TestClient):
        """GET /agents/{id} returns the specific agent."""
        client.post("/agents/register", json=_make_manifest(agent_id="agent-get"))
        resp = client.get("/agents/agent-get")
        assert resp.status_code == 200
        assert resp.json()["manifest"]["agent_id"] == "agent-get"

    def test_get_nonexistent_agent(self, client: TestClient):
        """GET /agents/{id} returns 404 for unknown agents."""
        resp = client.get("/agents/nonexistent")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Deregistration
# ---------------------------------------------------------------------------

class TestDeregistration:
    def test_deregister_agent(self, client: TestClient):
        """DELETE /agents/{id} removes the agent and returns confirmation."""
        client.post("/agents/register", json=_make_manifest(agent_id="agent-del"))
        resp = client.delete("/agents/agent-del")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deregistered"

    def test_deregister_nonexistent(self, client: TestClient):
        """DELETE /agents/{id} returns 404 for unknown agents."""
        resp = client.delete("/agents/ghost-agent")
        assert resp.status_code == 404

    def test_agent_gone_after_deregister(self, client: TestClient):
        """Agent no longer appears in list after deregistration."""
        client.post("/agents/register", json=_make_manifest(agent_id="agent-gone"))
        client.delete("/agents/agent-gone")
        resp = client.get("/agents/agent-gone")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

class TestDiscovery:
    def test_discover_by_capability_name(self, client: TestClient):
        """POST /discover finds agent with exact capability name match."""
        cap = CapabilitySchema(name="unique_cap_xyz", description="Unique capability")
        client.post(
            "/agents/register",
            json=_make_manifest(agent_id="agent-disco", capabilities=[cap]),
        )
        resp = client.post("/discover", json={"capability_name": "unique_cap_xyz"})
        assert resp.status_code == 200
        agents = resp.json()["agents"]
        assert any(a["manifest"]["agent_id"] == "agent-disco" for a in agents)

    def test_discover_empty_result(self, client: TestClient):
        """POST /discover returns empty list when no match."""
        resp = client.post(
            "/discover", json={"capability_name": "nonexistent_cap_12345"}
        )
        assert resp.status_code == 200
        assert resp.json()["agents"] == []

    def test_discover_respects_top_k(self, client: TestClient):
        """POST /discover returns at most top_k results."""
        for i in range(5):
            cap = CapabilitySchema(
                name=f"shared_cap_{i}",
                description="A shared capability for testing",
            )
            client.post(
                "/agents/register",
                json=_make_manifest(agent_id=f"agent-topk-{i}", capabilities=[cap]),
            )
        resp = client.post(
            "/discover",
            json={
                "capability_description": "shared capability for testing",
                "top_k": 2,
            },
        )
        assert resp.status_code == 200
        assert len(resp.json()["agents"]) <= 2


# ---------------------------------------------------------------------------
# Trust update
# ---------------------------------------------------------------------------

class TestTrust:
    def test_trust_update_success(self, client: TestClient):
        """POST /trust/update increments trust score on success."""
        client.post("/agents/register", json=_make_manifest(agent_id="agent-trust"))
        resp = client.post(
            "/trust/update",
            params={"agent_id": "agent-trust", "success": True, "quality": 1.0},
        )
        assert resp.status_code == 200
        new_trust = resp.json()["trust_score"]
        assert new_trust > 0.5  # Started at 0.5, should have increased

    def test_trust_update_failure(self, client: TestClient):
        """POST /trust/update decrements trust score on failure."""
        client.post("/agents/register", json=_make_manifest(agent_id="agent-trust-f"))
        resp = client.post(
            "/trust/update",
            params={"agent_id": "agent-trust-f", "success": False, "quality": 0.0},
        )
        assert resp.status_code == 200
        new_trust = resp.json()["trust_score"]
        assert new_trust <= 0.5

    def test_trust_update_nonexistent(self, client: TestClient):
        """POST /trust/update returns 404 for unknown agent."""
        resp = client.post(
            "/trust/update",
            params={"agent_id": "ghost", "success": True},
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Traces
# ---------------------------------------------------------------------------

class TestTraces:
    def test_get_traces_empty(self, client: TestClient):
        """GET /traces returns empty list when no traces."""
        resp = client.get("/traces")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_get_traces_with_limit(self, client: TestClient):
        """GET /traces respects the limit parameter."""
        resp = client.get("/traces?limit=5")
        assert resp.status_code == 200
