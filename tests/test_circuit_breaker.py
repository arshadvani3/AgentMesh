"""Tests for circuit breaker behaviour in the trust/availability system."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from mesh.registry import app, _failure_streaks, _degraded_since
from mesh.models import AgentManifest, CapabilitySchema


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _register(client: TestClient, agent_id: str, capability_name: str = "cap") -> dict:
    """Register a test agent and return the response JSON.

    Args:
        client: TestClient instance.
        agent_id: Agent identifier.
        capability_name: Name of the single capability to advertise.

    Returns:
        Parsed JSON response.
    """
    m = AgentManifest(
        agent_id=agent_id,
        name=agent_id,
        endpoint="ws://localhost:9999",
        capabilities=[CapabilitySchema(name=capability_name, description="test capability")],
    )
    resp = client.post("/agents/register", json=m.model_dump(mode="json"))
    assert resp.status_code == 200
    return resp.json()


def _fail(client: TestClient, agent_id: str, times: int = 1) -> None:
    """Post failure trust updates to an agent.

    Args:
        client: TestClient instance.
        agent_id: Target agent.
        times: Number of consecutive failures to submit.
    """
    for _ in range(times):
        resp = client.post(
            "/trust/update",
            params={"agent_id": agent_id, "success": False, "quality": 0.0},
        )
        assert resp.status_code == 200


def _succeed(client: TestClient, agent_id: str) -> None:
    """Post a successful trust update to an agent.

    Args:
        client: TestClient instance.
        agent_id: Target agent.
    """
    resp = client.post(
        "/trust/update",
        params={"agent_id": agent_id, "success": True, "quality": 1.0},
    )
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    # Clear circuit breaker state between tests so tests are isolated
    _failure_streaks.clear()
    _degraded_since.clear()
    with TestClient(app) as c:
        yield c
    _failure_streaks.clear()
    _degraded_since.clear()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCircuitBreaker:
    def test_three_failures_trigger_degraded(self, client: TestClient):
        """Three consecutive failures should open the circuit and set status to degraded."""
        _register(client, "cb-degrade-agent")

        # Two failures should not yet degrade
        _fail(client, "cb-degrade-agent", times=2)
        record = client.get("/agents/cb-degrade-agent").json()
        assert record["status"] == "healthy"

        # Third failure crosses the threshold
        _fail(client, "cb-degrade-agent", times=1)
        record = client.get("/agents/cb-degrade-agent").json()
        assert record["status"] == "degraded"

    def test_success_resets_circuit(self, client: TestClient):
        """A success after three failures should close the circuit and restore healthy status."""
        _register(client, "cb-reset-agent")

        # Open the circuit
        _fail(client, "cb-reset-agent", times=3)
        record = client.get("/agents/cb-reset-agent").json()
        assert record["status"] == "degraded"

        # One success resets everything
        _succeed(client, "cb-reset-agent")
        record = client.get("/agents/cb-reset-agent").json()
        assert record["status"] == "healthy"
        # Internal streak counter should also be cleared
        assert _failure_streaks.get("cb-reset-agent", 0) == 0
        assert "cb-reset-agent" not in _degraded_since

    def test_degraded_agent_still_discoverable(self, client: TestClient):
        """A degraded agent should still appear in discovery results (just ranked lower)."""
        _register(client, "cb-disc-healthy", capability_name="shared_cap")
        _register(client, "cb-disc-degraded", capability_name="shared_cap")

        # Degrade one of them
        _fail(client, "cb-disc-degraded", times=3)
        record = client.get("/agents/cb-disc-degraded").json()
        assert record["status"] == "degraded"

        # Both should show up in discovery
        resp = client.post(
            "/discover",
            json={"capability_name": "shared_cap", "top_k": 10},
        )
        assert resp.status_code == 200
        returned_ids = {a["manifest"]["agent_id"] for a in resp.json()["agents"]}
        assert "cb-disc-healthy" in returned_ids
        assert "cb-disc-degraded" in returned_ids

    def test_invalid_quality_returns_422(self, client: TestClient):
        """Quality values outside [0.0, 1.0] should be rejected with 422."""
        _register(client, "cb-quality-agent")

        resp = client.post(
            "/trust/update",
            params={"agent_id": "cb-quality-agent", "success": True, "quality": 1.5},
        )
        assert resp.status_code == 422

        resp = client.post(
            "/trust/update",
            params={"agent_id": "cb-quality-agent", "success": True, "quality": -0.1},
        )
        assert resp.status_code == 422

    def test_trust_update_nonexistent_agent_returns_404(self, client: TestClient):
        """Trust updates for unknown agents should return 404."""
        resp = client.post(
            "/trust/update",
            params={"agent_id": "ghost-agent", "success": True, "quality": 0.5},
        )
        assert resp.status_code == 404
