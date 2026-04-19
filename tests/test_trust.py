"""Tests for trust score updates -- ELO-style convergence behavior."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from mesh.models import AgentManifest, CapabilitySchema
from mesh.registry import app

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _register(client: TestClient, agent_id: str) -> dict:
    """Register a test agent and return the response JSON.

    Args:
        client: TestClient instance.
        agent_id: Agent identifier.

    Returns:
        Parsed JSON response.
    """
    m = AgentManifest(
        agent_id=agent_id,
        name=agent_id,
        endpoint="ws://localhost:9999",
        capabilities=[CapabilitySchema(name="cap", description="desc")],
    )
    resp = client.post("/agents/register", json=m.model_dump(mode="json"))
    assert resp.status_code == 200
    return resp.json()


def _update(client: TestClient, agent_id: str, success: bool, quality: float) -> float:
    """Send a trust update and return the new trust score.

    Args:
        client: TestClient instance.
        agent_id: Target agent.
        success: Whether the task succeeded.
        quality: Quality score [0, 1].

    Returns:
        Updated trust_score.
    """
    resp = client.post(
        "/trust/update",
        params={"agent_id": agent_id, "success": success, "quality": quality},
    )
    assert resp.status_code == 200
    return resp.json()["trust_score"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


class TestTrustUpdates:
    def test_success_increases_trust(self, client: TestClient):
        """Trust score increases after a successful high-quality task."""
        _register(client, "agent-trust-up")
        new_trust = _update(client, "agent-trust-up", success=True, quality=1.0)
        assert new_trust > 0.5

    def test_failure_decreases_trust(self, client: TestClient):
        """Trust score decreases after a failed task."""
        _register(client, "agent-trust-down")
        new_trust = _update(client, "agent-trust-down", success=False, quality=0.0)
        assert new_trust < 0.5

    def test_trust_bounded_above(self, client: TestClient):
        """Trust score never exceeds 1.0."""
        _register(client, "agent-bounded-hi")
        trust = 0.5
        for _ in range(50):
            trust = _update(client, "agent-bounded-hi", success=True, quality=1.0)
        assert trust <= 1.0

    def test_trust_bounded_below(self, client: TestClient):
        """Trust score never drops below 0.0."""
        _register(client, "agent-bounded-lo")
        trust = 0.5
        for _ in range(50):
            trust = _update(client, "agent-bounded-lo", success=False, quality=0.0)
        assert trust >= 0.0

    def test_convergence_to_high_with_perfect_quality(self, client: TestClient):
        """Repeated perfect tasks push trust toward 1.0."""
        _register(client, "agent-conv-hi")
        trust = 0.5
        for _ in range(30):
            trust = _update(client, "agent-conv-hi", success=True, quality=1.0)
        assert trust > 0.8

    def test_convergence_to_low_with_failures(self, client: TestClient):
        """Repeated failures push trust below 0.3."""
        _register(client, "agent-conv-lo")
        trust = 0.5
        for _ in range(30):
            trust = _update(client, "agent-conv-lo", success=False, quality=0.0)
        assert trust < 0.3

    def test_mixed_results_stabilize(self, client: TestClient):
        """Alternating success and failure with equal quality stabilizes near 0.5."""
        _register(client, "agent-stable")
        trust = 0.5
        for i in range(20):
            if i % 2 == 0:
                trust = _update(client, "agent-stable", success=True, quality=0.5)
            else:
                trust = _update(client, "agent-stable", success=False, quality=0.5)
        # Should remain somewhere in the middle range
        assert 0.2 <= trust <= 0.8

    def test_low_quality_success_has_small_effect(self, client: TestClient):
        """Success with quality below current trust still moves the score.

        quality=0.1 < starting trust=0.5, so the ELO update pulls trust down
        slightly (actual < expected). The change should be small (within 0.1).
        """
        _register(client, "agent-low-qual")
        trust_after = _update(client, "agent-low-qual", success=True, quality=0.1)
        assert abs(trust_after - 0.5) < 0.1  # small change either direction

    def test_task_counters_increment(self, client: TestClient):
        """Completing tasks increments the correct counter."""
        _register(client, "agent-counter")
        _update(client, "agent-counter", success=True, quality=1.0)
        _update(client, "agent-counter", success=True, quality=1.0)
        _update(client, "agent-counter", success=False, quality=0.0)

        resp = client.get("/agents/agent-counter")
        record = resp.json()
        assert record["tasks_completed"] == 2
        assert record["tasks_failed"] == 1
