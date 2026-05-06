"""Tests for live latency tracking and max_latency_ms routing filter."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from mesh.models import AgentManifest, CapabilitySchema, DiscoveryQuery
from mesh.registry import _latency_samples, app
from mesh.router import TaskRouter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _register(
    client: TestClient,
    agent_id: str,
    capability_name: str = "cap",
    avg_latency_ms: float | None = None,
) -> dict:
    cap = CapabilitySchema(
        name=capability_name,
        description="test capability for latency",
        avg_latency_ms=avg_latency_ms,
    )
    m = AgentManifest(
        agent_id=agent_id,
        name=agent_id,
        endpoint="ws://localhost:9999",
        capabilities=[cap],
    )
    resp = client.post("/agents/register", json=m.model_dump(mode="json"))
    assert resp.status_code == 200
    return resp.json()


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    _latency_samples.clear()
    with TestClient(app) as c:
        yield c
    _latency_samples.clear()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLatencyFilter:
    def test_max_latency_excludes_slow_agents(self, client: TestClient):
        """Agents with avg_latency_ms above max_latency_ms should be excluded."""
        _register(client, "lat-fast", capability_name="lat_cap", avg_latency_ms=500.0)
        _register(client, "lat-slow", capability_name="lat_cap", avg_latency_ms=9000.0)

        resp = client.post(
            "/discover",
            json={"capability_name": "lat_cap", "max_latency_ms": 1000.0, "top_k": 10},
        )
        assert resp.status_code == 200
        returned_ids = {a["manifest"]["agent_id"] for a in resp.json()["agents"]}
        assert "lat-fast" in returned_ids
        assert "lat-slow" not in returned_ids

    def test_no_latency_filter_returns_all(self, client: TestClient):
        """Without max_latency_ms, both fast and slow agents are returned."""
        _register(client, "lat-all-fast", capability_name="lat_all_cap", avg_latency_ms=200.0)
        _register(client, "lat-all-slow", capability_name="lat_all_cap", avg_latency_ms=15000.0)

        resp = client.post(
            "/discover",
            json={"capability_name": "lat_all_cap", "top_k": 10},
        )
        assert resp.status_code == 200
        returned_ids = {a["manifest"]["agent_id"] for a in resp.json()["agents"]}
        assert "lat-all-fast" in returned_ids
        assert "lat-all-slow" in returned_ids

    def test_agents_without_latency_data_not_filtered(self, client: TestClient):
        """Agents with no avg_latency_ms set should pass the latency filter."""
        _register(client, "lat-unknown", capability_name="lat_unk_cap", avg_latency_ms=None)

        resp = client.post(
            "/discover",
            json={"capability_name": "lat_unk_cap", "max_latency_ms": 100.0, "top_k": 10},
        )
        assert resp.status_code == 200
        returned_ids = {a["manifest"]["agent_id"] for a in resp.json()["agents"]}
        # No latency data → not filtered out
        assert "lat-unknown" in returned_ids


class TestLatencyRollingAverage:
    def test_router_latency_filter_direct(self):
        """Router's latency filter works correctly on the indexed data."""
        router = TaskRouter()

        fast_manifest = AgentManifest(
            agent_id="router-lat-fast",
            name="fast",
            endpoint="ws://localhost:9001",
            capabilities=[CapabilitySchema(name="speed_cap", description="fast agent", avg_latency_ms=300.0)],
        )
        slow_manifest = AgentManifest(
            agent_id="router-lat-slow",
            name="slow",
            endpoint="ws://localhost:9002",
            capabilities=[CapabilitySchema(name="speed_cap", description="fast agent", avg_latency_ms=8000.0)],
        )

        from mesh.models import AgentRecord
        fast_record = AgentRecord(manifest=fast_manifest, trust_score=0.7)
        slow_record = AgentRecord(manifest=slow_manifest, trust_score=0.7)

        import asyncio

        async def run():
            await router.index_agent(fast_record)
            await router.index_agent(slow_record)
            query = DiscoveryQuery(capability_name="speed_cap", max_latency_ms=1000.0)
            results = await router.match(query, [fast_record, slow_record])
            return results

        results = asyncio.run(run())
        result_ids = [r.manifest.agent_id for r in results]
        assert "router-lat-fast" in result_ids
        assert "router-lat-slow" not in result_ids
