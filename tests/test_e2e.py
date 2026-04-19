"""End-to-end test: register two agents, discover one, delegate a task, verify result.

This test exercises the full path:
  1. Register agent A (with capability) on the registry
  2. Register agent B (the orchestrator) on the registry
  3. B discovers A via POST /discover
  4. B delegates a task to A via WebSocket
  5. Verify the TaskResult is COMPLETED with expected output
"""

from __future__ import annotations

import asyncio
import json

import pytest
import pytest_asyncio
import websockets
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

from mesh.models import (
    AgentManifest,
    CapabilitySchema,
    DiscoveryQuery,
    NegotiationResponse,
    TaskRequest,
    TaskResult,
    TaskStatus,
)
from mesh.registry import app
from sdk.agent import MeshAgent, capability

# ---------------------------------------------------------------------------
# Worker agent -- runs a real WebSocket server
# ---------------------------------------------------------------------------

class EchoAgent(MeshAgent):
    """Simple agent that echoes its input as output."""

    @capability(
        name="echo",
        description="Echoes the input data back as output",
        avg_latency_ms=50,
    )
    async def echo(self, input_data: dict) -> dict:
        """Return input_data unchanged.

        Args:
            input_data: Arbitrary input dict.

        Returns:
            Same dict with an 'echoed' flag.
        """
        return {**input_data, "echoed": True}


async def _run_echo_server(agent: EchoAgent) -> websockets.WebSocketServer:
    """Start the echo agent's WebSocket server.

    Args:
        agent: The EchoAgent instance.

    Returns:
        Running WebSocket server.
    """
    async def handler(ws):
        async for msg in ws:
            data = json.loads(msg)
            if data.get("method") == "task.request":
                request = TaskRequest(**data["params"])
                await agent._handle_task_request(request, ws)

    return await websockets.serve(handler, "127.0.0.1", agent.ws_port)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sync_client():
    with TestClient(app) as c:
        yield c


@pytest_asyncio.fixture
async def async_client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


# ---------------------------------------------------------------------------
# E2E test
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_full_delegation_flow(async_client: AsyncClient):
    """Full flow: register, discover, delegate, receive result."""

    # -- Step 1: Register the worker agent --
    worker = EchoAgent(
        name="Echo Worker",
        registry_url="http://localhost:8000",
        ws_port=19100,
        agent_id="agent-echo-worker",
        tags=["echo", "test"],
    )
    server = await _run_echo_server(worker)

    try:
        manifest = worker._build_manifest()
        resp = await async_client.post(
            "/agents/register",
            json=manifest.model_dump(mode="json"),
        )
        assert resp.status_code == 200

        # -- Step 2: Register the orchestrator --
        orchestrator_manifest = AgentManifest(
            agent_id="agent-orchestrator-e2e",
            name="Orchestrator",
            endpoint="ws://localhost:19200",
        )
        resp2 = await async_client.post(
            "/agents/register",
            json=orchestrator_manifest.model_dump(mode="json"),
        )
        assert resp2.status_code == 200

        # -- Step 3: Discover agent with 'echo' capability --
        query = DiscoveryQuery(capability_name="echo", top_k=5)
        resp3 = await async_client.post(
            "/discover", json=query.model_dump(mode="json")
        )
        assert resp3.status_code == 200
        discovered = resp3.json()["agents"]
        assert len(discovered) >= 1
        assert any(
            a["manifest"]["agent_id"] == "agent-echo-worker" for a in discovered
        )

        # -- Step 4: Delegate task via WebSocket --
        request = TaskRequest(
            capability="echo",
            input_data={"message": "hello mesh", "count": 42},
            requester_id="agent-orchestrator-e2e",
            target_id="agent-echo-worker",
        )

        async with websockets.connect("ws://127.0.0.1:19100") as ws:
            await ws.send(json.dumps({
                "method": "task.request",
                "params": request.model_dump(mode="json"),
            }))

            # Negotiation
            neg_raw = await asyncio.wait_for(ws.recv(), timeout=5)
            neg = NegotiationResponse(**json.loads(neg_raw)["result"])
            assert neg.status == TaskStatus.ACCEPTED

            # Result
            result_raw = await asyncio.wait_for(ws.recv(), timeout=10)
            result = TaskResult(**json.loads(result_raw)["params"])

            # -- Step 5: Verify --
            assert result.status == TaskStatus.COMPLETED
            assert result.output["message"] == "hello mesh"
            assert result.output["count"] == 42
            assert result.output["echoed"] is True
            assert result.execution_time_ms >= 0

        # -- Verify trust update flow --
        resp4 = await async_client.post(
            "/trust/update",
            params={
                "agent_id": "agent-echo-worker",
                "success": True,
                "quality": 1.0,
            },
        )
        assert resp4.status_code == 200
        assert resp4.json()["trust_score"] > 0.5

    finally:
        server.close()
        await server.wait_closed()

        # Cleanup registry
        await async_client.delete("/agents/agent-echo-worker")
        await async_client.delete("/agents/agent-orchestrator-e2e")


@pytest.mark.asyncio
async def test_discovery_finds_no_agent_when_none_registered(async_client: AsyncClient):
    """Discovery returns empty list when no matching agents are registered."""
    query = DiscoveryQuery(capability_name="nonexistent_unique_cap_99999")
    resp = await async_client.post("/discover", json=query.model_dump(mode="json"))
    assert resp.status_code == 200
    assert resp.json()["agents"] == []


@pytest.mark.asyncio
async def test_delegation_rejected_for_wrong_capability(async_client: AsyncClient):
    """Agent correctly rejects tasks for capabilities it does not own."""
    worker = EchoAgent(
        name="Echo Worker 2",
        registry_url="http://localhost:8000",
        ws_port=19101,
        agent_id="agent-echo-worker-2",
    )
    server = await _run_echo_server(worker)

    try:
        manifest = worker._build_manifest()
        await async_client.post("/agents/register", json=manifest.model_dump(mode="json"))

        request = TaskRequest(
            capability="analyze_csv",  # Wrong capability
            input_data={},
            requester_id="test",
        )
        async with websockets.connect("ws://127.0.0.1:19101") as ws:
            await ws.send(json.dumps({
                "method": "task.request",
                "params": request.model_dump(mode="json"),
            }))
            neg_raw = await asyncio.wait_for(ws.recv(), timeout=5)
            neg = NegotiationResponse(**json.loads(neg_raw)["result"])
            assert neg.status == TaskStatus.REJECTED
    finally:
        server.close()
        await server.wait_closed()
        await async_client.delete("/agents/agent-echo-worker-2")
