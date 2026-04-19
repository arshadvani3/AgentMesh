"""Tests for task negotiation flows -- accept, reject, and counter via WebSocket.

These tests exercise the MeshAgent._handle_task_request logic by standing up
a real WebSocket server (via asyncio) and sending task.request messages to it.
"""

from __future__ import annotations

import asyncio
import json

import pytest
import websockets

from mesh.models import TaskRequest, TaskStatus
from sdk.agent import MeshAgent, capability

# ---------------------------------------------------------------------------
# Test agents
# ---------------------------------------------------------------------------

class AcceptingAgent(MeshAgent):
    """Always accepts and completes tasks."""

    @capability(
        name="simple_task",
        description="A simple task that always succeeds",
        avg_latency_ms=100,
    )
    async def simple_task(self, input_data: dict) -> dict:
        """Execute a trivially simple task.

        Args:
            input_data: Arbitrary input dict.

        Returns:
            Echo of input with success flag.
        """
        return {"echo": input_data, "success": True}


class RejectingAgent(MeshAgent):
    """Has no capabilities -- always rejects."""


class ErrorAgent(MeshAgent):
    """Raises an exception during execution."""

    @capability(name="failing_task", description="A task that always fails")
    async def failing_task(self, input_data: dict) -> dict:
        """Simulate a task execution failure.

        Args:
            input_data: Arbitrary input dict.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError("Simulated failure")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


async def _start_agent(agent: MeshAgent):
    """Run the agent's WebSocket server for task listening only (no heartbeat/registry)."""
    async def handler(websocket):
        async for message in websocket:
            data = json.loads(message)
            if data.get("method") == "task.request":
                request = TaskRequest(**data["params"])
                await agent._handle_task_request(request, websocket)

    server = await websockets.serve(handler, "127.0.0.1", agent.ws_port)
    return server


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_accept_and_complete():
    """Agent with matching capability accepts and completes the task."""
    agent = AcceptingAgent(
        name="Accepting",
        registry_url="http://localhost:8000",
        ws_port=19001,
        agent_id="agent-accepting",
    )
    server = await _start_agent(agent)

    try:
        async with websockets.connect("ws://127.0.0.1:19001") as ws:
            request = TaskRequest(
                capability="simple_task",
                input_data={"key": "value"},
                requester_id="test-requester",
            )
            await ws.send(json.dumps({
                "method": "task.request",
                "params": request.model_dump(mode="json"),
            }))

            # Negotiation response
            neg_raw = await asyncio.wait_for(ws.recv(), timeout=5)
            neg = json.loads(neg_raw)["result"]
            assert neg["status"] == TaskStatus.ACCEPTED

            # Task result
            result_raw = await asyncio.wait_for(ws.recv(), timeout=10)
            result = json.loads(result_raw)["params"]
            assert result["status"] == TaskStatus.COMPLETED
            assert result["output"]["success"] is True
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_reject_unknown_capability():
    """Agent rejects tasks for capabilities it does not have."""
    agent = RejectingAgent(
        name="Rejecting",
        registry_url="http://localhost:8000",
        ws_port=19002,
        agent_id="agent-rejecting",
    )
    server = await _start_agent(agent)

    try:
        async with websockets.connect("ws://127.0.0.1:19002") as ws:
            request = TaskRequest(
                capability="nonexistent_capability",
                input_data={},
                requester_id="test-requester",
            )
            await ws.send(json.dumps({
                "method": "task.request",
                "params": request.model_dump(mode="json"),
            }))

            neg_raw = await asyncio.wait_for(ws.recv(), timeout=5)
            neg = json.loads(neg_raw)["result"]
            assert neg["status"] == TaskStatus.REJECTED
            assert "nonexistent_capability" in neg["reason"]
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_execution_failure_returns_failed_result():
    """Agent returns FAILED status when capability raises an exception."""
    agent = ErrorAgent(
        name="Error",
        registry_url="http://localhost:8000",
        ws_port=19003,
        agent_id="agent-error",
    )
    server = await _start_agent(agent)

    try:
        async with websockets.connect("ws://127.0.0.1:19003") as ws:
            request = TaskRequest(
                capability="failing_task",
                input_data={},
                requester_id="test-requester",
            )
            await ws.send(json.dumps({
                "method": "task.request",
                "params": request.model_dump(mode="json"),
            }))

            # Negotiation -- should be accepted first
            neg_raw = await asyncio.wait_for(ws.recv(), timeout=5)
            neg = json.loads(neg_raw)["result"]
            assert neg["status"] == TaskStatus.ACCEPTED

            # Result -- should be FAILED
            result_raw = await asyncio.wait_for(ws.recv(), timeout=10)
            result = json.loads(result_raw)["params"]
            assert result["status"] == TaskStatus.FAILED
            assert "Simulated failure" in result["error"]
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_multiple_sequential_tasks():
    """Agent can handle multiple tasks sequentially on the same port."""
    agent = AcceptingAgent(
        name="Accepting2",
        registry_url="http://localhost:8000",
        ws_port=19004,
        agent_id="agent-accepting2",
    )
    server = await _start_agent(agent)

    try:
        for i in range(3):
            async with websockets.connect("ws://127.0.0.1:19004") as ws:
                request = TaskRequest(
                    capability="simple_task",
                    input_data={"iteration": i},
                    requester_id="test-requester",
                )
                await ws.send(json.dumps({
                    "method": "task.request",
                    "params": request.model_dump(mode="json"),
                }))
                await asyncio.wait_for(ws.recv(), timeout=5)   # neg
                result_raw = await asyncio.wait_for(ws.recv(), timeout=10)
                result = json.loads(result_raw)["params"]
                assert result["status"] == TaskStatus.COMPLETED
    finally:
        server.close()
        await server.wait_closed()


# ---------------------------------------------------------------------------
# Counter-proposal tests
# ---------------------------------------------------------------------------

class TestCounterProposal:
    """Tests for the counter-proposal negotiation flow."""

    @pytest.mark.asyncio
    async def test_counter_proposal_when_at_capacity(self):
        """Agent returns COUNTERED with a deadline counter_proposal when at capacity."""
        agent = AcceptingAgent(
            name="Capacity1",
            registry_url="http://localhost:8000",
            ws_port=19005,
            agent_id="agent-capacity1",
            max_concurrent_tasks=1,
        )
        # Simulate being at capacity by pre-filling _active_tasks
        agent._active_tasks = {"fake-task-id"}

        server = await _start_agent(agent)
        try:
            async with websockets.connect("ws://127.0.0.1:19005") as ws:
                request = TaskRequest(
                    capability="simple_task",
                    input_data={"key": "value"},
                    requester_id="test-requester",
                    deadline_ms=30000,
                )
                await ws.send(json.dumps({
                    "method": "task.request",
                    "params": request.model_dump(mode="json"),
                }))

                neg_raw = await asyncio.wait_for(ws.recv(), timeout=5)
                neg = json.loads(neg_raw)["result"]

                assert neg["status"] == TaskStatus.COUNTERED
                assert neg["counter_proposal"] is not None
                assert "deadline_ms" in neg["counter_proposal"]
                assert neg["counter_proposal"]["reason"] == "queue_full"
                assert neg["counter_proposal"]["deadline_ms"] > 30000
        finally:
            server.close()
            await server.wait_closed()

    @pytest.mark.asyncio
    async def test_active_tasks_tracked(self):
        """_active_tasks set starts empty and supports add/discard operations."""
        agent = AcceptingAgent(
            name="Capacity2",
            registry_url="http://localhost:8000",
            ws_port=19006,
            agent_id="agent-capacity2",
            max_concurrent_tasks=5,
        )

        # Starts empty
        assert len(agent._active_tasks) == 0
        assert isinstance(agent._active_tasks, set)

        # Supports add
        agent._active_tasks.add("task-abc")
        assert "task-abc" in agent._active_tasks
        assert len(agent._active_tasks) == 1

        # Supports discard
        agent._active_tasks.discard("task-abc")
        assert "task-abc" not in agent._active_tasks
        assert len(agent._active_tasks) == 0

        # Discard on missing key is a no-op (does not raise)
        agent._active_tasks.discard("nonexistent-task")
        assert len(agent._active_tasks) == 0
