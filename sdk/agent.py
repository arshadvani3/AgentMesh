"""AgentMesh SDK — import this to make any agent mesh-aware.

Usage:
    from sdk.agent import MeshAgent, capability

    class MyAgent(MeshAgent):
        @capability(
            name="analyze_csv",
            description="Statistical analysis of CSV datasets",
            input_schema={"type": "object", "properties": {"data_url": {"type": "string"}}},
            output_schema={"type": "object", "properties": {"report": {"type": "string"}}},
        )
        async def analyze(self, input_data: dict) -> dict:
            # ... your logic here ...
            return {"report": "..."}

    agent = MyAgent(name="Data Agent", registry_url="http://localhost:8000")
    await agent.start()
"""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.parse
from collections.abc import Callable, Coroutine
from datetime import datetime
from typing import Any

import httpx
import websockets

from mesh.models import (
    AgentManifest,
    AgentRecord,
    CapabilitySchema,
    DiscoveryQuery,
    DiscoveryResult,
    NegotiationResponse,
    TaskRequest,
    TaskResult,
    TaskStatus,
)

logger = logging.getLogger("agentmesh.sdk")


# ---------------------------------------------------------------------------
# @capability decorator
# ---------------------------------------------------------------------------

def capability(
    name: str,
    description: str,
    input_schema: dict | None = None,
    output_schema: dict | None = None,
    avg_latency_ms: float | None = None,
    cost_per_call_usd: float | None = None,
):
    """Decorator to register a method as a mesh capability."""

    def decorator(func: Callable[..., Coroutine]):
        func._mesh_capability = CapabilitySchema(  # type: ignore[attr-defined]
            name=name,
            description=description,
            input_schema=input_schema or {},
            output_schema=output_schema or {},
            avg_latency_ms=avg_latency_ms,
            cost_per_call_usd=cost_per_call_usd,
        )
        return func

    return decorator


# ---------------------------------------------------------------------------
# MeshAgent base class
# ---------------------------------------------------------------------------

class MeshAgent:
    """Base class for all mesh-aware agents.

    Subclass this, add @capability methods, and call start().
    The agent will:
    1. Register with the mesh registry
    2. Maintain a heartbeat
    3. Listen for incoming task requests
    4. Provide discover() and delegate() for outbound collaboration
    """

    def __init__(
        self,
        name: str,
        registry_url: str = "http://localhost:8000",
        ws_port: int = 9000,
        agent_id: str | None = None,
        tags: list[str] | None = None,
        mcp_servers: list[str] | None = None,
        max_concurrent_tasks: int = 3,
    ):
        self.name = name
        self.registry_url = registry_url.rstrip("/")
        self.ws_port = ws_port
        self.tags = tags or []
        self.mcp_servers = mcp_servers or []
        self._capabilities: dict[str, Callable] = {}
        self._agent_id = agent_id
        self._record: AgentRecord | None = None
        self._heartbeat_ws = None
        self._running = False
        self._max_concurrent_tasks = max_concurrent_tasks
        self._active_tasks: set[str] = set()

        # Auto-discover @capability decorated methods
        for attr_name in dir(self):
            attr = getattr(self, attr_name, None)
            if callable(attr) and hasattr(attr, "_mesh_capability"):
                cap: CapabilitySchema = attr._mesh_capability
                self._capabilities[cap.name] = attr

    @property
    def agent_id(self) -> str:
        return self._agent_id or (self._record.manifest.agent_id if self._record else "unregistered")

    def _build_manifest(self) -> AgentManifest:
        caps = [fn._mesh_capability for fn in self._capabilities.values()]  # type: ignore[attr-defined]
        return AgentManifest(  # type: ignore[call-arg]
            agent_id=self._agent_id or f"agent-{self.name.lower().replace(' ', '-')}",
            name=self.name,
            capabilities=caps,
            mcp_servers=self.mcp_servers,
            endpoint=f"ws://localhost:{self.ws_port}",
            tags=self.tags,
            max_concurrent_tasks=self._max_concurrent_tasks,
        )

    # --- Lifecycle ---

    async def start(self):
        """Register with mesh and start listening."""
        manifest = self._build_manifest()
        self._agent_id = manifest.agent_id

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.registry_url}/agents/register",
                json=manifest.model_dump(mode="json"),
            )
            resp.raise_for_status()
            self._record = AgentRecord(**resp.json())

        logger.info(f"[{self.name}] Registered as {self.agent_id}")

        self._running = True
        await asyncio.gather(
            self._heartbeat_loop(),
            self._listen_for_tasks(),
        )

    async def stop(self):
        """Deregister from the mesh."""
        self._running = False
        async with httpx.AsyncClient() as client:
            await client.delete(f"{self.registry_url}/agents/{self.agent_id}")
        logger.info(f"[{self.name}] Deregistered")

    # --- Heartbeat ---

    async def _heartbeat_loop(self):
        """Send periodic heartbeats to the registry."""
        uri = f"{self.registry_url.replace('http', 'ws')}/ws/heartbeat/{self.agent_id}"
        try:
            async with websockets.connect(uri) as ws:
                self._heartbeat_ws = ws
                while self._running:
                    await ws.send(json.dumps({"type": "heartbeat"}))
                    resp = await ws.recv()
                    logger.debug(f"[{self.name}] Heartbeat ack: {resp}")
                    await asyncio.sleep(10)
        except Exception as e:
            logger.warning(f"[{self.name}] Heartbeat connection lost: {e}")

    # --- Incoming Task Handling ---

    async def _listen_for_tasks(self):
        """WebSocket server that receives task requests from other agents."""
        async def handler(websocket):
            async for message in websocket:
                data = json.loads(message)
                method = data.get("method")

                if method == "task.request":
                    request = TaskRequest(**data["params"])
                    await self._handle_task_request(request, websocket)

        async with websockets.serve(handler, "0.0.0.0", self.ws_port):
            logger.info(f"[{self.name}] Listening for tasks on port {self.ws_port}")
            while self._running:
                await asyncio.sleep(1)

    async def _handle_task_request(self, request: TaskRequest, ws) -> None:
        """Process an incoming task request: negotiate, execute, return result."""
        cap_name = request.capability
        handler = self._capabilities.get(cap_name)

        if not handler:
            # Reject — we don't have this capability
            neg = NegotiationResponse(
                task_id=request.task_id,
                responder_id=self.agent_id,
                status=TaskStatus.REJECTED,
                reason=f"Capability '{cap_name}' not available",
                counter_proposal=None,
            )
            await ws.send(json.dumps({"result": neg.model_dump(mode="json")}))
            return

        # Check if at capacity
        if len(self._active_tasks) >= self._max_concurrent_tasks:
            avg_ms = handler._mesh_capability.avg_latency_ms or 5000  # type: ignore[attr-defined]
            proposed_deadline = request.deadline_ms + int(avg_ms)
            neg = NegotiationResponse(  # type: ignore[call-arg]
                task_id=request.task_id,
                responder_id=self.agent_id,
                status=TaskStatus.COUNTERED,
                counter_proposal={
                    "deadline_ms": proposed_deadline,
                    "reason": "queue_full",
                    "queue_depth": len(self._active_tasks),
                },
            )
            await ws.send(json.dumps({"result": neg.model_dump(mode="json")}))
            return

        # Accept
        neg = NegotiationResponse(
            task_id=request.task_id,
            responder_id=self.agent_id,
            status=TaskStatus.ACCEPTED,
            estimated_latency_ms=int(
                handler._mesh_capability.avg_latency_ms or 5000  # type: ignore[attr-defined]
            ),
            counter_proposal=None,
            reason=None,
        )
        await ws.send(json.dumps({"result": neg.model_dump(mode="json")}))
        self._active_tasks.add(request.task_id)

        # Notify registry that a task is starting
        if self._heartbeat_ws:
            try:
                await self._heartbeat_ws.send(json.dumps({"type": "task_start", "task_id": request.task_id}))
            except Exception:
                pass  # heartbeat loss shouldn't fail task execution

        # Execute
        start = datetime.utcnow()
        try:
            output = await handler(request.input_data)
            elapsed = int((datetime.utcnow() - start).total_seconds() * 1000)

            result = TaskResult(
                task_id=request.task_id,
                executor_id=self.agent_id,
                status=TaskStatus.COMPLETED,
                output=output if isinstance(output, dict) else {"result": output},
                execution_time_ms=elapsed,
            )

            # Notify registry that the task has ended
            if self._heartbeat_ws:
                try:
                    await self._heartbeat_ws.send(json.dumps({"type": "task_end", "task_id": request.task_id}))
                except Exception:
                    pass
        except Exception as e:
            elapsed = int((datetime.utcnow() - start).total_seconds() * 1000)
            result = TaskResult(
                task_id=request.task_id,
                executor_id=self.agent_id,
                status=TaskStatus.FAILED,
                error=str(e),
                execution_time_ms=elapsed,
            )

            # Notify registry that the task has ended (even on failure)
            if self._heartbeat_ws:
                try:
                    await self._heartbeat_ws.send(json.dumps({"type": "task_end", "task_id": request.task_id}))
                except Exception:
                    pass
        finally:
            self._active_tasks.discard(request.task_id)

        await ws.send(json.dumps({
            "method": "task.result",
            "params": result.model_dump(mode="json"),
        }))

    # --- Outbound: Discovery & Delegation ---

    async def discover(
        self,
        capability_name: str | None = None,
        description: str | None = None,
        min_trust: float = 0.0,
        top_k: int = 5,
        max_cost_usd: float | None = None,
    ) -> list[AgentRecord]:
        """Find agents on the mesh that can help with a task."""
        query = DiscoveryQuery(
            capability_name=capability_name,
            capability_description=description,
            min_trust_score=min_trust,
            top_k=top_k,
            max_cost_usd=max_cost_usd,
        )
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.registry_url}/discover",
                json=query.model_dump(mode="json"),
            )
            resp.raise_for_status()
            result = DiscoveryResult(**resp.json())
            return result.agents

    async def delegate(
        self,
        capability: str,
        input_data: dict[str, Any],
        target: AgentRecord | None = None,
        deadline_ms: int = 30000,
    ) -> TaskResult:
        """Delegate a task to another agent on the mesh.

        If no target is specified, auto-discovers the best match.
        Handles the full negotiation + execution + result cycle.
        """
        # Auto-discover if no target
        if target is None:
            candidates = await self.discover(capability_name=capability)
            if not candidates:
                raise RuntimeError(f"No agents found for capability '{capability}'")
            target = candidates[0]

        endpoint = target.manifest.endpoint

        # security: validate endpoint scheme to prevent SSRF via malicious registry entries
        parsed = urllib.parse.urlparse(endpoint)
        if parsed.scheme not in ("ws", "wss"):
            raise ValueError(f"Invalid endpoint scheme: {parsed.scheme}")

        request = TaskRequest(  # type: ignore[call-arg]
            capability=capability,
            input_data=input_data,
            requester_id=self.agent_id,
            target_id=target.manifest.agent_id,
            deadline_ms=deadline_ms,
        )

        # Send task request via WebSocket
        async with websockets.connect(endpoint) as ws:
            await ws.send(json.dumps({
                "method": "task.request",
                "params": request.model_dump(mode="json"),
            }))

            # Wait for negotiation response
            neg_raw = await asyncio.wait_for(ws.recv(), timeout=10)
            neg_data = json.loads(neg_raw)["result"]
            neg = NegotiationResponse(**neg_data)

            if neg.status == TaskStatus.REJECTED:
                raise RuntimeError(
                    f"Agent {target.manifest.agent_id} rejected: {neg.reason}"
                )

            if neg.status == TaskStatus.COUNTERED:
                proposed = neg.counter_proposal or {}
                proposed_deadline_ms = proposed.get("deadline_ms", deadline_ms * 2)
                if proposed_deadline_ms <= deadline_ms * 1.5:
                    # Re-send with proposed deadline
                    request = TaskRequest(  # type: ignore[call-arg]
                        capability=capability,
                        input_data=input_data,
                        requester_id=self.agent_id,
                        target_id=target.manifest.agent_id,
                        deadline_ms=int(proposed_deadline_ms),
                    )
                    await ws.send(json.dumps({
                        "method": "task.request",
                        "params": request.model_dump(mode="json"),
                    }))
                    neg_raw2 = await asyncio.wait_for(ws.recv(), timeout=10)
                    neg = NegotiationResponse(**json.loads(neg_raw2)["result"])
                    if neg.status != TaskStatus.ACCEPTED:
                        raise RuntimeError(f"Agent rejected after counter-proposal: {neg.reason}")
                else:
                    raise RuntimeError(
                        f"Counter-proposal deadline too far: {proposed_deadline_ms}ms (max acceptable: {deadline_ms * 1.5}ms)"
                    )

            # Wait for task result
            timeout = deadline_ms / 1000
            result_raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
            result_data = json.loads(result_raw)["params"]
            result = TaskResult(**result_data)

            # Report trust update
            await self._report_trust(
                agent_id=target.manifest.agent_id,
                task_id=result.task_id,
                success=result.status == TaskStatus.COMPLETED,
                quality=0.8 if result.status == TaskStatus.COMPLETED else 0.2,
            )

            return result

    async def _report_trust(
        self, agent_id: str, task_id: str, success: bool, quality: float
    ):
        """Report a trust update to the registry."""
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{self.registry_url}/trust/update",
                params={
                    "agent_id": agent_id,
                    "success": success,
                    "quality": quality,
                },
            )
