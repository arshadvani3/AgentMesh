"""AgentMesh Registry -- the central nervous system of the mesh.

Agents register here, advertise capabilities, and discover collaborators.
This is NOT a central orchestrator -- it is a phone book. Agents still
communicate peer-to-peer via WebSocket after discovery.

Persistence:
  - PostgreSQL (via asyncpg) when DATABASE_URL is set
  - In-memory dict fallback for development without Docker
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from .db import get_db
from .models import (
    AgentManifest,
    AgentRecord,
    DiscoveryQuery,
    DiscoveryResult,
    TraceEvent,
)
from .router import TaskRouter

# ---------------------------------------------------------------------------
# JWT configuration
# ---------------------------------------------------------------------------

_JWT_SECRET = os.environ.get("AGENT_SECRET", "agentmesh-dev-secret-change-in-prod")
_JWT_ALGORITHM = "HS256"
_JWT_EXPIRE_HOURS = 24

_http_bearer = HTTPBearer(auto_error=False)

logger = logging.getLogger("agentmesh.registry")

# WebSocket connections keyed by agent_id (always in-memory -- not persisted)
_ws_connections: dict[str, WebSocket] = {}
_dashboard_ws: list[WebSocket] = []

# Real-time task load tracking (in-memory)
_active_task_counts: dict[str, int] = {}
_failure_streaks: dict[str, int] = {}
_degraded_since: dict[str, datetime] = {}
_latency_samples: dict[str, deque] = {}  # agent_id -> deque of last 20 elapsed_ms values
CIRCUIT_OPEN_THRESHOLD = 3
CIRCUIT_COOLDOWN_SECONDS = 60
LATENCY_WINDOW = 20  # number of samples to keep in rolling average

HEARTBEAT_TIMEOUT = timedelta(seconds=30)
HEALTH_CHECK_INTERVAL = 10  # seconds


async def _health_check_loop():
    """Background task: mark agents as offline when heartbeat becomes stale.

    Also restores degraded agents to healthy once CIRCUIT_COOLDOWN_SECONDS
    has elapsed since the circuit opened.
    """
    while True:
        try:
            db = await get_db()
            agents = await db.list_agents()
            now = datetime.utcnow()
            for record in agents:
                agent_id = record.manifest.agent_id
                if now - record.last_heartbeat > HEARTBEAT_TIMEOUT:
                    if record.status != "offline":
                        record.status = "offline"
                        await db.update_heartbeat(agent_id, "offline")

            # Circuit breaker cooldown: restore degraded agents after timeout
            for agent_id, degraded_at in list(_degraded_since.items()):
                elapsed = (now - degraded_at).total_seconds()
                if elapsed > CIRCUIT_COOLDOWN_SECONDS:
                    await db.update_heartbeat(agent_id, "healthy")
                    _failure_streaks.pop(agent_id, None)
                    del _degraded_since[agent_id]
                    logger.info(f"Circuit breaker reset for {agent_id} after cooldown")
        except Exception as e:
            logger.warning(f"Health check error: {e}")
        await asyncio.sleep(HEALTH_CHECK_INTERVAL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: start background health-check loop."""
    task = asyncio.create_task(_health_check_loop())
    yield
    task.cancel()


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AgentMesh Registry",
    description="Peer-to-peer agent discovery and communication network",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

router_engine = TaskRouter()


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def _create_token(agent_id: str) -> str:
    """Issue a signed JWT for an agent."""
    payload = {
        "sub": agent_id,
        "exp": datetime.utcnow() + timedelta(hours=_JWT_EXPIRE_HOURS),
    }
    return jwt.encode(payload, _JWT_SECRET, algorithm=_JWT_ALGORITHM)


async def _verify_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(_http_bearer),  # noqa: B008
) -> str:
    """FastAPI dependency: validate Bearer token, return agent_id.

    Returns "anonymous" when AGENT_SECRET is not set (dev mode without auth).
    Raises HTTP 401 when auth is enabled but token is missing or invalid.
    """
    if _JWT_SECRET == "agentmesh-dev-secret-change-in-prod":
        return "anonymous"  # auth disabled in dev
    if not credentials:
        raise HTTPException(401, "Missing Authorization header")
    try:
        payload = jwt.decode(credentials.credentials, _JWT_SECRET, algorithms=[_JWT_ALGORITHM])
        return payload["sub"]
    except JWTError as exc:
        raise HTTPException(401, "Invalid or expired token") from exc


@app.post("/auth/token")
async def get_token(agent_id: str, secret: str) -> dict:
    """Issue a JWT for an agent.

    The agent must supply its agent_id and the shared AGENT_SECRET configured
    on the registry. Returns a bearer token valid for 24 hours.

    Args:
        agent_id: The agent requesting a token.
        secret: The pre-shared secret (must match AGENT_SECRET env var).

    Returns:
        Dict with 'access_token' and 'token_type'.

    Raises:
        HTTPException 401: If secret is incorrect.
        HTTPException 404: If agent is not registered.
    """
    if secret != _JWT_SECRET:
        raise HTTPException(401, "Invalid secret")
    db = await get_db()
    if not await db.get_agent(agent_id):
        raise HTTPException(404, f"Agent {agent_id} not found")
    token = _create_token(agent_id)
    return {"access_token": token, "token_type": "bearer"}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

@app.post("/agents/register", response_model=AgentRecord)
async def register_agent(manifest: AgentManifest) -> AgentRecord:
    """Register a new agent on the mesh.

    Args:
        manifest: The agent's full capability manifest.

    Returns:
        Newly created AgentRecord.

    Raises:
        HTTPException 409: If an agent with this ID is already registered.
    """
    db = await get_db()
    existing = await db.get_agent(manifest.agent_id)
    if existing:
        raise HTTPException(409, f"Agent {manifest.agent_id} already registered")

    record = AgentRecord(manifest=manifest)
    await db.save_agent(record)

    # Index capabilities for semantic search
    await router_engine.index_agent(record)

    logger.info(f"Registered agent: {manifest.name} ({manifest.agent_id})")
    return record


@app.delete("/agents/{agent_id}")
async def deregister_agent(agent_id: str, _: str = Depends(_verify_token)) -> dict:
    """Remove an agent from the mesh.

    Args:
        agent_id: The agent to deregister.

    Returns:
        Confirmation dict.

    Raises:
        HTTPException 404: If the agent is not found.
    """
    db = await get_db()
    removed = await db.delete_agent(agent_id)
    if not removed:
        raise HTTPException(404, f"Agent {agent_id} not found")

    router_engine.remove_agent(agent_id)
    _ws_connections.pop(agent_id, None)

    logger.info(f"Deregistered agent: {agent_id}")
    return {"status": "deregistered", "agent_id": agent_id}


@app.get("/agents", response_model=list[AgentRecord])
async def list_agents() -> list[AgentRecord]:
    """List all registered agents.

    Returns:
        All AgentRecord objects in the registry.
    """
    db = await get_db()
    return await db.list_agents()


@app.get("/agents/{agent_id}", response_model=AgentRecord)
async def get_agent(agent_id: str) -> AgentRecord:
    """Get a specific agent's record.

    Args:
        agent_id: The agent to fetch.

    Returns:
        The AgentRecord.

    Raises:
        HTTPException 404: If the agent is not found.
    """
    db = await get_db()
    record = await db.get_agent(agent_id)
    if not record:
        raise HTTPException(404, f"Agent {agent_id} not found")
    return record


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

@app.post("/discover", response_model=DiscoveryResult)
async def discover_agents(query: DiscoveryQuery) -> DiscoveryResult:
    """Find agents matching a capability query.

    Uses dual matching:
    1. Exact name match on capability_name
    2. Semantic similarity on capability_description
    Results are ranked by composite score: match * 0.4 + trust * 0.4 + availability * 0.2

    Args:
        query: DiscoveryQuery parameters.

    Returns:
        Ranked DiscoveryResult.
    """
    db = await get_db()
    all_agents = await db.list_agents()
    # Include healthy and degraded agents; router handles offline exclusion
    routable = [r for r in all_agents if r.status != "offline"]
    results = await router_engine.match(query, routable, task_counts=_active_task_counts)
    return DiscoveryResult(agents=results, query=query)


# ---------------------------------------------------------------------------
# Heartbeat (WebSocket)
# ---------------------------------------------------------------------------

@app.websocket("/ws/heartbeat/{agent_id}")
async def heartbeat(websocket: WebSocket, agent_id: str):
    """Persistent WebSocket for agent heartbeats and real-time event submission.

    Accepts JSON messages of type 'heartbeat' (updates last_heartbeat) or
    'trace' (stores + broadcasts a TraceEvent to dashboard subscribers).

    Args:
        websocket: The incoming WebSocket connection.
        agent_id: The registering agent's ID (path param).
    """
    db = await get_db()
    record = await db.get_agent(agent_id)
    if not record:
        await websocket.close(code=4004, reason="Agent not registered")
        return

    await websocket.accept()
    _ws_connections[agent_id] = websocket
    await db.update_heartbeat(agent_id, "healthy")

    try:
        while True:
            data = await websocket.receive_json()

            if data.get("type") == "heartbeat":
                await db.update_heartbeat(agent_id, "healthy")
                await websocket.send_json({"type": "heartbeat_ack"})

            elif data.get("type") == "task_start":
                _active_task_counts[agent_id] = _active_task_counts.get(agent_id, 0) + 1

            elif data.get("type") == "task_end":
                _active_task_counts[agent_id] = max(0, _active_task_counts.get(agent_id, 0) - 1)
                elapsed_ms = data.get("elapsed_ms")
                if elapsed_ms is not None:
                    if agent_id not in _latency_samples:
                        _latency_samples[agent_id] = deque(maxlen=LATENCY_WINDOW)
                    _latency_samples[agent_id].append(float(elapsed_ms))
                    avg = sum(_latency_samples[agent_id]) / len(_latency_samples[agent_id])
                    # Update avg_latency_ms on all capabilities for this agent
                    rec = await db.get_agent(agent_id)
                    if rec:
                        for cap in rec.manifest.capabilities:
                            cap.avg_latency_ms = round(avg, 1)
                        await db.save_agent(rec)

            elif data.get("type") == "trace":
                event = TraceEvent(**data["payload"])
                await db.save_trace(event)
                await _broadcast_trace(event)

    except WebSocketDisconnect:
        await db.update_heartbeat(agent_id, "offline")
        _ws_connections.pop(agent_id, None)


# ---------------------------------------------------------------------------
# Traces / Observability
# ---------------------------------------------------------------------------

@app.websocket("/ws/dashboard")
async def dashboard_stream(websocket: WebSocket):
    """WebSocket stream for the dashboard to receive real-time trace events.

    Args:
        websocket: The dashboard client WebSocket.
    """
    await websocket.accept()
    _dashboard_ws.append(websocket)
    try:
        while True:
            await websocket.receive_text()  # keep-alive
    except WebSocketDisconnect:
        _dashboard_ws.remove(websocket)


@app.get("/traces", response_model=list[TraceEvent])
async def get_traces(task_id: str | None = None, limit: int = 100) -> list[TraceEvent]:
    """Query stored trace events.

    Args:
        task_id: Optional filter by task ID.
        limit: Maximum events to return (default 100).

    Returns:
        List of TraceEvent objects.
    """
    db = await get_db()
    return await db.list_traces(task_id=task_id, limit=limit)


async def _broadcast_trace(event: TraceEvent):
    """Push a trace event to all connected dashboard WebSocket clients.

    Args:
        event: The TraceEvent to broadcast.
    """
    dead = []
    for ws in _dashboard_ws:
        try:
            await ws.send_json(event.model_dump(mode="json"))
        except Exception:
            dead.append(ws)
    for ws in dead:
        _dashboard_ws.remove(ws)


# ---------------------------------------------------------------------------
# Trust
# ---------------------------------------------------------------------------

@app.post("/trust/update")
async def update_trust(agent_id: str, success: bool, quality: float = 0.5, _: str = Depends(_verify_token)):
    """Update an agent's trust score after a task completes.

    Uses an ELO-inspired update rule: new = old + k * (actual - expected)

    Args:
        agent_id: The agent whose score to update.
        success: Whether the task succeeded.
        quality: Quality of output between 0 and 1.

    Returns:
        Updated trust score dict.

    Raises:
        HTTPException 404: If the agent is not found.
        HTTPException 422: If quality is outside [0.0, 1.0].
    """
    if not (0.0 <= quality <= 1.0):
        raise HTTPException(422, "quality must be between 0.0 and 1.0")

    db = await get_db()
    record = await db.get_agent(agent_id)
    if not record:
        raise HTTPException(404, f"Agent {agent_id} not found")

    if success:
        record.tasks_completed += 1
    else:
        record.tasks_failed += 1

    k = 0.1
    expected = record.trust_score
    actual = quality if success else max(0.0, quality - 0.3)
    new_trust = max(0.0, min(1.0, expected + k * (actual - expected)))

    await db.update_trust(
        agent_id,
        new_trust,
        record.tasks_completed,
        record.tasks_failed,
    )

    # Circuit breaker logic
    if not success:
        _failure_streaks[agent_id] = _failure_streaks.get(agent_id, 0) + 1
        if _failure_streaks[agent_id] >= CIRCUIT_OPEN_THRESHOLD:
            await db.update_heartbeat(agent_id, "degraded")
            _degraded_since[agent_id] = datetime.utcnow()
            logger.warning(f"Circuit opened for {agent_id} after {_failure_streaks[agent_id]} failures")
    else:
        _failure_streaks[agent_id] = 0
        if agent_id in _degraded_since:
            await db.update_heartbeat(agent_id, "healthy")
            del _degraded_since[agent_id]
            logger.info(f"Circuit closed for {agent_id} after successful task")

    logger.debug(f"Trust update for {agent_id}: {expected:.3f} -> {new_trust:.3f}")
    return {"agent_id": agent_id, "trust_score": new_trust}
