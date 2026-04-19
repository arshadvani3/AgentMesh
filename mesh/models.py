"""Core data models for AgentMesh.

These schemas define the protocol: how agents describe themselves,
how they request and deliver work, and how trust is tracked.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Capability & Agent Identity
# ---------------------------------------------------------------------------

class CapabilitySchema(BaseModel):
    """A single capability an agent advertises on the mesh."""

    name: str = Field(..., description="Machine-readable capability name, e.g. 'analyze_csv'")
    description: str = Field(..., description="Human-readable description for semantic matching")
    input_schema: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON Schema describing expected input",
    )
    output_schema: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON Schema describing expected output",
    )
    avg_latency_ms: float | None = Field(None, description="Rolling average execution time")
    cost_per_call_usd: float | None = Field(None, description="Estimated cost per invocation")

    @field_validator("cost_per_call_usd")
    @classmethod
    def validate_cost_per_call(cls, v):
        if v is not None and v < 0:
            raise ValueError("cost_per_call_usd must be non-negative")
        return v


class AgentManifest(BaseModel):
    """The full identity + capability manifest an agent registers with."""

    agent_id: str = Field(default_factory=lambda: f"agent-{uuid.uuid4().hex[:8]}")
    # security: enforce max lengths to prevent registry bloat and injection via oversized strings
    name: str = Field(..., max_length=128)
    version: str = "0.1.0"
    description: str = Field("", max_length=1024)
    capabilities: list[CapabilitySchema] = Field(default_factory=list)
    mcp_servers: list[str] = Field(default_factory=list, description="Connected MCP server names")
    max_concurrent_tasks: int = 3
    endpoint: str = Field(..., description="WebSocket URL where this agent listens")
    tags: list[str] = Field(default_factory=list)


class AgentRecord(BaseModel):
    """Internal registry record — manifest + runtime state."""

    manifest: AgentManifest
    trust_score: float = 0.5
    status: str = "healthy"  # healthy | degraded | offline
    registered_at: datetime = Field(default_factory=datetime.utcnow)
    last_heartbeat: datetime = Field(default_factory=datetime.utcnow)
    tasks_completed: int = 0
    tasks_failed: int = 0


# ---------------------------------------------------------------------------
# Task Delegation Protocol
# ---------------------------------------------------------------------------

class TaskStatus(StrEnum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    COUNTERED = "countered"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


class TaskRequest(BaseModel):
    """A request from one agent to another to perform a task."""

    task_id: str = Field(default_factory=lambda: f"task-{uuid.uuid4().hex[:8]}")
    capability: str = Field(..., description="Name of the capability being requested")
    input_data: dict[str, Any] = Field(default_factory=dict)
    requester_id: str = Field(..., description="agent_id of the requesting agent")
    target_id: str | None = Field(None, description="Specific agent to target, or None for auto-route")
    deadline_ms: int = Field(30000, description="Max time allowed for task completion")
    priority: int = Field(1, ge=1, le=5, description="1=lowest, 5=highest")
    context: str = Field("", description="Additional context for the agent")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class NegotiationResponse(BaseModel):
    """Response to a TaskRequest during the negotiation phase."""

    task_id: str
    responder_id: str
    status: TaskStatus
    estimated_latency_ms: int | None = None
    counter_proposal: dict[str, Any] | None = Field(
        None,
        description="If status=countered, what the agent needs instead",
    )
    reason: str | None = Field(None, description="Why rejected or countered")


class TaskResult(BaseModel):
    """The final output of a completed (or failed) task."""

    task_id: str
    executor_id: str
    status: TaskStatus  # completed | failed
    output: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    tokens_used: int = 0
    execution_time_ms: int = 0
    completed_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Trust & Reputation
# ---------------------------------------------------------------------------

class TrustUpdate(BaseModel):
    """Submitted after a task completes to update an agent's trust score."""

    agent_id: str
    task_id: str
    success: bool
    quality_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="How good was the output? 0=terrible, 1=perfect",
    )
    latency_ms: int
    reviewer_id: str


# ---------------------------------------------------------------------------
# Trace / Observability
# ---------------------------------------------------------------------------

class TraceEvent(BaseModel):
    """A single event in a cross-agent workflow trace."""

    trace_id: str = Field(default_factory=lambda: f"trace-{uuid.uuid4().hex[:8]}")
    task_id: str
    event_type: str  # request_sent | accepted | rejected | executing | completed | failed
    from_agent: str
    to_agent: str
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

class DiscoveryQuery(BaseModel):
    """Query the mesh to find agents with matching capabilities."""

    capability_name: str | None = None
    capability_description: str | None = Field(
        None, description="Free-text description for semantic matching"
    )
    min_trust_score: float = 0.0
    max_latency_ms: float | None = None
    max_cost_usd: float | None = None
    tags: list[str] = Field(default_factory=list)
    top_k: int = Field(5, ge=1, le=20)

    @field_validator("max_cost_usd")
    @classmethod
    def validate_max_cost(cls, v):
        if v is not None and v < 0:
            raise ValueError("max_cost_usd must be non-negative")
        return v


class DiscoveryResult(BaseModel):
    """Ranked list of agents matching a discovery query."""

    agents: list[AgentRecord]
    query: DiscoveryQuery
    matched_at: datetime = Field(default_factory=datetime.utcnow)
