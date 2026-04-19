"""Tests for mesh/models.py -- schema validation and serialization round-trips."""

from __future__ import annotations

import json
from datetime import datetime

import pytest

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
    TraceEvent,
    TrustUpdate,
)


# ---------------------------------------------------------------------------
# CapabilitySchema
# ---------------------------------------------------------------------------

class TestCapabilitySchema:
    def test_basic_creation(self):
        """CapabilitySchema accepts required fields."""
        cap = CapabilitySchema(
            name="analyze_csv",
            description="Analyze tabular data",
        )
        assert cap.name == "analyze_csv"
        assert cap.avg_latency_ms is None

    def test_full_schema(self):
        """CapabilitySchema accepts all optional fields."""
        cap = CapabilitySchema(
            name="fetch_code",
            description="Retrieve code examples",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            avg_latency_ms=3000.0,
            cost_per_call_usd=0.001,
        )
        assert cap.avg_latency_ms == 3000.0
        assert cap.cost_per_call_usd == 0.001

    def test_json_roundtrip(self):
        """CapabilitySchema serializes and deserializes correctly."""
        cap = CapabilitySchema(
            name="write_report",
            description="Write a report",
            avg_latency_ms=5000.0,
        )
        data = cap.model_dump(mode="json")
        restored = CapabilitySchema(**data)
        assert restored.name == cap.name
        assert restored.avg_latency_ms == cap.avg_latency_ms


# ---------------------------------------------------------------------------
# AgentManifest
# ---------------------------------------------------------------------------

class TestAgentManifest:
    def test_auto_agent_id(self):
        """AgentManifest auto-generates agent_id when not provided."""
        m = AgentManifest(name="Test Agent", endpoint="ws://localhost:9000")
        assert m.agent_id.startswith("agent-")

    def test_explicit_agent_id(self):
        """AgentManifest respects an explicit agent_id."""
        m = AgentManifest(
            agent_id="agent-explicit",
            name="Test",
            endpoint="ws://localhost:9000",
        )
        assert m.agent_id == "agent-explicit"

    def test_defaults(self):
        """AgentManifest has sensible defaults."""
        m = AgentManifest(name="Test", endpoint="ws://localhost:9000")
        assert m.version == "0.1.0"
        assert m.max_concurrent_tasks == 3
        assert m.capabilities == []
        assert m.tags == []

    def test_with_capabilities(self):
        """AgentManifest stores capability list."""
        cap = CapabilitySchema(name="cap1", description="desc1")
        m = AgentManifest(
            name="Agent",
            endpoint="ws://localhost:9000",
            capabilities=[cap],
            tags=["data"],
        )
        assert len(m.capabilities) == 1
        assert m.capabilities[0].name == "cap1"


# ---------------------------------------------------------------------------
# AgentRecord
# ---------------------------------------------------------------------------

class TestAgentRecord:
    def test_defaults(self):
        """AgentRecord starts with trust_score=0.5 and status=healthy."""
        manifest = AgentManifest(name="Test", endpoint="ws://localhost:9000")
        record = AgentRecord(manifest=manifest)
        assert record.trust_score == 0.5
        assert record.status == "healthy"
        assert record.tasks_completed == 0
        assert record.tasks_failed == 0

    def test_json_roundtrip(self):
        """AgentRecord round-trips through JSON cleanly."""
        manifest = AgentManifest(
            agent_id="agent-abc",
            name="Test",
            endpoint="ws://localhost:9000",
        )
        record = AgentRecord(manifest=manifest, trust_score=0.8)
        data = json.loads(record.model_dump_json())
        restored = AgentRecord(**data)
        assert restored.manifest.agent_id == "agent-abc"
        assert restored.trust_score == 0.8


# ---------------------------------------------------------------------------
# TaskRequest
# ---------------------------------------------------------------------------

class TestTaskRequest:
    def test_auto_task_id(self):
        """TaskRequest auto-generates task_id."""
        req = TaskRequest(capability="analyze_csv", requester_id="agent-1")
        assert req.task_id.startswith("task-")

    def test_priority_bounds(self):
        """TaskRequest rejects priority outside 1-5."""
        with pytest.raises(Exception):
            TaskRequest(capability="x", requester_id="y", priority=6)
        with pytest.raises(Exception):
            TaskRequest(capability="x", requester_id="y", priority=0)

    def test_valid_priorities(self):
        """TaskRequest accepts all valid priority values."""
        for p in range(1, 6):
            req = TaskRequest(capability="x", requester_id="y", priority=p)
            assert req.priority == p


# ---------------------------------------------------------------------------
# NegotiationResponse
# ---------------------------------------------------------------------------

class TestNegotiationResponse:
    def test_accepted(self):
        """NegotiationResponse with ACCEPTED status."""
        neg = NegotiationResponse(
            task_id="task-001",
            responder_id="agent-1",
            status=TaskStatus.ACCEPTED,
            estimated_latency_ms=3000,
        )
        assert neg.status == TaskStatus.ACCEPTED

    def test_rejected_with_reason(self):
        """NegotiationResponse with REJECTED status and reason."""
        neg = NegotiationResponse(
            task_id="task-001",
            responder_id="agent-1",
            status=TaskStatus.REJECTED,
            reason="Capability not available",
        )
        assert neg.reason == "Capability not available"

    def test_countered_with_proposal(self):
        """NegotiationResponse with COUNTERED status and counter_proposal."""
        neg = NegotiationResponse(
            task_id="task-001",
            responder_id="agent-1",
            status=TaskStatus.COUNTERED,
            counter_proposal={"needs": "more_context"},
        )
        assert neg.counter_proposal == {"needs": "more_context"}


# ---------------------------------------------------------------------------
# TaskResult
# ---------------------------------------------------------------------------

class TestTaskResult:
    def test_completed(self):
        """TaskResult tracks completion metadata."""
        result = TaskResult(
            task_id="task-001",
            executor_id="agent-2",
            status=TaskStatus.COMPLETED,
            output={"report": "Analysis complete"},
            tokens_used=150,
            execution_time_ms=3200,
        )
        assert result.status == TaskStatus.COMPLETED
        assert result.tokens_used == 150

    def test_failed(self):
        """TaskResult captures error messages on failure."""
        result = TaskResult(
            task_id="task-002",
            executor_id="agent-2",
            status=TaskStatus.FAILED,
            error="LLM timeout",
        )
        assert result.error == "LLM timeout"
        assert result.output == {}


# ---------------------------------------------------------------------------
# TrustUpdate
# ---------------------------------------------------------------------------

class TestTrustUpdate:
    def test_quality_bounds(self):
        """TrustUpdate rejects quality_score outside [0, 1]."""
        with pytest.raises(Exception):
            TrustUpdate(
                agent_id="a", task_id="t", success=True,
                quality_score=1.5, latency_ms=100, reviewer_id="r",
            )
        with pytest.raises(Exception):
            TrustUpdate(
                agent_id="a", task_id="t", success=True,
                quality_score=-0.1, latency_ms=100, reviewer_id="r",
            )

    def test_valid(self):
        """TrustUpdate accepts valid quality scores."""
        update = TrustUpdate(
            agent_id="agent-1", task_id="task-1", success=True,
            quality_score=0.9, latency_ms=500, reviewer_id="agent-0",
        )
        assert update.quality_score == 0.9


# ---------------------------------------------------------------------------
# TraceEvent
# ---------------------------------------------------------------------------

class TestTraceEvent:
    def test_creation(self):
        """TraceEvent populates defaults correctly."""
        ev = TraceEvent(
            task_id="task-001",
            event_type="request_sent",
            from_agent="agent-1",
            to_agent="agent-2",
        )
        assert ev.trace_id.startswith("trace-")
        assert isinstance(ev.timestamp, datetime)

    def test_json_roundtrip(self):
        """TraceEvent serializes datetimes as ISO strings."""
        ev = TraceEvent(
            task_id="task-001",
            event_type="completed",
            from_agent="a1",
            to_agent="a2",
            payload={"tokens": 42},
        )
        data = ev.model_dump(mode="json")
        assert isinstance(data["timestamp"], str)
        assert data["payload"]["tokens"] == 42


# ---------------------------------------------------------------------------
# DiscoveryQuery
# ---------------------------------------------------------------------------

class TestDiscoveryQuery:
    def test_defaults(self):
        """DiscoveryQuery has sensible defaults."""
        q = DiscoveryQuery()
        assert q.min_trust_score == 0.0
        assert q.top_k == 5
        assert q.tags == []

    def test_top_k_bounds(self):
        """DiscoveryQuery rejects top_k outside [1, 20]."""
        with pytest.raises(Exception):
            DiscoveryQuery(top_k=0)
        with pytest.raises(Exception):
            DiscoveryQuery(top_k=21)
