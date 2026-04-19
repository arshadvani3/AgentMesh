"""Tests for mesh/router.py -- exact match, semantic match, ranking, thresholds."""

from __future__ import annotations

import pytest

from mesh.models import AgentManifest, AgentRecord, CapabilitySchema, DiscoveryQuery
from mesh.router import TaskRouter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(
    agent_id: str,
    caps: list[tuple[str, str]],
    trust_score: float = 0.5,
    tags: list[str] | None = None,
) -> AgentRecord:
    """Build an AgentRecord with the given capabilities.

    Args:
        agent_id: Unique identifier.
        caps: List of (capability_name, description) tuples.
        trust_score: Initial trust score.
        tags: Optional tag list.

    Returns:
        AgentRecord ready for indexing.
    """
    capabilities = [
        CapabilitySchema(name=n, description=d) for n, d in caps
    ]
    manifest = AgentManifest(
        agent_id=agent_id,
        name=agent_id,
        endpoint=f"ws://localhost:9000",
        capabilities=capabilities,
        tags=tags or [],
    )
    return AgentRecord(manifest=manifest, trust_score=trust_score)


def _make_record_with_cost(
    agent_id: str,
    caps: list[tuple[str, str, float | None]],
    trust_score: float = 0.5,
) -> AgentRecord:
    """Build an AgentRecord with capabilities that include cost_per_call_usd.

    Args:
        agent_id: Unique identifier.
        caps: List of (capability_name, description, cost_per_call_usd) tuples.
        trust_score: Initial trust score.

    Returns:
        AgentRecord ready for indexing.
    """
    capabilities = [
        CapabilitySchema(name=n, description=d, cost_per_call_usd=cost)
        for n, d, cost in caps
    ]
    manifest = AgentManifest(
        agent_id=agent_id,
        name=agent_id,
        endpoint="ws://localhost:9000",
        capabilities=capabilities,
    )
    return AgentRecord(manifest=manifest, trust_score=trust_score)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestTaskRouter:
    async def test_exact_name_match(self):
        """Exact capability name match returns the correct agent."""
        router = TaskRouter()
        record = _make_record("agent-1", [("analyze_csv", "Analyze spreadsheet data")])
        await router.index_agent(record)

        query = DiscoveryQuery(capability_name="analyze_csv")
        results = await router.match(query, [record])
        assert len(results) == 1
        assert results[0].manifest.agent_id == "agent-1"

    async def test_exact_match_beats_semantic(self):
        """Exact name match agent ranks above semantic-only match."""
        router = TaskRouter()
        exact = _make_record("agent-exact", [("fetch_code", "Retrieve code from GitHub")])
        semantic = _make_record(
            "agent-semantic",
            [("code_search", "Search for code examples and programming patterns")],
        )
        await router.index_agent(exact)
        await router.index_agent(semantic)

        query = DiscoveryQuery(capability_name="fetch_code", top_k=5)
        results = await router.match(query, [exact, semantic])
        assert results[0].manifest.agent_id == "agent-exact"

    async def test_semantic_match_returns_relevant_agent(self):
        """Semantic query on description returns a relevant agent."""
        router = TaskRouter()
        record = _make_record(
            "agent-writer",
            [("write_report", "Creates formatted markdown reports and documentation")],
        )
        await router.index_agent(record)

        query = DiscoveryQuery(
            capability_description="write a professional document or report"
        )
        results = await router.match(query, [record])
        assert len(results) == 1

    async def test_trust_score_filter(self):
        """min_trust_score filter excludes low-trust agents."""
        router = TaskRouter()
        low_trust = _make_record(
            "agent-low", [("analyze_csv", "Analyze CSV data")], trust_score=0.2
        )
        high_trust = _make_record(
            "agent-high", [("analyze_csv", "Analyze CSV data")], trust_score=0.8
        )
        await router.index_agent(low_trust)
        await router.index_agent(high_trust)

        query = DiscoveryQuery(capability_name="analyze_csv", min_trust_score=0.5)
        results = await router.match(query, [low_trust, high_trust])
        ids = [r.manifest.agent_id for r in results]
        assert "agent-high" in ids
        assert "agent-low" not in ids

    async def test_ranking_order(self):
        """Higher trust score results in higher composite ranking."""
        router = TaskRouter()
        low = _make_record(
            "agent-low-trust",
            [("analyze_csv", "Analyze CSV data")],
            trust_score=0.3,
        )
        high = _make_record(
            "agent-high-trust",
            [("analyze_csv", "Analyze CSV data")],
            trust_score=0.9,
        )
        await router.index_agent(low)
        await router.index_agent(high)

        query = DiscoveryQuery(capability_name="analyze_csv", top_k=5)
        results = await router.match(query, [low, high])
        # High-trust agent should rank first
        assert results[0].manifest.agent_id == "agent-high-trust"

    async def test_threshold_filtering(self):
        """Agents with similarity below 0.3 threshold are excluded."""
        router = TaskRouter()
        irrelevant = _make_record(
            "agent-irrelevant",
            [("send_email", "Send notifications via email")],
        )
        await router.index_agent(irrelevant)

        query = DiscoveryQuery(
            capability_description="deep statistical analysis of CSV spreadsheet data"
        )
        results = await router.match(query, [irrelevant])
        # Should not match a completely unrelated capability
        assert len(results) == 0

    async def test_tag_filter(self):
        """Tags filter excludes agents without matching tags."""
        router = TaskRouter()
        tagged = _make_record(
            "agent-tagged",
            [("analyze_csv", "Analyze data")],
            tags=["data", "csv"],
        )
        untagged = _make_record(
            "agent-untagged",
            [("analyze_csv", "Analyze data")],
            tags=["writing"],
        )
        await router.index_agent(tagged)
        await router.index_agent(untagged)

        query = DiscoveryQuery(capability_name="analyze_csv", tags=["data"])
        results = await router.match(query, [tagged, untagged])
        ids = [r.manifest.agent_id for r in results]
        assert "agent-tagged" in ids
        assert "agent-untagged" not in ids

    async def test_top_k_limit(self):
        """Results are capped at top_k."""
        router = TaskRouter()
        for i in range(5):
            rec = _make_record(
                f"agent-topk-{i}",
                [("analyze_csv", "Analyze CSV data")],
            )
            await router.index_agent(rec)

        all_agents = [
            _make_record(f"agent-topk-{i}", [("analyze_csv", "Analyze CSV data")])
            for i in range(5)
        ]
        for rec in all_agents:
            await router.index_agent(rec)

        query = DiscoveryQuery(capability_name="analyze_csv", top_k=2)
        results = await router.match(query, all_agents)
        assert len(results) <= 2

    async def test_remove_agent(self):
        """Removed agent no longer appears in search results."""
        router = TaskRouter()
        record = _make_record("agent-remove", [("analyze_csv", "Analyze data")])
        await router.index_agent(record)
        router.remove_agent("agent-remove")

        query = DiscoveryQuery(capability_name="analyze_csv")
        results = await router.match(query, [record])
        assert len(results) == 0

    async def test_empty_candidates(self):
        """match() with empty candidates list returns empty."""
        router = TaskRouter()
        query = DiscoveryQuery(capability_name="analyze_csv")
        results = await router.match(query, [])
        assert results == []


@pytest.mark.asyncio
class TestCostAwareRouting:
    async def test_budget_excludes_expensive_agent(self):
        """Agent whose cost exceeds max_cost_usd is excluded from results."""
        router = TaskRouter()
        cheap = _make_record_with_cost(
            "agent-cheap", [("analyze_csv", "Analyze CSV data", 0.001)]
        )
        expensive = _make_record_with_cost(
            "agent-expensive", [("analyze_csv", "Analyze CSV data", 0.01)]
        )
        await router.index_agent(cheap)
        await router.index_agent(expensive)

        query = DiscoveryQuery(capability_name="analyze_csv", max_cost_usd=0.005, top_k=5)
        results = await router.match(query, [cheap, expensive])
        ids = [r.manifest.agent_id for r in results]
        assert "agent-cheap" in ids
        assert "agent-expensive" not in ids

    async def test_no_budget_returns_both(self):
        """Without max_cost_usd, all cost agents are returned."""
        router = TaskRouter()
        cheap = _make_record_with_cost(
            "agent-cheap2", [("analyze_csv", "Analyze CSV data", 0.001)]
        )
        expensive = _make_record_with_cost(
            "agent-expensive2", [("analyze_csv", "Analyze CSV data", 0.01)]
        )
        await router.index_agent(cheap)
        await router.index_agent(expensive)

        query = DiscoveryQuery(capability_name="analyze_csv", top_k=5)
        results = await router.match(query, [cheap, expensive])
        ids = [r.manifest.agent_id for r in results]
        assert "agent-cheap2" in ids
        assert "agent-expensive2" in ids

    async def test_cheaper_agent_ranks_first(self):
        """Cheaper agent ranks above more expensive agent when both pass the budget."""
        router = TaskRouter()
        cheap = _make_record_with_cost(
            "agent-low-cost", [("analyze_csv", "Analyze CSV data", 0.001)],
            trust_score=0.5,
        )
        pricey = _make_record_with_cost(
            "agent-high-cost", [("analyze_csv", "Analyze CSV data", 0.009)],
            trust_score=0.5,
        )
        await router.index_agent(cheap)
        await router.index_agent(pricey)

        query = DiscoveryQuery(capability_name="analyze_csv", max_cost_usd=0.01, top_k=5)
        results = await router.match(query, [cheap, pricey])
        assert len(results) == 2
        assert results[0].manifest.agent_id == "agent-low-cost"
