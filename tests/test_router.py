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
        endpoint="ws://localhost:9000",
        capabilities=capabilities,
        tags=tags or [],
    )
    return AgentRecord(manifest=manifest, trust_score=trust_score)


def _make_record_with_endpoint(
    agent_id: str,
    caps: list[tuple[str, str]],
    endpoint: str,
    trust_score: float = 0.5,
    tasks_completed: int = 20,
) -> AgentRecord:
    capabilities = [CapabilitySchema(name=n, description=d) for n, d in caps]
    manifest = AgentManifest(
        agent_id=agent_id,
        name=agent_id,
        endpoint=endpoint,
        capabilities=capabilities,
    )
    record = AgentRecord(manifest=manifest, trust_score=trust_score)
    record.tasks_completed = tasks_completed
    return record


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
        """Agent with matching tags ranks above one with no overlap (soft scoring).

        Uses a semantic query (not exact name) so the tag_overlap boost affects match_score.
        Exact-name matches always get match_score=1.0, making tag overlap irrelevant there.
        """
        router = TaskRouter()
        tagged = _make_record(
            "agent-tagged",
            [("analyze_csv", "Statistical analysis of tabular data")],
            tags=["data", "csv"],
        )
        tagged.tasks_completed = 20
        no_overlap = _make_record(
            "agent-no-overlap",
            [("analyze_csv", "Statistical analysis of tabular data")],
            tags=["writing"],
        )
        no_overlap.tasks_completed = 20
        await router.index_agent(tagged)
        await router.index_agent(no_overlap)

        # Semantic query + tags — agent with matching tags should rank first
        query = DiscoveryQuery(
            capability_description="analyze CSV spreadsheet data",
            tags=["data"],
            top_k=5,
        )
        results = await router.match(query, [tagged, no_overlap])
        assert len(results) >= 1
        assert results[0].manifest.agent_id == "agent-tagged"

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


@pytest.mark.asyncio
class TestConfidenceWeightedTrust:
    async def test_new_agent_gets_cold_start_floor(self):
        """Agent with 0 tasks gets 30% of face-value trust."""
        router = TaskRouter()
        record = _make_record("agent-cold", [("analyze_csv", "x")], trust_score=1.0)
        record.tasks_completed = 0
        assert abs(router._effective_trust(record) - 0.3) < 1e-9

    async def test_ten_tasks_gives_full_trust(self):
        """Agent with 10 completed tasks gets full face-value trust."""
        router = TaskRouter()
        record = _make_record("agent-warm", [("analyze_csv", "x")], trust_score=0.8)
        record.tasks_completed = 10
        assert abs(router._effective_trust(record) - 0.8) < 1e-9

    async def test_proven_agent_beats_new_same_raw_trust(self):
        """Proven agent (50 tasks) ranks above brand-new agent with identical trust."""
        router = TaskRouter()
        proven = _make_record("agent-proven", [("analyze_csv", "Analyze CSV data")], trust_score=0.8)
        proven.tasks_completed = 50
        new_agent = _make_record("agent-new", [("analyze_csv", "Analyze CSV data")], trust_score=0.8)
        new_agent.tasks_completed = 0

        await router.index_agent(proven)
        await router.index_agent(new_agent)

        query = DiscoveryQuery(capability_name="analyze_csv", top_k=5)
        results = await router.match(query, [proven, new_agent])
        assert results[0].manifest.agent_id == "agent-proven"

    async def test_partial_confidence_is_between_floor_and_full(self):
        """Agent with 5 tasks has effective trust between 30% and 100% of face-value."""
        router = TaskRouter()
        record = _make_record("agent-mid", [("analyze_csv", "x")], trust_score=1.0)
        record.tasks_completed = 5
        effective = router._effective_trust(record)
        assert 0.3 < effective < 1.0


@pytest.mark.asyncio
class TestDynamicWeights:
    async def test_latency_sensitive_prefers_idle_agent(self):
        """With max_latency_ms set, an idle low-trust agent beats a loaded high-trust agent."""
        router = TaskRouter()
        loaded = _make_record("agent-loaded", [("analyze_csv", "Analyze CSV data")], trust_score=0.9)
        loaded.tasks_completed = 20
        idle = _make_record("agent-idle", [("analyze_csv", "Analyze CSV data")], trust_score=0.6)
        idle.tasks_completed = 20

        await router.index_agent(loaded)
        await router.index_agent(idle)

        task_counts = {"agent-loaded": 3, "agent-idle": 0}  # loaded is at capacity
        query = DiscoveryQuery(capability_name="analyze_csv", max_latency_ms=5000, top_k=5)
        results = await router.match(query, [loaded, idle], task_counts=task_counts)
        assert results[0].manifest.agent_id == "agent-idle"

    async def test_cost_sensitive_prefers_cheap_agent(self):
        """With max_cost_usd set, cost weight increases — cheap agent beats pricey one."""
        router = TaskRouter()
        cheap = _make_record_with_cost(
            "agent-cheap-dyn", [("analyze_csv", "Analyze CSV data", 0.001)], trust_score=0.5
        )
        pricey = _make_record_with_cost(
            "agent-pricey-dyn", [("analyze_csv", "Analyze CSV data", 0.008)], trust_score=0.5
        )
        cheap.tasks_completed = 20
        pricey.tasks_completed = 20

        await router.index_agent(cheap)
        await router.index_agent(pricey)

        query = DiscoveryQuery(capability_name="analyze_csv", max_cost_usd=0.01, top_k=5)
        results = await router.match(query, [cheap, pricey])
        assert results[0].manifest.agent_id == "agent-cheap-dyn"

    async def test_weights_sum_to_one_no_signals(self):
        """Base weights sum to 1.0 when no latency or cost signals are set."""
        query = DiscoveryQuery(capability_name="analyze_csv")
        w_match, w_trust, w_avail, w_cost = TaskRouter._dynamic_weights(query)
        assert abs(w_match + w_trust + w_avail + w_cost - 1.0) < 1e-9

    async def test_weights_sum_to_one_with_signals(self):
        """Weights still sum to 1.0 when both latency and cost signals are present."""
        query = DiscoveryQuery(capability_name="analyze_csv", max_latency_ms=1000, max_cost_usd=0.005)
        w_match, w_trust, w_avail, w_cost = TaskRouter._dynamic_weights(query)
        assert abs(w_match + w_trust + w_avail + w_cost - 1.0) < 1e-9


@pytest.mark.asyncio
class TestTagSoftScoring:
    async def test_tag_overlap_boosts_rank(self):
        """Agent with more tag overlap ranks above one with less, same trust and capability."""
        router = TaskRouter()
        good_tags = _make_record(
            "agent-good-tags",
            [("analyze_csv", "Analyze CSV data")],
            trust_score=0.5,
            tags=["data", "csv", "analysis"],
        )
        good_tags.tasks_completed = 20
        weak_tags = _make_record(
            "agent-weak-tags",
            [("analyze_csv", "Analyze CSV data")],
            trust_score=0.5,
            tags=["writing"],
        )
        weak_tags.tasks_completed = 20

        await router.index_agent(good_tags)
        await router.index_agent(weak_tags)

        query = DiscoveryQuery(capability_name="analyze_csv", tags=["data", "csv"], top_k=5)
        results = await router.match(query, [good_tags, weak_tags])
        assert results[0].manifest.agent_id == "agent-good-tags"

    async def test_partial_tag_overlap_still_included(self):
        """Agent with partial tag match still appears — tags are no longer a hard filter."""
        router = TaskRouter()
        partial = _make_record(
            "agent-partial",
            [("analyze_csv", "Analyze CSV data")],
            trust_score=0.5,
            tags=["data"],
        )
        partial.tasks_completed = 20
        await router.index_agent(partial)

        query = DiscoveryQuery(capability_name="analyze_csv", tags=["data", "csv"], top_k=5)
        results = await router.match(query, [partial])
        assert len(results) == 1
        assert results[0].manifest.agent_id == "agent-partial"


@pytest.mark.asyncio
class TestDiversityReranking:
    async def test_prefers_different_hosts(self):
        """When top candidates share a host, a lower-scored agent on a fresh host is preferred."""
        router = TaskRouter()
        host_a_1 = _make_record_with_endpoint(
            "agent-a1", [("analyze_csv", "Analyze CSV data")],
            endpoint="ws://host-a:9001", trust_score=0.9,
        )
        host_a_2 = _make_record_with_endpoint(
            "agent-a2", [("analyze_csv", "Analyze CSV data")],
            endpoint="ws://host-a:9002", trust_score=0.85,
        )
        host_b = _make_record_with_endpoint(
            "agent-b", [("analyze_csv", "Analyze CSV data")],
            endpoint="ws://host-b:9001", trust_score=0.7,
        )
        await router.index_agent(host_a_1)
        await router.index_agent(host_a_2)
        await router.index_agent(host_b)

        query = DiscoveryQuery(capability_name="analyze_csv", top_k=2)
        results = await router.match(query, [host_a_1, host_a_2, host_b])
        ids = [r.manifest.agent_id for r in results]

        # First result must be host-a1 (highest score)
        assert ids[0] == "agent-a1"
        # Second result must be host-b (diversity penalty knocks host-a2 below host-b)
        assert ids[1] == "agent-b"

    async def test_same_host_still_selected_if_only_option(self):
        """If all candidates share a host, they are still returned (diversity is a preference)."""
        router = TaskRouter()
        a1 = _make_record_with_endpoint(
            "agent-same-1", [("analyze_csv", "Analyze CSV data")],
            endpoint="ws://same-host:9001", trust_score=0.9,
        )
        a2 = _make_record_with_endpoint(
            "agent-same-2", [("analyze_csv", "Analyze CSV data")],
            endpoint="ws://same-host:9002", trust_score=0.8,
        )
        await router.index_agent(a1)
        await router.index_agent(a2)

        query = DiscoveryQuery(capability_name="analyze_csv", top_k=2)
        results = await router.match(query, [a1, a2])
        assert len(results) == 2
