"""Task Router — finds the best agent for a given task.

Dual matching strategy:
1. Exact match on capability name  → match_score = 1.0
2. Semantic similarity on capability descriptions via sentence-transformers
3. Tag overlap as a soft boost on top of semantic similarity

Results are ranked by a composite score with dynamic weights:

  match_score    = semantic_similarity * 0.85 + tag_overlap * 0.15
  effective_trust = trust_score weighted by interaction volume
                   (new agents start humble; full trust after ~10 tasks)

  Base weights:  match=0.40  trust=0.30  availability=0.15  cost=0.15
  Adjustments:
    max_latency_ms set  → availability +0.10, trust -0.10  (latency-sensitive caller)
    max_cost_usd set    → cost        +0.10, trust -0.10  (budget-sensitive caller)
"""

from __future__ import annotations

import asyncio
import functools
import urllib.parse
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from .models import AgentRecord, AgentStatus, DiscoveryQuery


class TaskRouter:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model: SentenceTransformer | None = None
        self._model_name = model_name

        # agent_id -> list of (capability_name, description, embedding, cost_per_call_usd, avg_latency_ms)
        self._index: dict[str, list[tuple[str, str, Any, float | None, float | None]]] = {}

    def _ensure_model(self):
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)

    async def _encode(self, text: str) -> Any:
        """Run model.encode() in a thread so it doesn't block the event loop."""
        self._ensure_model()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, functools.partial(self._model.encode, text, normalize_embeddings=True)
        )

    async def index_agent(self, record: AgentRecord):
        """Index an agent's capabilities for semantic search."""
        entries = []
        for cap in record.manifest.capabilities:
            text = f"{cap.name}: {cap.description}"
            embedding = await self._encode(text)
            entries.append((cap.name, cap.description, embedding, cap.cost_per_call_usd, cap.avg_latency_ms))

        self._index[record.manifest.agent_id] = entries

    def remove_agent(self, agent_id: str):
        """Remove an agent from the search index."""
        self._index.pop(agent_id, None)

    @staticmethod
    def _dynamic_weights(query: DiscoveryQuery) -> tuple[float, float, float, float]:
        """Compute routing weights based on what the caller signals they care about.

        Base: match=0.40  trust=0.30  availability=0.15  cost=0.15
        When max_latency_ms is set, availability matters more (latency-sensitive).
        When max_cost_usd is set, cost matters more (budget-sensitive).
        Weights always sum to 1.0.
        """
        w_match = 0.40
        w_trust = 0.30
        w_avail = 0.15
        w_cost  = 0.15

        if query.max_latency_ms is not None:
            w_avail += 0.10
            w_trust -= 0.10

        if query.max_cost_usd is not None:
            w_cost  += 0.10
            w_trust -= 0.10

        return w_match, w_trust, w_avail, w_cost

    @staticmethod
    def _effective_trust(record: AgentRecord) -> float:
        """Confidence-weighted trust score.

        New agents (few completed tasks) are discounted so a high trust score
        from a single lucky task doesn't dominate over a proven agent.
        Agents with 0 tasks get 30% of face-value trust (avoids a cold-start cliff).
        Full face-value trust is reached at ~10 completed tasks.
        """
        confidence = min(record.tasks_completed / 10.0, 1.0)
        return record.trust_score * confidence + record.trust_score * 0.3 * (1.0 - confidence)

    async def match(
        self,
        query: DiscoveryQuery,
        candidates: list[AgentRecord],
        task_counts: dict[str, int] | None = None,
    ) -> list[AgentRecord]:
        """Find and rank agents matching a discovery query.

        Returns up to query.top_k agents, sorted by composite score.
        """
        # security: limit input length to prevent DoS via huge embeddings
        if query.capability_description and len(query.capability_description) > 1000:
            query = query.model_copy(update={"capability_description": query.capability_description[:1000]})

        scored: list[tuple[float, AgentRecord]] = []

        # Encode query once — reused for every candidate capability comparison
        query_emb = None
        if query.capability_description:
            query_emb = await self._encode(query.capability_description)

        w_match, w_trust, w_avail, w_cost = self._dynamic_weights(query)

        for record in candidates:
            agent_id = record.manifest.agent_id

            if record.status == AgentStatus.OFFLINE:
                continue

            if agent_id not in self._index:
                continue

            if record.trust_score < query.min_trust_score:
                continue

            # Tags: soft overlap score (0.0–1.0) used to boost match_score
            # No longer a hard filter — partial overlap still contributes
            tag_overlap = 0.0
            if query.tags and record.manifest.tags:
                matched_tags = len(set(query.tags) & set(record.manifest.tags))
                tag_overlap = matched_tags / len(query.tags)
            elif query.tags and not record.manifest.tags:
                # Caller specified tags but agent has none — small penalty
                tag_overlap = 0.0

            best_semantic = 0.0
            best_cost: float | None = None
            exact_hit = False

            for cap_name, _cap_desc, cap_emb, cap_cost, cap_latency in self._index[agent_id]:
                # Cost filter: skip capability if it exceeds budget
                if (
                    cap_cost is not None
                    and query.max_cost_usd is not None
                    and cap_cost > query.max_cost_usd
                ):
                    continue

                # Latency filter: skip capability if tracked latency exceeds max
                if (
                    cap_latency is not None
                    and query.max_latency_ms is not None
                    and cap_latency > query.max_latency_ms
                ):
                    continue

                # 1. Exact name match — always wins
                if query.capability_name and query.capability_name == cap_name:
                    exact_hit = True
                    if cap_cost is not None:
                        best_cost = cap_cost
                    continue

                # 2. Semantic similarity
                if query_emb is not None:
                    similarity = float(np.dot(query_emb, cap_emb))
                    best_semantic = max(best_semantic, similarity)

                # 3. Partial name match fallback
                if query.capability_name and query.capability_name in cap_name:
                    best_semantic = max(best_semantic, 0.7)

                if cap_cost is not None:
                    best_cost = cap_cost

            # Compute match_score: exact hit = 1.0, otherwise blend semantic + tag overlap
            if exact_hit:
                match_score = 1.0
            else:
                if best_semantic < 0.3:
                    continue
                match_score = best_semantic * 0.85 + tag_overlap * 0.15

            # Availability: live task load + degraded penalty
            max_tasks = record.manifest.max_concurrent_tasks or 3
            active = (task_counts or {}).get(agent_id, 0)
            availability = max(0.0, 1.0 - (active / max_tasks))
            if record.status == AgentStatus.DEGRADED:
                availability *= 0.3

            # Cost factor: cheaper relative to budget scores higher
            if query.max_cost_usd and query.max_cost_usd > 0 and best_cost is not None:
                cost_factor = 1.0 - min(best_cost / query.max_cost_usd, 1.0)
            else:
                cost_factor = 1.0

            # Confidence-weighted trust: discounts agents with few completed tasks
            effective_trust = self._effective_trust(record)

            composite = (
                match_score    * w_match
                + effective_trust * w_trust
                + availability    * w_avail
                + cost_factor     * w_cost
            )
            scored.append((composite, record))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Diversity re-ranking: iteratively select candidates, penalising any
        # remaining candidate whose host was already selected. A second agent on
        # the same host must score 0.15 higher than a fresh-host candidate to win.
        selected: list[AgentRecord] = []
        selected_hosts: set[str] = set()
        remaining = list(scored)

        while len(selected) < query.top_k and remaining:
            penalised = [
                (
                    score - (0.15 if (urllib.parse.urlparse(r.manifest.endpoint).hostname or "") in selected_hosts else 0.0),
                    r,
                )
                for score, r in remaining
            ]
            penalised.sort(key=lambda x: x[0], reverse=True)
            _, best = penalised[0]
            selected.append(best)
            selected_hosts.add(urllib.parse.urlparse(best.manifest.endpoint).hostname or "")
            remaining = [(s, r) for s, r in remaining if r is not best]

        return selected
