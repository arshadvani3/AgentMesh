"""Task Router — finds the best agent for a given task.

Dual matching strategy:
1. Exact match on capability name
2. Semantic similarity on capability descriptions using sentence-transformers

Results are ranked by a composite score:
  score = (match_confidence * 0.35) + (trust_score * 0.35) + (availability * 0.15) + (cost_factor * 0.15)
"""

from __future__ import annotations

import asyncio
import functools
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

        for record in candidates:
            agent_id = record.manifest.agent_id

            # Skip offline agents entirely
            if record.status == AgentStatus.OFFLINE:
                continue

            if agent_id not in self._index:
                continue

            # Apply filters
            if record.trust_score < query.min_trust_score:
                continue
            if query.tags and not set(query.tags).intersection(record.manifest.tags):
                continue

            best_match = 0.0
            best_cost: float | None = None

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

                # 1. Exact name match
                if query.capability_name and query.capability_name == cap_name:
                    best_match = max(best_match, 1.0)
                    if cap_cost is not None:
                        best_cost = cap_cost
                    continue

                # 2. Semantic similarity
                if query_emb is not None:
                    similarity = float(np.dot(query_emb, cap_emb))
                    best_match = max(best_match, similarity)

                # 3. Partial name match
                if query.capability_name and query.capability_name in cap_name:
                    best_match = max(best_match, 0.7)

                if cap_cost is not None:
                    best_cost = cap_cost

            if best_match < 0.3:  # threshold
                continue

            # Compute real availability based on active task load
            max_tasks = record.manifest.max_concurrent_tasks or 3
            active = (task_counts or {}).get(agent_id, 0)
            availability = max(0.0, 1.0 - (active / max_tasks))

            # Degraded agents are still routable but penalised
            if record.status == AgentStatus.DEGRADED:
                availability *= 0.3

            # Compute cost factor: cheaper relative to budget scores higher
            if query.max_cost_usd and query.max_cost_usd > 0 and best_cost is not None:
                cost_factor = 1.0 - min(best_cost / query.max_cost_usd, 1.0)
            else:
                cost_factor = 1.0

            composite = (best_match * 0.35) + (record.trust_score * 0.35) + (availability * 0.15) + (cost_factor * 0.15)
            scored.append((composite, record))

        # Sort descending by score
        scored.sort(key=lambda x: x[0], reverse=True)
        return [record for _, record in scored[: query.top_k]]
