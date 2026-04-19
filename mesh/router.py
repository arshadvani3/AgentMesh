"""Task Router — finds the best agent for a given task.

Dual matching strategy:
1. Exact match on capability name
2. Semantic similarity on capability descriptions using sentence-transformers

Results are ranked by a composite score:
  score = (match_confidence * 0.35) + (trust_score * 0.35) + (availability * 0.15) + (cost_factor * 0.15)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from .models import AgentRecord, DiscoveryQuery


class TaskRouter:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model: SentenceTransformer | None = None
        self._model_name = model_name

        # agent_id -> list of (capability_name, description, embedding, cost_per_call_usd)
        self._index: dict[str, list[tuple[str, str, Any, float | None]]] = {}

    def _ensure_model(self):
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)

    async def index_agent(self, record: AgentRecord):
        """Index an agent's capabilities for semantic search."""
        self._ensure_model()

        entries = []
        for cap in record.manifest.capabilities:
            text = f"{cap.name}: {cap.description}"
            embedding = self._model.encode(text, normalize_embeddings=True)
            entries.append((cap.name, cap.description, embedding, cap.cost_per_call_usd))

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

        self._ensure_model()
        scored: list[tuple[float, AgentRecord]] = []

        for record in candidates:
            agent_id = record.manifest.agent_id

            # Skip offline agents entirely
            if record.status == "offline":
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

            for cap_name, _cap_desc, cap_emb, cap_cost in self._index[agent_id]:
                # Cost filter: skip capability if it exceeds budget
                if (
                    cap_cost is not None
                    and query.max_cost_usd is not None
                    and cap_cost > query.max_cost_usd
                ):
                    continue

                # 1. Exact name match
                if query.capability_name and query.capability_name == cap_name:
                    best_match = max(best_match, 1.0)
                    if cap_cost is not None:
                        best_cost = cap_cost
                    continue

                # 2. Semantic similarity
                if query.capability_description:
                    query_emb = self._model.encode(
                        query.capability_description, normalize_embeddings=True
                    )
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
            if record.status == "degraded":
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
