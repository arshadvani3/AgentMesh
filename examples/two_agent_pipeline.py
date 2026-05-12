"""Two-agent pipeline — shows discover() and delegate() without LangGraph.

Demonstrates the SDK working standalone: one agent discovers another
and delegates a task directly, no orchestration framework needed.

Setup:
    # Terminal 1 — start the registry:
    uvicorn mesh.registry:app --port 8000

    # Terminal 2 — start the worker agent:
    python examples/two_agent_pipeline.py worker

    # Terminal 3 — run the requester (delegates a task):
    python examples/two_agent_pipeline.py requester

The requester discovers the worker on the mesh, delegates a summarise
task, and prints the result. No hardcoded endpoints — pure discovery.
"""

from __future__ import annotations

import asyncio
import os
import sys

from sdk.agent import MeshAgent, capability

# ---------------------------------------------------------------------------
# Worker agent — provides the "summarise_text" capability
# ---------------------------------------------------------------------------

class WorkerAgent(MeshAgent):
    @capability(
        name="summarise_text",
        description="Summarises a block of text into a single sentence.",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        output_schema={
            "type": "object",
            "properties": {"summary": {"type": "string"}},
        },
        avg_latency_ms=200,
        cost_per_call_usd=0.0,
    )
    async def summarise_text(self, input_data: dict) -> dict:
        text = input_data.get("text", "")
        words = text.split()
        summary = " ".join(words[:15]) + ("..." if len(words) > 15 else "")
        return {"summary": summary}


# ---------------------------------------------------------------------------
# Requester agent — discovers the worker and delegates to it
# ---------------------------------------------------------------------------

class RequesterAgent(MeshAgent):
    async def run_task(self, text: str) -> str:
        print("[Requester] Discovering agents that can summarise text...")
        agents = await self.discover(
            capability_name="summarise_text",
            min_trust=0.0,
        )

        if not agents:
            return "No agents found for 'summarise_text'"

        print(f"[Requester] Found {len(agents)} agent(s). Delegating to: {agents[0].manifest.name}")
        result = await self.delegate(
            capability="summarise_text",
            input_data={"text": text},
            target=agents[0],
        )

        if result.status.value == "completed":
            return result.output.get("summary", "")
        return f"Task failed: {result.error}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def run_worker():
    agent = WorkerAgent(
        name="Summariser Worker",
        registry_url=os.environ.get("REGISTRY_URL", "http://localhost:8000"),
        ws_port=9020,
        tags=["summarisation", "nlp", "example"],
    )
    print("[Worker] Registered. Waiting for tasks...")
    await agent.start()


async def run_requester():
    agent = RequesterAgent(
        name="Pipeline Requester",
        registry_url=os.environ.get("REGISTRY_URL", "http://localhost:8000"),
        ws_port=9021,
        tags=["example"],
    )

    # Register so we can use discover() and delegate()
    import httpx

    from mesh.models import AgentRecord
    manifest = agent._build_manifest()
    agent._agent_id = manifest.agent_id
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{agent.registry_url}/agents/register",
            json=manifest.model_dump(mode="json"),
        )
        resp.raise_for_status()
        agent._record = AgentRecord(**resp.json())

    sample_text = (
        "AgentMesh is a peer-to-peer agent discovery framework where agents register "
        "capabilities and discover each other at runtime using semantic search, then "
        "negotiate task contracts over WebSocket and earn ELO trust scores."
    )

    print("[Requester] Delegating summarisation task...")
    summary = await agent.run_task(sample_text)
    print(f"\n[Result] {summary}\n")

    async with httpx.AsyncClient() as client:
        await client.delete(f"{agent.registry_url}/agents/{agent.agent_id}")


if __name__ == "__main__":
    role = sys.argv[1] if len(sys.argv) > 1 else "worker"
    if role == "worker":
        asyncio.run(run_worker())
    elif role == "requester":
        asyncio.run(run_requester())
    else:
        print("Usage: python examples/two_agent_pipeline.py [worker|requester]")
