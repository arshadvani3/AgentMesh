"""Minimal AgentMesh example — a hello-world agent.

Shows the bare minimum to register a capability on the mesh,
respond to requests, and shut down cleanly.

Usage:
    # 1. Start the registry (in a separate terminal):
    #    uvicorn mesh.registry:app --port 8000

    # 2. Run this agent:
    python examples/hello_agent.py
"""

from __future__ import annotations

import asyncio
import os

from sdk.agent import MeshAgent, capability


class HelloAgent(MeshAgent):
    @capability(
        name="say_hello",
        description="Returns a friendly greeting for a given name.",
        input_schema={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
        output_schema={
            "type": "object",
            "properties": {"message": {"type": "string"}},
        },
        avg_latency_ms=50,
        cost_per_call_usd=0.0,
    )
    async def say_hello(self, input_data: dict) -> dict:
        name = input_data.get("name", "world")
        return {"message": f"Hello, {name}! I am a mesh agent."}


async def main():
    agent = HelloAgent(
        name="Hello Agent",
        registry_url=os.environ.get("REGISTRY_URL", "http://localhost:8000"),
        ws_port=9010,
        tags=["example", "hello"],
    )
    print("Registering Hello Agent on the mesh...")
    await agent.start()


if __name__ == "__main__":
    asyncio.run(main())
