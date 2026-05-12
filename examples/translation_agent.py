"""Translation Agent example — shows cost and latency annotations.

A mesh agent that translates text using Groq. Demonstrates how to
annotate a capability with cost_per_call_usd so the router can
apply budget-aware routing.

Usage:
    export GROQ_API_KEY=gsk_...

    # 1. Start the registry:
    #    uvicorn mesh.registry:app --port 8000

    # 2. Run this agent:
    python examples/translation_agent.py

    # 3. From any other MeshAgent, discover and delegate:
    #    agents = await self.discover(description="translate text between languages")
    #    result = await self.delegate("translate_text", {
    #        "text": "Hello, world!", "target_lang": "es"
    #    }, target=agents[0])
    #    print(result.output["translated"])  # "¡Hola, mundo!"
"""

from __future__ import annotations

import asyncio
import os

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from sdk.agent import MeshAgent, capability


class TranslationAgent(MeshAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        api_key = os.environ.get("GROQ_API_KEY", "")
        self._llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=api_key,  # type: ignore[arg-type]
            temperature=0.1,
        ) if api_key else None

    @capability(
        name="translate_text",
        description=(
            "Translates text between languages. Supports 50+ languages. "
            "Returns the translated text and detected source language."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "text":        {"type": "string", "description": "Text to translate"},
                "target_lang": {"type": "string", "description": "Target language (e.g. 'Spanish', 'French', 'ja')"},
                "source_lang": {"type": "string", "description": "Source language (optional, auto-detected if omitted)"},
            },
            "required": ["text", "target_lang"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "translated":       {"type": "string"},
                "detected_source":  {"type": "string"},
                "target_lang":      {"type": "string"},
            },
        },
        avg_latency_ms=1500,
        cost_per_call_usd=0.001,
    )
    async def translate_text(self, input_data: dict) -> dict:
        text = input_data.get("text", "")
        target = input_data.get("target_lang", "English")
        source = input_data.get("source_lang", "auto")

        if not self._llm:
            return {
                "translated": f"[Translation unavailable — GROQ_API_KEY not set] {text}",
                "detected_source": source,
                "target_lang": target,
            }

        prompt = (
            f"Translate the following text to {target}. "
            f"{'Detect the source language automatically.' if source == 'auto' else f'Source language: {source}.'}\n\n"
            f"Return only the translated text, no explanation.\n\n"
            f"Text: {text}"
        )
        response = await self._llm.ainvoke([HumanMessage(content=prompt)])
        return {
            "translated": response.content.strip(),
            "detected_source": source,
            "target_lang": target,
        }


async def main():
    agent = TranslationAgent(
        name="Translation Agent",
        registry_url=os.environ.get("REGISTRY_URL", "http://localhost:8000"),
        ws_port=9011,
        max_concurrent_tasks=5,
        tags=["translation", "nlp", "language"],
    )
    print("Registering Translation Agent on the mesh...")
    await agent.start()


if __name__ == "__main__":
    asyncio.run(main())
