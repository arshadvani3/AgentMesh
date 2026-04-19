"""Code Agent -- retrieves and generates relevant code examples via LLM reasoning.

Registers on the mesh with the 'fetch_code' capability. For the MVP the agent
uses Groq (Llama 3.3 70B) to generate high-quality code examples based on
natural-language queries. Future: connect to GitHub via MCP for real repo search.

Run standalone:
    python -m agents.code_agent
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from sdk.agent import MeshAgent, capability

logger = logging.getLogger("agentmesh.agents.code")


class CodeAgent(MeshAgent):
    """Generates and retrieves code examples for programming queries."""

    def __init__(self, **kwargs):
        """Initialize CodeAgent with a Groq LLM backend."""
        super().__init__(**kwargs)
        self._llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.environ["GROQ_API_KEY"],
            temperature=0.2,
        )

    @capability(
        name="fetch_code",
        description=(
            "Fetches or generates relevant code examples and patterns for a given query. "
            "Handles questions about code architecture, library usage, tool calling, "
            "framework comparisons, and best practices. "
            "Returns annotated code snippets with explanations."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What code patterns or examples to find",
                },
                "language": {
                    "type": "string",
                    "description": "Programming language (default: python)",
                },
                "framework": {
                    "type": "string",
                    "description": "Specific framework or library to target",
                },
                "num_examples": {
                    "type": "integer",
                    "description": "Number of examples to return (default: 2)",
                },
            },
            "required": ["query"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "examples": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "code": {"type": "string"},
                            "explanation": {"type": "string"},
                            "language": {"type": "string"},
                        },
                    },
                },
                "summary": {"type": "string"},
            },
        },
        avg_latency_ms=10000,
        cost_per_call_usd=0.003,
    )
    async def fetch_code(self, input_data: dict) -> dict:
        """Generate annotated code examples for the given query.

        Args:
            input_data: Dict with keys: query (required), language (optional),
                        framework (optional), num_examples (optional).

        Returns:
            Dict with examples list (title, code, explanation, language) and summary.
        """
        query = input_data.get("query", "")
        language = input_data.get("language", "python")
        framework = input_data.get("framework", "")
        num_examples = input_data.get("num_examples", 2)

        framework_clause = f" using {framework}" if framework else ""

        system_prompt = (
            "You are an expert software engineer with deep knowledge of AI frameworks, "
            "Python, TypeScript, and modern development patterns. "
            "You produce clean, well-commented, production-quality code examples. "
            "Always include realistic, runnable code -- not pseudocode."
        )

        user_prompt = (
            f"Generate {num_examples} code example(s) for:\n"
            f"Query: {query}\n"
            f"Language: {language}{framework_clause}\n\n"
            "Return a JSON object with:\n"
            "- 'examples': array of objects, each with 'title', 'code', 'explanation', 'language'\n"
            "- 'summary': one paragraph describing the overall pattern\n\n"
            "Make code examples realistic and production-ready. "
            "Include imports and all necessary context."
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        logger.info(f"[CodeAgent] Fetching code for: {query[:80]}...")
        response = await self._llm.ainvoke(messages)
        raw = response.content.strip()

        # Extract JSON block
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                return {
                    "examples": parsed.get("examples", []),
                    "summary": parsed.get("summary", ""),
                }
            except json.JSONDecodeError:
                pass

        # Fallback: wrap raw content as a single example
        return {
            "examples": [
                {
                    "title": f"Example: {query[:60]}",
                    "code": raw,
                    "explanation": "Generated by CodeAgent",
                    "language": language,
                }
            ],
            "summary": f"Code examples for: {query}",
        }


# ---------------------------------------------------------------------------
# Run standalone
# ---------------------------------------------------------------------------

async def main():
    """Start the CodeAgent and register it on the mesh."""
    agent = CodeAgent(
        name="Code Agent",
        registry_url=os.environ.get("REGISTRY_URL", "http://localhost:8000"),
        ws_port=9003,
        tags=["code", "github", "programming"],
    )
    await agent.start()


if __name__ == "__main__":
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )
    asyncio.run(main())
