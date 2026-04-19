"""Writer Agent -- synthesizes multi-source data into polished markdown reports.

Registers on the mesh with the 'write_report' capability. Uses Groq (Llama 3.3 70B)
to weave together data from data_agent, code_agent, and other sources into a
cohesive, professionally structured document.

Run standalone:
    python -m agents.writer_agent
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

logger = logging.getLogger("agentmesh.agents.writer")


class WriterAgent(MeshAgent):
    """Produces formatted markdown reports by synthesizing multi-source data."""

    def __init__(self, **kwargs):
        """Initialize WriterAgent with a Groq LLM backend."""
        super().__init__(**kwargs)
        self._llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.environ["GROQ_API_KEY"],
            temperature=0.3,
        )

    @capability(
        name="write_report",
        description=(
            "Produces a well-structured, professional markdown report on any topic. "
            "Synthesizes data gathered from multiple agents (data analysis, code examples, "
            "research findings) into a cohesive document with clear sections, "
            "executive summary, and actionable conclusions."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The main subject of the report",
                },
                "data": {
                    "type": "object",
                    "description": "Gathered data from other agents keyed by source type",
                    "additionalProperties": True,
                },
                "report_type": {
                    "type": "string",
                    "enum": ["analysis", "comparison", "tutorial", "summary"],
                    "description": "Style of report to produce (default: analysis)",
                },
                "audience": {
                    "type": "string",
                    "description": "Target audience (default: technical)",
                },
            },
            "required": ["topic"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "report": {"type": "string", "description": "Full markdown report"},
                "word_count": {"type": "integer"},
                "sections": {"type": "array", "items": {"type": "string"}},
            },
        },
        avg_latency_ms=12000,
        cost_per_call_usd=0.005,
    )
    async def write_report(self, input_data: dict) -> dict:
        """Synthesize gathered data into a structured markdown report.

        Args:
            input_data: Dict with keys: topic (required), data (optional),
                        report_type (optional), audience (optional).

        Returns:
            Dict with full markdown report, word count, and section list.
        """
        topic = input_data.get("topic", "")
        data = input_data.get("data", {})
        report_type = input_data.get("report_type", "analysis")
        audience = input_data.get("audience", "technical")

        # Build context from gathered data
        context_sections = []
        if "data_analysis" in data:
            da = data["data_analysis"]
            if isinstance(da, dict):
                context_sections.append(
                    f"DATA ANALYSIS:\n{da.get('analysis', str(da))}"
                )
            else:
                context_sections.append(f"DATA ANALYSIS:\n{da}")

        if "code_examples" in data:
            ce = data["code_examples"]
            if isinstance(ce, dict):
                examples = ce.get("examples", [])
                code_text = "\n\n".join(
                    f"### {ex.get('title', 'Example')}\n```{ex.get('language','')}\n{ex.get('code','')}\n```\n{ex.get('explanation','')}"
                    for ex in examples
                )
                context_sections.append(f"CODE EXAMPLES:\n{code_text}")
            else:
                context_sections.append(f"CODE EXAMPLES:\n{ce}")

        # Add any other data sources
        for key, value in data.items():
            if key not in ("data_analysis", "code_examples"):
                context_sections.append(f"{key.upper().replace('_', ' ')}:\n{value}")

        context_text = "\n\n---\n\n".join(context_sections) if context_sections else "No external data provided."

        system_prompt = (
            "You are a senior technical writer with expertise in AI/ML, software engineering, "
            "and business analysis. You produce clear, well-structured reports that are "
            "accurate, insightful, and engaging. You always include an executive summary, "
            "organized sections with proper markdown headers, and concrete conclusions."
        )

        user_prompt = (
            f"Write a {report_type} report on the following topic:\n"
            f"Topic: {topic}\n"
            f"Audience: {audience}\n\n"
            f"Use the following gathered data as source material:\n\n"
            f"{context_text}\n\n"
            "Requirements:\n"
            "- Use proper markdown with ## headers\n"
            "- Start with an Executive Summary\n"
            "- Include all relevant data from the provided sources\n"
            "- End with Conclusions and/or Recommendations\n"
            "- Be comprehensive but concise -- aim for 800-1200 words\n"
            "- If code examples are provided, include them with proper fencing\n\n"
            "Return a JSON object with:\n"
            "- 'report': the full markdown report as a string\n"
            "- 'sections': array of section header names\n"
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        logger.info(f"[WriterAgent] Writing {report_type} report on: {topic[:80]}...")
        response = await self._llm.ainvoke(messages)
        raw = response.content.strip()

        # Extract JSON block
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                report_text = parsed.get("report", raw)
                sections = parsed.get("sections", [])
                return {
                    "report": report_text,
                    "word_count": len(report_text.split()),
                    "sections": sections,
                }
            except json.JSONDecodeError:
                pass

        # Fallback: the raw content is the report
        return {
            "report": raw,
            "word_count": len(raw.split()),
            "sections": [],
        }


# ---------------------------------------------------------------------------
# Run standalone
# ---------------------------------------------------------------------------

async def main():
    """Start the WriterAgent and register it on the mesh."""
    agent = WriterAgent(
        name="Writer Agent",
        registry_url=os.environ.get("REGISTRY_URL", "http://localhost:8000"),
        ws_port=9004,
        tags=["writing", "reports", "documentation"],
    )
    await agent.start()


if __name__ == "__main__":
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )
    asyncio.run(main())
