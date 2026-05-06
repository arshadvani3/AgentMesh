"""Data Agent -- analyzes CSV/tabular data via LLM-powered statistical reasoning.

Registers on the mesh with the 'analyze_csv' capability. For the MVP the agent
uses Groq (Llama 3.3 70B) to interpret and summarize inline datasets or
structured queries. Future: connect to Google Sheets via MCP.

Run standalone:
    python -m agents.data_agent
"""

from __future__ import annotations

import asyncio
import logging
import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from agents.utils import extract_json
from sdk.agent import MeshAgent, capability

logger = logging.getLogger("agentmesh.agents.data")


class DataAgent(MeshAgent):
    """Performs statistical analysis and data summarization on tabular data."""

    def __init__(self, **kwargs):
        """Initialize DataAgent with a Groq LLM backend."""
        super().__init__(**kwargs)
        self._llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.environ["GROQ_API_KEY"],
            temperature=0.1,
        )

    @capability(
        name="analyze_csv",
        description=(
            "Statistical analysis of CSV or tabular data. "
            "Accepts a natural-language query describing what to find, "
            "plus optional inline data or a dataset description. "
            "Returns summary statistics, patterns, and insights."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to analyze or find in the data",
                },
                "data": {
                    "type": "string",
                    "description": "Inline CSV text or a description of the dataset",
                },
                "analysis_type": {
                    "type": "string",
                    "enum": ["summary", "trends", "comparison", "anomalies"],
                    "description": "Type of analysis to perform",
                },
            },
            "required": ["query"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "analysis": {"type": "string"},
                "key_findings": {"type": "array", "items": {"type": "string"}},
                "data_summary": {"type": "string"},
            },
        },
        avg_latency_ms=8000,
        cost_per_call_usd=0.002,
    )
    async def analyze_csv(self, input_data: dict) -> dict:
        """Run LLM-powered analysis on the provided data and query.

        Args:
            input_data: Dict with keys: query (required), data (optional),
                        analysis_type (optional).

        Returns:
            Dict with analysis narrative, key findings list, and data summary.
        """
        query = input_data.get("query", "")
        data = input_data.get("data", "No inline data provided.")
        analysis_type = input_data.get("analysis_type", "summary")

        system_prompt = (
            "You are a data analyst specializing in statistical analysis of tabular data. "
            "When given a dataset and a query, you provide clear, structured analysis "
            "including summary statistics, key patterns, and actionable insights. "
            "IMPORTANT: Respond with raw JSON only. No markdown fences, no explanation outside the JSON."
        )

        user_prompt = (
            f"Analysis type: {analysis_type}\n\n"
            f"Query: {query}\n\n"
            f"Dataset:\n{data}\n\n"
            "Return a JSON object with exactly these keys:\n"
            '- "analysis": comprehensive narrative analysis (string)\n'
            '- "key_findings": list of 3-5 key findings (array of strings)\n'
            '- "data_summary": one-sentence dataset description (string)\n\n'
            "Respond with raw JSON only."
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        logger.info(f"[DataAgent] Analyzing query: {query[:80]}...")
        response = await self._llm.ainvoke(messages)
        raw = response.content.strip()

        try:
            parsed = extract_json(raw)
            if isinstance(parsed, dict):
                return {
                    "analysis": parsed.get("analysis", raw),
                    "key_findings": parsed.get("key_findings", []),
                    "data_summary": parsed.get("data_summary", ""),
                }
        except ValueError:
            logger.warning("[DataAgent] Could not parse JSON from LLM response, returning raw text")

        return {
            "analysis": raw,
            "key_findings": [],
            "data_summary": f"Analysis of: {query}",
        }


# ---------------------------------------------------------------------------
# Run standalone
# ---------------------------------------------------------------------------

async def main():
    """Start the DataAgent and register it on the mesh."""
    agent = DataAgent(
        name="Data Agent",
        registry_url=os.environ.get("REGISTRY_URL", "http://localhost:8000"),
        ws_port=9002,
        tags=["data", "analysis", "csv", "spreadsheet"],
    )
    await agent.start()


if __name__ == "__main__":
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )
    asyncio.run(main())
