"""Research Agent -- orchestrates multi-agent research via a LangGraph workflow.

This agent receives complex queries, uses a LangGraph StateGraph to decompose
them into subtasks, discovers specialized agents on the mesh, delegates work,
and synthesizes the results into a final report.

Key design: the registry is just a phone book. This agent discovers collaborators
dynamically at runtime -- no hardwired connections.

Run standalone:
    python -m agents.research_agent
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from agents.utils import extract_json
from sdk.agent import MeshAgent, capability

logger = logging.getLogger("agentmesh.agents.research")


# ---------------------------------------------------------------------------
# LangGraph State
# ---------------------------------------------------------------------------

class ResearchState(TypedDict):
    """Shared state flowing through the LangGraph research pipeline."""

    original_query: str
    subtasks: list[dict[str, str]]       # [{capability, description, context}]
    discovered_agents: dict[str, Any]    # capability -> AgentRecord
    delegation_results: dict[str, Any]   # capability -> TaskResult.output
    final_report: str
    errors: list[str]


# ---------------------------------------------------------------------------
# ResearchAgent
# ---------------------------------------------------------------------------

class ResearchAgent(MeshAgent):
    """Decomposes research tasks and delegates to mesh collaborators via LangGraph."""

    def __init__(self, **kwargs):
        """Initialize ResearchAgent with Groq LLM and LangGraph workflow."""
        super().__init__(**kwargs)
        self._llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.environ["GROQ_API_KEY"],
            temperature=0.2,
        )
        self._graph = self._build_graph()

    # ---------------------------------------------------------------------------
    # LangGraph nodes
    # ---------------------------------------------------------------------------

    async def _plan_node(self, state: ResearchState) -> dict:
        """Decompose the research query into subtasks based on live mesh capabilities.

        Discovers what agents are currently registered on the mesh, then asks
        the LLM to decompose the query into subtasks that match available capabilities.
        This makes the Research Agent truly dynamic — it adapts to whatever is online.

        Args:
            state: Current ResearchState with original_query set.

        Returns:
            Dict with 'subtasks' list of capability/description/context dicts.
        """
        query = state["original_query"]
        logger.info(f"[Research/plan] Planning subtasks for: {query[:80]}...")

        # Discover live capabilities from the mesh instead of hardcoding them
        capability_list = ""
        try:
            available = await self.discover(top_k=10)
            if available:
                cap_lines = []
                for rec in available:
                    for cap in rec.manifest.capabilities:
                        cap_lines.append(f"- '{cap.name}': {cap.description}")
                capability_list = "\n".join(cap_lines)
                logger.info(f"[Research/plan] Found {len(cap_lines)} capabilities on mesh")
        except Exception as e:
            logger.warning(f"[Research/plan] Could not fetch live capabilities: {e}")

        # Fall back to known defaults if discovery failed
        if not capability_list:
            capability_list = (
                "- 'analyze_csv': statistical analysis and data research\n"
                "- 'fetch_code': code examples and implementation patterns\n"
                "- 'write_report': final report synthesis"
            )

        system_prompt = (
            "You are a research coordinator. Given a complex research query and a list "
            "of available agent capabilities, break the query into 2-4 atomic subtasks. "
            "Each subtask must use one of the listed capability names exactly as shown. "
            "Always include 'write_report' as the last subtask if it is available. "
            "IMPORTANT: Respond with raw JSON only. No markdown fences, no explanation."
        )

        user_prompt = (
            f"Query: {query}\n\n"
            f"Available capabilities on the mesh:\n{capability_list}\n\n"
            "Return a JSON array of subtasks, each with:\n"
            '- "capability": exact capability name from the list above\n'
            '- "description": specific instruction for that agent\n'
            '- "context": additional context (can be empty string)\n\n'
            "Respond with raw JSON only."
        )

        response = await self._llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])

        raw = response.content.strip()
        subtasks = []

        try:
            parsed = extract_json(raw)
            if isinstance(parsed, list):
                subtasks = parsed
        except ValueError:
            logger.warning("[Research/plan] Could not parse subtasks JSON, using defaults")

        if not subtasks:
            subtasks = [
                {"capability": "fetch_code", "description": query, "context": ""},
                {"capability": "write_report", "description": query, "context": ""},
            ]

        logger.info(f"[Research/plan] Created {len(subtasks)} subtasks")
        return {"subtasks": subtasks}

    async def _discover_node(self, state: ResearchState) -> dict:
        """Query the mesh for each subtask capability.

        Skips subtasks where no matching agent is found (graceful degradation).

        Args:
            state: Current ResearchState with subtasks set.

        Returns:
            Dict with 'discovered_agents' mapping capability -> AgentRecord.
        """
        discovered: dict[str, Any] = {}

        for subtask in state["subtasks"]:
            cap = subtask["capability"]
            desc = subtask["description"]
            logger.info(f"[Research/discover] Looking for '{cap}' agent...")

            try:
                agents = await self.discover(
                    capability_name=cap,
                    description=desc,
                    min_trust=0.3,
                    top_k=1,
                )
                if agents:
                    discovered[cap] = agents[0]
                    logger.info(
                        f"[Research/discover] Found: {agents[0].manifest.name} "
                        f"(trust={agents[0].trust_score:.2f})"
                    )
                else:
                    logger.warning(f"[Research/discover] No agent found for '{cap}' -- skipping")
            except Exception as e:
                logger.warning(f"[Research/discover] Discovery failed for '{cap}': {e}")

        return {"discovered_agents": discovered}

    async def _delegate_node(self, state: ResearchState) -> dict:
        """Delegate subtasks (except write_report) to discovered agents in parallel.

        Uses asyncio.gather() to fan out all subtasks concurrently, capped at
        3 in-flight tasks to avoid overwhelming agents.

        Args:
            state: Current ResearchState with discovered_agents set.

        Returns:
            Dict with 'delegation_results' and 'errors'.
        """
        results: dict[str, Any] = {}
        errors: list[str] = []
        discovered = state["discovered_agents"]

        # Build list of (cap, subtask, target) to delegate — skip write_report
        work = []
        for subtask in state["subtasks"]:
            cap = subtask["capability"]
            if cap == "write_report":
                continue
            if cap not in discovered:
                logger.info(f"[Research/delegate] Skipping '{cap}' -- no agent available")
                continue
            work.append((cap, subtask, discovered[cap]))

        if not work:
            return {"delegation_results": results, "errors": errors}

        logger.info(f"[Research/delegate] Delegating {len(work)} subtasks in parallel")

        async def _run_one(cap: str, subtask: dict, target: Any):
            try:
                result = await self.delegate(
                    capability=cap,
                    input_data={
                        "query": subtask["description"],
                        "context": subtask.get("context", ""),
                    },
                    target=target,
                )
                logger.info(f"[Research/delegate] '{cap}' completed in {result.execution_time_ms}ms")
                return cap, result.output, None
            except Exception as e:
                err = f"Delegation to '{cap}' failed: {e}"
                logger.warning(f"[Research/delegate] {err}")
                return cap, None, err

        # Cap at 3 concurrent delegations to avoid overwhelming agents
        max_parallel = 3
        for i in range(0, len(work), max_parallel):
            batch = work[i:i + max_parallel]
            batch_results = await asyncio.gather(*[_run_one(c, s, t) for c, s, t in batch])
            for cap, output, err in batch_results:
                if err:
                    errors.append(err)
                else:
                    results[cap] = output

        return {"delegation_results": results, "errors": errors}

    async def _synthesize_node(self, state: ResearchState) -> dict:
        """Combine all results into a final report.

        Delegates to WriterAgent if available; falls back to local Groq synthesis.

        Args:
            state: Current ResearchState with delegation_results set.

        Returns:
            Dict with 'final_report' as a markdown string.
        """
        query = state["original_query"]
        results = state["delegation_results"]
        discovered = state["discovered_agents"]

        if "write_report" in discovered:
            logger.info("[Research/synthesize] Delegating final synthesis to WriterAgent")
            try:
                result = await self.delegate(
                    capability="write_report",
                    input_data={"topic": query, "data": results},
                    target=discovered["write_report"],
                )
                return {"final_report": result.output.get("report", str(result.output))}
            except Exception as e:
                logger.warning(f"[Research/synthesize] WriterAgent failed: {e} -- using LLM fallback")

        # Local Groq fallback
        logger.info("[Research/synthesize] Synthesizing locally via Groq")
        context_parts = [f"Query: {query}\n"]

        if "analyze_csv" in results:
            da = results["analyze_csv"]
            context_parts.append(f"## Data Analysis\n{da.get('analysis', str(da))}")

        if "fetch_code" in results:
            ce = results["fetch_code"]
            examples = ce.get("examples", [])
            code_section = "\n\n".join(
                f"### {ex.get('title', 'Example')}\n"
                f"```{ex.get('language','')}\n{ex.get('code','')}\n```"
                for ex in examples
            )
            context_parts.append(f"## Code Examples\n{code_section}")

        if not results:
            context_parts.append("No data was gathered from collaborating agents.")

        context = "\n\n".join(context_parts)

        response = await self._llm.ainvoke([
            SystemMessage(content=(
                "You are a senior technical writer. Synthesize the provided research "
                "data into a comprehensive, well-structured markdown report. "
                "Include an executive summary and clear conclusions."
            )),
            HumanMessage(content=f"Write a comprehensive report using this data:\n\n{context}"),
        ])

        return {"final_report": response.content.strip()}

    # ---------------------------------------------------------------------------
    # Conditional edges
    # ---------------------------------------------------------------------------

    def _should_delegate(self, state: ResearchState) -> str:
        """Route to delegate only if agents were discovered, else go to synthesize.

        Args:
            state: Current ResearchState after discover node.

        Returns:
            'delegate' or 'synthesize' routing key.
        """
        if state["discovered_agents"]:
            return "delegate"
        logger.info("[Research] No agents discovered -- skipping delegation")
        return "synthesize"

    # ---------------------------------------------------------------------------
    # Graph construction
    # ---------------------------------------------------------------------------

    def _build_graph(self) -> Any:
        """Assemble the LangGraph StateGraph for the research pipeline.

        Returns:
            Compiled LangGraph runnable.
        """
        graph = StateGraph(ResearchState)

        graph.add_node("plan", self._plan_node)
        graph.add_node("discover", self._discover_node)
        graph.add_node("delegate", self._delegate_node)
        graph.add_node("synthesize", self._synthesize_node)

        graph.set_entry_point("plan")
        graph.add_edge("plan", "discover")
        graph.add_conditional_edges(
            "discover",
            self._should_delegate,
            {"delegate": "delegate", "synthesize": "synthesize"},
        )
        graph.add_edge("delegate", "synthesize")
        graph.add_edge("synthesize", END)

        return graph.compile()

    # ---------------------------------------------------------------------------
    # @capability handler
    # ---------------------------------------------------------------------------

    @capability(
        name="research",
        description=(
            "Comprehensive research and competitive analysis on any topic. "
            "Coordinates with data, code, and writing agents on the mesh to "
            "gather information and synthesize a complete report."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The research question or topic"},
                "sources": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional: specific data sources to consult",
                },
            },
            "required": ["query"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "report": {"type": "string"},
                "sources_used": {"type": "array"},
                "agents_consulted": {"type": "array"},
            },
        },
        avg_latency_ms=30000,
    )
    async def research(self, input_data: dict) -> dict:
        """Run the full LangGraph research pipeline for the given query.

        Args:
            input_data: Dict with key 'query' (required) and optional 'sources'.

        Returns:
            Dict with final markdown report, sources used, and agents consulted.
        """
        query = input_data["query"]
        logger.info(f"[Research] Starting LangGraph pipeline for: {query[:80]}...")

        initial_state: ResearchState = {
            "original_query": query,
            "subtasks": [],
            "discovered_agents": {},
            "delegation_results": {},
            "final_report": "",
            "errors": [],
        }

        final_state = await self._graph.ainvoke(initial_state)

        agents_consulted = [
            rec.manifest.name
            for rec in final_state["discovered_agents"].values()
        ]

        return {
            "report": final_state["final_report"],
            "sources_used": list(final_state["delegation_results"].keys()),
            "agents_consulted": agents_consulted,
        }


# ---------------------------------------------------------------------------
# Run standalone
# ---------------------------------------------------------------------------

async def main():
    """Start the ResearchAgent and register it on the mesh."""
    agent = ResearchAgent(
        name="Research Agent",
        registry_url=os.environ.get("REGISTRY_URL", "http://localhost:8000"),
        ws_port=9001,
        tags=["research", "analysis", "coordination"],
    )
    await agent.start()


if __name__ == "__main__":
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )
    asyncio.run(main())
