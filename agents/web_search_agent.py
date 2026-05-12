"""WebSearchAgent — searches the web via the brave-search MCP server.

This agent demonstrates real MCP tool usage inside a MeshAgent. It advertises
a 'web_search' capability and fulfils requests by calling the brave_web_search
tool on the brave-search MCP server.

Setup (optional — works in mock mode without this):
    1. Get a Brave Search API key: https://api.search.brave.com/
    2. Add to ~/.agentmesh/mcp_servers.json:
       {
         "brave-search": {
           "command": "npx",
           "args": ["-y", "@modelcontextprotocol/server-brave-search"],
           "env": {"BRAVE_API_KEY": "BSA...your key here..."}
         }
       }

Without setup, the agent runs in mock mode and returns stub results. The mesh
still routes correctly — useful for demos and tests without a Brave API key.

Run standalone:
    python -m agents.web_search_agent
"""

from __future__ import annotations

import asyncio
import logging
import os

from sdk.agent import MeshAgent, capability

logger = logging.getLogger("agentmesh.agents.web_search")


class WebSearchAgent(MeshAgent):
    """Searches the web using the brave-search MCP server."""

    @capability(
        name="web_search",
        description=(
            "Search the web for current information, news, recent events, and real-time data. "
            "Returns ranked results with titles, URLs, and snippets. "
            "Uses Brave Search MCP server."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "count": {
                    "type": "integer",
                    "description": "Number of results to return (default 5)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "results": {"type": "string", "description": "Search results text"},
                "query": {"type": "string"},
                "source": {"type": "string", "description": "MCP server used"},
                "mock": {"type": "boolean", "description": "True if mock mode"},
            },
        },
        avg_latency_ms=2000,
        cost_per_call_usd=0.001,
    )
    async def web_search(self, input_data: dict) -> dict:
        """Execute a web search via brave-search MCP server.

        Args:
            input_data: Dict with 'query' (required) and optional 'count'.

        Returns:
            Dict with search results, query, and source metadata.
        """
        query = input_data.get("query", "")
        count = int(input_data.get("count", 5))

        if not query:
            return {
                "results": "",
                "query": query,
                "source": "none",
                "mock": False,
                "error": "Empty query",
            }

        logger.info(f"[WebSearch] Searching: {query[:80]}")

        try:
            client = await self.get_mcp_tool("brave-search")
            raw_results = await client.call_tool(
                "brave_web_search",
                {"query": query, "count": count},
            )
            is_mock = client._mock_mode
        except Exception as e:
            logger.warning(f"[WebSearch] MCP call failed: {e}")
            raw_results = f"Search failed: {e}"
            is_mock = True

        return {
            "results": raw_results,
            "query": query,
            "source": "brave-search-mcp",
            "mock": is_mock,
        }


# ---------------------------------------------------------------------------
# Run standalone
# ---------------------------------------------------------------------------

async def main():
    """Start the WebSearchAgent and register it on the mesh."""
    agent = WebSearchAgent(
        name="Web Search Agent",
        registry_url=os.environ.get("REGISTRY_URL", "http://localhost:8000"),
        ws_port=9005,
        mcp_servers=["brave-search"],
        tags=["search", "web", "realtime", "news"],
        max_concurrent_tasks=5,
    )
    await agent.start()


if __name__ == "__main__":
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )
    asyncio.run(main())
