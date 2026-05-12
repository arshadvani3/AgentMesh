"""MCPToolClient — wraps an MCP server connection for use inside MeshAgents.

Agents declare which MCP servers they use via the `mcp_servers` list in their
AgentManifest. At runtime, the SDK wires each name to an actual server process
using configuration from ~/.agentmesh/mcp_servers.json.

Config file format (~/.agentmesh/mcp_servers.json):
    {
      "brave-search": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-brave-search"],
        "env": {"BRAVE_API_KEY": "BSA..."}
      },
      "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
      }
    }

If the config file does not exist, or a server isn't configured, MCPToolClient
falls back to a mock mode that returns empty tool lists and stub responses.
This ensures the mesh runs and tests pass without requiring MCP servers to be
installed.

Usage:
    client = MCPToolClient("brave-search")
    await client.connect()
    tools = await client.list_tools()
    result = await client.call_tool("brave_web_search", {"query": "AI news"})
    await client.close()
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger("agentmesh.mcp_client")

_CONFIG_PATH = Path.home() / ".agentmesh" / "mcp_servers.json"


def _load_server_config(server_name: str) -> dict | None:
    """Load MCP server config from ~/.agentmesh/mcp_servers.json."""
    if not _CONFIG_PATH.exists():
        return None
    try:
        with open(_CONFIG_PATH) as f:
            config = json.load(f)
        return config.get(server_name)
    except Exception as e:
        logger.warning(f"MCPToolClient: failed to read config: {e}")
        return None


class MCPToolClient:
    """Manages a connection to a single MCP server.

    Lazy-connects on first call_tool(). Falls back to mock mode when the
    server is not configured or the connection fails.
    """

    def __init__(self, server_name: str):
        self.server_name = server_name
        self._session: Any = None          # mcp.ClientSession
        self._cm: Any = None               # async context manager stack
        self._tools: list[dict] | None = None
        self._mock_mode = False

    async def connect(self) -> None:
        """Open connection to the MCP server. Falls back to mock on failure."""
        config = _load_server_config(self.server_name)
        if not config:
            logger.info(
                f"MCPToolClient: no config for '{self.server_name}', using mock mode"
            )
            self._mock_mode = True
            return

        try:
            from mcp import ClientSession, StdioServerParameters, stdio_client  # noqa: PLC0415

            env = {**os.environ, **config.get("env", {})}
            server_params = StdioServerParameters(
                command=config["command"],
                args=config.get("args", []),
                env=env,
            )

            self._cm = stdio_client(server_params)
            read, write = await self._cm.__aenter__()
            self._session = ClientSession(read, write)
            await self._session.__aenter__()
            await self._session.initialize()
            logger.info(f"MCPToolClient: connected to '{self.server_name}'")
        except Exception as e:
            logger.warning(
                f"MCPToolClient: could not connect to '{self.server_name}' ({e}), using mock mode"
            )
            self._mock_mode = True
            self._session = None

    async def list_tools(self) -> list[dict]:
        """Return list of tools provided by this MCP server.

        Each tool is a dict: {name, description, inputSchema}.
        Returns empty list in mock mode.
        """
        if self._mock_mode:
            return []

        if self._session is None:
            await self.connect()
            if self._mock_mode:
                return []

        if self._tools is not None:
            return self._tools

        try:
            result = await self._session.list_tools()
            self._tools = [
                {
                    "name": t.name,
                    "description": t.description or "",
                    "inputSchema": t.inputSchema if hasattr(t, "inputSchema") else {},
                }
                for t in result.tools
            ]
            return self._tools
        except Exception as e:
            logger.warning(f"MCPToolClient: list_tools failed: {e}")
            return []

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Call a tool on the MCP server. Returns the text result.

        In mock mode, returns a stub response for known tool patterns.
        """
        if self._mock_mode:
            return _mock_tool_response(self.server_name, tool_name, arguments)

        if self._session is None:
            await self.connect()
            if self._mock_mode:
                return _mock_tool_response(self.server_name, tool_name, arguments)

        try:
            result = await self._session.call_tool(tool_name, arguments)
            # Extract text content from result
            parts: list[str] = []
            for content in result.content:
                if hasattr(content, "text"):
                    parts.append(content.text)
                else:
                    parts.append(str(content))
            return "\n".join(parts) if parts else ""
        except Exception as e:
            logger.warning(f"MCPToolClient: call_tool '{tool_name}' failed: {e}")
            raise

    async def close(self) -> None:
        """Close the MCP server connection."""
        if self._session is not None:
            try:
                await self._session.__aexit__(None, None, None)
            except Exception:
                pass
            self._session = None
        if self._cm is not None:
            try:
                await self._cm.__aexit__(None, None, None)
            except Exception:
                pass
            self._cm = None


# ---------------------------------------------------------------------------
# Mock responses for development without real MCP servers
# ---------------------------------------------------------------------------

def _mock_tool_response(server_name: str, tool_name: str, arguments: dict) -> str:
    """Return a plausible stub response for known MCP tool patterns."""
    query = arguments.get("query", arguments.get("q", "unknown query"))

    if "search" in tool_name.lower() or "search" in server_name.lower():
        return (
            f"[Mock search results for: {query}]\n\n"
            "1. Example Result — This is a mock result returned because no real MCP "
            "server is configured. To get real results, add a 'brave-search' entry to "
            "~/.agentmesh/mcp_servers.json.\n\n"
            "2. Another Mock Result — Configure BRAVE_API_KEY in your mcp_servers.json "
            "to enable live web search.\n"
        )

    if "filesystem" in server_name.lower() or "read" in tool_name.lower():
        path = arguments.get("path", "/unknown")
        return f"[Mock filesystem response for path: {path}] File contents would appear here."

    return f"[Mock response from {server_name}.{tool_name}({arguments})]"


# ---------------------------------------------------------------------------
# Registry helper: build capability enrichment from MCP tools
# ---------------------------------------------------------------------------

async def get_tool_descriptions(server_name: str) -> str:
    """Return a comma-separated string of tool names for a server.

    Used by sdk/agent.py to enrich capability descriptions with available tools.
    Returns empty string if server is not configured or fails.
    """
    client = MCPToolClient(server_name)
    try:
        await client.connect()
        tools = await client.list_tools()
        return ", ".join(t["name"] for t in tools)
    except Exception:
        return ""
    finally:
        await client.close()
