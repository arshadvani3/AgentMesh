"""Tests for MCPToolClient — MCP server connection with mock fallback."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mesh.mcp_client import (
    MCPToolClient,
    _mock_tool_response,
    get_tool_descriptions,
)


class TestMockMode:
    """Tests for mock fallback behaviour (no real MCP server required)."""

    @pytest.mark.asyncio
    async def test_no_config_enters_mock_mode(self):
        """Client enters mock mode when server has no config entry."""
        with patch("mesh.mcp_client._CONFIG_PATH", Path("/nonexistent/path.json")):
            client = MCPToolClient("brave-search")
            await client.connect()
            assert client._mock_mode is True

    @pytest.mark.asyncio
    async def test_mock_list_tools_returns_empty(self):
        """list_tools() returns [] in mock mode."""
        client = MCPToolClient("brave-search")
        client._mock_mode = True
        tools = await client.list_tools()
        assert tools == []

    @pytest.mark.asyncio
    async def test_mock_call_tool_returns_string(self):
        """call_tool() returns a stub string in mock mode."""
        client = MCPToolClient("brave-search")
        client._mock_mode = True
        result = await client.call_tool("brave_web_search", {"query": "AI agents"})
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_mock_search_mentions_query(self):
        """Mock search result references the query."""
        client = MCPToolClient("brave-search")
        client._mock_mode = True
        result = await client.call_tool("brave_web_search", {"query": "Python asyncio"})
        assert "Python asyncio" in result

    @pytest.mark.asyncio
    async def test_close_in_mock_mode_is_noop(self):
        """close() does not raise in mock mode."""
        client = MCPToolClient("brave-search")
        client._mock_mode = True
        await client.close()  # should not raise


class TestMockToolResponse:
    """Tests for the _mock_tool_response helper."""

    def test_search_server_returns_search_stub(self):
        result = _mock_tool_response("brave-search", "brave_web_search", {"query": "test"})
        assert "test" in result
        assert len(result) > 20

    def test_filesystem_server_returns_file_stub(self):
        result = _mock_tool_response("filesystem", "read_file", {"path": "/tmp/foo.txt"})
        assert "/tmp/foo.txt" in result

    def test_unknown_server_returns_generic_stub(self):
        result = _mock_tool_response("my-server", "my_tool", {"x": 1})
        assert "my-server" in result
        assert "my_tool" in result


class TestConfigLoading:
    """Tests for config file loading."""

    @pytest.mark.asyncio
    async def test_valid_config_sets_not_mock(self):
        """When valid config exists but connection fails, still falls back to mock."""
        config = {
            "test-server": {
                "command": "nonexistent-command-xyz",
                "args": [],
            }
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(config, f)
            tmp_path = Path(f.name)

        try:
            with patch("mesh.mcp_client._CONFIG_PATH", tmp_path):
                client = MCPToolClient("test-server")
                await client.connect()
                # Connection to nonexistent command fails → mock mode
                assert client._mock_mode is True
        finally:
            tmp_path.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_missing_server_name_in_config_uses_mock(self):
        """Server name not in config → mock mode."""
        config = {"other-server": {"command": "echo", "args": []}}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(config, f)
            tmp_path = Path(f.name)

        try:
            with patch("mesh.mcp_client._CONFIG_PATH", tmp_path):
                client = MCPToolClient("brave-search")
                await client.connect()
                assert client._mock_mode is True
        finally:
            tmp_path.unlink(missing_ok=True)


class TestGetToolDescriptions:
    """Tests for the get_tool_descriptions helper."""

    @pytest.mark.asyncio
    async def test_returns_empty_string_in_mock_mode(self):
        """Returns '' when no config exists (mock mode)."""
        with patch("mesh.mcp_client._CONFIG_PATH", Path("/nonexistent/path.json")):
            result = await get_tool_descriptions("brave-search")
            assert result == ""

    @pytest.mark.asyncio
    async def test_returns_comma_separated_tool_names(self):
        """Returns tool names when tools are listed."""
        client_mock = MagicMock()
        client_mock.connect = AsyncMock()
        client_mock.list_tools = AsyncMock(
            return_value=[
                {"name": "brave_web_search", "description": "Search"},
                {"name": "brave_local_search", "description": "Local"},
            ]
        )
        client_mock.close = AsyncMock()

        with patch("mesh.mcp_client.MCPToolClient", return_value=client_mock):
            result = await get_tool_descriptions("brave-search")
            assert "brave_web_search" in result
            assert "brave_local_search" in result
