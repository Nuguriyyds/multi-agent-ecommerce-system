from __future__ import annotations

import logging
from typing import Any, Protocol

from .mcp_types import MCPContentBlock, MCPToolCallResult, MCPToolDefinition


class MCPClientError(RuntimeError):
    """Raised when the transport returns an invalid MCP response."""


class MCPTransport(Protocol):
    async def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute one MCP-style request and return the raw response payload."""


class InProcessMCPTransport:
    def __init__(self, server: Any | None = None) -> None:
        if server is None:
            from .mcp_mock_server import MockMCPServer

            server = MockMCPServer()
        self._server = server

    @property
    def server(self) -> Any:
        return self._server

    async def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return await self._server.handle_request(method, params or {})


class MCPClient:
    def __init__(self, *, transport: MCPTransport | None = None, server: Any | None = None) -> None:
        if transport is not None and server is not None:
            raise ValueError("Specify either transport or server, not both.")
        self._transport = transport or InProcessMCPTransport(server)
        self._logger = logging.getLogger(__name__)

    async def list_tools(self) -> list[MCPToolDefinition]:
        self._logger.info("MCPClient tools.list start")
        response = await self._transport.request("tools.list", {})
        raw_tools = response.get("tools")
        if not isinstance(raw_tools, list):
            raise MCPClientError("tools.list response must contain a tools array")

        tools = [MCPToolDefinition.model_validate(item) for item in raw_tools]
        self._logger.info("MCPClient tools.list success total=%s", len(tools))
        return tools

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> MCPToolCallResult:
        self._logger.info("MCPClient tools.call start name=%s args=%s", tool_name, arguments)
        response = await self._transport.request(
            "tools.call",
            {
                "name": tool_name,
                "arguments": dict(arguments),
            },
        )
        raw_content = response.get("content")
        if not isinstance(raw_content, list):
            raise MCPClientError("tools.call response must contain a content array")

        result = MCPToolCallResult(
            tool_name=tool_name,
            content=[MCPContentBlock.model_validate(item) for item in raw_content],
            is_error=bool(response.get("is_error", False)),
        )
        self._logger.info(
            "MCPClient tools.call success name=%s content_blocks=%s",
            tool_name,
            len(result.content),
        )
        return result


__all__ = [
    "InProcessMCPTransport",
    "MCPClient",
    "MCPClientError",
    "MCPContentBlock",
    "MCPToolCallResult",
    "MCPToolDefinition",
]
