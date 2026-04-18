from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any
from uuid import uuid4

from app.v3.config import Settings, get_settings
from app.v3.models import CapabilityDescriptor, CapabilityKind, Observation, Product
from app.v3.registry import CapabilityRegistry, MCPProvider

from .mcp_client import InProcessMCPTransport, MCPClient, MCPToolDefinition
from .mcp_mock_server import MockMCPServer

_MCP_TOOL_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "tool_name": {"type": "string"},
        "arguments": {"type": "object"},
        "content": {"type": "array", "items": {"type": "object"}},
        "snippet_count": {"type": "integer"},
        "snippets": {"type": "array", "items": {"type": "object"}},
    },
    "required": ["tool_name", "arguments", "content", "snippet_count", "snippets"],
    "additionalProperties": False,
}


class MCPToolProvider(MCPProvider):
    def __init__(
        self,
        tool_definition: MCPToolDefinition,
        *,
        client: MCPClient,
        timeout: float = 4.0,
    ) -> None:
        self._client = client
        super().__init__(
            CapabilityDescriptor(
                name=tool_definition.name,
                kind=CapabilityKind.mcp_tool,
                input_schema=dict(tool_definition.input_schema),
                output_schema=_MCP_TOOL_OUTPUT_SCHEMA,
                timeout=timeout,
                permission_tag=f"mcp.{tool_definition.name}",
                description=tool_definition.description,
            )
        )
        self._logger = logging.getLogger(__name__)

    async def invoke(self, args: dict[str, Any]) -> Observation:
        self._logger.info("mcp_tool start name=%s args=%s", self.name, args)
        try:
            result = await self._client.call_tool(self.name, args)
        except Exception:
            self._logger.exception("mcp_tool failed name=%s", self.name)
            raise

        snippets = result.json_items()
        observation = Observation(
            observation_id=f"obs-{uuid4().hex[:12]}",
            source=self.name,
            status="error" if result.is_error else ("ok" if snippets else "partial"),
            summary=f"Retrieved {len(snippets)} knowledge snippets from MCP tool {self.name}.",
            payload={
                "tool_name": self.name,
                "arguments": dict(args),
                "content": [block.model_dump(mode="json") for block in result.content],
                "snippet_count": len(snippets),
                "snippets": snippets,
            },
            evidence_source=f"mcp:{self.name}",
        )
        self._logger.info(
            "mcp_tool success name=%s snippet_count=%s observation_id=%s",
            self.name,
            len(snippets),
            observation.observation_id,
        )
        return observation


def build_mock_mcp_tool_providers(
    *,
    settings: Settings | None = None,
    catalog: Sequence[Product] | None = None,
    server: MockMCPServer | None = None,
) -> list[MCPToolProvider]:
    resolved_settings = settings or get_settings()
    if not resolved_settings.mcp_mock_enabled:
        return []

    resolved_server = server or MockMCPServer(catalog=catalog)
    client = MCPClient(transport=InProcessMCPTransport(resolved_server))
    return [
        MCPToolProvider(tool_definition=tool_definition, client=client)
        for tool_definition in resolved_server.list_tool_definitions()
    ]


def register_mock_mcp_tool_providers(
    registry: CapabilityRegistry,
    *,
    settings: Settings | None = None,
    catalog: Sequence[Product] | None = None,
    server: MockMCPServer | None = None,
) -> list[CapabilityDescriptor]:
    descriptors: list[CapabilityDescriptor] = []
    for provider in build_mock_mcp_tool_providers(
        settings=settings,
        catalog=catalog,
        server=server,
    ):
        descriptors.append(registry.register(provider))
    return descriptors


__all__ = [
    "MCPToolProvider",
    "build_mock_mcp_tool_providers",
    "register_mock_mcp_tool_providers",
]
