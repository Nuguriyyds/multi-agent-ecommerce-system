from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.v3.config import Settings
from app.v3.models import AgentRole, CapabilityKind, SpecialistBrief
from app.v3.registry import CapabilityRegistry, MCPProvider, ToolProvider
from app.v3.specialists import Specialist
from app.v3.tools import (
    MCPClient,
    MockMCPServer,
    build_mock_mcp_tool_providers,
    register_mock_mcp_tool_providers,
)


def make_brief(*, allowed_capabilities: list[str] | None = None) -> SpecialistBrief:
    return SpecialistBrief(
        brief_id="brief-rag-1",
        task_id="task-rag-1",
        role=AgentRole.recommendation_rationale,
        goal="Use external product knowledge to support the recommendation.",
        allowed_capabilities=allowed_capabilities or [],
    )


class RAGSpecialist(Specialist):
    def __init__(self, *, registry: CapabilityRegistry) -> None:
        super().__init__(
            role=AgentRole.recommendation_rationale,
            name="recommendation_rationale_specialist",
            registry=registry,
            allowed_capabilities=["rag_product_knowledge"],
        )

    async def execute(self, brief: SpecialistBrief):
        observation = await self.invoke_tool(
            brief,
            capability_name="rag_product_knowledge",
            arguments={"query": "通勤 3000 左右降噪耳机", "limit": 4},
        )
        return self.build_observation(
            brief,
            summary="Collected MCP-backed product knowledge.",
            payload={"knowledge_snippets": observation.payload["snippets"]},
            evidence_source=observation.evidence_source,
        )


@pytest.mark.asyncio
async def test_mcp_client_lists_mock_server_tools() -> None:
    client = MCPClient(server=MockMCPServer())

    tools = await client.list_tools()

    assert [tool.name for tool in tools] == ["rag_product_knowledge"]
    assert tools[0].input_schema["required"] == ["query"]


@pytest.mark.asyncio
async def test_mcp_provider_registers_and_returns_rag_observation() -> None:
    registry = CapabilityRegistry()
    descriptors = register_mock_mcp_tool_providers(
        registry,
        settings=Settings(mcp_mock_enabled=True),
    )

    assert [descriptor.name for descriptor in descriptors] == ["rag_product_knowledge"]
    assert [descriptor.kind for descriptor in registry.list(CapabilityKind.mcp_tool)] == [
        CapabilityKind.mcp_tool
    ]

    provider = registry.get("rag_product_knowledge")
    observation = await provider.invoke({"query": "Sony 通勤降噪耳机", "limit": 4})

    assert isinstance(provider, ToolProvider)
    assert isinstance(provider, MCPProvider)
    assert observation.evidence_source == "mcp:rag_product_knowledge"
    assert observation.payload["tool_name"] == "rag_product_knowledge"
    assert observation.payload["snippet_count"] == 4
    assert len(observation.payload["snippets"]) == 4
    assert any("Sony" in item["product_name"] for item in observation.payload["snippets"])


@pytest.mark.asyncio
async def test_specialist_can_invoke_registered_mcp_tool_like_local_tool() -> None:
    registry = CapabilityRegistry()
    register_mock_mcp_tool_providers(
        registry,
        settings=Settings(mcp_mock_enabled=True),
    )
    specialist = RAGSpecialist(registry=registry)

    observation = await specialist.invoke(make_brief(allowed_capabilities=["rag_product_knowledge"]))

    assert observation.role is AgentRole.recommendation_rationale
    assert observation.evidence_source == "mcp:rag_product_knowledge"
    assert len(observation.payload["knowledge_snippets"]) == 4


def test_mcp_registration_is_skipped_when_mock_is_disabled() -> None:
    registry = CapabilityRegistry()
    descriptors = register_mock_mcp_tool_providers(
        registry,
        settings=Settings(mcp_mock_enabled=False),
    )

    assert descriptors == []
    assert registry.list(CapabilityKind.mcp_tool) == []
    assert build_mock_mcp_tool_providers(settings=Settings(mcp_mock_enabled=False)) == []


@pytest.mark.asyncio
async def test_mcp_tool_rejects_invalid_args() -> None:
    registry = CapabilityRegistry()
    register_mock_mcp_tool_providers(
        registry,
        settings=Settings(mcp_mock_enabled=True),
    )
    provider = registry.get("rag_product_knowledge")

    with pytest.raises(ValidationError):
        await provider.invoke({"query": "   "})

