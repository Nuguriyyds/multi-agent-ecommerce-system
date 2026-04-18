from __future__ import annotations

from typing import Any

import pytest

from app.v3.models import (
    AgentRole,
    CapabilityDescriptor,
    CapabilityKind,
    Observation,
    SpecialistBrief,
    SpecialistObservation,
)
from app.v3.registry import CapabilityAlreadyRegistered, CapabilityRegistry, MCPProvider, SubAgentProvider, ToolProvider


def make_tool_descriptor(name: str) -> CapabilityDescriptor:
    return CapabilityDescriptor(
        name=name,
        kind=CapabilityKind.tool,
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        timeout=3.0,
        permission_tag=f"{name}.read",
    )


def make_sub_agent_descriptor(name: str) -> CapabilityDescriptor:
    return CapabilityDescriptor(
        name=name,
        kind=CapabilityKind.sub_agent,
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        timeout=5.0,
        permission_tag=f"{name}.invoke",
    )


def make_mcp_descriptor(name: str) -> CapabilityDescriptor:
    return CapabilityDescriptor(
        name=name,
        kind=CapabilityKind.mcp_tool,
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        timeout=4.0,
        permission_tag=f"{name}.read",
    )


def make_brief() -> SpecialistBrief:
    return SpecialistBrief(
        brief_id="brief-1",
        task_id="task-1",
        role=AgentRole.candidate_analysis,
        goal="Summarize shortlist fit.",
    )


class MockToolProvider(ToolProvider):
    async def invoke(self, args: dict[str, Any]) -> Observation:
        return Observation(
            observation_id=f"obs-{self.name}",
            source=self.name,
            summary=f"{self.name} completed.",
            payload=args,
            evidence_source=f"tool:{self.name}",
        )


class MockSpecialistProvider(SubAgentProvider):
    async def invoke(self, brief: SpecialistBrief) -> SpecialistObservation:
        return SpecialistObservation(
            observation_id=f"obs-{self.name}",
            source=self.name,
            role=brief.role,
            brief_id=brief.brief_id,
            summary=f"{self.name} completed.",
            payload={"goal": brief.goal},
            evidence_source=f"specialist:{self.name}",
        )


class MockMCPProvider(MCPProvider):
    async def invoke(self, args: dict[str, Any]) -> Observation:
        return Observation(
            observation_id=f"obs-{self.name}",
            source=self.name,
            summary=f"{self.name} completed.",
            payload=args,
            evidence_source=f"mcp:{self.name}",
        )


def test_registry_lists_tools_and_specialists() -> None:
    registry = CapabilityRegistry()
    registry.register(MockToolProvider(make_tool_descriptor("catalog_search")))
    registry.register(MockToolProvider(make_tool_descriptor("inventory_check")))
    registry.register(MockSpecialistProvider(make_sub_agent_descriptor("candidate_analysis")))

    tools = registry.list(CapabilityKind.tool)
    specialists = registry.list("specialist")

    assert [descriptor.name for descriptor in tools] == ["catalog_search", "inventory_check"]
    assert [descriptor.name for descriptor in specialists] == ["candidate_analysis"]


def test_registry_rejects_duplicate_capabilities() -> None:
    registry = CapabilityRegistry()
    registry.register(MockToolProvider(make_tool_descriptor("catalog_search")))

    with pytest.raises(CapabilityAlreadyRegistered):
        registry.register(MockToolProvider(make_tool_descriptor("catalog_search")))


@pytest.mark.asyncio
async def test_mcp_provider_is_tool_provider() -> None:
    registry = CapabilityRegistry()
    registry.register(MockMCPProvider(make_mcp_descriptor("rag_product_knowledge")))

    provider = registry.get("rag_product_knowledge")
    observation = await provider.invoke({"sku": "sku-1"})

    assert isinstance(provider, ToolProvider)
    assert isinstance(provider, MCPProvider)
    assert observation.evidence_source == "mcp:rag_product_knowledge"


@pytest.mark.asyncio
async def test_sub_agent_provider_returns_specialist_observation() -> None:
    registry = CapabilityRegistry()
    registry.register(MockSpecialistProvider(make_sub_agent_descriptor("candidate_analysis")))

    provider = registry.get("candidate_analysis")
    observation = await provider.invoke(make_brief())

    assert isinstance(observation, SpecialistObservation)
    assert observation.role is AgentRole.candidate_analysis
