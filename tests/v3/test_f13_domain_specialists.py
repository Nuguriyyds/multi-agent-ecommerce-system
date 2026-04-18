from __future__ import annotations

import pytest

from app.v3.config import Settings
from app.v3.hardening import HardeningGate
from app.v3.models import AgentRole, Observation, ReplyToUserAction, SpecialistBrief
from app.v3.prompts import PromptRegistry
from app.v3.registry import CapabilityRegistry
from app.v3.specialists import (
    CandidateAnalysisSpecialist,
    ComparisonSpecialist,
    RecommendationRationaleSpecialist,
    ShoppingBriefSpecialist,
)
from app.v3.tools import (
    get_seed_catalog,
    register_mock_mcp_tool_providers,
    register_mock_tool_providers,
)


def make_brief(
    *,
    role: AgentRole,
    goal: str,
    constraints: dict | None = None,
    allowed_capabilities: list[str] | None = None,
) -> SpecialistBrief:
    return SpecialistBrief(
        brief_id=f"brief-{role.value}",
        task_id=f"task-{role.value}",
        role=role,
        goal=goal,
        constraints=constraints or {},
        allowed_capabilities=allowed_capabilities or [],
    )


def make_registry() -> CapabilityRegistry:
    registry = CapabilityRegistry()
    catalog = get_seed_catalog()
    register_mock_tool_providers(registry, catalog=catalog)
    register_mock_mcp_tool_providers(
        registry,
        settings=Settings(mcp_mock_enabled=True),
        catalog=catalog,
    )
    return registry


@pytest.mark.asyncio
async def test_shopping_brief_identifies_missing_slots_and_registers_role_prompt() -> None:
    prompt_registry = PromptRegistry()
    specialist = ShoppingBriefSpecialist(prompt_registry=prompt_registry)

    observation = await specialist.invoke(
        make_brief(
            role=AgentRole.shopping_brief,
            goal="Extract the user's need.",
            constraints={
                "raw_user_need": "我想买通勤用的降噪耳机，不要 Beats。",
            },
        )
    )

    assert specialist.allowed_capabilities == []
    assert prompt_registry.get("role", "shopping_brief_specialist").text
    assert observation.status == "partial"
    assert observation.payload["category"] == "earphones"
    assert observation.payload["scene"] == "commute"
    assert observation.payload["exclusions"] == ["Beats"]
    assert observation.payload["slots_missing"] == ["budget"]


@pytest.mark.asyncio
async def test_candidate_analysis_adds_fit_reasons_with_catalog_observation_ids() -> None:
    registry = make_registry()
    prompt_registry = PromptRegistry()
    specialist = CandidateAnalysisSpecialist(
        registry=registry,
        prompt_registry=prompt_registry,
    )

    observation = await specialist.invoke(
        make_brief(
            role=AgentRole.candidate_analysis,
            goal="Analyze commute ANC headphone candidates.",
            allowed_capabilities=["catalog_search"],
            constraints={
                "query": "3000 左右的通勤降噪耳机",
                "category": "earphones",
                "scene": "commute",
                "budget_max": 3500,
                "exclude_brands": ["Beats"],
                "limit": 4,
            },
        )
    )

    source_observation_id = observation.payload["source_observation_id"]
    candidates = observation.payload["candidates"]
    assert specialist.allowed_capabilities == ["catalog_search"]
    assert prompt_registry.get("role", "candidate_analysis_specialist").text
    assert len(candidates) == 4
    assert {candidate["name"] for candidate in candidates} >= {
        "Sony WH-1000XM5",
        "Bose QuietComfort Ultra Headphones",
    }
    assert all(
        reason["observation_id"] == source_observation_id
        for candidate in candidates
        for reason in candidate["fit_reasons"]
    )


@pytest.mark.asyncio
async def test_comparison_filters_non_whitelisted_dimensions_before_tool_calls() -> None:
    registry = make_registry()
    prompt_registry = PromptRegistry()
    specialist = ComparisonSpecialist(registry=registry, prompt_registry=prompt_registry)

    observation = await specialist.invoke(
        make_brief(
            role=AgentRole.comparison,
            goal="Compare the two headphones.",
            allowed_capabilities=["product_compare", "inventory_check"],
            constraints={
                "skus": ["EAR-SON-WH1000XM5", "EAR-BOS-QCUH"],
                "dimensions": ["price", "battery", "latency", "camera", "brand"],
            },
        )
    )

    comparison_payload = observation.payload["comparisons"][0]["payload"]
    compared_dimensions = [
        item["dimension"] for item in comparison_payload["dimensions"]
    ]
    assert specialist.allowed_capabilities == ["product_compare", "inventory_check"]
    assert prompt_registry.get("role", "comparison_specialist").text
    assert observation.payload["accepted_dimensions"] == ["price", "battery", "brand"]
    assert observation.payload["ignored_dimensions"] == ["latency", "camera"]
    assert compared_dimensions == ["price", "battery", "brand"]
    assert len(observation.payload["inventory"]) == 2


@pytest.mark.asyncio
async def test_recommendation_rationale_cites_evidence_and_passes_gate_rule() -> None:
    registry = make_registry()
    prompt_registry = PromptRegistry()
    catalog_observation = await registry.get("catalog_search").invoke(
        {
            "query": "Sony WH-1000XM5 通勤降噪耳机",
            "filters": {
                "category": "earphones",
                "scene": "commute",
                "price_max": 3500,
                "limit": 4,
            },
        }
    )
    comparison_observation = await registry.get("product_compare").invoke(
        {
            "sku_a": "EAR-SON-WH1000XM5",
            "sku_b": "EAR-BOS-QCUH",
            "dimensions": ["price", "battery", "noise_cancel"],
        }
    )
    inventory_observation = await registry.get("inventory_check").invoke(
        {"sku": "EAR-SON-WH1000XM5"}
    )
    specialist = RecommendationRationaleSpecialist(
        registry=registry,
        prompt_registry=prompt_registry,
    )

    observation = await specialist.invoke(
        make_brief(
            role=AgentRole.recommendation_rationale,
            goal="Explain why Sony WH-1000XM5 is the final pick.",
            allowed_capabilities=[
                "catalog_search",
                "inventory_check",
                "product_compare",
                "rag_product_knowledge",
            ],
            constraints={
                "pick_sku": "EAR-SON-WH1000XM5",
                "query": "Sony WH-1000XM5 通勤 降噪",
                "observations": [
                    catalog_observation.model_dump(mode="json"),
                    comparison_observation.model_dump(mode="json"),
                    inventory_observation.model_dump(mode="json"),
                ],
            },
        )
    )

    rationales = observation.payload["rationales"]
    supporting_observations = [
        Observation.model_validate(item)
        for item in observation.payload["supporting_observations"]
    ]
    evidence_ids = [item["observation_id"] for item in rationales]
    gate_result = HardeningGate().evaluate(
        ReplyToUserAction(
            message="Sony WH-1000XM5 is the best fit based on the cited evidence.",
            observation_ids=evidence_ids,
        ),
        current_node="advice",
        topic="advice",
        observations=supporting_observations,
    )

    assert specialist.allowed_capabilities == [
        "catalog_search",
        "inventory_check",
        "product_compare",
        "rag_product_knowledge",
    ]
    assert prompt_registry.get("role", "recommendation_rationale_specialist").text
    assert observation.payload["evidence_sufficient"] is True
    assert evidence_ids
    assert all(item["observation_id"] for item in rationales)
    assert "mcp:rag_product_knowledge" in {
        item.evidence_source for item in supporting_observations
    }
    assert gate_result.decision == "allow"
