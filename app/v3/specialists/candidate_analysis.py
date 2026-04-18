from __future__ import annotations

import logging
from typing import Any

from pydantic import Field

from app.v3.models import AgentRole, Observation, Product, PromptLayer, SpecialistBrief
from app.v3.models.base import V3Model
from app.v3.prompts import PromptAlreadyRegistered, PromptRegistry

from .base import Specialist

ROLE_PROMPT_NAME = "candidate_analysis_specialist"
ROLE_PROMPT_VERSION = "1"
ROLE_PROMPT_TEXT = (
    "Analyze catalog-backed candidate products against the shopping brief. "
    "Every fit reason must cite the candidate observation_id."
)

_LOGGER = logging.getLogger(__name__)


class CandidateFitReason(V3Model):
    reason: str
    observation_id: str


class CandidateAnalysisItem(V3Model):
    sku: str
    name: str
    brand: str
    fit_score: int = Field(ge=0)
    fit_reasons: list[CandidateFitReason] = Field(default_factory=list)


class CandidateAnalysisPayload(V3Model):
    source_observation_id: str
    candidates: list[CandidateAnalysisItem] = Field(default_factory=list)


def register_candidate_analysis_prompt(registry: PromptRegistry) -> None:
    try:
        registry.register(
            PromptLayer.role,
            ROLE_PROMPT_NAME,
            ROLE_PROMPT_VERSION,
            ROLE_PROMPT_TEXT,
        )
    except PromptAlreadyRegistered:
        _LOGGER.debug("candidate analysis role prompt already registered")


class CandidateAnalysisSpecialist(Specialist):
    def __init__(
        self,
        *,
        registry=None,
        prompt_registry: PromptRegistry | None = None,
    ) -> None:
        super().__init__(
            role=AgentRole.candidate_analysis,
            name=ROLE_PROMPT_NAME,
            description="Annotate catalog candidates with tool-backed fit reasons.",
            registry=registry,
            allowed_capabilities=("catalog_search",),
        )
        self._prompt_registry = prompt_registry
        if prompt_registry is not None:
            register_candidate_analysis_prompt(prompt_registry)

    async def execute(self, brief: SpecialistBrief):
        candidate_observation = await self._resolve_candidate_observation(brief)
        products = _products_from_observation(candidate_observation)
        analysis_items = [
            _analyze_product(product, brief.constraints, candidate_observation.observation_id)
            for product in products
        ]
        payload = CandidateAnalysisPayload(
            source_observation_id=candidate_observation.observation_id,
            candidates=analysis_items,
        )
        status = "ok" if analysis_items else "partial"
        return self.build_observation(
            brief,
            summary=f"Analyzed {len(analysis_items)} catalog-backed candidates.",
            status=status,
            payload=payload.model_dump(mode="json"),
            evidence_source=candidate_observation.evidence_source,
        )

    async def _resolve_candidate_observation(self, brief: SpecialistBrief) -> Observation:
        raw_observation = brief.constraints.get("candidate_observation")
        if isinstance(raw_observation, Observation):
            return raw_observation.model_copy(deep=True)
        if isinstance(raw_observation, dict):
            return Observation.model_validate(raw_observation)

        query = str(brief.constraints.get("query") or brief.goal)
        filters = _build_catalog_filters(brief.constraints)
        return await self.invoke_tool(
            brief,
            capability_name="catalog_search",
            arguments={"query": query, "filters": filters},
        )


def _build_catalog_filters(constraints: dict[str, Any]) -> dict[str, Any]:
    filters = dict(constraints.get("filters") or {})
    structured_brief = constraints.get("shopping_brief")
    if isinstance(structured_brief, dict):
        filters.setdefault("category", structured_brief.get("category"))
        filters.setdefault("scene", structured_brief.get("scene"))
        budget = structured_brief.get("budget")
        if isinstance(budget, dict):
            filters.setdefault("price_min", budget.get("min"))
            filters.setdefault("price_max", budget.get("max"))
        exclusions = structured_brief.get("exclusions")
        if exclusions:
            filters.setdefault("exclude_brands", exclusions)

    if "category" in constraints:
        filters.setdefault("category", constraints["category"])
    if "scene" in constraints:
        filters.setdefault("scene", constraints["scene"])
    if "budget_max" in constraints:
        filters.setdefault("price_max", constraints["budget_max"])
    if "budget_min" in constraints:
        filters.setdefault("price_min", constraints["budget_min"])
    if "exclude_brands" in constraints:
        filters.setdefault("exclude_brands", constraints["exclude_brands"])
    filters.setdefault("limit", constraints.get("limit", 4))
    return {key: value for key, value in filters.items() if value is not None}


def _products_from_observation(observation: Observation) -> list[Product]:
    raw_results = observation.payload.get("results", [])
    if not isinstance(raw_results, list):
        return []
    products: list[Product] = []
    for raw_product in raw_results:
        if isinstance(raw_product, Product):
            products.append(raw_product.model_copy(deep=True))
        elif isinstance(raw_product, dict):
            products.append(Product.model_validate(raw_product))
    return products


def _analyze_product(
    product: Product,
    constraints: dict[str, Any],
    observation_id: str,
) -> CandidateAnalysisItem:
    reasons: list[CandidateFitReason] = [
        CandidateFitReason(
            reason="Candidate came from the catalog_search tool result.",
            observation_id=observation_id,
        )
    ]
    budget_max = _budget_max(constraints)
    if budget_max is not None and product.price <= budget_max:
        reasons.append(
            CandidateFitReason(
                reason=f"Price {product.price} is within the requested budget ceiling {budget_max}.",
                observation_id=observation_id,
            )
        )

    scene = _scene_constraint(constraints)
    if scene and scene in product.scene_tags:
        reasons.append(
            CandidateFitReason(
                reason=f"Scene tag {scene!r} matches the product usage tags.",
                observation_id=observation_id,
            )
        )

    category = constraints.get("category")
    if category is not None and str(category) == product.category.value:
        reasons.append(
            CandidateFitReason(
                reason=f"Category {product.category.value!r} matches the requested category.",
                observation_id=observation_id,
            )
        )

    if product.rating >= 4.7:
        reasons.append(
            CandidateFitReason(
                reason=f"Rating {product.rating} is strong in the mock catalog.",
                observation_id=observation_id,
            )
        )

    return CandidateAnalysisItem(
        sku=product.sku,
        name=product.name,
        brand=product.brand,
        fit_score=len(reasons),
        fit_reasons=reasons,
    )


def _budget_max(constraints: dict[str, Any]) -> int | None:
    if "budget_max" in constraints:
        try:
            return int(constraints["budget_max"])
        except (TypeError, ValueError):
            return None
    structured_brief = constraints.get("shopping_brief")
    if isinstance(structured_brief, dict):
        budget = structured_brief.get("budget")
        if isinstance(budget, dict) and budget.get("max") is not None:
            return int(budget["max"])
    return None


def _scene_constraint(constraints: dict[str, Any]) -> str | None:
    if isinstance(constraints.get("scene"), str):
        return str(constraints["scene"])
    structured_brief = constraints.get("shopping_brief")
    if isinstance(structured_brief, dict) and isinstance(structured_brief.get("scene"), str):
        return str(structured_brief["scene"])
    return None


__all__ = [
    "CandidateAnalysisItem",
    "CandidateAnalysisPayload",
    "CandidateAnalysisSpecialist",
    "CandidateFitReason",
    "register_candidate_analysis_prompt",
]
