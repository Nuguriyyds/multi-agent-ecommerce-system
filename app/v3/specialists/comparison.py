from __future__ import annotations

import asyncio
import logging
from typing import Any

from pydantic import Field

from app.v3.models import AgentRole, ComparisonDimension, Observation, PromptLayer, SpecialistBrief
from app.v3.models.base import V3Model
from app.v3.prompts import PromptAlreadyRegistered, PromptRegistry

from .base import Specialist

ROLE_PROMPT_NAME = "comparison_specialist"
ROLE_PROMPT_VERSION = "1"
ROLE_PROMPT_TEXT = (
    "Compare candidate products only on approved dimensions: price, battery, "
    "noise_cancel, weight, warranty, and brand. Drop all other requested dimensions."
)

ALLOWED_COMPARISON_DIMENSIONS = frozenset(
    {
        ComparisonDimension.price,
        ComparisonDimension.battery,
        ComparisonDimension.noise_cancel,
        ComparisonDimension.weight,
        ComparisonDimension.warranty,
        ComparisonDimension.brand,
    }
)
DEFAULT_COMPARISON_DIMENSIONS = (
    ComparisonDimension.price,
    ComparisonDimension.battery,
    ComparisonDimension.noise_cancel,
    ComparisonDimension.weight,
)

_LOGGER = logging.getLogger(__name__)


class ComparisonObservationRef(V3Model):
    observation_id: str
    payload: dict[str, Any] = Field(default_factory=dict)


class DomainComparisonPayload(V3Model):
    skus: list[str]
    accepted_dimensions: list[str] = Field(default_factory=list)
    ignored_dimensions: list[str] = Field(default_factory=list)
    comparisons: list[ComparisonObservationRef] = Field(default_factory=list)
    inventory: list[ComparisonObservationRef] = Field(default_factory=list)


def register_comparison_prompt(registry: PromptRegistry) -> None:
    try:
        registry.register(
            PromptLayer.role,
            ROLE_PROMPT_NAME,
            ROLE_PROMPT_VERSION,
            ROLE_PROMPT_TEXT,
        )
    except PromptAlreadyRegistered:
        _LOGGER.debug("comparison role prompt already registered")


class ComparisonSpecialist(Specialist):
    def __init__(
        self,
        *,
        registry=None,
        prompt_registry: PromptRegistry | None = None,
    ) -> None:
        super().__init__(
            role=AgentRole.comparison,
            name=ROLE_PROMPT_NAME,
            description="Compare products across whitelisted dimensions.",
            registry=registry,
            allowed_capabilities=("product_compare", "inventory_check"),
        )
        self._prompt_registry = prompt_registry
        if prompt_registry is not None:
            register_comparison_prompt(prompt_registry)

    async def execute(self, brief: SpecialistBrief):
        skus = _extract_skus(brief.constraints)
        accepted_dimensions, ignored_dimensions = _filter_dimensions(
            brief.constraints.get("dimensions")
        )
        if len(skus) < 2 or not accepted_dimensions:
            payload = DomainComparisonPayload(
                skus=skus,
                accepted_dimensions=[dimension.value for dimension in accepted_dimensions],
                ignored_dimensions=ignored_dimensions,
            )
            return self.build_observation(
                brief,
                summary="Comparison requires at least two products and one allowed dimension.",
                status="partial",
                payload=payload.model_dump(mode="json"),
            )

        comparison_observations = await asyncio.gather(
            *[
                self.invoke_tool(
                    brief,
                    capability_name="product_compare",
                    arguments={
                        "sku_a": skus[0],
                        "sku_b": sku_b,
                        "dimensions": [dimension.value for dimension in accepted_dimensions],
                    },
                )
                for sku_b in skus[1:]
            ]
        )

        inventory_observations: list[Observation] = []
        if "inventory_check" in brief.allowed_capabilities:
            inventory_observations = list(
                await asyncio.gather(
                    *[
                        self.invoke_tool(
                            brief,
                            capability_name="inventory_check",
                            arguments={"sku": sku},
                        )
                        for sku in skus
                    ]
                )
            )

        payload = DomainComparisonPayload(
            skus=skus,
            accepted_dimensions=[dimension.value for dimension in accepted_dimensions],
            ignored_dimensions=ignored_dimensions,
            comparisons=[
                ComparisonObservationRef(
                    observation_id=observation.observation_id,
                    payload=observation.payload,
                )
                for observation in comparison_observations
            ],
            inventory=[
                ComparisonObservationRef(
                    observation_id=observation.observation_id,
                    payload=observation.payload,
                )
                for observation in inventory_observations
            ],
        )
        return self.build_observation(
            brief,
            summary=f"Compared {len(skus)} products across {len(accepted_dimensions)} approved dimensions.",
            payload=payload.model_dump(mode="json"),
            evidence_source="tool:product_compare",
        )


def _extract_skus(constraints: dict[str, Any]) -> list[str]:
    raw_skus = constraints.get("skus")
    if isinstance(raw_skus, list):
        return [str(item).strip().upper() for item in raw_skus if str(item).strip()]

    candidates = constraints.get("candidates")
    if isinstance(candidates, list):
        skus: list[str] = []
        for candidate in candidates:
            if isinstance(candidate, dict) and candidate.get("sku"):
                skus.append(str(candidate["sku"]).strip().upper())
        return skus
    return []


def _filter_dimensions(raw_dimensions: Any) -> tuple[list[ComparisonDimension], list[str]]:
    supplied = raw_dimensions if isinstance(raw_dimensions, list) else []
    if not supplied:
        return list(DEFAULT_COMPARISON_DIMENSIONS), []

    accepted: list[ComparisonDimension] = []
    ignored: list[str] = []
    for raw_dimension in supplied:
        normalized = str(raw_dimension).strip()
        try:
            dimension = ComparisonDimension(normalized)
        except ValueError:
            ignored.append(normalized)
            continue
        if dimension in ALLOWED_COMPARISON_DIMENSIONS:
            if dimension not in accepted:
                accepted.append(dimension)
        else:
            ignored.append(normalized)
    return accepted, ignored


__all__ = [
    "ALLOWED_COMPARISON_DIMENSIONS",
    "ComparisonObservationRef",
    "ComparisonSpecialist",
    "DomainComparisonPayload",
    "register_comparison_prompt",
]
