from __future__ import annotations

import logging
from typing import Any

from pydantic import Field

from app.v3.models import AgentRole, Observation, PromptLayer, SpecialistBrief
from app.v3.models.base import V3Model
from app.v3.prompts import PromptAlreadyRegistered, PromptRegistry

from .base import Specialist

ROLE_PROMPT_NAME = "recommendation_rationale_specialist"
ROLE_PROMPT_VERSION = "1"
ROLE_PROMPT_TEXT = (
    "Write traceable recommendation rationales. Every rationale item must carry "
    "an observation_id from catalog, inventory, comparison, or MCP RAG evidence."
)

_LOGGER = logging.getLogger(__name__)
_ALL_RATIONALE_CAPABILITIES = (
    "catalog_search",
    "inventory_check",
    "product_compare",
    "rag_product_knowledge",
)


class RationaleItem(V3Model):
    reason: str
    observation_id: str


class RecommendationRationalePayload(V3Model):
    pick_sku: str | None = None
    rationales: list[RationaleItem] = Field(default_factory=list)
    evidence_observation_ids: list[str] = Field(default_factory=list)
    evidence_sufficient: bool = False
    supporting_observations: list[dict[str, Any]] = Field(default_factory=list)


def register_recommendation_rationale_prompt(registry: PromptRegistry) -> None:
    try:
        registry.register(
            PromptLayer.role,
            ROLE_PROMPT_NAME,
            ROLE_PROMPT_VERSION,
            ROLE_PROMPT_TEXT,
        )
    except PromptAlreadyRegistered:
        _LOGGER.debug("recommendation rationale role prompt already registered")


class RecommendationRationaleSpecialist(Specialist):
    def __init__(
        self,
        *,
        registry=None,
        prompt_registry: PromptRegistry | None = None,
    ) -> None:
        super().__init__(
            role=AgentRole.recommendation_rationale,
            name=ROLE_PROMPT_NAME,
            description="Generate evidence-cited recommendation rationale items.",
            registry=registry,
            allowed_capabilities=_ALL_RATIONALE_CAPABILITIES,
        )
        self._prompt_registry = prompt_registry
        if prompt_registry is not None:
            register_recommendation_rationale_prompt(prompt_registry)

    async def execute(self, brief: SpecialistBrief):
        observations = _extract_observations(brief.constraints)
        pick_sku = _pick_sku(brief.constraints, observations)

        if (
            brief.constraints.get("use_rag", True)
            and "rag_product_knowledge" in brief.allowed_capabilities
        ):
            rag_observation = await self._try_rag_lookup(brief, pick_sku)
            if rag_observation is not None:
                observations.append(rag_observation)

        rationales = _build_rationales(pick_sku, observations)
        evidence_ids = list(dict.fromkeys(item.observation_id for item in rationales))
        payload = RecommendationRationalePayload(
            pick_sku=pick_sku,
            rationales=rationales,
            evidence_observation_ids=evidence_ids,
            evidence_sufficient=bool(rationales),
            supporting_observations=[
                observation.model_dump(mode="json") for observation in observations
            ],
        )
        return self.build_observation(
            brief,
            summary=(
                f"Built {len(rationales)} traceable recommendation rationales."
                if rationales
                else "Recommendation rationale needs more evidence."
            ),
            status="ok" if rationales else "partial",
            payload=payload.model_dump(mode="json"),
        )

    async def _try_rag_lookup(
        self,
        brief: SpecialistBrief,
        pick_sku: str | None,
    ) -> Observation | None:
        query = str(brief.constraints.get("query") or pick_sku or brief.goal)
        try:
            return await self.invoke_tool(
                brief,
                capability_name="rag_product_knowledge",
                arguments={"query": query, "limit": 3},
            )
        except LookupError:
            _LOGGER.info("rag_product_knowledge is not registered; continuing without MCP evidence")
            return None


def _extract_observations(constraints: dict[str, Any]) -> list[Observation]:
    observations: list[Observation] = []
    for key in (
        "observations",
        "candidate_observations",
        "comparison_observations",
        "inventory_observations",
    ):
        raw_value = constraints.get(key)
        if isinstance(raw_value, list):
            observations.extend(_coerce_observation(item) for item in raw_value)

    for key in ("candidate_observation", "comparison_observation", "inventory_observation"):
        raw_value = constraints.get(key)
        if raw_value is not None:
            observations.append(_coerce_observation(raw_value))

    return observations


def _coerce_observation(raw_value: Any) -> Observation:
    if isinstance(raw_value, Observation):
        return raw_value.model_copy(deep=True)
    if isinstance(raw_value, dict):
        return Observation.model_validate(raw_value)
    raise TypeError(f"Unsupported observation payload: {type(raw_value).__name__}")


def _pick_sku(constraints: dict[str, Any], observations: list[Observation]) -> str | None:
    for key in ("pick_sku", "selected_sku", "sku"):
        raw_value = constraints.get(key)
        if isinstance(raw_value, str) and raw_value.strip():
            return raw_value.strip().upper()

    for observation in observations:
        results = observation.payload.get("results")
        if isinstance(results, list) and results:
            first = results[0]
            if isinstance(first, dict) and first.get("sku"):
                return str(first["sku"]).strip().upper()
    return None


def _build_rationales(
    pick_sku: str | None,
    observations: list[Observation],
) -> list[RationaleItem]:
    rationales: list[RationaleItem] = []
    for observation in observations:
        payload = observation.payload
        source = observation.evidence_source or observation.source
        if _catalog_supports_pick(payload, pick_sku):
            rationales.append(
                RationaleItem(
                    reason=f"{pick_sku} appears in catalog-backed candidate results.",
                    observation_id=observation.observation_id,
                )
            )
        elif _comparison_supports_pick(payload, pick_sku):
            rationales.append(
                RationaleItem(
                    reason=f"Comparison evidence supports considering {pick_sku}.",
                    observation_id=observation.observation_id,
                )
            )
        elif _inventory_supports_pick(payload, pick_sku):
            status = payload.get("status", "available")
            rationales.append(
                RationaleItem(
                    reason=f"Inventory evidence reports {pick_sku} as {status}.",
                    observation_id=observation.observation_id,
                )
            )
        elif _rag_supports_pick(payload, pick_sku):
            rationales.append(
                RationaleItem(
                    reason=f"MCP product knowledge adds context for {pick_sku}.",
                    observation_id=observation.observation_id,
                )
            )
        elif not pick_sku and source:
            rationales.append(
                RationaleItem(
                    reason=f"Evidence from {source} is available for recommendation synthesis.",
                    observation_id=observation.observation_id,
                )
            )

    deduped: list[RationaleItem] = []
    seen: set[tuple[str, str]] = set()
    for rationale in rationales:
        marker = (rationale.reason, rationale.observation_id)
        if marker not in seen:
            deduped.append(rationale)
            seen.add(marker)
    return deduped


def _catalog_supports_pick(payload: dict[str, Any], pick_sku: str | None) -> bool:
    if pick_sku is None:
        return False
    results = payload.get("results")
    if not isinstance(results, list):
        return False
    return any(
        isinstance(item, dict) and str(item.get("sku", "")).upper() == pick_sku
        for item in results
    )


def _comparison_supports_pick(payload: dict[str, Any], pick_sku: str | None) -> bool:
    if pick_sku is None:
        return False
    sku_values = {
        str(payload.get("sku_a", "")).upper(),
        str(payload.get("sku_b", "")).upper(),
    }
    if pick_sku in sku_values:
        return True

    comparisons = payload.get("comparisons")
    if not isinstance(comparisons, list):
        return False
    for comparison in comparisons:
        if not isinstance(comparison, dict):
            continue
        nested_payload = comparison.get("payload")
        if isinstance(nested_payload, dict) and _comparison_supports_pick(nested_payload, pick_sku):
            return True
    return False


def _inventory_supports_pick(payload: dict[str, Any], pick_sku: str | None) -> bool:
    return pick_sku is not None and str(payload.get("sku", "")).upper() == pick_sku


def _rag_supports_pick(payload: dict[str, Any], pick_sku: str | None) -> bool:
    if pick_sku is None:
        return False
    snippets = payload.get("snippets")
    if not isinstance(snippets, list):
        return False
    return any(
        isinstance(item, dict) and str(item.get("sku", "")).upper() == pick_sku
        for item in snippets
    )


__all__ = [
    "RecommendationRationalePayload",
    "RecommendationRationaleSpecialist",
    "RationaleItem",
    "register_recommendation_rationale_prompt",
]
