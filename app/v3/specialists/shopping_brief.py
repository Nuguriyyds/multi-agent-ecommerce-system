from __future__ import annotations

import logging
import re
from typing import Any

from pydantic import Field, ValidationError, field_validator, model_validator

from app.v3.models import AgentRole, ProductCategory, PromptLayer, SpecialistBrief
from app.v3.models.base import V3Model
from app.v3.prompts import PromptAlreadyRegistered, PromptRegistry

from .base import Specialist

ROLE_PROMPT_NAME = "shopping_brief_specialist"
ROLE_PROMPT_VERSION = "1"
ROLE_PROMPT_TEXT = (
    "Extract the user's shopping need into structured slots: budget, category, "
    "brand preference, scene, exclusions, and slots_missing. Do not recommend products."
)

_LOGGER = logging.getLogger(__name__)
_BRAND_KEYWORDS = (
    "Apple",
    "Sony",
    "Bose",
    "Sennheiser",
    "Huawei",
    "Xiaomi",
    "Samsung",
    "Beats",
    "Nothing",
)
_SCENE_KEYWORDS = {
    "通勤": "commute",
    "commute": "commute",
    "办公室": "office",
    "office": "office",
    "旅行": "travel",
    "travel": "travel",
    "礼物": "gift",
    "送礼": "gift",
    "gift": "gift",
    "运动": "gym",
    "健身": "gym",
    "gym": "gym",
    "游戏": "gaming",
    "gaming": "gaming",
    "daily": "daily",
    "日常": "daily",
}


class BudgetSlot(V3Model):
    min: int | None = Field(default=None, ge=0)
    max: int | None = Field(default=None, ge=0)
    currency: str = "CNY"

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: str) -> str:
        normalized = value.strip().upper()
        if not normalized:
            raise ValueError("currency must not be blank")
        return normalized

    @model_validator(mode="after")
    def validate_budget_range(self) -> "BudgetSlot":
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("budget min must be less than or equal to max")
        return self

    @property
    def is_complete(self) -> bool:
        return self.max is not None and self.currency.strip() != ""


class ShoppingBriefPayload(V3Model):
    budget: BudgetSlot | None = None
    category: ProductCategory | None = None
    brand: str | None = None
    scene: str | None = None
    exclusions: list[str] = Field(default_factory=list)
    slots_missing: list[str] = Field(default_factory=list)


def register_shopping_brief_prompt(registry: PromptRegistry) -> None:
    try:
        registry.register(
            PromptLayer.role,
            ROLE_PROMPT_NAME,
            ROLE_PROMPT_VERSION,
            ROLE_PROMPT_TEXT,
        )
    except PromptAlreadyRegistered:
        _LOGGER.debug("shopping brief role prompt already registered")


class ShoppingBriefSpecialist(Specialist):
    def __init__(
        self,
        *,
        prompt_registry: PromptRegistry | None = None,
    ) -> None:
        super().__init__(
            role=AgentRole.shopping_brief,
            name=ROLE_PROMPT_NAME,
            description="Extract a structured shopping brief from the user's raw need.",
            allowed_capabilities=(),
        )
        self._prompt_registry = prompt_registry
        if prompt_registry is not None:
            register_shopping_brief_prompt(prompt_registry)

    async def execute(self, brief: SpecialistBrief):
        raw_text = _raw_need_text(brief)
        budget, budget_missing = _extract_budget(brief.constraints, raw_text)
        category = _extract_category(brief.constraints, raw_text)
        scene = _extract_scene(brief.constraints, raw_text)
        brand = _extract_brand(brief.constraints, raw_text)
        exclusions = _extract_exclusions(brief.constraints, raw_text)

        slots_missing: list[str] = []
        if budget_missing or budget is None or not budget.is_complete:
            slots_missing.append("budget")
        if category is None:
            slots_missing.append("category")
        if scene is None:
            slots_missing.append("scene")

        payload = ShoppingBriefPayload(
            budget=budget,
            category=category,
            brand=brand,
            scene=scene,
            exclusions=exclusions,
            slots_missing=slots_missing,
        )
        status = "partial" if payload.slots_missing else "ok"
        summary = (
            "Shopping brief has missing slots: " + ", ".join(payload.slots_missing)
            if payload.slots_missing
            else "Shopping brief is complete enough for candidate search."
        )
        return self.build_observation(
            brief,
            summary=summary,
            status=status,
            payload=payload.model_dump(mode="json"),
        )


def _raw_need_text(brief: SpecialistBrief) -> str:
    for key in ("raw_user_need", "user_message", "latest_user_message"):
        value = brief.constraints.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if brief.context_packet is not None and brief.context_packet.latest_user_message.strip():
        return brief.context_packet.latest_user_message.strip()
    return brief.goal


def _extract_budget(
    constraints: dict[str, Any],
    raw_text: str,
) -> tuple[BudgetSlot | None, bool]:
    explicit_budget = constraints.get("budget")
    if explicit_budget is not None:
        try:
            if isinstance(explicit_budget, BudgetSlot):
                return explicit_budget, False
            if isinstance(explicit_budget, dict):
                return BudgetSlot.model_validate(explicit_budget), False
        except ValidationError:
            return None, True

    budget_min = constraints.get("budget_min")
    budget_max = constraints.get("budget_max")
    if budget_min is not None or budget_max is not None:
        try:
            return BudgetSlot(
                min=int(budget_min) if budget_min is not None else None,
                max=int(budget_max) if budget_max is not None else None,
                currency=str(constraints.get("currency", "CNY")),
            ), False
        except (TypeError, ValueError, ValidationError):
            return None, True

    if "预算" not in raw_text and "budget" not in raw_text.lower() and "左右" not in raw_text:
        return None, True

    number_matches = re.findall(r"\d{3,6}", raw_text.replace(",", ""))
    if not number_matches:
        return None, True

    budget_ceiling = int(number_matches[-1])
    return BudgetSlot(min=0, max=budget_ceiling, currency="CNY"), False


def _extract_category(
    constraints: dict[str, Any],
    raw_text: str,
) -> ProductCategory | None:
    raw_category = constraints.get("category")
    if raw_category is not None:
        try:
            return ProductCategory(str(raw_category))
        except ValueError:
            return None

    lowered = raw_text.lower()
    if any(token in lowered for token in ("earphone", "headphone", "earbuds")):
        return ProductCategory.earphones
    if any(token in raw_text for token in ("耳机", "降噪")):
        return ProductCategory.earphones
    if "phone" in lowered or "手机" in raw_text:
        return ProductCategory.phone
    return None


def _extract_scene(constraints: dict[str, Any], raw_text: str) -> str | None:
    raw_scene = constraints.get("scene")
    if isinstance(raw_scene, str) and raw_scene.strip():
        return raw_scene.strip()

    lowered = raw_text.lower()
    for keyword, scene in _SCENE_KEYWORDS.items():
        haystack = lowered if keyword.isascii() else raw_text
        if keyword in haystack:
            return scene
    return None


def _extract_brand(constraints: dict[str, Any], raw_text: str) -> str | None:
    raw_brand = constraints.get("brand")
    if isinstance(raw_brand, str) and raw_brand.strip():
        return raw_brand.strip()

    lowered = raw_text.lower()
    for brand in _BRAND_KEYWORDS:
        if brand.lower() in lowered:
            return brand
    return None


def _extract_exclusions(constraints: dict[str, Any], raw_text: str) -> list[str]:
    exclusions: list[str] = []
    for key in ("exclusions", "exclude_brands"):
        raw_value = constraints.get(key)
        if isinstance(raw_value, str) and raw_value.strip():
            exclusions.append(raw_value.strip())
        elif isinstance(raw_value, list):
            exclusions.extend(str(item).strip() for item in raw_value if str(item).strip())

    lowered = raw_text.lower()
    for brand in _BRAND_KEYWORDS:
        brand_index = lowered.find(brand.lower())
        if brand_index < 0:
            continue
        prefix = raw_text[max(0, brand_index - 8) : brand_index]
        if any(marker in prefix for marker in ("不要", "排除", "不考虑", "exclude", "not ")):
            exclusions.append(brand)

    return list(dict.fromkeys(exclusions))


__all__ = [
    "BudgetSlot",
    "ShoppingBriefPayload",
    "ShoppingBriefSpecialist",
    "register_shopping_brief_prompt",
]
