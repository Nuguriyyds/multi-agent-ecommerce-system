from __future__ import annotations

import logging
from collections.abc import Sequence
from uuid import uuid4

from app.v3.models import (
    CapabilityDescriptor,
    CapabilityKind,
    ComparisonDimension,
    ComparisonDimensionResult,
    ComparisonResult,
    Observation,
    Product,
    ProductCompareRequest,
)
from app.v3.models.catalog import ComparableValue
from app.v3.registry import ToolProvider

from .seed_data import find_product, get_seed_catalog

_PRODUCT_COMPARE_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "sku_a": {"type": "string"},
        "sku_b": {"type": "string"},
        "dimensions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["sku_a", "sku_b", "dimensions"],
    "additionalProperties": False,
}

_PRODUCT_COMPARE_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "sku_a": {"type": "string"},
        "sku_b": {"type": "string"},
        "product_a_name": {"type": "string"},
        "product_b_name": {"type": "string"},
        "summary": {"type": "string"},
        "dimensions": {"type": "array", "items": {"type": "object"}},
    },
    "required": ["sku_a", "sku_b", "product_a_name", "product_b_name", "summary", "dimensions"],
}


def _dimension_value(product: Product, dimension: ComparisonDimension) -> ComparableValue:
    if dimension is ComparisonDimension.price:
        return product.price
    if dimension is ComparisonDimension.battery:
        return product.features.get("battery_mah") or product.features.get("battery_hours")
    if dimension is ComparisonDimension.noise_cancel:
        return product.features.get("noise_cancel_score")
    if dimension is ComparisonDimension.weight:
        return product.features.get("weight_grams")
    if dimension is ComparisonDimension.warranty:
        return product.features.get("warranty_years")
    if dimension is ComparisonDimension.brand:
        return product.brand
    if dimension is ComparisonDimension.camera:
        return product.features.get("camera_score")
    if dimension is ComparisonDimension.charging:
        return product.features.get("charging_watts")
    return None


def _compare_values(
    dimension: ComparisonDimension,
    *,
    value_a: ComparableValue,
    value_b: ComparableValue,
) -> tuple[str, str]:
    if value_a is None or value_b is None:
        return ("not_applicable", f"{dimension.value} is not available for both products.")

    if dimension is ComparisonDimension.brand:
        if value_a == value_b:
            return ("tie", "Both products are from the same brand.")
        return ("not_applicable", "Brand is qualitative and does not produce a numerical winner.")

    if value_a == value_b:
        return ("tie", f"Both products are tied on {dimension.value}.")

    lower_is_better = {ComparisonDimension.price, ComparisonDimension.weight}
    if dimension in lower_is_better:
        winner = "sku_a" if value_a < value_b else "sku_b"
    else:
        winner = "sku_a" if value_a > value_b else "sku_b"
    return (winner, f"{winner} is stronger on {dimension.value}.")


def product_compare(
    sku_a: str,
    sku_b: str,
    dimensions: Sequence[str | ComparisonDimension],
    *,
    catalog: Sequence[Product] | None = None,
) -> ComparisonResult:
    request = ProductCompareRequest(
        sku_a=sku_a,
        sku_b=sku_b,
        dimensions=list(dimensions),
    )
    product_a = find_product(request.sku_a, catalog)
    product_b = find_product(request.sku_b, catalog)
    dimension_results: list[ComparisonDimensionResult] = []
    wins = {"sku_a": 0, "sku_b": 0}

    for dimension in request.dimensions:
        value_a = _dimension_value(product_a, dimension)
        value_b = _dimension_value(product_b, dimension)
        winner, rationale = _compare_values(dimension, value_a=value_a, value_b=value_b)
        if winner in wins:
            wins[winner] += 1
        dimension_results.append(
            ComparisonDimensionResult(
                dimension=dimension,
                value_a=value_a,
                value_b=value_b,
                winner=winner,
                rationale=rationale,
            )
        )

    if wins["sku_a"] > wins["sku_b"]:
        summary = f"{product_a.name} leads this comparison."
    elif wins["sku_b"] > wins["sku_a"]:
        summary = f"{product_b.name} leads this comparison."
    else:
        summary = "The two products are balanced across the requested dimensions."

    return ComparisonResult(
        sku_a=product_a.sku,
        sku_b=product_b.sku,
        product_a_name=product_a.name,
        product_b_name=product_b.name,
        category=product_a.category if product_a.category is product_b.category else None,
        dimensions=dimension_results,
        summary=summary,
    )


class ProductCompareProvider(ToolProvider):
    def __init__(self, *, catalog: Sequence[Product] | None = None) -> None:
        self._catalog = [product.model_copy(deep=True) for product in (catalog or get_seed_catalog())]
        super().__init__(
            CapabilityDescriptor(
                name="product_compare",
                kind=CapabilityKind.tool,
                input_schema=_PRODUCT_COMPARE_INPUT_SCHEMA,
                output_schema=_PRODUCT_COMPARE_OUTPUT_SCHEMA,
                timeout=3.0,
                permission_tag="catalog.compare",
                description="Compare two mock catalog products across requested dimensions.",
            )
        )
        self._logger = logging.getLogger(__name__)

    async def invoke(self, args: dict[str, object]) -> Observation:
        self._logger.info("product_compare start args=%s", args)
        try:
            request = ProductCompareRequest.model_validate(args)
            result = product_compare(
                request.sku_a,
                request.sku_b,
                request.dimensions,
                catalog=self._catalog,
            )
        except Exception:
            self._logger.exception("product_compare failed")
            raise

        observation = Observation(
            observation_id=f"obs-{uuid4().hex[:12]}",
            source=self.name,
            summary=result.summary,
            payload=result.model_dump(mode="json"),
            evidence_source=f"tool:{self.name}",
        )
        self._logger.info(
            "product_compare success sku_a=%s sku_b=%s observation_id=%s",
            result.sku_a,
            result.sku_b,
            observation.observation_id,
        )
        return observation


__all__ = ["ProductCompareProvider", "product_compare"]
