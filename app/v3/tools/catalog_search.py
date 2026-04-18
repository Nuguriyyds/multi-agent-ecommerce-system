from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from uuid import uuid4

from app.v3.models import (
    CapabilityDescriptor,
    CapabilityKind,
    CatalogSearchFilters,
    CatalogSearchRequest,
    Observation,
    Product,
)
from app.v3.registry import ToolProvider

from .seed_data import get_seed_catalog

_CATALOG_SEARCH_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "filters": {
            "type": "object",
            "properties": {
                "category": {"type": "string"},
                "subcategory": {"type": "string"},
                "brand": {"type": "string"},
                "exclude_brands": {"type": "array", "items": {"type": "string"}},
                "scene": {"type": "string"},
                "price_min": {"type": "integer"},
                "price_max": {"type": "integer"},
                "min_rating": {"type": "number"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "limit": {"type": "integer"},
            },
            "additionalProperties": False,
        },
    },
    "required": ["query"],
    "additionalProperties": False,
}

_CATALOG_SEARCH_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "filters": {"type": "object"},
        "total": {"type": "integer"},
        "results": {"type": "array", "items": {"type": "object"}},
    },
    "required": ["query", "filters", "total", "results"],
}


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _extract_query_terms(query: str) -> list[str]:
    return [match.group(0) for match in re.finditer(r"[a-z0-9]+|[\u4e00-\u9fff]+", query)]


def _product_search_text(product: Product) -> str:
    chunks = [
        product.name,
        product.brand,
        product.category.value,
        product.subcategory,
        product.description,
        *product.scene_tags,
        *product.tags,
        *product.aliases,
    ]
    return " ".join(_normalize_text(chunk) for chunk in chunks if chunk)


def _matches_filters(product: Product, filters: CatalogSearchFilters) -> bool:
    if filters.category is not None and product.category is not filters.category:
        return False
    if filters.subcategory is not None and _normalize_text(product.subcategory) != _normalize_text(filters.subcategory):
        return False
    if filters.brand is not None and _normalize_text(product.brand) != _normalize_text(filters.brand):
        return False
    if filters.exclude_brands and _normalize_text(product.brand) in {
        _normalize_text(item) for item in filters.exclude_brands
    }:
        return False
    if filters.scene is not None and _normalize_text(filters.scene) not in {
        _normalize_text(item) for item in product.scene_tags
    }:
        return False
    if filters.price_min is not None and product.price < filters.price_min:
        return False
    if filters.price_max is not None and product.price > filters.price_max:
        return False
    if filters.min_rating is not None and product.rating < filters.min_rating:
        return False
    if filters.tags and not {
        _normalize_text(item) for item in filters.tags
    }.issubset({_normalize_text(item) for item in product.tags}):
        return False
    return True


def _score_product(product: Product, query: str) -> int:
    normalized_query = _normalize_text(query)
    searchable = _product_search_text(product)
    score = 0
    if normalized_query in searchable:
        score += 12

    for term in _extract_query_terms(normalized_query):
        if term in searchable:
            score += 3

    for keyword in [product.brand, product.subcategory, *product.tags, *product.aliases, *product.scene_tags]:
        normalized_keyword = _normalize_text(keyword)
        if normalized_keyword and normalized_keyword in normalized_query:
            score += 2

    if str(product.price) in normalized_query:
        score += 1
    return score


def catalog_search(
    query: str,
    filters: CatalogSearchFilters | Mapping[str, object] | None = None,
    *,
    catalog: Sequence[Product] | None = None,
) -> list[Product]:
    request = CatalogSearchRequest(
        query=query,
        filters=filters if isinstance(filters, CatalogSearchFilters) else dict(filters or {}),
    )
    working_catalog = list(catalog or get_seed_catalog())
    target_price: int | None = None
    if request.filters.price_min is not None and request.filters.price_max is not None:
        target_price = (request.filters.price_min + request.filters.price_max) // 2
    elif request.filters.price_max is not None:
        target_price = request.filters.price_max
    elif request.filters.price_min is not None:
        target_price = request.filters.price_min

    ranked: list[tuple[int, int, Product]] = []
    for product in working_catalog:
        if not _matches_filters(product, request.filters):
            continue
        score = _score_product(product, request.query)
        distance = abs(product.price - target_price) if target_price is not None else 0
        ranked.append((score, distance, product))

    ranked.sort(key=lambda item: (-item[0], item[1], -item[2].rating, item[2].price, item[2].name))
    return [product.model_copy(deep=True) for _, _, product in ranked[: request.filters.limit]]


class CatalogSearchProvider(ToolProvider):
    def __init__(self, *, catalog: Sequence[Product] | None = None) -> None:
        self._catalog = [product.model_copy(deep=True) for product in (catalog or get_seed_catalog())]
        super().__init__(
            CapabilityDescriptor(
                name="catalog_search",
                kind=CapabilityKind.tool,
                input_schema=_CATALOG_SEARCH_INPUT_SCHEMA,
                output_schema=_CATALOG_SEARCH_OUTPUT_SCHEMA,
                timeout=3.0,
                permission_tag="catalog.read",
                description="Search the local mock product catalog by query and filters.",
            )
        )
        self._logger = logging.getLogger(__name__)

    async def invoke(self, args: dict[str, object]) -> Observation:
        self._logger.info("catalog_search start args=%s", args)
        try:
            request = CatalogSearchRequest.model_validate(args)
            results = catalog_search(
                request.query,
                request.filters,
                catalog=self._catalog,
            )
        except Exception:
            self._logger.exception("catalog_search failed")
            raise

        observation = Observation(
            observation_id=f"obs-{uuid4().hex[:12]}",
            source=self.name,
            status="ok" if results else "partial",
            summary=f"Found {len(results)} matching products in the local mock catalog.",
            payload={
                "query": request.query,
                "filters": request.filters.model_dump(mode="json"),
                "total": len(results),
                "results": [product.model_dump(mode="json") for product in results],
            },
            evidence_source=f"tool:{self.name}",
        )
        self._logger.info("catalog_search success total=%s observation_id=%s", len(results), observation.observation_id)
        return observation


__all__ = ["CatalogSearchProvider", "catalog_search"]
