from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import Field

from app.v3.models import Product
from app.v3.models.base import V3Model

from ..seed_data import get_seed_catalog


class KnowledgeSnippet(V3Model):
    snippet_id: str
    sku: str
    product_name: str
    brand: str
    category: str
    subcategory: str
    price: int = Field(ge=0)
    rating: float = Field(ge=0.0, le=5.0)
    title: str
    excerpt: str
    scene_tags: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    knowledge_type: str = "product_note"


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _extract_terms(query: str) -> list[str]:
    return [match.group(0) for match in re.finditer(r"[a-z0-9]+|[\u4e00-\u9fff]+", query)]


def _feature_summary(product: Product) -> str:
    chunks: list[str] = [f"{product.price} CNY"]
    if "battery_hours" in product.features:
        chunks.append(f"battery {product.features['battery_hours']}h")
    if "battery_mah" in product.features:
        chunks.append(f"battery {product.features['battery_mah']}mAh")
    if "noise_cancel_score" in product.features:
        chunks.append(f"ANC {product.features['noise_cancel_score']}/10")
    if "camera_score" in product.features:
        chunks.append(f"camera {product.features['camera_score']}")
    if "charging_watts" in product.features:
        chunks.append(f"charging {product.features['charging_watts']}W")
    if "weight_grams" in product.features:
        chunks.append(f"weight {product.features['weight_grams']}g")
    return ", ".join(chunks)


def build_knowledge_base(*, catalog: Sequence[Product] | None = None) -> list[KnowledgeSnippet]:
    snippets: list[KnowledgeSnippet] = []
    for product in catalog or get_seed_catalog():
        excerpt = (
            f"{product.description} "
            f"Best for {', '.join(product.scene_tags)}. "
            f"Highlights: {_feature_summary(product)}."
        )
        snippets.append(
            KnowledgeSnippet(
                snippet_id=f"kb-{product.sku.lower()}",
                sku=product.sku,
                product_name=product.name,
                brand=product.brand,
                category=product.category.value,
                subcategory=product.subcategory,
                price=product.price,
                rating=product.rating,
                title=f"{product.name} buying notes",
                excerpt=excerpt,
                scene_tags=list(product.scene_tags),
                tags=list(product.tags),
                aliases=list(product.aliases),
            )
        )
    return snippets


def _searchable_text(snippet: KnowledgeSnippet) -> str:
    chunks = [
        snippet.product_name,
        snippet.brand,
        snippet.category,
        snippet.subcategory,
        snippet.title,
        snippet.excerpt,
        *snippet.scene_tags,
        *snippet.tags,
        *snippet.aliases,
    ]
    return " ".join(_normalize_text(chunk) for chunk in chunks if chunk)


def _score(snippet: KnowledgeSnippet, query: str) -> int:
    normalized_query = _normalize_text(query)
    searchable = _searchable_text(snippet)
    score = 0
    if normalized_query in searchable:
        score += 12

    for term in _extract_terms(normalized_query):
        if term in searchable:
            score += 3

    for keyword in [snippet.brand, snippet.subcategory, *snippet.scene_tags, *snippet.tags, *snippet.aliases]:
        normalized_keyword = _normalize_text(keyword)
        if normalized_keyword and normalized_keyword in normalized_query:
            score += 2

    return score


def search_product_knowledge(
    query: str,
    *,
    limit: int = 4,
    knowledge_base: Sequence[KnowledgeSnippet] | None = None,
) -> list[KnowledgeSnippet]:
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query must not be empty")

    ranked = sorted(
        (snippet.model_copy(deep=True) for snippet in (knowledge_base or build_knowledge_base())),
        key=lambda snippet: (
            -_score(snippet, normalized_query),
            -snippet.rating,
            snippet.price,
            snippet.product_name,
        ),
    )
    return ranked[:limit]


__all__ = ["KnowledgeSnippet", "build_knowledge_base", "search_product_knowledge"]
