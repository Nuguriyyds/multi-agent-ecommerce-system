from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import Field, field_validator

from app.v3.models import Product
from app.v3.models.base import V3Model

from ..mcp_types import MCPToolDefinition
from .knowledge_base import KnowledgeSnippet, build_knowledge_base, search_product_knowledge

_RAG_PRODUCT_KNOWLEDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "limit": {"type": "integer", "minimum": 3, "maximum": 5},
    },
    "required": ["query"],
    "additionalProperties": False,
}


class RagProductKnowledgeRequest(V3Model):
    query: str = Field(min_length=1)
    limit: int = Field(default=4, ge=3, le=5)

    @field_validator("query")
    @classmethod
    def strip_query(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("query must not be empty")
        return normalized


class MockMCPServer:
    def __init__(
        self,
        *,
        catalog: Sequence[Product] | None = None,
        knowledge_base: Sequence[KnowledgeSnippet] | None = None,
    ) -> None:
        self._knowledge_base = [
            snippet.model_copy(deep=True)
            for snippet in (knowledge_base or build_knowledge_base(catalog=catalog))
        ]
        self._tool_definition = MCPToolDefinition(
            name="rag_product_knowledge",
            description="Search product-buying knowledge snippets derived from the local mock catalog.",
            input_schema=_RAG_PRODUCT_KNOWLEDGE_SCHEMA,
        )

    def list_tool_definitions(self) -> list[MCPToolDefinition]:
        return [self._tool_definition.model_copy(deep=True)]

    async def handle_request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "tools.list":
            return {
                "tools": [
                    tool.model_dump(mode="json")
                    for tool in self.list_tool_definitions()
                ]
            }
        if method == "tools.call":
            return await self._handle_tools_call(params)
        raise ValueError(f"Unsupported MCP method: {method}")

    async def _handle_tools_call(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        arguments = params.get("arguments", {})
        if name != self._tool_definition.name:
            raise LookupError(f"Unknown MCP tool: {name}")
        if not isinstance(arguments, dict):
            raise TypeError("MCP tool arguments must be an object")

        request = RagProductKnowledgeRequest.model_validate(arguments)
        snippets = search_product_knowledge(
            request.query,
            limit=request.limit,
            knowledge_base=self._knowledge_base,
        )
        return {
            "content": [
                {
                    "type": "json",
                    "data": snippet.model_dump(mode="json"),
                }
                for snippet in snippets
            ],
            "is_error": False,
        }


__all__ = ["MockMCPServer", "RagProductKnowledgeRequest"]
