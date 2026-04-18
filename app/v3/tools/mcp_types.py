from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from app.v3.models.base import V3Model


class MCPToolDefinition(V3Model):
    name: str
    description: str | None = None
    input_schema: dict[str, Any] = Field(default_factory=dict)


class MCPContentBlock(V3Model):
    type: Literal["text", "json"]
    text: str | None = None
    data: dict[str, Any] | list[Any] | None = None

    @model_validator(mode="after")
    def validate_payload(self) -> "MCPContentBlock":
        if self.type == "text" and not self.text:
            raise ValueError("text blocks require text")
        if self.type == "json" and self.data is None:
            raise ValueError("json blocks require data")
        return self


class MCPToolCallResult(V3Model):
    tool_name: str
    content: list[MCPContentBlock] = Field(default_factory=list)
    is_error: bool = False

    def json_items(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for block in self.content:
            if block.type != "json" or not isinstance(block.data, dict):
                continue
            items.append(dict(block.data))
        return items


__all__ = [
    "MCPContentBlock",
    "MCPToolCallResult",
    "MCPToolDefinition",
]
