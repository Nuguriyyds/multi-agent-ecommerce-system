"""V3 mock tools and MCP adapters."""

from __future__ import annotations

from collections.abc import Sequence

from app.v3.models import CapabilityDescriptor, Product
from app.v3.registry import CapabilityRegistry, ToolProvider

from .catalog_search import CatalogSearchProvider, catalog_search
from .inventory_check import InventoryCheckProvider, inventory_check
from .mcp_client import InProcessMCPTransport, MCPClient, MCPToolCallResult, MCPToolDefinition
from .mcp_mock_server import MockMCPServer, build_knowledge_base, search_product_knowledge
from .mcp_provider import (
    MCPToolProvider,
    build_mock_mcp_tool_providers,
    register_mock_mcp_tool_providers,
)
from .mcp_types import MCPContentBlock
from .product_compare import ProductCompareProvider, product_compare
from .seed_data import get_seed_catalog, seed_counts


def build_mock_tool_providers(*, catalog: Sequence[Product] | None = None) -> list[ToolProvider]:
    return [
        CatalogSearchProvider(catalog=catalog),
        InventoryCheckProvider(catalog=catalog),
        ProductCompareProvider(catalog=catalog),
    ]


def register_mock_tool_providers(
    registry: CapabilityRegistry,
    *,
    catalog: Sequence[Product] | None = None,
) -> list[CapabilityDescriptor]:
    descriptors: list[CapabilityDescriptor] = []
    for provider in build_mock_tool_providers(catalog=catalog):
        descriptors.append(registry.register(provider))
    return descriptors


__all__ = [
    "CatalogSearchProvider",
    "InProcessMCPTransport",
    "InventoryCheckProvider",
    "MCPClient",
    "MCPContentBlock",
    "MCPToolCallResult",
    "MCPToolDefinition",
    "MCPToolProvider",
    "MockMCPServer",
    "ProductCompareProvider",
    "build_knowledge_base",
    "build_mock_mcp_tool_providers",
    "build_mock_tool_providers",
    "catalog_search",
    "get_seed_catalog",
    "inventory_check",
    "product_compare",
    "register_mock_mcp_tool_providers",
    "register_mock_tool_providers",
    "search_product_knowledge",
    "seed_counts",
]
