"""In-process mock MCP server and RAG-style product knowledge search."""

from .knowledge_base import KnowledgeSnippet, build_knowledge_base, search_product_knowledge
from .server import MockMCPServer, RagProductKnowledgeRequest

__all__ = [
    "KnowledgeSnippet",
    "MockMCPServer",
    "RagProductKnowledgeRequest",
    "build_knowledge_base",
    "search_product_knowledge",
]
