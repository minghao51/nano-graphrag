"""Base retriever protocol and result types."""

from dataclasses import dataclass, field
from typing import Protocol, Any
from nano_graphrag import GraphRAG
from nano_graphrag.base import QueryParam


@dataclass
class RetrieverResult:
    """Result from a retrieval operation."""
    context: str  # Retrieved context text
    entities: list[str] = field(default_factory=list)  # Entities discovered
    hops: int = 0  # Number of hops taken
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HopState:
    """State for a single hop in multi-hop retrieval."""
    sub_question: str  # Question for this hop
    retrieved_entities: list[str] = field(default_factory=list)
    context_chunks: list[str] = field(default_factory=list)
    answer_fragment: str = ""  # Partial answer from this hop


class Retriever(Protocol):
    """Protocol for retrieval strategies."""

    async def __call__(
        self,
        query: str,
        graph_rag: GraphRAG,
        param: QueryParam,
        **kwargs,
    ) -> str:
        """Retrieve context for a query.

        Args:
            query: User question
            graph_rag: GraphRAG instance
            param: Query parameters
            **kwargs: Additional parameters

        Returns:
            Retrieved context as string
        """
        ...
