"""HippoRAG-style Personalized PageRank retriever for multi-hop discovery.

Uses Personalized PageRank (PPR) over the knowledge graph to discover
multi-hop bridging paths in a single graph operation, without iterative
LLM calls.

Based on: HippoRAG (Gutierrez et al., 2024)
https://arxiv.org/abs/2405.14831

Example:
    retriever = HippoRAGRetriever(alpha=0.85, top_k_result=20)
    context = await retriever("Who is connected to both X and Y?", graph_rag)
"""

from __future__ import annotations

from typing import Any, List

import networkx as nx

from nano_graphrag import GraphRAG
from nano_graphrag.base import QueryParam


class HippoRAGRetriever:
    """Personalized PageRank retrieval over the nano-graphrag knowledge graph.

    Uses PPR to discover multi-hop bridging entities in a single graph operation.
    Seed entities are identified via query embedding similarity, then PPR is run
    to find related entities through graph traversal.

    Args:
        alpha: PPR damping factor (0.0-1.0). Higher = more global exploration.
            Defaults to 0.85 (standard PageRank value).
        top_k_seed: Number of seed entities from query embedding.
            Defaults to 5.
        top_k_result: Number of top-scoring nodes to return.
            Defaults to 20.

    Attributes:
        alpha: PPR damping factor.
        top_k_seed: Number of seed entities.
        top_k_result: Number of result nodes.
    """

    def __init__(
        self,
        alpha: float = 0.85,
        top_k_seed: int = 5,
        top_k_result: int = 20,
    ) -> None:
        self._alpha = alpha
        self._top_k_seed = top_k_seed
        self._top_k_result = top_k_result

    async def __call__(
        self,
        query: str,
        graph_rag: GraphRAG,
        param: QueryParam,
        **kwargs,
    ) -> str:
        """Retrieve context using PPR-based multi-hop discovery.

        Args:
            query: User question.
            graph_rag: GraphRAG instance.
            param: Query parameters (unused, kept for protocol compatibility).
            **kwargs: Additional parameters.

        Returns:
            Retrieved context as string.
        """
        # Get the NetworkX graph from storage
        graph = graph_rag.chunk_entity_relation_graph._graph

        if graph.number_of_nodes() == 0:
            # Empty graph, fallback to local mode
            return await graph_rag.aquery(query, param=QueryParam(mode="local"))

        # Find seed entities via embedding similarity
        seed_entities = await self._find_seed_entities(query, graph_rag)

        if not seed_entities:
            # No seeds found, use high-degree nodes as fallback
            seed_entities = [
                node
                for node, _ in sorted(graph.degree(), key=lambda x: x[1], reverse=True)[
                    : self._top_k_seed
                ]
            ]

        # Run PPR from seed entities
        scores = nx.pagerank(
            graph,
            alpha=self._alpha,
            personalization={
                node: 1.0 / len(seed_entities) for node in seed_entities if node in graph
            },
            weight="weight",
            dangling=dict.fromkeys(
                [node for node in seed_entities if node in graph], 1.0 / max(len(seed_entities), 1)
            ),
        )

        # Get top-scoring nodes
        top_nodes = sorted(scores, key=scores.get, reverse=True)[: self._top_k_result]

        # Convert nodes to context
        return await self._nodes_to_context(top_nodes, graph_rag)

    async def _find_seed_entities(self, query: str, graph_rag: GraphRAG) -> List[str]:
        """Find seed entities via query embedding similarity.

        Args:
            query: User question.
            graph_rag: GraphRAG instance.

        Returns:
            List of seed entity names.
        """
        if graph_rag.entities_vdb is None:
            return []

        # Query the entity vector store for closest entities
        try:
            results = await graph_rag.entities_vdb.query(query, top_k=self._top_k_seed)
            return [
                r.get("entity_name", r.get("id", ""))
                for r in results
                if r.get("entity_name") or r.get("id")
            ]
        except Exception:
            # If vector query fails, return empty list
            return []

    async def _nodes_to_context(self, nodes: List[str], graph_rag: GraphRAG) -> str:
        """Convert graph nodes to context text.

        Args:
            nodes: List of node names.
            graph_rag: GraphRAG instance.

        Returns:
            Context text string.
        """
        # Get node data and associated chunks
        context_parts = []

        for node in nodes:
            node_data = await graph_rag.chunk_entity_relation_graph.get_node(node)
            if node_data:
                # Add node name/description
                description = node_data.get("description", node)
                source_id = node_data.get("source_id", "")
                if source_id:
                    context_parts.append(f"- {node} (from {source_id}): {description}")
                else:
                    context_parts.append(f"- {node}: {description}")

                # Get associated chunk if available
                chunk_id = node_data.get("chunk_id")
                if chunk_id:
                    chunk_data = await graph_rag.text_chunks.get_by_id(chunk_id)
                    if chunk_data:
                        content = (
                            chunk_data.get("content", "")
                            if isinstance(chunk_data, dict)
                            else str(chunk_data)
                        )
                        context_parts.append(f"  {content[:300]}")

        return "\n".join(context_parts)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "HippoRAGRetriever":
        """Create retriever from configuration dict.

        Args:
            config: Configuration dict with keys:
                - alpha (float): PPR damping factor
                - top_k_seed (int): Number of seed entities
                - top_k_result (int): Number of result nodes

        Returns:
            Configured HippoRAGRetriever instance.
        """
        return cls(
            alpha=config.get("alpha", 0.85),
            top_k_seed=config.get("top_k_seed", 5),
            top_k_result=config.get("top_k_result", 20),
        )
