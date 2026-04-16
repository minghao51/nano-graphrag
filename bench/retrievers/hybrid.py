"""Hybrid retriever that fuses results from multiple retrieval strategies.

Combines results from multiple retrievers using configurable fusion strategies:
- weighted_avg: Weighted average of scores
- reciprocal_rank: Reciprocal rank fusion (RRF)
- rrf: Same as reciprocal_rank

Example:
    retriever = HybridRetriever(
        retrievers=["local", "hipporag"],
        weights=[0.6, 0.4],
        fusion="weighted_avg"
    )
    context = await retriever("What is the capital of France?", graph_rag)
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from nano_graphrag import GraphRAG
from nano_graphrag.base import QueryParam


class HybridRetriever:
    """Fuses results from multiple retrieval strategies.

    Runs multiple retrievers in parallel and combines their results using
    the specified fusion strategy.

    Args:
        retrievers: List of retriever names to use.
        weights: Weights for each retriever (used in weighted_avg fusion).
        fusion: Fusion strategy: "weighted_avg", "reciprocal_rank", or "rrf".
        top_k: Number of top results to return after fusion.

    Attributes:
        retrievers: List of retriever names.
        weights: Weights for each retriever.
        fusion: Fusion strategy.
        top_k: Number of results to return.
    """

    FUSION_STRATEGIES = ["weighted_avg", "reciprocal_rank", "rrf"]

    def __init__(
        self,
        retrievers: List[str],
        weights: Optional[List[float]] = None,
        fusion: str = "weighted_avg",
        top_k: int = 20,
    ) -> None:
        if fusion not in self.FUSION_STRATEGIES:
            raise ValueError(
                f"Unknown fusion strategy '{fusion}'. Available: {self.FUSION_STRATEGIES}"
            )

        if not retrievers:
            raise ValueError("At least one retriever must be specified")

        if weights is None:
            weights = [1.0 / len(retrievers)] * len(retrievers)

        if len(weights) != len(retrievers):
            raise ValueError(
                f"Number of weights ({len(weights)}) must match "
                f"number of retrievers ({len(retrievers)})"
            )

        self._retrievers = retrievers
        self._weights = weights
        self._fusion = fusion
        self._top_k = top_k

    async def __call__(
        self,
        query: str,
        graph_rag: GraphRAG,
        param: QueryParam,
        **kwargs,
    ) -> str:
        """Retrieve and fuse results from multiple retrievers.

        Args:
            query: User question.
            graph_rag: GraphRAG instance.
            param: Query parameters.
            **kwargs: Additional parameters.

        Returns:
            Fused context as string.
        """
        from bench.registry import resolve

        # Create retriever instances
        retriever_instances = []
        for name in self._retrievers:
            retriever_class = resolve("retriever", name)
            retriever_instances.append(retriever_class())

        # Run retrievers in parallel
        results = await asyncio.gather(
            *[retriever(query, graph_rag, param, **kwargs) for retriever in retriever_instances],
            return_exceptions=True,
        )

        # Filter out exceptions and empty results
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                continue
            if result and isinstance(result, str):
                valid_results.append((self._weights[i], result))

        if not valid_results:
            # All retrievers failed, return empty string
            return ""

        # Fuse results
        if self._fusion == "weighted_avg":
            return self._weighted_avg_fusion(valid_results)
        else:  # reciprocal_rank or rrf
            return self._reciprocal_rank_fusion(valid_results)

    def _weighted_avg_fusion(self, weighted_results: List[tuple[float, str]]) -> str:
        """Combine results using weighted average of passage scores.

        Args:
            weighted_results: List of (weight, result) tuples.

        Returns:
            Combined context string.
        """
        # Split each result into passages
        all_passages: List[tuple[float, str]] = []

        for weight, result in weighted_results:
            # Split by double newlines to get passages
            passages = [p.strip() for p in result.split("\n\n") if p.strip()]
            for passage in passages:
                all_passages.append((weight, passage))

        # Sort by weight (descending) and return top_k
        all_passages.sort(key=lambda x: x[0], reverse=True)
        top_passages = all_passages[: self._top_k]

        return "\n\n".join(p[1] for p in top_passages)

    def _reciprocal_rank_fusion(self, weighted_results: List[tuple[float, str]]) -> str:
        """Combine results using reciprocal rank fusion (RRF).

        RRF assigns scores based on rank position: 1 / (k + rank)
        where k is a constant (default 60).

        Args:
            weighted_results: List of (weight, result) tuples.

        Returns:
            Combined context string.
        """
        k = 60  # RRF constant

        # Split results into passages and calculate RRF scores
        passage_scores: Dict[str, float] = {}

        for weight, result in weighted_results:
            passages = [p.strip() for p in result.split("\n\n") if p.strip()]
            for rank, passage in enumerate(passages):
                # RRF score weighted by retriever weight
                rrf_score = weight / (k + rank + 1)
                if passage in passage_scores:
                    passage_scores[passage] += rrf_score
                else:
                    passage_scores[passage] = rrf_score

        # Sort by score and return top_k
        sorted_passages = sorted(passage_scores.items(), key=lambda x: x[1], reverse=True)
        top_passages = sorted_passages[: self._top_k]

        return "\n\n".join(p[0] for p in top_passages)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "HybridRetriever":
        """Create retriever from configuration dict.

        Args:
            config: Configuration dict with keys:
                - retrievers (list[str]): List of retriever names
                - weights (list[float] | None): Weights for each retriever
                - fusion (str): Fusion strategy
                - top_k (int): Number of results to return

        Returns:
            Configured HybridRetriever instance.
        """
        return cls(
            retrievers=config.get("retrievers", ["local", "global"]),
            weights=config.get("weights"),
            fusion=config.get("fusion", "weighted_avg"),
            top_k=config.get("top_k", 20),
        )
