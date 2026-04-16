"""Advanced retrieval and optimization techniques for nano-graphrag benchmarks.

This module contains experimental techniques for improving multi-hop RAG performance:
- Cross-encoder reranking
- Adaptive mode routing
- Edge confidence weighting
- DSPy prompt tuning
- RAPTOR hierarchical summaries
"""

from .adaptive_router import AdaptiveRouter
from .edge_confidence import (
    create_edge_confidence_hook,
    get_edge_weight,
    score_edges_by_confidence,
)
from .raptor import RaptorRetriever
from .reranker import CrossEncoderReranker

__all__ = [
    "AdaptiveRouter",
    "CrossEncoderReranker",
    "RaptorRetriever",
    "create_edge_confidence_hook",
    "get_edge_weight",
    "score_edges_by_confidence",
]
