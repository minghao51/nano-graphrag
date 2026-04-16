"""Edge confidence weighting for knowledge graph edges.

Provides a post-insert hook that normalizes and stores edge confidence
weights based on extraction confidence and corpus frequency signals.

This improves multi-hop retrieval precision by preferring high-confidence
edges during graph traversal.

Example:
    await score_edges_by_confidence(graph_storage)
"""

from __future__ import annotations

from typing import Any, Optional


async def score_edges_by_confidence(
    graph_storage: Any,
    confidence_weight: float = 0.7,
    frequency_weight: float = 0.3,
    max_frequency_cap: int = 10,
) -> None:
    """Post-process step: normalize and store edge confidence weights.

    Combines two signals:
    1. Extraction confidence: LLM-provided confidence score (0.0-1.0)
    2. Corpus frequency: How often the edge appears (normalized)

    Args:
        graph_storage: Graph storage instance (BaseGraphStorage).
        confidence_weight: Weight for extraction confidence (default: 0.7).
        frequency_weight: Weight for corpus frequency (default: 0.3).
        max_frequency_cap: Maximum frequency for normalization (default: 10).
    """
    # Get the underlying NetworkX graph
    if not hasattr(graph_storage, "_graph"):
        return

    graph = graph_storage._graph

    # Process each edge
    for src, dst, data in graph.edges(data=True):
        # Get raw confidence (default to 1.0 if not provided)
        raw_conf = float(data.get("confidence", 1.0))

        # Get occurrence count (default to 1 if not provided)
        freq = int(data.get("occurrence_count", 1))

        # Normalize frequency by max cap
        normalized_freq = min(freq / max_frequency_cap, 1.0)

        # Combine signals
        weight = confidence_weight * raw_conf + frequency_weight * normalized_freq

        # Update edge data with weight
        await graph_storage.upsert_edge(src, dst, {**data, "weight": str(weight)})


async def get_edge_weight(
    graph_storage: Any, source_node_id: str, target_node_id: str
) -> Optional[float]:
    """Get the weighted confidence score for an edge.

    Args:
        graph_storage: Graph storage instance.
        source_node_id: Source node ID.
        target_node_id: Target node ID.

    Returns:
        Edge weight as float, or None if edge doesn't exist or has no weight.
    """
    edge_data = await graph_storage.get_edge(source_node_id, target_node_id)
    if edge_data is None:
        return None

    weight_str = edge_data.get("weight")
    if weight_str is None:
        return None

    try:
        return float(weight_str)
    except (ValueError, TypeError):
        return None


def create_edge_confidence_hook(
    confidence_weight: float = 0.7,
    frequency_weight: float = 0.3,
    max_frequency_cap: int = 10,
):
    """Create a post-insert hook for edge confidence scoring.

    Args:
        confidence_weight: Weight for extraction confidence.
        frequency_weight: Weight for corpus frequency.
        max_frequency_cap: Maximum frequency for normalization.

    Returns:
        Async function that can be used as a post-insert hook.

    Example:
        hook = create_edge_confidence_hook()
        await hook(graph_storage)
    """

    async def hook(graph_storage: Any) -> None:
        await score_edges_by_confidence(
            graph_storage,
            confidence_weight=confidence_weight,
            frequency_weight=frequency_weight,
            max_frequency_cap=max_frequency_cap,
        )

    return hook
