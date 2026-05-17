from __future__ import annotations

import asyncio
import math

import networkx as nx

from .._utils import logger


def _compute_structural_features_sync(
    graph: nx.MultiGraph,
) -> dict[str, dict[str, float]]:
    if graph.number_of_nodes() == 0:
        return {}
    features: dict[str, dict[str, float]] = {}
    try:
        degree = nx.degree_centrality(graph)
    except Exception:
        degree = {}
    try:
        pagerank = nx.pagerank(graph, alpha=0.85)
    except Exception:
        pagerank = {}
    n = graph.number_of_nodes()
    k = max(1, int(math.sqrt(n)))
    try:
        betweenness = nx.betweenness_centrality(graph, k=k, normalized=True)
    except Exception:
        betweenness = {}
    try:
        clustering = nx.clustering(graph)
    except Exception:
        clustering = {}
    all_vals = {
        "degree_centrality": degree,
        "pagerank": pagerank,
        "betweenness_centrality": betweenness,
        "clustering": clustering,
    }
    for node in graph.nodes():
        features[node] = {}
        for key, vals in all_vals.items():
            features[node][key] = vals.get(node, 0.0)
    for key in all_vals:
        vals = list(all_vals[key].values())
        if not vals:
            continue
        mn, mx = min(vals), max(vals)
        rng = mx - mn
        if rng > 1e-10:
            for node in features:
                features[node][key] = (features[node][key] - mn) / rng
        else:
            for node in features:
                features[node][key] = 0.5 if rng == 0 else features[node][key]
    return features


async def compute_structural_features(
    graph: nx.MultiGraph,
) -> dict[str, dict[str, float]]:
    features = await asyncio.to_thread(_compute_structural_features_sync, graph)
    logger.debug("structural_features_computed", nodes=len(features))
    return features


async def ensure_structural_features(
    knowledge_graph_inst,
    document_index=None,
) -> dict[str, dict[str, float]] | None:
    if not hasattr(knowledge_graph_inst, "_graph"):
        return None
    graph = knowledge_graph_inst._graph
    if graph is None:
        return None
    sf = getattr(graph, "_structural_features", None)
    if sf is not None:
        return sf
    if document_index is not None:
        try:
            payload = await document_index.get_by_id("structural_features_payload")
            if isinstance(payload, dict) and payload:
                graph._structural_features = payload
                logger.debug("structural_features_loaded_from_index", nodes=len(payload))
                return payload
        except Exception:
            pass
    if graph.number_of_nodes() == 0:
        return None
    sf = await compute_structural_features(graph)
    graph._structural_features = sf
    return sf


def compute_composite_rank(
    structural_features: dict[str, float] | None,
    weights: list[float],
) -> float:
    if not structural_features:
        return 0.0
    w_deg, w_pr, w_bet, w_clust = weights
    return (
        w_deg * structural_features.get("degree_centrality", 0.0)
        + w_pr * structural_features.get("pagerank", 0.0)
        + w_bet * structural_features.get("betweenness_centrality", 0.0)
        + w_clust * structural_features.get("clustering", 0.0)
    )
