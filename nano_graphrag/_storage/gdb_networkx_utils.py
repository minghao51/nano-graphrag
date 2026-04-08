import html
import os
from typing import Any

import networkx as nx

from .._utils import logger


def load_nx_graph(file_name) -> nx.Graph:
    if os.path.exists(file_name):
        return nx.read_graphml(file_name)
    return None


def write_nx_graph(graph: nx.Graph, file_name):
    logger.info(
        f"Writing graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
    )
    tmp_file = f"{file_name}.tmp"
    nx.write_graphml(graph, tmp_file)
    os.replace(tmp_file, file_name)


def stabilize_graph(graph: nx.Graph) -> nx.Graph:
    fixed_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()

    sorted_nodes = sorted(graph.nodes(data=True), key=lambda x: x[0])
    fixed_graph.add_nodes_from(sorted_nodes)
    edges = list(graph.edges(data=True))

    if not graph.is_directed():

        def _sort_source_target(edge):
            source, target, edge_data = edge
            if source > target:
                source, target = target, source
            return source, target, edge_data

        edges = [_sort_source_target(edge) for edge in edges]

    def _get_edge_key(source: Any, target: Any) -> str:
        return f"{source} -> {target}"

    edges = sorted(edges, key=lambda x: _get_edge_key(x[0], x[1]))
    fixed_graph.add_edges_from(edges)
    return fixed_graph


def stable_largest_connected_component(graph: nx.Graph) -> nx.Graph:
    graph = graph.copy()
    if graph.is_directed():
        lcc_nodes = max(nx.weakly_connected_components(graph), key=len)
    else:
        lcc_nodes = max(nx.connected_components(graph), key=len)
    graph = graph.subgraph(lcc_nodes).copy()
    node_mapping = {node: html.unescape(str(node).strip()) for node in graph.nodes()}
    graph = nx.relabel_nodes(graph, node_mapping)
    return stabilize_graph(graph)


def to_igraph(nx_graph: nx.Graph):
    import igraph as ig

    node_list = list(nx_graph.nodes())
    ig_graph = ig.Graph.from_networkx(nx_graph)
    return ig_graph, node_list
