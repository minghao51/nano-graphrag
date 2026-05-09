from __future__ import annotations

import html
import os

import networkx as nx

from .._utils import logger


def load_nx_graph(file_name) -> nx.MultiGraph | None:
    if os.path.exists(file_name):
        return nx.read_graphml(file_name, node_type=str, force_multigraph=True)
    return None


def write_nx_graph(graph: nx.Graph, file_name):
    logger.info(
        "graph_write",
        nodes=graph.number_of_nodes(),
        edges=graph.number_of_edges(),
    )
    tmp_file = f"{file_name}.tmp"
    nx.write_graphml(graph, tmp_file)
    os.replace(tmp_file, file_name)


def stabilize_graph(graph):
    is_multigraph = graph.is_multigraph()
    if graph.is_directed():
        fixed_graph = nx.MultiDiGraph() if is_multigraph else nx.DiGraph()
    else:
        fixed_graph = nx.MultiGraph() if is_multigraph else nx.Graph()

    sorted_nodes = sorted(graph.nodes(data=True), key=lambda x: x[0])
    fixed_graph.add_nodes_from(sorted_nodes)

    if is_multigraph:
        edges = list(graph.edges(data=True, keys=True))
        if not graph.is_directed():

            def _sort_source_target_mg(edge):
                u, v, key, data = edge
                if u > v:
                    u, v = v, u
                return u, v, key, data

            edges = [_sort_source_target_mg(e) for e in edges]

        def _get_edge_key_mg(edge):
            u, v, key, _ = edge
            return f"{u} -> {v} -> {key}"

        edges = sorted(edges, key=lambda x: _get_edge_key_mg(x))
        for u, v, key, data in edges:
            fixed_graph.add_edge(u, v, key=key, **data)
    else:
        edges = list(graph.edges(data=True))

        if not graph.is_directed():

            def _sort_source_target(edge):
                source, target, edge_data = edge
                if source > target:
                    source, target = target, source
                return source, target, edge_data

            edges = [_sort_source_target(edge) for edge in edges]

        def _get_edge_key(source, target):
            return f"{source} -> {target}"

        edges = sorted(edges, key=lambda x: _get_edge_key(x[0], x[1]))
        fixed_graph.add_edges_from(edges)

    return fixed_graph


def stable_largest_connected_component(graph):
    graph = graph.copy()
    if graph.is_directed():
        lcc_nodes = max(nx.weakly_connected_components(graph), key=len)
    else:
        lcc_nodes = max(nx.connected_components(graph), key=len)
    graph = graph.subgraph(lcc_nodes).copy()
    node_mapping = {node: html.unescape(str(node).strip()) for node in graph.nodes()}
    graph = nx.relabel_nodes(graph, node_mapping)
    return stabilize_graph(graph)


def to_igraph(nx_graph):
    import igraph as ig

    node_list = list(nx_graph.nodes())
    if nx_graph.is_multigraph():
        simple_graph = nx.Graph()
        for u, v, data in nx_graph.edges(data=True):
            if simple_graph.has_edge(u, v):
                simple_graph[u][v]["weight"] = simple_graph[u][v].get("weight", 0) + data.get(
                    "weight", 1
                )
            else:
                simple_graph.add_edge(u, v, **data)
        ig_graph = ig.Graph.from_networkx(simple_graph)
    else:
        ig_graph = ig.Graph.from_networkx(nx_graph)
    return ig_graph, node_list
