import asyncio
import html
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, List, Union, cast

import networkx as nx
import numpy as np

from .._utils import logger
from ..base import (
    BaseGraphStorage,
    SingleCommunitySchema,
)
from ..prompt import GRAPH_FIELD_SEP


@dataclass
class NetworkXStorage(BaseGraphStorage):
    @staticmethod
    def load_nx_graph(file_name) -> nx.Graph:
        if os.path.exists(file_name):
            return nx.read_graphml(file_name)
        return None

    @staticmethod
    def write_nx_graph(graph: nx.Graph, file_name):
        logger.info(
            f"Writing graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )
        nx.write_graphml(graph, file_name)

    @staticmethod
    def stable_largest_connected_component(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Return the largest connected component of the graph, with nodes and edges sorted in a stable way.
        """
        from graspologic.utils import largest_connected_component

        graph = graph.copy()
        graph = cast(nx.Graph, largest_connected_component(graph))
        # Preserve stable hashed node IDs exactly as stored. Uppercasing here breaks
        # incremental entity identities because SHA-256 ids are case-sensitive.
        node_mapping = {node: html.unescape(str(node).strip()) for node in graph.nodes()}  # type: ignore
        graph = nx.relabel_nodes(graph, node_mapping)
        return NetworkXStorage._stabilize_graph(graph)

    @staticmethod
    def _stabilize_graph(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Ensure an undirected graph with the same relationships will always be read the same way.
        """
        fixed_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()

        sorted_nodes = graph.nodes(data=True)
        sorted_nodes = sorted(sorted_nodes, key=lambda x: x[0])

        fixed_graph.add_nodes_from(sorted_nodes)
        edges = list(graph.edges(data=True))

        if not graph.is_directed():

            def _sort_source_target(edge):
                source, target, edge_data = edge
                if source > target:
                    temp = source
                    source = target
                    target = temp
                return source, target, edge_data

            edges = [_sort_source_target(edge) for edge in edges]

        def _get_edge_key(source: Any, target: Any) -> str:
            return f"{source} -> {target}"

        edges = sorted(edges, key=lambda x: _get_edge_key(x[0], x[1]))

        fixed_graph.add_edges_from(edges)
        return fixed_graph

    def __post_init__(self):
        self._graphml_xml_file = os.path.join(
            self.global_config["working_dir"], f"graph_{self.namespace}.graphml"
        )
        preloaded_graph = NetworkXStorage.load_nx_graph(self._graphml_xml_file)
        if preloaded_graph is not None:
            logger.info(
                f"Loaded graph from {self._graphml_xml_file} with {preloaded_graph.number_of_nodes()} nodes, {preloaded_graph.number_of_edges()} edges"
            )
        self._graph = preloaded_graph or nx.Graph()
        self._clustering_algorithms = {
            "leiden": self._leiden_clustering,
        }
        self._node_embed_algorithms = {
            "node2vec": self._node2vec_embed,
        }
        self._last_affected_community_ids = set()
        self._last_clustering_was_incremental = False

    async def index_done_callback(self):
        NetworkXStorage.write_nx_graph(self._graph, self._graphml_xml_file)

    async def has_node(self, node_id: str) -> bool:
        return self._graph.has_node(node_id)

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        return self._graph.has_edge(source_node_id, target_node_id)

    async def get_node(self, node_id: str) -> Union[dict, None]:
        return self._graph.nodes.get(node_id)

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, Union[dict, None]]:
        return await asyncio.gather(*[self.get_node(node_id) for node_id in node_ids])

    async def node_degree(self, node_id: str) -> int:
        # [numberchiffre]: node_id not part of graph returns `DegreeView({})` instead of 0
        return self._graph.degree(node_id) if self._graph.has_node(node_id) else 0

    async def node_degrees_batch(self, node_ids: List[str]) -> List[str]:
        return await asyncio.gather(*[self.node_degree(node_id) for node_id in node_ids])

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        return (self._graph.degree(src_id) if self._graph.has_node(src_id) else 0) + (
            self._graph.degree(tgt_id) if self._graph.has_node(tgt_id) else 0
        )

    async def edge_degrees_batch(self, edge_pairs: list[tuple[str, str]]) -> list[int]:
        return await asyncio.gather(
            *[self.edge_degree(src_id, tgt_id) for src_id, tgt_id in edge_pairs]
        )

    async def get_edge(self, source_node_id: str, target_node_id: str) -> Union[dict, None]:
        return self._graph.edges.get((source_node_id, target_node_id))

    async def get_edges_batch(self, edge_pairs: list[tuple[str, str]]) -> list[Union[dict, None]]:
        return await asyncio.gather(
            *[
                self.get_edge(source_node_id, target_node_id)
                for source_node_id, target_node_id in edge_pairs
            ]
        )

    async def get_node_edges(self, source_node_id: str):
        if self._graph.has_node(source_node_id):
            return list(self._graph.edges(source_node_id))
        return None

    async def get_nodes_edges_batch(self, node_ids: list[str]) -> list[list[tuple[str, str]]]:
        return await asyncio.gather(*[self.get_node_edges(node_id) for node_id in node_ids])

    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        self._graph.add_node(node_id, **node_data)

    async def upsert_nodes_batch(self, nodes_data: list[tuple[str, dict[str, str]]]):
        await asyncio.gather(
            *[self.upsert_node(node_id, node_data) for node_id, node_data in nodes_data]
        )

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        self._graph.add_edge(source_node_id, target_node_id, **edge_data)

    async def upsert_edges_batch(self, edges_data: list[tuple[str, str, dict[str, str]]]):
        await asyncio.gather(
            *[
                self.upsert_edge(source_node_id, target_node_id, edge_data)
                for source_node_id, target_node_id, edge_data in edges_data
            ]
        )

    async def delete_node(self, node_id: str):
        if self._graph.has_node(node_id):
            self._graph.remove_node(node_id)

    async def delete_nodes_batch(self, node_ids: list[str]):
        for node_id in node_ids:
            if self._graph.has_node(node_id):
                self._graph.remove_node(node_id)

    async def delete_edge(self, source_node_id: str, target_node_id: str):
        if self._graph.has_edge(source_node_id, target_node_id):
            self._graph.remove_edge(source_node_id, target_node_id)

    async def delete_edges_batch(self, edge_pairs: list[tuple[str, str]]):
        for source_node_id, target_node_id in edge_pairs:
            if self._graph.has_edge(source_node_id, target_node_id):
                self._graph.remove_edge(source_node_id, target_node_id)

    async def clustering(self, algorithm: str, affected_node_ids: Union[set[str], None] = None):
        if algorithm not in self._clustering_algorithms:
            raise ValueError(f"Clustering algorithm {algorithm} not supported")
        await self._clustering_algorithms[algorithm](affected_node_ids=affected_node_ids)

    async def community_schema(self) -> dict[str, SingleCommunitySchema]:
        results = defaultdict(
            lambda: dict(
                level=None,
                title=None,
                edges=set(),
                nodes=set(),
                chunk_ids=set(),
                occurrence=0.0,
                sub_communities=[],
            )
        )
        max_num_ids = 0
        levels = defaultdict(set)
        for node_id, node_data in self._graph.nodes(data=True):
            if "clusters" not in node_data:
                continue
            clusters = json.loads(node_data["clusters"])
            this_node_edges = self._graph.edges(node_id)

            for cluster in clusters:
                level = cluster["level"]
                cluster_key = str(cluster["cluster"])
                levels[level].add(cluster_key)
                results[cluster_key]["level"] = level
                results[cluster_key]["title"] = f"Cluster {cluster_key}"
                results[cluster_key]["nodes"].add(node_id)
                results[cluster_key]["edges"].update([tuple(sorted(e)) for e in this_node_edges])
                results[cluster_key]["chunk_ids"].update(
                    node_data["source_id"].split(GRAPH_FIELD_SEP)
                )
                max_num_ids = max(max_num_ids, len(results[cluster_key]["chunk_ids"]))

        ordered_levels = sorted(levels.keys())
        for i, curr_level in enumerate(ordered_levels[:-1]):
            next_level = ordered_levels[i + 1]
            this_level_comms = levels[curr_level]
            next_level_comms = levels[next_level]
            # compute the sub-communities by nodes intersection
            for comm in this_level_comms:
                results[comm]["sub_communities"] = [
                    c
                    for c in next_level_comms
                    if results[c]["nodes"].issubset(results[comm]["nodes"])
                ]

        for k, v in results.items():
            v["edges"] = list(v["edges"])
            v["edges"] = [list(e) for e in v["edges"]]
            v["nodes"] = list(v["nodes"])
            v["chunk_ids"] = list(v["chunk_ids"])
            v["occurrence"] = len(v["chunk_ids"]) / max_num_ids
        return dict(results)

    def _cluster_data_to_subgraphs(self, cluster_data: dict[str, list[dict[str, str]]]):
        for node_id, clusters in cluster_data.items():
            self._graph.nodes[node_id]["clusters"] = json.dumps(clusters)

    def _next_incremental_cluster_prefix(self) -> str:
        counter = int(self._graph.graph.get("community_update_counter", 0)) + 1
        self._graph.graph["community_update_counter"] = counter
        return f"inc-{counter}"

    def _extract_existing_clusters(self, node_id: str) -> list[dict[str, str]]:
        node_data = self._graph.nodes.get(node_id, {})
        raw_clusters = node_data.get("clusters")
        if not raw_clusters:
            return []
        return json.loads(raw_clusters)

    def _compute_frontier_nodes(self, affected_node_ids: set[str]) -> set[str]:
        radius = self.global_config["addon_params"].get("community_update_neighbor_radius", 1)
        frontier = {node_id for node_id in affected_node_ids if self._graph.has_node(node_id)}
        current_layer = set(frontier)
        for _ in range(radius):
            next_layer = set()
            for node_id in current_layer:
                next_layer.update(self._graph.neighbors(node_id))
            frontier.update(next_layer)
            current_layer = next_layer
        return frontier

    def _collect_level0_starting_communities(self, frontier_nodes: set[str]) -> dict[str, int]:
        cluster_lookup = {}
        next_cluster_id = 0
        starting = {}
        for node_id in frontier_nodes:
            level0 = next(
                (
                    cluster["cluster"]
                    for cluster in self._extract_existing_clusters(node_id)
                    if cluster["level"] == 0
                ),
                None,
            )
            if level0 is None:
                continue
            if level0 not in cluster_lookup:
                cluster_lookup[level0] = next_cluster_id
                next_cluster_id += 1
            starting[node_id] = cluster_lookup[level0]
        return starting

    def _graph_has_multi_level_clusters(self) -> bool:
        for node_id in self._graph.nodes:
            if any(cluster["level"] > 0 for cluster in self._extract_existing_clusters(node_id)):
                return True
        return False

    def _assign_singleton_clusters(self):
        node_communities = {
            node_id: [{"level": 0, "cluster": f"singleton-{index}"}]
            for index, node_id in enumerate(sorted(self._graph.nodes()))
        }
        self._cluster_data_to_subgraphs(node_communities)
        self._last_affected_community_ids = {
            cluster["cluster"]
            for clusters in node_communities.values()
            for cluster in clusters
        }
        self._last_clustering_was_incremental = False

    async def _leiden_clustering(self, affected_node_ids: Union[set[str], None] = None):
        from graspologic.partition import hierarchical_leiden

        if self._graph.number_of_nodes() == 0:
            self._last_affected_community_ids = set()
            self._last_clustering_was_incremental = False
            return
        if self._graph.number_of_edges() == 0:
            self._assign_singleton_clusters()
            return

        frontier_ratio_limit = self.global_config["addon_params"].get(
            "community_update_max_frontier_ratio", 0.5
        )
        should_try_incremental = bool(affected_node_ids)
        if should_try_incremental:
            if self._graph_has_multi_level_clusters():
                should_try_incremental = False
            frontier_nodes = self._compute_frontier_nodes(affected_node_ids or set())
            if should_try_incremental:
                has_existing_clusters = any(
                    self._graph.nodes[node_id].get("clusters")
                    for node_id in frontier_nodes
                    if self._graph.has_node(node_id)
                )
                if (
                    not frontier_nodes
                    or len(frontier_nodes) < 2
                    or len(frontier_nodes) / max(1, self._graph.number_of_nodes()) > frontier_ratio_limit
                    or not has_existing_clusters
                ):
                    should_try_incremental = False

        if should_try_incremental:
            old_community_ids = {
                str(cluster["cluster"])
                for node_id in frontier_nodes
                for cluster in self._extract_existing_clusters(node_id)
            }
            frontier_graph = NetworkXStorage._stabilize_graph(self._graph.subgraph(frontier_nodes).copy())
            starting_communities = self._collect_level0_starting_communities(frontier_nodes)
            community_mapping = hierarchical_leiden(
                frontier_graph,
                max_cluster_size=self.global_config["max_graph_cluster_size"],
                random_seed=self.global_config["graph_cluster_seed"],
                starting_communities=starting_communities or None,
                extra_forced_iterations=1,
            )
            cluster_prefix = self._next_incremental_cluster_prefix()
            new_level0_ids = {}
            for partition in community_mapping:
                if partition.level != 0:
                    continue
                node_id = partition.node
                level0_cluster = f"{cluster_prefix}-l0-{partition.cluster}"
                self._graph.nodes[node_id]["clusters"] = json.dumps(
                    [{"level": 0, "cluster": level0_cluster}]
                )
                new_level0_ids[node_id] = level0_cluster
            self._last_affected_community_ids = old_community_ids.union(new_level0_ids.values())
            self._last_clustering_was_incremental = True
            logger.info(
                f"Incremental Leiden updated {len(frontier_nodes)} frontier nodes across {len(self._last_affected_community_ids)} communities"
            )
            return

        graph = NetworkXStorage.stable_largest_connected_component(self._graph)
        community_mapping = hierarchical_leiden(
            graph,
            max_cluster_size=self.global_config["max_graph_cluster_size"],
            random_seed=self.global_config["graph_cluster_seed"],
        )

        node_communities: dict[str, list[dict[str, str]]] = defaultdict(list)
        __levels = defaultdict(set)
        for partition in community_mapping:
            level_key = partition.level
            cluster_id = partition.cluster
            node_communities[partition.node].append({"level": level_key, "cluster": cluster_id})
            __levels[level_key].add(cluster_id)
        node_communities = dict(node_communities)
        __levels = {k: len(v) for k, v in __levels.items()}
        logger.info(f"Each level has communities: {dict(__levels)}")
        self._cluster_data_to_subgraphs(node_communities)
        self._last_affected_community_ids = {
            str(cluster["cluster"])
            for clusters in node_communities.values()
            for cluster in clusters
        }
        self._last_clustering_was_incremental = False

    async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        if algorithm not in self._node_embed_algorithms:
            raise ValueError(f"Node embedding algorithm {algorithm} not supported")
        return await self._node_embed_algorithms[algorithm]()

    async def _node2vec_embed(self):
        from graspologic import embed

        embeddings, nodes = embed.node2vec_embed(
            self._graph,
            **self.global_config["node2vec_params"],
        )

        nodes_ids = [self._graph.nodes[node_id]["id"] for node_id in nodes]
        return embeddings, nodes_ids
