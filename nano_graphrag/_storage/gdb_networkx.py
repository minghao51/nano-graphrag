import asyncio
import html
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, List, Union

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
        # Write to temp file first, then atomic rename
        tmp_file = f"{file_name}.tmp"
        nx.write_graphml(graph, tmp_file)
        os.replace(tmp_file, file_name)  # Atomic on POSIX

    @staticmethod
    def stable_largest_connected_component(graph: nx.Graph) -> nx.Graph:
        """Return the largest connected component using pure NetworkX."""
        graph = graph.copy()
        # Use weakly_connected_components for directed graphs, connected_components for undirected
        if graph.is_directed():
            lcc_nodes = max(nx.weakly_connected_components(graph), key=len)
        else:
            lcc_nodes = max(nx.connected_components(graph), key=len)
        graph = graph.subgraph(lcc_nodes).copy()
        # Preserve stable hashed node IDs
        node_mapping = {node: html.unescape(str(node).strip()) for node in graph.nodes()}
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
            cluster["cluster"] for clusters in node_communities.values() for cluster in clusters
        }
        self._last_clustering_was_incremental = False

    @staticmethod
    def _to_igraph(nx_graph: nx.Graph):
        """Convert a NetworkX graph to igraph, preserving node names."""
        import igraph as ig

        node_list = list(nx_graph.nodes())
        ig_graph = ig.Graph.from_networkx(nx_graph)
        return ig_graph, node_list

    def _leiden_seed(self) -> int:
        """Get graph cluster seed clamped to signed 32-bit range for leidenalg C compatibility."""
        seed = self.global_config.get("graph_cluster_seed", 0xDEADBEEF)
        # leidenalg C code uses signed 32-bit integers; clamp to valid range
        return seed % (2**31 - 1)

    async def _leiden_clustering(self, affected_node_ids: Union[set[str], None] = None):
        import igraph as ig
        import leidenalg as la

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
            frontier_nodes = self._compute_frontier_nodes(affected_node_ids or set())
            has_existing_clusters = any(
                self._graph.nodes[node_id].get("clusters")
                for node_id in frontier_nodes
                if self._graph.has_node(node_id)
            )
            if (
                not frontier_nodes
                or len(frontier_nodes) < 2
                or len(frontier_nodes) / max(1, self._graph.number_of_nodes())
                > frontier_ratio_limit
                or not has_existing_clusters
            ):
                should_try_incremental = False

        if should_try_incremental:
            old_community_ids = {
                str(cluster["cluster"])
                for node_id in frontier_nodes
                for cluster in self._extract_existing_clusters(node_id)
            }
            frontier_graph = NetworkXStorage._stabilize_graph(
                self._graph.subgraph(frontier_nodes).copy()
            )
            starting_communities = self._collect_level0_starting_communities(frontier_nodes)

            ig_graph, _ = NetworkXStorage._to_igraph(frontier_graph)
            next_membership_id = max(starting_communities.values(), default=-1) + 1
            initial_membership = []
            for vertex in ig_graph.vs:
                node_id = vertex["_nx_name"]
                if node_id in starting_communities:
                    initial_membership.append(starting_communities[node_id])
                else:
                    initial_membership.append(next_membership_id)
                    next_membership_id += 1

            # Run Leiden with starting communities
            partition = la.find_partition(
                ig_graph,
                la.ModularityVertexPartition,
                seed=self._leiden_seed(),
                n_iterations=8,
                initial_membership=initial_membership,
            )

            cluster_prefix = self._next_incremental_cluster_prefix()
            new_level0_ids = {}
            for cluster_idx, cluster_nodes in enumerate(partition):
                level0_cluster = f"{cluster_prefix}-l0-{cluster_idx}"
                for node_idx in cluster_nodes:
                    nx_node_id = ig_graph.vs[node_idx]["_nx_name"]
                    self._graph.nodes[nx_node_id]["clusters"] = json.dumps(
                        [{"level": 0, "cluster": level0_cluster}]
                    )
                    new_level0_ids[nx_node_id] = level0_cluster
            self._last_affected_community_ids = old_community_ids.union(new_level0_ids.values())
            self._last_clustering_was_incremental = True
            logger.info(
                f"Incremental Leiden updated {len(frontier_nodes)} frontier nodes across {len(self._last_affected_community_ids)} communities"
            )
            return

        # Full clustering using igraph + leidenalg
        graph_to_cluster = NetworkXStorage.stable_largest_connected_component(self._graph)

        ig_graph, node_list = NetworkXStorage._to_igraph(graph_to_cluster)

        # Run Leiden at multiple resolutions for hierarchical effect
        # Higher resolution = more communities (finer granularity)
        # Level 0 should be finest (most communities), higher levels coarser
        resolutions = self.global_config.get("leiden_resolutions", [2.0, 1.0, 0.5])

        node_communities = defaultdict(list)
        __levels = defaultdict(set)

        for level, resolution in enumerate(resolutions):
            # Use RBConfigurationVertexPartition which supports resolution_parameter
            partition = la.find_partition(
                ig_graph,
                la.RBConfigurationVertexPartition,
                resolution_parameter=resolution,
                seed=self._leiden_seed() + level,
                n_iterations=8,
            )

            for cluster_idx, cluster_nodes in enumerate(partition):
                cluster_id = f"l{level}_c{cluster_idx}"
                __levels[level].add(cluster_id)
                for node_idx in cluster_nodes:
                    nx_node_id = ig_graph.vs[node_idx]["_nx_name"]
                    node_communities[nx_node_id].append({"level": level, "cluster": cluster_id})

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
        from node2vec import Node2Vec

        # Separate Node2Vec init params from Word2Vec fit params
        all_params = self.global_config["node2vec_params"].copy()

        # Node2Vec init params
        n2v_init_params = {}
        if "dimensions" in all_params:
            n2v_init_params["dimensions"] = all_params.pop("dimensions")
        if "num_walks" in all_params:
            n2v_init_params["num_walks"] = all_params.pop("num_walks")
        if "walk_length" in all_params:
            n2v_init_params["walk_length"] = all_params.pop("walk_length")
        if "random_seed" in all_params:
            n2v_init_params["seed"] = all_params.pop("random_seed")
        # p, q, weight_key, workers, etc. pass through
        for key in ("p", "q", "weight_key", "workers", "sampling_strategy", "quiet", "temp_folder"):
            if key in all_params:
                n2v_init_params[key] = all_params.pop(key)

        # Word2Vec fit params (remaining keys)
        w2v_fit_params = all_params  # whatever's left (window_size, iterations, etc.)
        if "window_size" in w2v_fit_params:
            w2v_fit_params["window"] = w2v_fit_params.pop("window_size")
        if "iterations" in w2v_fit_params:
            w2v_fit_params["epochs"] = w2v_fit_params.pop("iterations")

        node2vec = Node2Vec(self._graph, **n2v_init_params)
        model = node2vec.fit(**w2v_fit_params)
        embeddings = model.wv.vectors
        nodes = model.wv.index_to_key

        nodes_ids = [self._graph.nodes[node_id]["id"] for node_id in nodes]
        return embeddings, nodes_ids
