import asyncio
import os
from dataclasses import dataclass
from typing import List, Union

import networkx as nx
import numpy as np

from ..base import BaseGraphStorage
from .gdb_networkx_clustering import (
    LeidenClusteringBackend,
    build_community_schema,
)
from .gdb_networkx_utils import (
    load_nx_graph,
    stable_largest_connected_component,
    write_nx_graph,
)


@dataclass
class NetworkXStorage(BaseGraphStorage):
    load_nx_graph = staticmethod(load_nx_graph)
    write_nx_graph = staticmethod(write_nx_graph)
    stable_largest_connected_component = staticmethod(stable_largest_connected_component)

    def __post_init__(self):
        self._graphml_xml_file = os.path.join(
            self.global_config["working_dir"], f"graph_{self.namespace}.graphml"
        )
        preloaded_graph = load_nx_graph(self._graphml_xml_file)
        if preloaded_graph is not None:
            from .._utils import logger

            logger.info(
                f"Loaded graph from {self._graphml_xml_file} with {preloaded_graph.number_of_nodes()} nodes, {preloaded_graph.number_of_edges()} edges"
            )
        self._graph = preloaded_graph or nx.Graph()
        self._clustering_algorithms = {
            "leiden": LeidenClusteringBackend(),
        }
        self._node_embed_algorithms = {
            "node2vec": self._node2vec_embed,
        }
        self._last_affected_community_ids = set()
        self._last_clustering_was_incremental = False

    async def index_done_callback(self):
        write_nx_graph(self._graph, self._graphml_xml_file)

    async def has_node(self, node_id: str) -> bool:
        return self._graph.has_node(node_id)

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        return self._graph.has_edge(source_node_id, target_node_id)

    async def get_node(self, node_id: str) -> Union[dict, None]:
        return self._graph.nodes.get(node_id)

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, Union[dict, None]]:
        return await asyncio.gather(*[self.get_node(node_id) for node_id in node_ids])

    async def node_degree(self, node_id: str) -> int:
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
        await self._clustering_algorithms[algorithm].cluster(
            self, affected_node_ids=affected_node_ids
        )

    async def community_schema(self):
        return build_community_schema(self)

    async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        if algorithm not in self._node_embed_algorithms:
            raise ValueError(f"Node embedding algorithm {algorithm} not supported")
        return await self._node_embed_algorithms[algorithm]()

    async def _node2vec_embed(self):
        from node2vec import Node2Vec

        all_params = self.global_config["node2vec_params"].copy()

        n2v_init_params = {}
        if "dimensions" in all_params:
            n2v_init_params["dimensions"] = all_params.pop("dimensions")
        if "num_walks" in all_params:
            n2v_init_params["num_walks"] = all_params.pop("num_walks")
        if "walk_length" in all_params:
            n2v_init_params["walk_length"] = all_params.pop("walk_length")
        if "random_seed" in all_params:
            n2v_init_params["seed"] = all_params.pop("random_seed")
        for key in ("p", "q", "weight_key", "workers", "sampling_strategy", "quiet", "temp_folder"):
            if key in all_params:
                n2v_init_params[key] = all_params.pop(key)

        w2v_fit_params = all_params
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
