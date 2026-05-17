from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np

from .._utils import AsyncRWLock, logger
from ..base import BaseGraphStorage
from ..prompt import GRAPH_FIELD_SEP
from .gdb_networkx_clustering import (
    LeidenClusteringBackend,
    LouvainClusteringBackend,
    build_community_schema,
)
from .gdb_networkx_utils import (
    load_nx_graph,
    stable_largest_connected_component,
    write_nx_graph,
)


def _merge_multi_edge_data(edges) -> dict[str, Any]:
    combined: dict[str, Any] = {}
    all_descriptions = []
    total_weight = 0.0
    all_source_ids = []
    temporal_context = None
    valid_from = None
    valid_to = None
    for _key, data in edges.items():
        if not isinstance(data, dict):
            continue
        if "description" in data:
            all_descriptions.append(data["description"])
        total_weight += data.get("weight", 0.0)
        if "source_id" in data:
            all_source_ids.append(data["source_id"])
        if not temporal_context and data.get("temporal_context"):
            temporal_context = data["temporal_context"]
        if not valid_from and data.get("valid_from"):
            valid_from = data["valid_from"]
        if not valid_to and data.get("valid_to"):
            valid_to = data["valid_to"]
    combined["description"] = GRAPH_FIELD_SEP.join(sorted({d for d in all_descriptions if d}))
    combined["weight"] = total_weight
    combined["source_id"] = GRAPH_FIELD_SEP.join(sorted({s for s in all_source_ids if s}))
    combined["temporal_context"] = temporal_context
    combined["valid_from"] = valid_from
    combined["valid_to"] = valid_to
    if all_source_ids:
        combined["order"] = 1
    return combined


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
            logger.info(
                "graph_loaded",
                path=self._graphml_xml_file,
                nodes=preloaded_graph.number_of_nodes(),
                edges=preloaded_graph.number_of_edges(),
            )
        self._graph = preloaded_graph or nx.MultiGraph()
        self._clustering_algorithms = {
            "leiden": LeidenClusteringBackend(),
            "louvain": LouvainClusteringBackend(),
        }
        self._node_embed_algorithms = {
            "node2vec": self._node2vec_embed,
        }
        self._last_affected_community_ids = set()
        self._last_clustering_was_incremental = False
        self._graph_lock = AsyncRWLock()
        self._community_schema_cache = None

    async def index_done_callback(self):
        async with self._graph_lock.write_lock():
            write_nx_graph(self._graph, self._graphml_xml_file)

    async def _snapshot_graph(self) -> str:
        """Create a snapshot of the current graph state.

        Returns:
            Path to the snapshot file.
        """
        import time

        from .._utils import logger

        async with self._graph_lock.write_lock():
            snapshot_dir = os.path.join(self.global_config["working_dir"], "snapshots")
            os.makedirs(snapshot_dir, exist_ok=True)
            snapshot_path = os.path.join(
                snapshot_dir, f"graph_{self.namespace}_snapshot_{int(time.time() * 1000)}.graphml"
            )
            write_nx_graph(self._graph, snapshot_path)
            logger.debug("graph_snapshot_created", path=snapshot_path)
            return snapshot_path

    async def _restore_graph(self, snapshot_path: str) -> None:
        async with self._graph_lock.write_lock():
            if not os.path.exists(snapshot_path):
                logger.warning("graph_snapshot_not_found", path=snapshot_path)
                return
            restored_graph = load_nx_graph(snapshot_path)
            if restored_graph is not None:
                self._graph = restored_graph
                logger.info("graph_restored", path=snapshot_path)
            else:
                logger.error("graph_restore_failed", path=snapshot_path)

    async def has_node(self, node_id: str) -> bool:
        async with self._graph_lock.read_lock():
            return self._graph.has_node(node_id)

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        async with self._graph_lock.read_lock():
            return self._graph.has_edge(source_node_id, target_node_id)

    async def get_node(self, node_id: str) -> dict | None:
        async with self._graph_lock.read_lock():
            return self._graph.nodes.get(node_id)

    async def get_nodes_batch(self, node_ids: list[str]) -> list[dict | None]:
        async with self._graph_lock.read_lock():
            return [self._graph.nodes.get(node_id) for node_id in node_ids]

    async def node_degree(self, node_id: str) -> int:
        async with self._graph_lock.read_lock():
            if not self._graph.has_node(node_id):
                return 0
            return len(self._graph[node_id])

    async def node_degrees_batch(self, node_ids: list[str]) -> list[int]:
        async with self._graph_lock.read_lock():
            return [
                len(self._graph[node_id]) if self._graph.has_node(node_id) else 0
                for node_id in node_ids
            ]

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        async with self._graph_lock.read_lock():
            return (self._graph.degree(src_id) if self._graph.has_node(src_id) else 0) + (
                self._graph.degree(tgt_id) if self._graph.has_node(tgt_id) else 0
            )

    async def edge_degrees_batch(self, edge_pairs: list[tuple[str, str]]) -> list[int]:
        async with self._graph_lock.read_lock():
            return [
                (self._graph.degree(src_id) if self._graph.has_node(src_id) else 0)
                + (self._graph.degree(tgt_id) if self._graph.has_node(tgt_id) else 0)
                for src_id, tgt_id in edge_pairs
            ]

    async def get_edge(self, source_node_id: str, target_node_id: str) -> dict | None:
        """Get edge data between two nodes.

        When multiple edges exist between the same node pair (temporal multi-edges),
        they are merged into a single dict with aggregated descriptions/weights and
        first-found temporal fields. Use direct storage access for per-edge temporal detail.
        """
        async with self._graph_lock.read_lock():
            if not self._graph.has_edge(source_node_id, target_node_id):
                return None
            edges = self._graph.get_edge_data(source_node_id, target_node_id)
            if not edges:
                return None
            if len(edges) == 1:
                return dict(next(iter(edges.values())))
            return _merge_multi_edge_data(edges)

    async def get_edges_batch(self, edge_pairs: list[tuple[str, str]]) -> list[dict | None]:
        def _get_edge_sync(source_node_id: str, target_node_id: str) -> dict | None:
            if not self._graph.has_edge(source_node_id, target_node_id):
                return None
            edges = self._graph.get_edge_data(source_node_id, target_node_id)
            if not edges:
                return None
            if len(edges) == 1:
                return dict(next(iter(edges.values())))
            return _merge_multi_edge_data(edges)

        async with self._graph_lock.read_lock():
            return [
                _get_edge_sync(source_node_id, target_node_id)
                for source_node_id, target_node_id in edge_pairs
            ]

    async def get_node_edges(self, source_node_id: str):
        async with self._graph_lock.read_lock():
            if self._graph.has_node(source_node_id):
                return list({(u, v) for u, v, k in self._graph.edges(source_node_id, keys=True)})
            return None

    async def get_nodes_edges_batch(self, node_ids: list[str]) -> list[list[tuple[str, str]]]:
        async with self._graph_lock.read_lock():
            results = []
            for node_id in node_ids:
                if self._graph.has_node(node_id):
                    results.append(
                        list({(u, v) for u, v, k in self._graph.edges(node_id, keys=True)})
                    )
                else:
                    results.append([])
            return results

    def _upsert_node_unsafe(self, node_id: str, node_data: dict[str, str]):
        """Must be called with _graph_lock held."""
        filtered = {}
        for k, v in node_data.items():
            if v is None:
                continue
            if isinstance(v, str | int | float | bool):
                filtered[k] = v
            else:
                filtered[k] = str(v)
        self._graph.add_node(node_id, **filtered)

    def _upsert_edge_unsafe(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        """Must be called with _graph_lock held."""
        filtered = {}
        for k, v in edge_data.items():
            if v is None:
                continue
            if isinstance(v, str | int | float | bool):
                filtered[k] = v
            else:
                filtered[k] = str(v)
        edge_key = filtered.get("relationship_id")
        if edge_key is None:
            from .._utils import compute_sha256_id

            edge_key = compute_sha256_id(
                f"{source_node_id}|{target_node_id}|{filtered.get('description', '')}",
                prefix="edge_",
            )
            filtered["relationship_id"] = edge_key
        if self._graph.has_edge(source_node_id, target_node_id, key=edge_key):
            existing = self._graph[source_node_id][target_node_id][edge_key]
            existing.update(filtered)
        else:
            self._graph.add_edge(source_node_id, target_node_id, key=edge_key, **filtered)

    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        async with self._graph_lock.write_lock():
            self._upsert_node_unsafe(node_id, node_data)
            self._community_schema_cache = None

    async def upsert_nodes_batch(self, nodes_data: list[tuple[str, dict[str, str]]]):
        async with self._graph_lock.write_lock():
            for node_id, node_data in nodes_data:
                self._upsert_node_unsafe(node_id, node_data)
            self._community_schema_cache = None

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        async with self._graph_lock.write_lock():
            self._upsert_edge_unsafe(source_node_id, target_node_id, edge_data)
            self._community_schema_cache = None

    async def upsert_edges_batch(self, edges_data: list[tuple[str, str, dict[str, str]]]):
        async with self._graph_lock.write_lock():
            for source_node_id, target_node_id, edge_data in edges_data:
                self._upsert_edge_unsafe(source_node_id, target_node_id, edge_data)
            self._community_schema_cache = None

    async def delete_node(self, node_id: str):
        async with self._graph_lock.write_lock():
            if self._graph.has_node(node_id):
                self._graph.remove_node(node_id)
                self._community_schema_cache = None

    async def delete_nodes_batch(self, node_ids: list[str]):
        async with self._graph_lock.write_lock():
            for node_id in node_ids:
                if self._graph.has_node(node_id):
                    self._graph.remove_node(node_id)
            self._community_schema_cache = None

    async def delete_edge(self, source_node_id: str, target_node_id: str):
        async with self._graph_lock.write_lock():
            if self._graph.has_edge(source_node_id, target_node_id):
                keys = list(self._graph[source_node_id][target_node_id].keys())
                for key in keys:
                    self._graph.remove_edge(source_node_id, target_node_id, key=key)
                self._community_schema_cache = None

    async def delete_edges_batch(self, edge_pairs: list[tuple[str, str]]):
        async with self._graph_lock.write_lock():
            for source_node_id, target_node_id in edge_pairs:
                if self._graph.has_edge(source_node_id, target_node_id):
                    keys = list(self._graph[source_node_id][target_node_id].keys())
                    for key in keys:
                        self._graph.remove_edge(source_node_id, target_node_id, key=key)
            self._community_schema_cache = None

    async def clustering(self, algorithm: str, affected_node_ids: set[str] | None = None):
        async with self._graph_lock.write_lock():
            if algorithm not in self._clustering_algorithms:
                raise ValueError(f"Clustering algorithm {algorithm} not supported")
            await self._clustering_algorithms[algorithm].cluster(
                self, affected_node_ids=affected_node_ids
            )
            self._community_schema_cache = None

    async def community_schema(self):
        if self._community_schema_cache is not None:
            return self._community_schema_cache
        result = build_community_schema(self)
        self._community_schema_cache = result
        return result

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

        async with self._graph_lock.read_lock():
            graph_copy = self._graph.copy()

        node2vec = Node2Vec(graph_copy, **n2v_init_params)
        model = node2vec.fit(**w2v_fit_params)
        embeddings = model.wv.vectors
        nodes = model.wv.index_to_key

        async with self._graph_lock.read_lock():
            nodes_ids = [self._graph.nodes[node_id]["id"] for node_id in nodes]
        return embeddings, nodes_ids
