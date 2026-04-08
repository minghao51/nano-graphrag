import json
from collections import defaultdict
from typing import Optional, Protocol

from .._utils import logger
from ..base import SingleCommunitySchema
from ..prompt import GRAPH_FIELD_SEP
from .gdb_networkx_utils import stabilize_graph, stable_largest_connected_component, to_igraph


class ClusteringBackend(Protocol):
    async def cluster(self, storage, affected_node_ids: Optional[set[str]] = None) -> None:
        ...


def build_community_schema(storage) -> dict[str, SingleCommunitySchema]:
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
    for node_id, node_data in storage._graph.nodes(data=True):
        if "clusters" not in node_data:
            continue
        clusters = json.loads(node_data["clusters"])
        this_node_edges = storage._graph.edges(node_id)

        for cluster in clusters:
            level = cluster["level"]
            cluster_key = str(cluster["cluster"])
            levels[level].add(cluster_key)
            results[cluster_key]["level"] = level
            results[cluster_key]["title"] = f"Cluster {cluster_key}"
            results[cluster_key]["nodes"].add(node_id)
            results[cluster_key]["edges"].update([tuple(sorted(e)) for e in this_node_edges])
            results[cluster_key]["chunk_ids"].update(node_data["source_id"].split(GRAPH_FIELD_SEP))
            max_num_ids = max(max_num_ids, len(results[cluster_key]["chunk_ids"]))

    ordered_levels = sorted(levels.keys())
    for i, curr_level in enumerate(ordered_levels[:-1]):
        next_level = ordered_levels[i + 1]
        this_level_comms = levels[curr_level]
        next_level_comms = levels[next_level]
        for comm in this_level_comms:
            results[comm]["sub_communities"] = [
                c for c in next_level_comms if results[c]["nodes"].issubset(results[comm]["nodes"])
            ]

    for _, value in results.items():
        value["edges"] = [list(e) for e in list(value["edges"])]
        value["nodes"] = list(value["nodes"])
        value["chunk_ids"] = list(value["chunk_ids"])
        value["occurrence"] = len(value["chunk_ids"]) / max_num_ids
    return dict(results)


class LeidenClusteringBackend:
    def _cluster_data_to_subgraphs(self, storage, cluster_data: dict[str, list[dict[str, str]]]):
        for node_id, clusters in cluster_data.items():
            storage._graph.nodes[node_id]["clusters"] = json.dumps(clusters)

    def _next_incremental_cluster_prefix(self, storage) -> str:
        counter = int(storage._graph.graph.get("community_update_counter", 0)) + 1
        storage._graph.graph["community_update_counter"] = counter
        return f"inc-{counter}"

    def _extract_existing_clusters(self, storage, node_id: str) -> list[dict[str, str]]:
        node_data = storage._graph.nodes.get(node_id, {})
        raw_clusters = node_data.get("clusters")
        if not raw_clusters:
            return []
        return json.loads(raw_clusters)

    def _compute_frontier_nodes(self, storage, affected_node_ids: set[str]) -> set[str]:
        radius = storage.global_config["addon_params"].get("community_update_neighbor_radius", 1)
        frontier = {node_id for node_id in affected_node_ids if storage._graph.has_node(node_id)}
        current_layer = set(frontier)
        for _ in range(radius):
            next_layer = set()
            for node_id in current_layer:
                next_layer.update(storage._graph.neighbors(node_id))
            frontier.update(next_layer)
            current_layer = next_layer
        return frontier

    def _collect_level0_starting_communities(self, storage, frontier_nodes: set[str]) -> dict[str, int]:
        cluster_lookup = {}
        next_cluster_id = 0
        starting = {}
        for node_id in frontier_nodes:
            level0 = next(
                (
                    cluster["cluster"]
                    for cluster in self._extract_existing_clusters(storage, node_id)
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

    def _assign_singleton_clusters(self, storage):
        node_communities = {
            node_id: [{"level": 0, "cluster": f"singleton-{index}"}]
            for index, node_id in enumerate(sorted(storage._graph.nodes()))
        }
        self._cluster_data_to_subgraphs(storage, node_communities)
        storage._last_affected_community_ids = {
            cluster["cluster"] for clusters in node_communities.values() for cluster in clusters
        }
        storage._last_clustering_was_incremental = False

    def _leiden_seed(self, storage) -> int:
        seed = storage.global_config.get("graph_cluster_seed", 0xDEADBEEF)
        return seed % (2**31 - 1)

    async def cluster(self, storage, affected_node_ids: Optional[set[str]] = None) -> None:
        import leidenalg as la

        if storage._graph.number_of_nodes() == 0:
            storage._last_affected_community_ids = set()
            storage._last_clustering_was_incremental = False
            return
        if storage._graph.number_of_edges() == 0:
            self._assign_singleton_clusters(storage)
            return

        frontier_ratio_limit = storage.global_config["addon_params"].get(
            "community_update_max_frontier_ratio", 0.5
        )
        should_try_incremental = bool(affected_node_ids)
        if should_try_incremental:
            frontier_nodes = self._compute_frontier_nodes(storage, affected_node_ids or set())
            has_existing_clusters = any(
                storage._graph.nodes[node_id].get("clusters")
                for node_id in frontier_nodes
                if storage._graph.has_node(node_id)
            )
            if (
                not frontier_nodes
                or len(frontier_nodes) < 2
                or len(frontier_nodes) / max(1, storage._graph.number_of_nodes())
                > frontier_ratio_limit
                or not has_existing_clusters
            ):
                should_try_incremental = False

        if should_try_incremental:
            old_community_ids = {
                str(cluster["cluster"])
                for node_id in frontier_nodes
                for cluster in self._extract_existing_clusters(storage, node_id)
            }
            frontier_graph = stabilize_graph(storage._graph.subgraph(frontier_nodes).copy())
            starting_communities = self._collect_level0_starting_communities(storage, frontier_nodes)

            ig_graph, _ = to_igraph(frontier_graph)
            next_membership_id = max(starting_communities.values(), default=-1) + 1
            initial_membership = []
            for vertex in ig_graph.vs:
                node_id = vertex["_nx_name"]
                if node_id in starting_communities:
                    initial_membership.append(starting_communities[node_id])
                else:
                    initial_membership.append(next_membership_id)
                    next_membership_id += 1

            partition = la.find_partition(
                ig_graph,
                la.ModularityVertexPartition,
                seed=self._leiden_seed(storage),
                n_iterations=8,
                initial_membership=initial_membership,
            )

            cluster_prefix = self._next_incremental_cluster_prefix(storage)
            new_level0_ids = {}
            for cluster_idx, cluster_nodes in enumerate(partition):
                level0_cluster = f"{cluster_prefix}-l0-{cluster_idx}"
                for node_idx in cluster_nodes:
                    nx_node_id = ig_graph.vs[node_idx]["_nx_name"]
                    storage._graph.nodes[nx_node_id]["clusters"] = json.dumps(
                        [{"level": 0, "cluster": level0_cluster}]
                    )
                    new_level0_ids[nx_node_id] = level0_cluster
            storage._last_affected_community_ids = old_community_ids.union(new_level0_ids.values())
            storage._last_clustering_was_incremental = True
            logger.info(
                f"Incremental Leiden updated {len(frontier_nodes)} frontier nodes across {len(storage._last_affected_community_ids)} communities"
            )
            return

        graph_to_cluster = stable_largest_connected_component(storage._graph)
        ig_graph, _ = to_igraph(graph_to_cluster)
        resolutions = storage.global_config.get("leiden_resolutions", [2.0, 1.0, 0.5])

        node_communities = defaultdict(list)
        levels = defaultdict(set)

        for level, resolution in enumerate(resolutions):
            partition = la.find_partition(
                ig_graph,
                la.RBConfigurationVertexPartition,
                resolution_parameter=resolution,
                seed=self._leiden_seed(storage) + level,
                n_iterations=8,
            )

            for cluster_idx, cluster_nodes in enumerate(partition):
                cluster_id = f"l{level}_c{cluster_idx}"
                levels[level].add(cluster_id)
                for node_idx in cluster_nodes:
                    nx_node_id = ig_graph.vs[node_idx]["_nx_name"]
                    node_communities[nx_node_id].append({"level": level, "cluster": cluster_id})

        node_communities = dict(node_communities)
        logger.info(f"Each level has communities: { {k: len(v) for k, v in levels.items()} }")
        self._cluster_data_to_subgraphs(storage, node_communities)
        storage._last_affected_community_ids = {
            str(cluster["cluster"])
            for clusters in node_communities.values()
            for cluster in clusters
        }
        storage._last_clustering_was_incremental = False
