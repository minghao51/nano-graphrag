import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import networkx as nx

from .._utils import logger
from ..base import BaseGraphStorage
from .gdb_networkx_clustering import LeidenClusteringBackend, build_community_schema


def _canonical_edge(source_node_id: str, target_node_id: str) -> tuple[str, str]:
    return tuple(sorted((source_node_id, target_node_id)))


@dataclass
class SQLiteGraphStorage(BaseGraphStorage):
    _conn: Any = field(default=None, repr=False)

    def close(self):
        if self._conn is None:
            return
        self._conn.close()
        self._conn = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __post_init__(self):
        working_dir = self.global_config["working_dir"]
        os.makedirs(working_dir, exist_ok=True)
        self._db_file = os.path.join(working_dir, f"graph_{self.namespace}.db")
        self._conn = sqlite3.connect(self._db_file)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._ensure_schema()
        self._clustering_algorithms = {
            "leiden": LeidenClusteringBackend(),
            "louvain": LeidenClusteringBackend(),
        }
        self._last_affected_community_ids = set()
        self._last_clustering_was_incremental = False
        logger.info(f"Loaded SQLite graph store {self.namespace} from {self._db_file}")

    def _ensure_schema(self):
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                data TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS edges (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                data TEXT NOT NULL,
                PRIMARY KEY (source_id, target_id)
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS graph_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_source_id ON edges(source_id)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_target_id ON edges(target_id)")
        self._conn.commit()

    def _set_meta(self, key: str, value: Any):
        self._conn.execute(
            "INSERT OR REPLACE INTO graph_meta (key, value) VALUES (?, ?)",
            (key, json.dumps(value)),
        )

    def _get_meta(self, key: str, default: Any = None) -> Any:
        row = self._conn.execute("SELECT value FROM graph_meta WHERE key = ?", (key,)).fetchone()
        if row is None:
            return default
        return json.loads(row[0])

    def _build_projection(self) -> nx.Graph:
        graph = nx.Graph()
        node_rows = self._conn.execute("SELECT id, data FROM nodes").fetchall()
        for node_id, raw_data in node_rows:
            graph.add_node(node_id, **json.loads(raw_data))

        edge_rows = self._conn.execute("SELECT source_id, target_id, data FROM edges").fetchall()
        for source_id, target_id, raw_data in edge_rows:
            graph.add_edge(source_id, target_id, **json.loads(raw_data))

        graph.graph["community_update_counter"] = self._get_meta("community_update_counter", 0)
        return graph

    def _write_clusters_from_projection(self, graph: nx.Graph):
        for node_id, node_data in graph.nodes(data=True):
            existing = self._conn.execute(
                "SELECT 1 FROM nodes WHERE id = ?",
                (node_id,),
            ).fetchone()
            if existing is None:
                continue
            self._conn.execute(
                "UPDATE nodes SET data = ? WHERE id = ?",
                (json.dumps(dict(node_data)), node_id),
            )
        self._set_meta("community_update_counter", graph.graph.get("community_update_counter", 0))

    async def index_start_callback(self):
        self._ensure_schema()

    async def index_done_callback(self):
        self._conn.commit()

    async def _snapshot_graph(self) -> str:
        snapshot_dir = os.path.join(self.global_config["working_dir"], "snapshots")
        os.makedirs(snapshot_dir, exist_ok=True)
        snapshot_path = os.path.join(
            snapshot_dir, f"graph_{self.namespace}_snapshot_{int(time.time() * 1000)}.db"
        )
        snapshot_conn = sqlite3.connect(snapshot_path)
        try:
            self._conn.commit()
            self._conn.backup(snapshot_conn)
        finally:
            snapshot_conn.close()
        logger.debug(f"SQLite graph snapshot created at {snapshot_path}")
        return snapshot_path

    async def _restore_graph(self, snapshot_path: str) -> None:
        if not os.path.exists(snapshot_path):
            logger.warning(f"Snapshot file not found: {snapshot_path}")
            return
        snapshot_conn = sqlite3.connect(snapshot_path)
        try:
            self._conn.close()
            self._conn = sqlite3.connect(self._db_file)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            snapshot_conn.backup(self._conn)
            self._conn.commit()
            logger.info(f"SQLite graph restored from snapshot: {snapshot_path}")
        finally:
            snapshot_conn.close()

    async def has_node(self, node_id: str) -> bool:
        row = self._conn.execute("SELECT 1 FROM nodes WHERE id = ?", (node_id,)).fetchone()
        return row is not None

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        source_id, target_id = _canonical_edge(source_node_id, target_node_id)
        row = self._conn.execute(
            "SELECT 1 FROM edges WHERE source_id = ? AND target_id = ?",
            (source_id, target_id),
        ).fetchone()
        return row is not None

    async def node_degree(self, node_id: str) -> int:
        row = self._conn.execute(
            """
            SELECT COUNT(*)
            FROM edges
            WHERE source_id = ? OR target_id = ?
            """,
            (node_id, node_id),
        ).fetchone()
        return int(row[0]) if row is not None else 0

    async def node_degrees_batch(self, node_ids: list[str]) -> list[int]:
        return [await self.node_degree(node_id) for node_id in node_ids]

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        return (await self.node_degree(src_id)) + (await self.node_degree(tgt_id))

    async def edge_degrees_batch(self, edge_pairs: list[tuple[str, str]]) -> list[int]:
        return [await self.edge_degree(src_id, tgt_id) for src_id, tgt_id in edge_pairs]

    async def get_node(self, node_id: str) -> Optional[dict]:
        row = self._conn.execute("SELECT data FROM nodes WHERE id = ?", (node_id,)).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    async def get_nodes_batch(self, node_ids: list[str]) -> list[Optional[dict]]:
        return [await self.get_node(node_id) for node_id in node_ids]

    async def get_edge(self, source_node_id: str, target_node_id: str) -> Optional[dict]:
        source_id, target_id = _canonical_edge(source_node_id, target_node_id)
        row = self._conn.execute(
            "SELECT data FROM edges WHERE source_id = ? AND target_id = ?",
            (source_id, target_id),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    async def get_edges_batch(self, edge_pairs: list[tuple[str, str]]) -> list[Optional[dict]]:
        return [await self.get_edge(source_id, target_id) for source_id, target_id in edge_pairs]

    async def get_node_edges(self, source_node_id: str) -> Optional[list[tuple[str, str]]]:
        if not await self.has_node(source_node_id):
            return None
        rows = self._conn.execute(
            """
            SELECT source_id, target_id
            FROM edges
            WHERE source_id = ? OR target_id = ?
            ORDER BY source_id, target_id
            """,
            (source_node_id, source_node_id),
        ).fetchall()
        return [(row[0], row[1]) for row in rows]

    async def get_nodes_edges_batch(self, node_ids: list[str]) -> list[list[tuple[str, str]]]:
        return [(await self.get_node_edges(node_id)) or [] for node_id in node_ids]

    async def upsert_node(self, node_id: str, node_data: dict[str, Any]):
        self._conn.execute(
            "INSERT OR REPLACE INTO nodes (id, data) VALUES (?, ?)",
            (node_id, json.dumps(node_data)),
        )

    async def upsert_nodes_batch(self, nodes_data: list[tuple[str, dict[str, Any]]]):
        for node_id, node_data in nodes_data:
            await self.upsert_node(node_id, node_data)

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, Any]
    ):
        source_id, target_id = _canonical_edge(source_node_id, target_node_id)
        self._conn.execute(
            """
            INSERT OR REPLACE INTO edges (source_id, target_id, data)
            VALUES (?, ?, ?)
            """,
            (source_id, target_id, json.dumps(edge_data)),
        )

    async def upsert_edges_batch(self, edges_data: list[tuple[str, str, dict[str, Any]]]):
        for source_id, target_id, edge_data in edges_data:
            await self.upsert_edge(source_id, target_id, edge_data)

    async def delete_node(self, node_id: str):
        self._conn.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
        self._conn.execute(
            "DELETE FROM edges WHERE source_id = ? OR target_id = ?",
            (node_id, node_id),
        )

    async def delete_nodes_batch(self, node_ids: list[str]):
        for node_id in node_ids:
            await self.delete_node(node_id)

    async def delete_edge(self, source_node_id: str, target_node_id: str):
        source_id, target_id = _canonical_edge(source_node_id, target_node_id)
        self._conn.execute(
            "DELETE FROM edges WHERE source_id = ? AND target_id = ?",
            (source_id, target_id),
        )

    async def delete_edges_batch(self, edge_pairs: list[tuple[str, str]]):
        for source_id, target_id in edge_pairs:
            await self.delete_edge(source_id, target_id)

    async def clustering(self, algorithm: str, affected_node_ids: Optional[set[str]] = None):
        if algorithm not in self._clustering_algorithms:
            raise ValueError(f"Clustering algorithm {algorithm} not supported")

        projection = self._build_projection()
        temp_storage = type("ProjectedStorage", (), {})()
        temp_storage._graph = projection
        temp_storage.global_config = self.global_config
        temp_storage._last_affected_community_ids = set()
        temp_storage._last_clustering_was_incremental = False
        await self._clustering_algorithms[algorithm].cluster(
            temp_storage, affected_node_ids=affected_node_ids
        )
        self._write_clusters_from_projection(temp_storage._graph)
        self._last_affected_community_ids = set(temp_storage._last_affected_community_ids)
        self._last_clustering_was_incremental = temp_storage._last_clustering_was_incremental

    async def community_schema(self) -> dict[str, Any]:
        projection = self._build_projection()
        temp_storage = type("ProjectedStorage", (), {})()
        temp_storage._graph = projection
        return build_community_schema(temp_storage)

    async def embed_nodes(self, algorithm: str):
        raise NotImplementedError("Node embedding is not supported in SQLiteGraphStorage.")
