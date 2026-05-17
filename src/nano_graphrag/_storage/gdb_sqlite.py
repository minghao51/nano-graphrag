from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, cast

import networkx as nx

from .._utils import logger
from ..base import BaseGraphStorage
from ..prompt import GRAPH_FIELD_SEP
from .gdb_networkx_clustering import LeidenClusteringBackend, build_community_schema


def _canonical_edge(source_node_id: str, target_node_id: str) -> tuple[str, str]:
    return tuple(sorted((source_node_id, target_node_id)))  # type: ignore[return-value]


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
        self._conn = sqlite3.connect(self._db_file, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._ensure_schema()
        self._clustering_algorithms = {
            "leiden": LeidenClusteringBackend(),
        }
        self._last_affected_community_ids = set()
        self._last_clustering_was_incremental = False
        self._projection_cache: nx.MultiGraph | None = None
        self._projection_dirty = True
        logger.info("sqlite_graph_loaded", namespace=self.namespace, path=self._db_file)

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
            CREATE TABLE IF NOT EXISTS graph_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS edges (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                edge_key TEXT NOT NULL DEFAULT '',
                data TEXT NOT NULL,
                PRIMARY KEY (source_id, target_id, edge_key)
            )
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_source_id ON edges(source_id)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_target_id ON edges(target_id)")
        self._migrate_edges_to_multigraph()
        self._conn.commit()

    def _fetchone_sync(self, sql: str, params=()) -> tuple | None:
        return self._conn.execute(sql, params).fetchone()

    def _fetchall_sync(self, sql: str, params=()) -> list[tuple]:
        return self._conn.execute(sql, params).fetchall()

    def _execute_sync(self, sql: str, params=()) -> None:
        self._conn.execute(sql, params)

    def _migrate_edges_to_multigraph(self):
        rows = self._conn.execute("PRAGMA table_info(edges)").fetchall()
        column_names = [row[1] for row in rows]
        if "edge_key" in column_names:
            return
        logger.info("sqlite_graph_migration_start")
        self._conn.execute("ALTER TABLE edges RENAME TO edges_old")
        self._conn.execute(
            """
            CREATE TABLE edges (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                edge_key TEXT NOT NULL DEFAULT '',
                data TEXT NOT NULL,
                PRIMARY KEY (source_id, target_id, edge_key)
            )
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_source_id ON edges(source_id)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_target_id ON edges(target_id)")
        old_rows = self._conn.execute("SELECT source_id, target_id, data FROM edges_old").fetchall()
        for source_id, target_id, raw_data in old_rows:
            data = json.loads(raw_data)
            edge_key = data.get("relationship_id", "")
            self._conn.execute(
                "INSERT OR IGNORE INTO edges (source_id, target_id, edge_key, data) VALUES (?, ?, ?, ?)",
                (source_id, target_id, edge_key, raw_data),
            )
        self._conn.execute("DROP TABLE edges_old")
        logger.info("sqlite_graph_migration_complete", edges_migrated=len(old_rows))

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

    def _merge_edge_data(self, data_list: list[dict]) -> dict[str, Any]:
        if len(data_list) == 1:
            return data_list[0]
        combined: dict[str, Any] = {}
        descriptions = set()
        total_weight = 0.0
        source_ids = set()
        temporal_context = None
        valid_from = None
        valid_to = None
        for data in data_list:
            if "description" in data:
                descriptions.add(data["description"])
            total_weight += data.get("weight", 0.0)
            if "source_id" in data:
                source_ids.add(data["source_id"])
            if not temporal_context and data.get("temporal_context"):
                temporal_context = data["temporal_context"]
            if not valid_from and data.get("valid_from"):
                valid_from = data["valid_from"]
            if not valid_to and data.get("valid_to"):
                valid_to = data["valid_to"]
        combined["description"] = GRAPH_FIELD_SEP.join(sorted(descriptions))
        combined["weight"] = total_weight
        combined["source_id"] = GRAPH_FIELD_SEP.join(sorted(source_ids))
        combined["temporal_context"] = temporal_context
        combined["valid_from"] = valid_from
        combined["valid_to"] = valid_to
        combined["order"] = 1
        return combined

    def _invalidate_projection_cache(self):
        self._projection_dirty = True
        self._projection_cache = None

    def _build_projection(self) -> nx.MultiGraph:
        if not self._projection_dirty and self._projection_cache is not None:
            return self._projection_cache.copy()
        graph: nx.MultiGraph = nx.MultiGraph()
        node_rows = self._conn.execute("SELECT id, data FROM nodes").fetchall()
        for node_id, raw_data in node_rows:
            graph.add_node(node_id, **json.loads(raw_data))

        edge_rows = self._conn.execute(
            "SELECT source_id, target_id, edge_key, data FROM edges"
        ).fetchall()
        for source_id, target_id, edge_key, raw_data in edge_rows:
            graph.add_edge(source_id, target_id, key=edge_key, **json.loads(raw_data))

        graph.graph["community_update_counter"] = self._get_meta("community_update_counter", 0)
        self._projection_cache = graph.copy()
        self._projection_dirty = False
        return graph

    def _write_clusters_from_projection(self, graph: nx.MultiGraph):
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
        self._invalidate_projection_cache()

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
        logger.debug("sqlite_graph_snapshot_created", path=snapshot_path)
        return snapshot_path

    async def _restore_graph(self, snapshot_path: str) -> None:
        if not os.path.exists(snapshot_path):
            logger.warning("sqlite_graph_snapshot_not_found", path=snapshot_path)
            return
        snapshot_conn = sqlite3.connect(snapshot_path)
        try:
            self._conn.close()
            self._conn = sqlite3.connect(self._db_file, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            snapshot_conn.backup(self._conn)
            self._conn.commit()
            logger.info("sqlite_graph_restored", path=snapshot_path)
        finally:
            snapshot_conn.close()

    async def has_node(self, node_id: str) -> bool:
        row = await asyncio.to_thread(
            self._fetchone_sync, "SELECT 1 FROM nodes WHERE id = ?", (node_id,)
        )
        return row is not None

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        source_id, target_id = _canonical_edge(source_node_id, target_node_id)
        row = await asyncio.to_thread(
            self._fetchone_sync,
            "SELECT 1 FROM edges WHERE source_id = ? AND target_id = ? LIMIT 1",
            (source_id, target_id),
        )
        return row is not None

    async def node_degree(self, node_id: str) -> int:
        row = await asyncio.to_thread(
            self._fetchone_sync,
            """
            SELECT COUNT(DISTINCT CASE WHEN source_id = ? THEN target_id ELSE source_id END)
            FROM edges
            WHERE source_id = ? OR target_id = ?
            """,
            (node_id, node_id, node_id),
        )
        return int(row[0]) if row is not None else 0

    async def node_degrees_batch(self, node_ids: list[str]) -> list[int]:
        if not node_ids:
            return []
        placeholders = ",".join("?" for _ in node_ids)
        sql = f"""
        SELECT id, COUNT(DISTINCT neighbor) AS degree FROM (
            SELECT source_id AS id, target_id AS neighbor FROM edges WHERE source_id IN ({placeholders})
            UNION ALL
            SELECT target_id AS id, source_id AS neighbor FROM edges WHERE target_id IN ({placeholders})
        ) GROUP BY id
        """
        rows = await asyncio.to_thread(self._fetchall_sync, sql, node_ids + node_ids)
        degree_map = {row[0]: row[1] for row in rows}
        return [degree_map.get(node_id, 0) for node_id in node_ids]

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        return (await self.node_degree(src_id)) + (await self.node_degree(tgt_id))

    async def edge_degrees_batch(self, edge_pairs: list[tuple[str, str]]) -> list[int]:
        node_ids = list({n for pair in edge_pairs for n in pair})
        degrees = await self.node_degrees_batch(node_ids)
        degree_map = dict(zip(node_ids, degrees, strict=False))
        return [degree_map.get(src, 0) + degree_map.get(tgt, 0) for src, tgt in edge_pairs]

    async def get_node(self, node_id: str) -> dict | None:
        row = await asyncio.to_thread(
            self._fetchone_sync, "SELECT data FROM nodes WHERE id = ?", (node_id,)
        )
        if row is None:
            return None
        return json.loads(row[0])

    async def get_nodes_batch(self, node_ids: list[str]) -> list[dict | None]:
        if not node_ids:
            return []
        placeholders = ",".join("?" for _ in node_ids)
        rows = await asyncio.to_thread(
            self._fetchall_sync,
            f"SELECT id, data FROM nodes WHERE id IN ({placeholders})",
            node_ids,
        )
        node_map = {row[0]: json.loads(row[1]) for row in rows}
        return [node_map.get(node_id) for node_id in node_ids]

    async def get_edge(self, source_node_id: str, target_node_id: str) -> dict | None:
        """Get edge data between two nodes.

        When multiple edges exist between the same node pair (temporal multi-edges),
        they are merged into a single dict with aggregated descriptions/weights and
        first-found temporal fields. Use direct storage access for per-edge temporal detail.
        """
        source_id, target_id = _canonical_edge(source_node_id, target_node_id)
        rows = await asyncio.to_thread(
            self._fetchall_sync,
            "SELECT data FROM edges WHERE source_id = ? AND target_id = ?",
            (source_id, target_id),
        )
        if not rows:
            return None
        return self._merge_edge_data([json.loads(r[0]) for r in rows])

    async def get_edges_batch(self, edge_pairs: list[tuple[str, str]]) -> list[dict | None]:
        if not edge_pairs:
            return []
        canonical = {tuple(sorted((s, t))): (s, t) for s, t in edge_pairs}
        ids = list(canonical.keys())
        conditions = " OR ".join("(source_id = ? AND target_id = ?)" for _ in ids)
        params = [v for pair in ids for v in pair]
        rows = await asyncio.to_thread(
            self._fetchall_sync,
            f"SELECT source_id, target_id, data FROM edges WHERE {conditions} ORDER BY source_id, target_id",
            params,
        )
        edge_map: dict[tuple[str, str], list[dict]] = {}
        for source_id, target_id, raw_data in rows:
            edge_map.setdefault((source_id, target_id), []).append(json.loads(raw_data))
        result: list[dict[str, Any] | None] = []
        for pair in edge_pairs:
            canon_key: tuple[str, str] = cast(tuple[str, str], tuple(sorted(pair)))
            data_list = edge_map.get(canon_key)
            if not data_list:
                result.append(None)
            else:
                result.append(self._merge_edge_data(data_list))
        return result

    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        if not await self.has_node(source_node_id):
            return None
        rows = await asyncio.to_thread(
            self._fetchall_sync,
            """
            SELECT source_id, target_id
            FROM edges
            WHERE source_id = ? OR target_id = ?
            ORDER BY source_id, target_id
            """,
            (source_node_id, source_node_id),
        )
        return [(row[0], row[1]) for row in rows]

    async def get_nodes_edges_batch(self, node_ids: list[str]) -> list[list[tuple[str, str]]]:
        if not node_ids:
            return []
        placeholders = ",".join("?" for _ in node_ids)
        rows = await asyncio.to_thread(
            self._fetchall_sync,
            f"SELECT source_id, target_id FROM edges WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders}) ORDER BY source_id, target_id",
            node_ids + node_ids,
        )
        edge_map: dict[str, set[tuple[str, str]]] = {n: set() for n in node_ids}
        for source_id, target_id in rows:
            if source_id in edge_map:
                edge_map[source_id].add((source_id, target_id))
            if target_id in edge_map:
                edge_map[target_id].add((source_id, target_id))
        return [sorted(edge_map[n]) for n in node_ids]

    async def upsert_node(self, node_id: str, node_data: dict[str, Any]):
        self._invalidate_projection_cache()
        await asyncio.to_thread(
            self._execute_sync,
            "INSERT OR REPLACE INTO nodes (id, data) VALUES (?, ?)",
            (node_id, json.dumps(node_data)),
        )

    async def upsert_nodes_batch(self, nodes_data: list[tuple[str, dict[str, Any]]]):
        if not nodes_data:
            return
        self._invalidate_projection_cache()

        def _bulk_upsert_nodes(data):
            self._conn.executemany(
                "INSERT OR REPLACE INTO nodes (id, data) VALUES (?, ?)",
                [(nid, json.dumps(ndata)) for nid, ndata in data],
            )

        await asyncio.to_thread(_bulk_upsert_nodes, nodes_data)

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, Any]
    ):
        source_id, target_id = _canonical_edge(source_node_id, target_node_id)
        edge_key = edge_data.get("relationship_id", "")
        self._invalidate_projection_cache()
        await asyncio.to_thread(
            self._execute_sync,
            """
            INSERT OR REPLACE INTO edges (source_id, target_id, edge_key, data)
            VALUES (?, ?, ?, ?)
            """,
            (source_id, target_id, edge_key, json.dumps(edge_data)),
        )

    async def upsert_edges_batch(self, edges_data: list[tuple[str, str, dict[str, Any]]]):
        if not edges_data:
            return
        self._invalidate_projection_cache()

        def _bulk_upsert_edges(data):
            self._conn.executemany(
                "INSERT OR REPLACE INTO edges (source_id, target_id, edge_key, data) VALUES (?, ?, ?, ?)",
                [
                    (*_canonical_edge(s, t), d.get("relationship_id", ""), json.dumps(d))
                    for s, t, d in data
                ],
            )

        await asyncio.to_thread(_bulk_upsert_edges, edges_data)

    async def delete_node(self, node_id: str):
        self._invalidate_projection_cache()
        await asyncio.to_thread(self._execute_sync, "DELETE FROM nodes WHERE id = ?", (node_id,))
        await asyncio.to_thread(
            self._execute_sync,
            "DELETE FROM edges WHERE source_id = ? OR target_id = ?",
            (node_id, node_id),
        )

    async def delete_nodes_batch(self, node_ids: list[str]):
        if not node_ids:
            return
        self._invalidate_projection_cache()
        placeholders = ",".join("?" for _ in node_ids)

        def _bulk_delete_nodes(ids):
            self._conn.execute(f"DELETE FROM nodes WHERE id IN ({placeholders})", ids)
            self._conn.execute(
                f"DELETE FROM edges WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})",
                ids + ids,
            )

        await asyncio.to_thread(_bulk_delete_nodes, node_ids)

    async def delete_edge(self, source_node_id: str, target_node_id: str):
        source_id, target_id = _canonical_edge(source_node_id, target_node_id)
        self._invalidate_projection_cache()
        await asyncio.to_thread(
            self._execute_sync,
            "DELETE FROM edges WHERE source_id = ? AND target_id = ?",
            (source_id, target_id),
        )

    async def delete_edges_batch(self, edge_pairs: list[tuple[str, str]]):
        if not edge_pairs:
            return
        self._invalidate_projection_cache()
        pairs = [tuple(sorted((s, t))) for s, t in edge_pairs]
        conditions = " OR ".join("(source_id = ? AND target_id = ?)" for _ in pairs)
        params = [v for pair in pairs for v in pair]

        def _bulk_delete_edges():
            self._conn.execute(f"DELETE FROM edges WHERE {conditions}", params)

        await asyncio.to_thread(_bulk_delete_edges)

    async def clustering(self, algorithm: str, affected_node_ids: set[str] | None = None):
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
