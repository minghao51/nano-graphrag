import asyncio
import json
import os
import sqlite3
from dataclasses import dataclass, field
from typing import Any

from .._utils import logger
from ..base import BaseKVStorage


@dataclass
class SQLiteKVStorage(BaseKVStorage):
    """SQLite-based key-value storage.

    Benefits:
    - No full-file load on init (query-based access)
    - Atomic writes via transactions
    - Scales to millions of entries
    - Thread-safe with WAL mode
    - All I/O offloaded to thread pool to avoid blocking the event loop
    """

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

    def _execute_sync(self, sql: str, params=()) -> list[tuple]:
        cursor = self._conn.execute(sql, params)
        return cursor.fetchall()

    def _execute_script_sync(self, sql: str) -> None:
        self._conn.execute(sql)
        self._conn.commit()

    def __post_init__(self):
        working_dir = self.global_config["working_dir"]
        self._db_file = os.path.join(working_dir, f"kv_store_{self.namespace}.db")
        self._legacy_file = os.path.join(working_dir, f"kv_store_{self.namespace}.json")
        os.makedirs(working_dir, exist_ok=True)
        self._conn = sqlite3.connect(self._db_file, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS kv_store (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self._conn.commit()
        self._migrate_legacy_json_store()
        count = self._conn.execute("SELECT COUNT(*) FROM kv_store").fetchone()[0]
        logger.info(f"Loaded SQLite KV {self.namespace} with {count} entries")

    def _migrate_legacy_json_store(self):
        if not os.path.exists(self._legacy_file):
            return
        existing_count = self._conn.execute("SELECT COUNT(*) FROM kv_store").fetchone()[0]
        if existing_count:
            return
        with open(self._legacy_file, encoding="utf-8") as f:
            legacy_data = json.load(f) or {}
        if not legacy_data:
            return
        self._conn.executemany(
            "INSERT OR REPLACE INTO kv_store (key, value) VALUES (?, ?)",
            [(key, json.dumps(value)) for key, value in legacy_data.items()],
        )
        self._conn.commit()
        logger.info(
            f"Migrated {len(legacy_data)} entries from legacy JSON KV store for {self.namespace}"
        )

    async def all_keys(self) -> list[str]:
        rows = await asyncio.to_thread(self._execute_sync, "SELECT key FROM kv_store")
        return [row[0] for row in rows]

    async def index_done_callback(self):
        await asyncio.to_thread(self._conn.commit)

    async def get_by_id(self, id: str):
        rows = await asyncio.to_thread(
            self._execute_sync, "SELECT value FROM kv_store WHERE key = ?", (id,)
        )
        if not rows:
            return None
        return json.loads(rows[0][0])

    async def get_by_ids(self, ids: list[str], fields=None):
        if not ids:
            return []
        data = {}
        batch_size = 900
        for i in range(0, len(ids), batch_size):
            batch = ids[i : i + batch_size]
            placeholders = ",".join("?" for _ in batch)
            rows = await asyncio.to_thread(
                self._execute_sync,
                f"SELECT key, value FROM kv_store WHERE key IN ({placeholders})",
                batch,
            )
            data.update({row[0]: json.loads(row[1]) for row in rows})
        if fields is None:
            return [data.get(id) for id in ids]
        return [
            {k: v for k, v in data[id].items() if k in fields} if id in data else None for id in ids
        ]

    async def filter_keys(self, data: list[str]) -> set[str]:
        if not data:
            return set()
        existing = set()
        batch_size = 900
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            placeholders = ",".join("?" for _ in batch)
            rows = await asyncio.to_thread(
                self._execute_sync,
                f"SELECT key FROM kv_store WHERE key IN ({placeholders})",
                batch,
            )
            existing.update({row[0] for row in rows})
        return set(data) - existing

    async def upsert(self, data: dict[str, Any]):
        rows = [(key, json.dumps(value)) for key, value in data.items()]

        def _upsert_sync():
            self._conn.executemany(
                "INSERT OR REPLACE INTO kv_store (key, value) VALUES (?, ?)", rows
            )
            self._conn.commit()

        await asyncio.to_thread(_upsert_sync)

    async def delete(self, ids: list[str]):
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)

        def _delete_sync():
            self._conn.execute(f"DELETE FROM kv_store WHERE key IN ({placeholders})", ids)
            self._conn.commit()

        await asyncio.to_thread(_delete_sync)

    async def drop(self):
        def _drop_sync():
            self._conn.execute("DELETE FROM kv_store")
            self._conn.commit()

        await asyncio.to_thread(_drop_sync)


# Backward-compatible alias
JsonKVStorage = SQLiteKVStorage
