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

    def __post_init__(self):
        working_dir = self.global_config["working_dir"]
        self._db_file = os.path.join(working_dir, f"kv_store_{self.namespace}.db")
        self._legacy_file = os.path.join(working_dir, f"kv_store_{self.namespace}.json")
        os.makedirs(working_dir, exist_ok=True)
        self._conn = sqlite3.connect(self._db_file)
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
        cursor = self._conn.execute("SELECT key FROM kv_store")
        return [row[0] for row in cursor.fetchall()]

    async def index_done_callback(self):
        self._conn.commit()

    async def get_by_id(self, id: str):
        cursor = self._conn.execute("SELECT value FROM kv_store WHERE key = ?", (id,))
        row = cursor.fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    async def get_by_ids(self, ids: list[str], fields=None):
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        cursor = self._conn.execute(
            f"SELECT key, value FROM kv_store WHERE key IN ({placeholders})", ids
        )
        data = {row[0]: json.loads(row[1]) for row in cursor.fetchall()}
        if fields is None:
            return [data.get(id) for id in ids]
        return [
            {k: v for k, v in data[id].items() if k in fields} if id in data else None
            for id in ids
        ]

    async def filter_keys(self, data: list[str]) -> set[str]:
        if not data:
            return set()
        placeholders = ",".join("?" for _ in data)
        cursor = self._conn.execute(
            f"SELECT key FROM kv_store WHERE key IN ({placeholders})", data
        )
        existing = {row[0] for row in cursor.fetchall()}
        return set(data) - existing

    async def upsert(self, data: dict[str, Any]):
        for key, value in data.items():
            self._conn.execute(
                "INSERT OR REPLACE INTO kv_store (key, value) VALUES (?, ?)",
                (key, json.dumps(value)),
            )
        self._conn.commit()

    async def delete(self, ids: list[str]):
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        self._conn.execute(f"DELETE FROM kv_store WHERE key IN ({placeholders})", ids)
        self._conn.commit()

    async def drop(self):
        self._conn.execute("DELETE FROM kv_store")
        self._conn.commit()


# Backward-compatible alias
JsonKVStorage = SQLiteKVStorage
