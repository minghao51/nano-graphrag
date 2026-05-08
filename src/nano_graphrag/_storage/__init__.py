from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .gdb_neo4j import Neo4jStorage
    from .vdb_nanovectordb import NanoVectorDBStorage

try:
    from .gdb_neo4j import Neo4jStorage
except ImportError:
    Neo4jStorage = None  # type: ignore[assignment,misc]

from .gdb_networkx import NetworkXStorage as NetworkXStorage
from .gdb_sqlite import SQLiteGraphStorage as SQLiteGraphStorage
from .kv_json import SQLiteKVStorage as JsonKVStorage  # backward-compat alias
from .vdb_hnswlib import HNSWVectorStorage as HNSWVectorStorage

try:
    from .vdb_nanovectordb import NanoVectorDBStorage
except ImportError:
    NanoVectorDBStorage = None  # type: ignore[assignment,misc]

__all__ = [
    "HNSWVectorStorage",
    "JsonKVStorage",
    "NanoVectorDBStorage",
    "Neo4jStorage",
    "NetworkXStorage",
    "SQLiteGraphStorage",
]
