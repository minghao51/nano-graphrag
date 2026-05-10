"""
Storage & Extraction — Scale and tune
======================================
Objective: Swap storage backends for your scale, do incremental inserts,
tune extraction quality vs speed, and enable entity linking.

Run:  dotenvx run -- uv run notebooks/04_storage_and_extraction.py
"""

import json
import logging
import sqlite3
import time
from pathlib import Path

logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.WARNING)

WORKING_DIR = Path("./_cache/04_storage_and_extraction")
WORKING_DIR.mkdir(parents=True, exist_ok=True)

# %% Imports
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag._storage import SQLiteGraphStorage

TEXT = (
    "Alice works at Wonderland Labs as a quantum researcher. "
    "Bob is the CTO of Wonderland Labs and oversees the AI division. "
    "Carol founded DataForge, a startup building graph databases. "
    "Dave joined DataForge as lead engineer after leaving CloudCorp. "
    "Eve is a security consultant who audits both Wonderland Labs and DataForge."
)

# ============================================================
# STORAGE BACKENDS
# ============================================================

# %% Default: NetworkX (GraphML file) + HNSWLib (vector index)
# Best for: development, prototyping, small-to-medium graphs
dir_default = WORKING_DIR / "default"
dir_default.mkdir(parents=True, exist_ok=True)

rag_default = GraphRAG(working_dir=str(dir_default))
start = time.time()
await rag_default.ainsert(TEXT)
t_default = time.time() - start
result = await rag_default.aquery("Who works at Wonderland Labs?", QueryParam())
print(f"[NetworkX] Insert: {t_default:.1f}s")
print(f"  Query: {result[:100]}...")
print(f"  Graph:  {rag_default.graph_storage_cls.__name__}")
print(f"  Vector: {rag_default.vector_db_storage_cls.__name__}")

# List files created
print("  Files:")
for f in sorted(dir_default.iterdir()):
    size = f.stat().st_size
    print(f"    {f.name:50s} {size / 1024:.1f} KB" if size > 1024 else f"    {f.name:50s} {size} B")

# %% SQLite graph backend
# Best for: crash safety, atomic writes, 100K+ edges
dir_sqlite = WORKING_DIR / "sqlite"
dir_sqlite.mkdir(parents=True, exist_ok=True)

rag_sqlite = GraphRAG(
    working_dir=str(dir_sqlite),
    graph_storage_cls=SQLiteGraphStorage,
)
start = time.time()
await rag_sqlite.ainsert(TEXT)
t_sqlite = time.time() - start
print(f"\n[SQLite] Insert: {t_sqlite:.1f}s")

# Inspect SQLite database
db_path = dir_sqlite / "graph_chunk_entity_relation.db"
conn = sqlite3.connect(str(db_path))
nodes = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
edges = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]

# Show some actual data
sample_nodes = conn.execute("SELECT id, data FROM nodes LIMIT 3").fetchall()
conn.close()

print(f"  Nodes: {nodes}, Edges: {edges}")
for nid, data_str in sample_nodes:
    data = json.loads(data_str)
    name = data.get("entity_name", nid)
    etype = data.get("entity_type", "?")
    print(f"  Node: {name} ({etype})")

# List files
print("  Files:")
for f in sorted(dir_sqlite.iterdir()):
    size = f.stat().st_size
    print(f"    {f.name:50s} {size / 1024:.1f} KB" if size > 1024 else f"    {f.name:50s} {size} B")

# %% Backend comparison
print("""
+--------------------+-----------------+------------------+---------------------------+
| Backend            | Storage         | Execution        | Best for                  |
+--------------------+-----------------+------------------+---------------------------+
| NetworkXStorage    | GraphML XML     | In-memory        | Dev, small graphs         |
| SQLiteGraphStorage | SQLite DB       | Disk (WAL)       | Medium, crash-safe        |
| Neo4jStorage       | Neo4j server    | External server  | Production, large graphs  |
+--------------------+-----------------+------------------+---------------------------+

Vector backends:
  HNSWVectorStorage    — fast ANN, default
  NanoVectorDBStorage  — pure Python, no C deps

Swap backends:
  rag = GraphRAG(graph_storage_cls=SQLiteGraphStorage)
  rag = GraphRAG(vector_db_storage_cls=NanoVectorDBStorage)
""")

# ============================================================
# INCREMENTAL INSERTION
# ============================================================

# %% ainsert_documents — add new docs without reprocessing old ones
# Uses content hashing to skip already-indexed documents.
PART1 = (
    "Alice is a quantum physicist at MIT. She published research on "
    "quantum entanglement in 2023. Bob is a computer scientist at MIT."
)
PART2 = (
    "Alice Chen collaborated with Carol on quantum error correction. "
    "Bob Smith also contributed. Dave is their department head at MIT."
)
PART3 = (
    "Carol's startup DataForge launched a new graph database product. "
    "Eve audited their security and found no issues."
)

dir_inc = WORKING_DIR / "incremental"
dir_inc.mkdir(parents=True, exist_ok=True)
rag_inc = GraphRAG(working_dir=str(dir_inc))

start = time.time()
await rag_inc.ainsert_documents({"doc-1": PART1})
t1 = time.time() - start
print(f"Round 1 (1 doc):  {t1:.1f}s")

start = time.time()
await rag_inc.ainsert_documents({"doc-2": PART2})
t2 = time.time() - start
print(f"Round 2 (1 new):  {t2:.1f}s (doc-1 skipped by content hash)")

start = time.time()
await rag_inc.ainsert_documents({"doc-3": PART3})
t3 = time.time() - start
print(f"Round 3 (1 new):  {t3:.1f}s (doc-1, doc-2 skipped)")

# Query across all documents
result = await rag_inc.aquery(
    "What do we know about Alice and her collaborators?",
    QueryParam(mode="local"),
)
print(f"\nCross-document query: {result[:200]}...")

# %% force_rebuild option
print("""
Force rebuild (reprocess all documents from scratch):
  await rag_inc.ainsert_documents({"doc-4": text}, force_rebuild=True)
""")

# ============================================================
# EXTRACTION TUNING
# ============================================================

# %% Balanced vs Fast quality
# balanced — main model + gleaning (more accurate)
# fast     — cheap model + no gleaning (cheaper, faster)

dir_balanced = WORKING_DIR / "balanced"
dir_balanced.mkdir(parents=True, exist_ok=True)
rag_balanced = GraphRAG(
    working_dir=str(dir_balanced),
    entity_extraction_quality="balanced",
)

dir_fast = WORKING_DIR / "fast"
dir_fast.mkdir(parents=True, exist_ok=True)
rag_fast = GraphRAG(
    working_dir=str(dir_fast),
    entity_extraction_quality="fast",
)

print("\nRunning extraction quality comparison...")
start = time.time()
await rag_balanced.ainsert(TEXT)
t_bal = time.time() - start

start = time.time()
await rag_fast.ainsert(TEXT)
t_fast = time.time() - start

print(f"  Balanced: {t_bal:.1f}s (main model + gleaning)")
print(f"  Fast:     {t_fast:.1f}s (cheap model, no gleaning)")

# Compare query results
q = "Who are the key people mentioned and what do they do?"
result_bal = await rag_balanced.aquery(q, QueryParam())
result_fast = await rag_fast.aquery(q, QueryParam())
print(f"\n  Balanced answer ({len(result_bal)} chars):")
print(f"    {result_bal[:150]}...")
print(f"\n  Fast answer ({len(result_fast)} chars):")
print(f"    {result_fast[:150]}...")

# %% Batch size tuning
print("\n--- Batch Size Effect ---")
for bs in [1, 3, 5]:
    dir_bs = WORKING_DIR / f"batch_{bs}"
    dir_bs.mkdir(parents=True, exist_ok=True)
    rag_bs = GraphRAG(
        working_dir=str(dir_bs),
        extraction_batch_size=bs,
    )
    start = time.time()
    await rag_bs.ainsert(TEXT)
    t_bs = time.time() - start
    print(f"  batch_size={bs}: {t_bs:.1f}s ({bs} chunks per LLM call)")

print("""
Guidelines:
  batch_size=1:   Most reliable, one chunk per call
  batch_size=3-5: Good balance (default: 5)
  batch_size=8+:  Best throughput, needs 32K+ context models
""")

# ============================================================
# ENTITY LINKING
# ============================================================

# %% Entity linking — merge duplicate entities across documents
# When enabled, "Alice" and "Alice Chen" get recognized as the same entity
dir_el = WORKING_DIR / "entity_linking"
dir_el.mkdir(parents=True, exist_ok=True)

rag_el = GraphRAG(
    working_dir=str(dir_el),
    enable_entity_linking=True,
    entity_linking_similarity_threshold=0.85,
)

await rag_el.ainsert_documents({"part1": PART1, "part2": PART2})

import networkx as nx
graphml = dir_el / "graph_chunk_entity_relation.graphml"
G = nx.read_graphml(str(graphml))

print("\nEntity linking enabled:")
print(f"  Graph nodes: {G.number_of_nodes()}")
for _nid, data in G.nodes(data=True):
    name = data.get("entity_name", "?")
    aliases = json.loads(data.get("aliases", "[]"))
    if aliases:
        print(f"  {name} → aliases: {aliases}")

print("""
Entity linking parameters:
  entity_linking_similarity_threshold    — min embedding similarity (default: 0.92)
  entity_linking_max_candidates          — max candidates to check (default: 3)
  entity_linking_iou_threshold           — min neighborhood overlap (default: 0.3)
  entity_linking_min_common_neighbors    — min shared neighbors (default: 2)

Pipeline: exact match → fuzzy match → neighborhood evidence → LLM disambiguation
""")

# ============================================================
# EXTRACTION CONFIG REFERENCE
# ============================================================

print("""
Key extraction parameters:
  extraction_batch_size              — chunks per LLM call (default: 5)
  extraction_max_async               — concurrent LLM calls (default: 16)
  doc_extraction_max_async           — concurrent docs (default: 4)
  entity_extraction_quality          — 'balanced' or 'fast'
  structured_output                  — use Pydantic JSON schema (default: True)
  fallback_to_parsing                — retry with text parser on failure (default: True)
  extraction_backend                 — 'llm' (default) or 'gliner' (local model, free)

GLiNER alternative (no LLM API, local neural model):
  rag = GraphRAG(extraction_backend='gliner')
  Requires: pip install gliner2
""")
