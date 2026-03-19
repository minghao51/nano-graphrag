# Nano-GraphRAG Architecture

## Overview

`nano-graphrag` is organized around one pipeline:

1. Ingest raw documents
2. Chunk text with a tokenizer-aware splitter
3. Extract entities and relationships
4. Store graph, chunks, and vectors in pluggable backends
5. Cluster the graph and generate community reports
6. Answer `global`, `local`, or `naive` queries from the persisted state

The minimal core favors:

- LiteLLM for provider access
- `networkx` for graph storage
- `nano-vectordb` for vector storage
- file-backed JSON/GraphML persistence in the working directory

## Main Runtime Pieces

### `GraphRAG`

`nano_graphrag.graphrag.GraphRAG` coordinates the pipeline.

- Normalizes runtime config and compatibility aliases
- Builds the tokenizer
- Builds LLM and embedding runtime wrappers
- Builds storage backends
- Runs insert and query flows

`GraphRAGConfig` is the canonical configuration object for runtime settings. Direct `GraphRAG(...)` kwargs still work as compatibility aliases for one release window.

### Operation Modules

Operational logic is split by responsibility under `nano_graphrag._ops`:

- `chunking.py`: chunk construction and built-in chunking helpers
- `extraction.py`: entity extraction, merge, summarize, and graph upsert flow
- `community.py`: community description packing and report generation
- `query.py`: local/global/naive query context building and answer generation

`nano_graphrag._op` is now a compatibility wrapper that re-exports the same public helpers.

### Storage Interfaces

The core storage contracts live in `nano_graphrag.base`:

- `BaseKVStorage`: document/chunk/report/cache storage
- `BaseVectorStorage`: embedding index
- `BaseGraphStorage`: graph read/write and clustering surface

Built-in implementations:

- KV: JSON files
- Vector: `nano-vectordb`, optional `hnswlib`
- Graph: `networkx`, optional Neo4j

Additional persisted state for incremental indexing:

- `full_docs`: logical document records keyed by caller-provided doc ID
- `document_index`: per-document chunk/entity/relationship manifest used to rebuild affected graph records on update

## Insert Flow

`GraphRAG.insert` / `ainsert` performs:

1. Normalize inputs into logical document IDs plus content hashes
   `insert` also checks for a legacy MD5-based doc ID before creating a new SHA-256 doc ID so older stores can be updated in place.
2. Skip unchanged logical documents by comparing stored content hashes
3. Build document-scoped chunks only for changed/new docs
4. Extract per-document entities/relationships into a manifest using stable SHA-256 entity IDs
5. Replace changed-document chunks and update the persisted `document_index`
6. Rebuild only affected graph nodes/edges by aggregating document manifests
   During rebuild, entities with the same normalized name can be remapped onto a canonical entity ID so a later `"UNKNOWN"` extraction does not fork the graph identity for an existing entity.
7. Recompute graph clustering
8. Regenerate only affected community reports when incremental clustering can stay local
9. Persist full docs, chunks, vectors, graph, manifests, and cache state

Important constraint:

- graph clustering falls back to full Leiden when there is no safe local frontier, but NetworkX now attempts frontier-only leaf reclustering for small affected neighborhoods
- community report regeneration can target only the affected community IDs after an incremental clustering pass

## Query Flow

### Local Query

- vector search over entity embeddings
- gather nearby entities, edges, text units, and relevant community reports
- build a structured context block
- ask the main model to answer with that context

### Global Query

- select community reports by level and occurrence
- map over community groups to extract support points
- reduce support points into the final answer

### Naive Query

- vector search over chunk embeddings
- truncate retrieved chunks to token budget
- answer directly from chunk context

## Extension Points

The intended extension points are:

- `embedding_func`
- `entity_extraction_func`
- `key_string_value_json_storage_cls`
- `vector_db_storage_cls`
- `graph_storage_cls`
- `chunk_func`

These hooks are supported, but they are considered advanced customization points rather than the default path.

## Optional Integrations

Optional integrations should stay outside the core runtime so the default package remains lightweight.
