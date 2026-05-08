# Nano-GraphRAG Pipeline Reference

This document describes the end-to-end data flow from raw documents to query answers. It covers what each stage does, how quality modes change behavior, and where the relevant code lives.

## Pipeline Overview

```
Documents
  │
  ▼
┌─────────────────┐
│  1. Ingest       │  Delta detection, content hashing
└────────┬────────┘
         ▼
┌─────────────────┐
│  2. Chunk        │  Tokenizer-aware text splitting
└────────┬────────┘
         ▼
┌─────────────────┐
│  3. Extract      │  Entity/relationship extraction (LLM or GLiNER)
└────────┬────────┘
         ▼
┌─────────────────┐
│  4. Link & Normalize │  Entity linking, alias enrichment, manifest assembly
└────────┬────────┘
         ▼
┌─────────────────┐
│  5. Writeback    │  Graph rebuild, vector store upsert
└────────┬────────┘
         ▼
┌─────────────────┐
│  6. Cluster      │  Leiden/Louvain community detection
└────────┬────────┘
         ▼
┌─────────────────┐
│  7. Report       │  Community report generation
└────────┬────────┘
         ▼
┌─────────────────┐
│  8. Query        │  Local / Global / Naive / Entity-Grounded
└─────────────────┘
```

---

## Stage 1: Ingest

**Entry point:** `GraphRAG.insert()` / `ainsert()`
**Code:** `graphrag_insert.py`

The insert flow normalizes input documents into a `dict[str, str]` mapping (doc_id → content). Each document gets a SHA-256 content hash. Delta detection compares stored hashes against new ones to skip unchanged documents.

Delta detection also checks the **extraction hash** — a hash of the extraction function name, LLM model, backend, and quality setting. If the extraction config changed since last insert (e.g., model switch), previously unchanged documents are re-extracted automatically.

### Insert Modes

| Mode | Trigger | Behavior |
|------|---------|----------|
| **Incremental** | Default | Processes only new/changed docs, rebuilds affected graph regions |
| **Force rebuild** | `force_rebuild=True` | Re-extracts all provided docs even if unchanged |
| **Legacy custom** | Non-builtin `entity_extraction_func` | Uses the old single-pass insert path |

### Concurrency Model

Two-layer semaphore:

```
doc_semaphore (doc_extraction_max_async=4)
  └── per-doc extraction
        └── chunk_semaphore (extraction_max_async=16)
              └── per-chunk/batch LLM call
```

Maximum concurrent LLM calls: 4 × 16 = 64.

### Progress Flushing

Extraction results flush to storage every `doc_flush_batch_size` (default 50) documents, enabling crash recovery. A graph snapshot is taken before rebuild so a failed rebuild can be rolled back.

### Integrity Check

After graph rebuild, an integrity check verifies that manifest entities actually exist in the graph. If zero entities are found, the insert rolls back and raises.

---

## Stage 2: Chunk

**Code:** `_ops/chunking.py`, `_splitter.py`

Documents are split into chunks using a tokenizer-aware `SeparatorSplitter`:

1. Split on paragraph boundaries (`\n\n`)
2. Split on line boundaries (`\n`)
3. Split on sentence boundaries (`. `, `! `, `? `)
4. Split on word boundaries (space)
5. Split character-by-character as last resort

Each chunk carries:
- `content`: the text
- `full_doc_id`: parent document ID
- `chunk_order_index`: position within the document

Chunks are token-bounded (`chunk_token_size`, default 1200) with configurable overlap (`chunk_overlap_token_size`, default 100).

---

## Stage 3: Extract

**Code:** `_ops/extraction.py` (router), `_ops/extraction_structured.py`, `_ops/extraction_legacy.py`, `_ops/extraction_gliner.py`, `_ops/extraction_common.py`

The router (`extract_document_entity_relationships`) chooses a backend based on config:

| Config | Backend | Method |
|--------|---------|--------|
| `_use_structured_extraction=True` | `extraction_structured.py` | Pydantic-structured JSON output from LLM |
| default | `extraction_legacy.py` | Prompt + free-text parsing |
| `extraction_backend="gliner"` | `extraction_gliner.py` | GLiNER2 model (no LLM) |

### Quality Modes

The `entity_extraction_quality` setting (`"fast"` or `"balanced"`) controls cost/quality tradeoffs throughout extraction:

| Behavior | `fast` | `balanced` |
|----------|--------|------------|
| **LLM model** | `cheap_model_func` | `best_model_func` |
| **System prompt** | Minimal (entity types + JSON schema) | Full assistant prompt with examples |
| **Aliases** | Not extracted | Extracted via alias enrichment pass |
| **Temporal fields** | Not extracted | Extracted if `enable_temporal_extraction=True` |
| **Gleaning** | Disabled (`max_gleaning=0`) | Enabled (default 1 round) |
| **Entity summary** | Truncate tokens | LLM-generated summary |
| **Batch size** | Auto-bumped to 8 if below | User-configured (default 5) |

### Structured Extraction Path

1. Chunks are grouped into batches of `extraction_batch_size`
2. Each batch is sent as a single LLM call with a numbered-chunk format
3. The LLM returns structured JSON parsed into `BatchedEntityExtractionOutput`
4. If structured output fails, falls back to single-chunk extraction
5. If single-chunk fails, falls back to legacy prompt parsing (with gleaning disabled in fast mode)

### Legacy Extraction Path

1. Each chunk gets the full `entity_extraction` prompt with few-shot examples
2. Gleaning loop: ask LLM "are there more entities?", up to `entity_extract_max_gleaning` rounds
3. Parse free-text output using tuple delimiters into entity/relationship records

### GLiNER Extraction Path

1. Load GLiNER2 model (lazy, cached globally)
2. Extract entities by type using the model's `extract()` method
3. Build relationships from GLiNER's relation extraction output
4. No LLM calls — entirely local inference

### Manifest Assembly

All extraction backends return a **manifest** — a dict with:

```python
{
    "chunk_ids": ["chunk-0", "chunk-1", ...],
    "entities": {
        "<entity_id>": {
            "entity_name": "...",
            "entity_type": "...",
            "descriptions": ["..."],
            "source_chunk_ids": ["..."],
            "aliases": ["..."],
        }
    },
    "relationships": {
        "<relationship_id>": {
            "src_entity_id": "...",
            "tgt_entity_id": "...",
            "relation_type": "...",
            "descriptions": ["..."],
            "weight": float,
            "source_chunk_ids": ["..."],
        }
    }
}
```

The shared helper `_merge_results_into_manifest()` (in `extraction_common.py`) accumulates per-chunk results into a single manifest, merging duplicate entities (by ID) and accumulating relationship weights. The manifest is then normalized via `_normalize_document_manifest()`.

### Shared Extraction Helpers

These helpers in `extraction_common.py` are used by all extraction backends:

| Helper | Purpose |
|--------|---------|
| `_merge_results_into_manifest()` | Accumulate per-chunk results into document manifest |
| `_run_gleaning_loop()` | Iterative "are there more?" LLM loop |
| `_build_extraction_system_prompt()` | Build quality-aware system prompt |
| `_build_temporal_instructions()` | Build temporal extraction instructions |
| `_ExtractionProgress` | Track and log chunk processing progress |
| `_handle_entity_relation_summary()` | Summarize long entity descriptions |

---

## Stage 4: Link & Normalize

**Code:** `_ops/extraction.py` (`_enrich_manifest_aliases`, `_apply_entity_linking`)

After extraction produces a raw manifest, two post-processing passes run:

### Alias Enrichment (balanced mode only)

For entities without aliases, a batch LLM call extracts alternative names, abbreviations, and nicknames from the source chunks. Skipped entirely in `fast` mode.

### Entity Linking

When `enable_entity_linking=True` and an `EntityRegistry` is available:

1. **Exact match** — check if entity name exists in registry
2. **Fuzzy candidates** — find similar names above `similarity_threshold` (default 0.92)
3. **Neighborhood evidence** — if a single candidate exists, compare graph neighbors using IoU. If IoU ≥ threshold and enough common neighbors exist, auto-link
4. **Heuristic resolution** — match on aliases or exact name + type
5. **LLM disambiguation** — for genuinely ambiguous cases (multiple candidates), ask the LLM to decide. Includes structural evidence (neighbor overlap) in the prompt. Skipped in `fast` mode

When entities are linked, their IDs are remapped across the manifest, and relationship endpoints are updated to use the canonical entity ID.

---

## Stage 5: Writeback

**Code:** `_ops/extraction_writeback.py`, `_ops/extraction_rebuild.py`

### Graph Rebuild

Rather than incrementally modifying the graph, the rebuild path:

1. Reads all document manifests from `document_index`
2. For each entity, aggregates descriptions and source chunk IDs across all documents that reference it
3. For each relationship, sums weights and collects all descriptions
4. Writes the aggregated state to the graph storage (nodes, edges)
5. Upserts entity embeddings into the vector store

The `graph_contribution_index` provides the reverse mapping: entity ID → contributing document IDs. This enables efficient incremental rebuilds that only touch entities affected by changed documents.

---

## Stage 6: Cluster

**Code:** `_storage/gdb_networkx_clustering.py`

Community detection via Leiden (default) or Louvain algorithm. The clustering produces a `community_schema` mapping community IDs to their member nodes, edges, sub-communities, and occurrence counts.

Incremental clustering: when `affected_node_ids` is provided, the algorithm attempts frontier-only leaf reclustering for small affected neighborhoods. Falls back to full reclustering when no safe local frontier exists.

---

## Stage 7: Report

**Code:** `_ops/community.py`

For each community, generate a structured report containing:
- Title, summary, impact severity rating
- Detailed findings (5-10 key insights)

The report prompt includes entity/relationship data packed as CSV. Sub-communities can optionally be included as context for higher-level reports (when `force_to_use_sub_communities=True`).

Incremental report generation: when `only_community_ids` is specified, only affected communities are regenerated.

---

## Stage 8: Query

**Code:** `graphrag_query.py` (dispatcher), `_ops/query.py` (implementations)

Four query modes share a context-builder + dispatcher pattern:

### Local Query

```
Query → vector search (entity embeddings) → top-k entities
      → gather related edges, text units, community reports
      → build CSV context (entities, relationships, reports, sources)
      → LLM answer generation
```

**Context builder:** `_build_local_query_context()` in `query.py`
Returns structured CSV context or `None` if no entities found.

### Global Query

```
Query → select community reports by level/occurrence
      → map: extract support points from each community group
      → reduce: rank and truncate points
      → LLM synthesis from analyst reports
```

**Context builder:** `_build_global_query_context()` in `query.py`
Returns formatted analyst report text or `None`.

The map step uses `_map_global_communities()` which batches community data into groups fitting within token limits, then extracts scored points from each group in parallel.

### Naive Query

```
Query → vector search (chunk embeddings) → top-k chunks
      → truncate to token budget
      → LLM answer from chunk context
```

**Context builder:** `_build_naive_query_context()` in `query.py`
Returns joined chunk text or `None`.

### Entity-Grounded Query

```
Query → extract entity mentions → resolve to canonical IDs
      → graph traversal for related entities
      → generate answer with formatting constraints
      → validate answer against retrieved entities
```

**Code:** `_entity_grounded_query.py`
Provides enhanced entity resolution and answer validation.

### Streaming

All four modes have a `*_stream` variant (`local_query_stream`, etc.) that:
- Shares the same context builder as the non-stream variant
- Streams the final LLM answer as text chunks
- Falls back to buffered completion if the model doesn't support streaming natively

---

## Storage Backends

| Type | Interface | Default | Alternatives |
|------|-----------|---------|-------------|
| Key-Value | `BaseKVStorage` | SQLite (`kv_json.py`) | — |
| Vector | `BaseVectorStorage` | HNSW (`vdb_hnswlib.py`) | NanoVectorDB (`vdb_nanovectordb.py`) |
| Graph | `BaseGraphStorage` | NetworkX (`gdb_networkx.py`) | SQLite (`gdb_sqlite.py`), Neo4j (`gdb_neo4j.py`) |

### Persisted State

| Store | Contents |
|-------|----------|
| `full_docs` | Document records keyed by doc ID |
| `text_chunks` | Chunk records with content + metadata |
| `document_index` | Per-document extraction manifest |
| `graph_contribution_index` | Entity/relationship → document ID reverse lookup |
| `entities_vdb` | Entity embedding vectors |
| `chunks_vdb` | Chunk embedding vectors (for naive query) |
| `community_reports` | Generated community reports |
| `llm_response_cache` | Cached LLM responses |
| `entity_registry.json` | Canonical entity ID → name/type/aliases |

---

## Configuration

**Code:** `_config.py` (`GraphRAGSettings`), `base.py` (`GraphRAGConfig`)

Config can be loaded from:
- Constructor kwargs: `GraphRAGConfig(llm_model="gpt-4o", ...)`
- YAML file: `GraphRAGConfig.from_yaml("config.yaml")`
- Environment variables: via pydantic-settings with `FLAT_FIELD_TO_ENV_VAR` mapping
- Defaults: `config/settings.yaml`

### Key Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `entity_extraction_quality` | `"balanced"` | `"fast"` (cheap, minimal) or `"balanced"` (full) |
| `extraction_backend` | `"llm"` | `"llm"` or `"gliner"` |
| `extraction_batch_size` | 5 | Chunks per LLM call (fast mode auto-bumps to 8) |
| `doc_extraction_max_async` | 4 | Max concurrent document extractions |
| `extraction_max_async` | 16 | Max concurrent chunk-level LLM calls |
| `graph_cluster_algorithm` | `"leiden"` | `"leiden"` or `"louvain"` |
| `enable_entity_linking` | `False` | Enable cross-document entity resolution |
| `enable_temporal_extraction` | `False` | Extract temporal fields (event_date, valid_from/to) |
| `enable_community_reports` | `True` | Generate community reports after clustering |
| `chunk_token_size` | 1200 | Max tokens per chunk |
| `chunk_overlap_token_size` | 100 | Overlap tokens between chunks |

---

## LLM Integration

**Code:** `_llm_litellm.py`

All LLM calls go through `litellm_completion()` which provides:
- Async execution via `litellm.acompletion()`
- Exponential backoff (3 retries, base delay 2s)
- Structured output via Pydantic models with provider-specific fallback
- Response caching via `llm_response_cache`

Two model functions are configured:
- `best_model_func` — used for balanced extraction, community reports, query answering
- `cheap_model_func` — used for fast extraction, entity summaries, entity linking disambiguation
