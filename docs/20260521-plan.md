# Nano-GraphRAG UX Improvement Plan

> Date: 2026-05-18
> Status: Draft
> Scope: Comprehensive UX/DX overhaul across API surface, CLI, observability, configuration, and documentation

## Executive Summary

This plan addresses every UX pain point identified through codebase analysis and competitive research against LlamaIndex, Haystack, LangChain, LightRAG, and Microsoft GraphRAG. Work is organized into 6 phases with dependency tracking. Phases 1-2 are foundational and should be completed before later phases.

---

## Current State Assessment

### Strengths
- Async-first API with `insert`/`query`/`refine`/`export_vault` lifecycle
- Incremental extraction with content-hash delta detection
- Structured logging via structlog
- Multiple storage backends (NetworkX, SQLite, Neo4j, HNSW)
- Batched entity extraction with two-layer concurrency
- Refinement pipeline with journal + rejection cache
- Vault export for Obsidian

### Critical UX Gaps (from analysis)

| Gap | Severity | Source |
|-----|----------|--------|
| Zero docstrings on `GraphRAG` public methods | HIGH | Code audit |
| No CLI — Python-only interaction | HIGH | Competitive: MS GraphRAG has full CLI |
| Query returns raw `str`, no metadata | HIGH | Competitive: all rivals return structured results |
| Triple config representation (`_ConfigFields` + `GraphRAG` + `GraphRAGSettings`) | MEDIUM | Code audit |
| No progress callbacks/hooks | MEDIUM | Competitive: Rich/tqdm integration is standard |
| No custom exception hierarchy | MEDIUM | Code audit |
| Mode gating errors at query time, not init | MEDIUM | Code audit: `graphrag_query.py:18-24` |
| Cost/token tracking logged but not returned | MEDIUM | Code audit |
| No `__all__` in `__init__.py` | LOW | Code audit |
| Examples don't cover new features | LOW | Code audit |
| No built-in visualization | LOW | Competitive: LightRAG has Web UI |

---

## Phase 1: Exception Hierarchy & Structured Results

> **Goal:** Give users typed errors they can catch and structured results they can inspect.
> **Files:** New `_exceptions.py`, modified `_schemas.py`, `graphrag_query.py`, `graphrag_insert.py`, `graphrag.py`
> **Breaking:** No — `QueryResult.__str__()` returns the answer text for backward compat.

### 1.1 Custom Exception Hierarchy (`_exceptions.py` — new file)

```
GraphRAGError (base)
├── ConfigError              — Invalid configuration (model names, batch sizes, etc.)
│   └── StorageConfigError   — Backend-specific config issues
├── ExtractionError          — Entity extraction failures
│   ├── LLMExtractionError   — LLM call failures during extraction
│   └── ParsingError         — Structured output parsing failures
├── QueryError               — Query-time failures
│   ├── ModeNotEnabledError  — Mode gating (enable_local, enable_naive_rag)
│   └── NoContextError        — No relevant context found
├── StorageError              — Storage backend failures
│   ├── GraphIntegrityError   — Graph/manifest mismatch
│   └── VectorDBError         — Vector DB failures
└── LLMError                  — LiteLLM wrapper errors
    ├── RateLimitError        — Rate limit exceeded (retries exhausted)
    └── AuthError             — API key issues
```

**Implementation notes:**
- Each exception carries `message`, `details: dict` (structured context), and optional `cause`.
- `LLMError` wraps litellm exceptions with user-friendly messages.
- `GraphIntegrityError` replaces the current `RuntimeError("Integrity check failed")` in `graphrag_insert.py`.

### 1.2 `QueryResult` Dataclass (in `_schemas.py`)

```python
@dataclass
class QueryResult:
    answer: str
    mode: str
    sources: list[QuerySource]         # entities, communities, chunks used
    tokens_used: TokenUsage | None     # prompt + completion tokens
    latency_ms: float
    metadata: dict                      # mode-specific extras

    def __str__(self) -> str:
        return self.answer
```

**`QuerySource` sub-model:**
```python
@dataclass
class QuerySource:
    source_type: Literal["entity", "community", "chunk"]
    id: str
    name: str | None          # entity name or community title
    relevance_score: float | None
    text_snippet: str | None  # first 200 chars
```

**`TokenUsage` sub-model:**
```python
@dataclass
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float | None
```

### 1.3 `InsertResult` Dataclass (in `_schemas.py`)

```python
@dataclass
class InsertResult:
    documents_processed: int
    documents_skipped: int       # unchanged via delta detection
    entities_created: int
    relationships_created: int
    communities_updated: int
    tokens_used: TokenUsage | None
    latency_ms: float
```

### 1.4 Eager Validation in `__post_init__`

Move validation from "fail at runtime" to "fail at construction":

```python
# In _normalize_settings (graphrag_runtime.py):
if self.entity_extraction_quality not in ("fast", "balanced"):
    raise ConfigError(...)
if self.extraction_batch_size < 1:
    raise ConfigError(...)
if self.embedding_dim < 1:
    raise ConfigError(...)
```

Mode gating moves from `_check_mode_permissions` (query time) to `__post_init__`:
```python
# Store requested modes; validate at init
if not self.enable_local and not self.enable_naive_rag:
    logger.warning("Both local and naive modes disabled; only global queries available.")
```

### 1.5 Wire Exceptions Throughout

Replace all `ValueError`/`RuntimeError`/raw litellm exceptions:
- `graphrag_query.py:18-24` → `ModeNotEnabledError`
- `graphrag_insert.py` integrity check → `GraphIntegrityError`
- `_llm_litellm.py` retry exhaustion → `RateLimitError`/`AuthError`
- `_config.py` validation → `ConfigError`

### 1.6 Tests

- Test each exception class has correct `message` and `details`.
- Test `QueryResult.__str__()` returns answer (backward compat).
- Test eager validation rejects invalid configs.

---

## Phase 2: Callback/Hook System & Progress Tracking

> **Goal:** Users can attach tqdm bars, Rich progress, web UI updates, or Langfuse tracing.
> **Files:** New `_callbacks.py`, modified `graphrag_insert.py`, `graphrag_query.py`, `graphrag.py`
> **Breaking:** No — callbacks are opt-in.

### 2.1 Callback Protocol (`_callbacks.py` — new file)

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class ExtractionCallback(Protocol):
    async def on_start(self, total_docs: int) -> None: ...
    async def on_doc_progress(self, committed: int, total: int) -> None: ...
    async def on_chunk_extracted(self, doc_id: str, chunks: int) -> None: ...
    async def on_community_report(self, level: int, count: int) -> None: ...
    async def on_complete(self, result: InsertResult) -> None: ...
    async def on_error(self, error: GraphRAGError) -> None: ...

@runtime_checkable
class QueryCallback(Protocol):
    async def on_start(self, query: str, mode: str) -> None: ...
    async def on_sources_found(self, sources: list[QuerySource]) -> None: ...
    async def on_complete(self, result: QueryResult) -> None: ...

@runtime_checkable
class LLMCallback(Protocol):
    async def on_call(self, model: str, prompt_tokens: int, completion_tokens: int, latency_ms: float) -> None: ...
```

### 2.2 Built-in Callback Implementations

```python
class RichProgressCallback:
    """Shows Rich progress bars during extraction."""

class TqdmProgressCallback:
    """Shows tqdm progress bars during extraction."""

class TokenTrackingCallback:
    """Accumulates token usage; accessible via rag.token_tracker.summary()."""

class LoggingCallback:
    """Default: emits structlog events (current behavior, no change)."""
```

### 2.3 Wiring into `GraphRAG`

Add `callbacks` parameter to `GraphRAG.__init__`:

```python
@dataclass
class GraphRAG(_ConfigFields):
    ...
    callbacks: list[ExtractionCallback | QueryCallback | LLMCallback] = field(default_factory=list)
```

In `_ainsert_documents`, emit events:
```python
for cb in self._extraction_callbacks:
    await cb.on_doc_progress(committed=committed, total=total)
```

### 2.4 Token Tracking Return

`TokenTrackingCallback` accumulates all LLM call metrics. After `insert()` or `query()`, users can:

```python
rag = GraphRAG(working_dir="...", callbacks=[TokenTrackingCallback()])
result = rag.insert("some text")
print(rag.token_tracker.summary())  # {"prompt_tokens": 5000, "completion_tokens": 1200, "cost_usd": 0.03}
```

### 2.5 Tests

- Test `LoggingCallback` emits correct structlog events.
- Test `TokenTrackingCallback` accumulates correctly.
- Test callbacks don't crash on user-side errors (wrapped in try/except).

---

## Phase 3: CLI with Typer + Rich

> **Goal:** `nano-graphrag insert/query/config/status` CLI for quick exploration.
> **Files:** New `src/nano_graphrag/_cli.py`, modified `pyproject.toml` (console_scripts)
> **Breaking:** No — CLI is additive.

### 3.1 Dependencies

Add to core dependencies (or optional `[cli]` extra):
```toml
dependencies = [
    ...,
    "typer>=0.12.0",
    "rich>=13.0.0",
]
```

Or as optional:
```toml
[project.optional-dependencies]
cli = ["typer>=0.12.0", "rich>=13.0.0"]
```

### 3.2 CLI Commands

```bash
# Initialize config
nano-graphrag config init                    # Writes settings.yaml with defaults + prompts for API key
nano-graphrag config show                    # Prints current config (redacts secrets)
nano-graphrag config validate                # Validates config without running

# Insert documents
nano-graphrag insert ./documents/            # Insert all .txt/.md files in directory
nano-graphrag insert document.txt            # Insert single file
nano-graphrag insert --stdin                 # Read from stdin
nano-graphrag insert ./docs/ --recursive     # Recursive directory scan

# Query
nano-graphrag query "What is GraphRAG?"      # Default global mode
nano-graphrag query "..." --mode local       # Local mode
nano-graphrag query "..." --mode naive       # Naive mode
nano-graphrag query "..." --stream           # Streaming output
nano-graphrag query "..." --explain          # Show sources + reasoning

# Status
nano-graphrag status                         # Graph stats: entities, communities, storage sizes
nano-graphrag status --verbose               # Detailed: entity types, community levels, last insert

# Refine
nano-graphrag refine                         # Run all refinement phases
nano-graphrag refine --phases merge,enrich   # Run specific phases

# Export
nano-graphrag export vault --output ./vault  # Export to Obsidian vault

# Graph operations
nano-graphrag rebuild                        # Rebuild graph from manifests
```

### 3.3 Rich Integration

- `insert` shows progress bar with document count
- `query --explain` shows source table with Rich `Table`
- `status` shows entity count bar chart with Rich `BarColumn`
- `config show` uses Rich `Panel` with syntax highlighting
- Errors use Rich tracebacks with error codes and documentation links

### 3.4 `console_scripts` Entry

```toml
[project.scripts]
nano-graphrag = "nano_graphrag._cli:app"
```

### 3.5 Tests

- Test each CLI command with `typer.testing.CliRunner`
- Test config init creates valid YAML
- Test status with empty/ populated working dir

---

## Phase 4: Config Simplification & Documentation

> **Goal:** Reduce config complexity and document every public API.
> **Files:** `base.py`, `_config.py`, `graphrag_runtime.py`, all public modules
> **Breaking:** Potentially yes — deprecated field removal. Guard behind `__future__` flag or version bump.

### 4.1 Remove Deprecated Fields

- Remove `embedding_batch_num` (deprecated, replaced by `embedding_batch_size`)
- Remove `llm_max_async` and `embedding_max_async` (override fields with non-obvious side effects)
- Users should set `best_model_max_async`/`cheap_model_max_async`/`embedding_func_max_async` directly

### 4.2 String-Based Storage Selection

Add storage registry for string-based selection:

```python
_STORAGE_REGISTRY = {
    "json": JsonKVStorage,
    "sqlite": SQLiteKVStorage,
    "hnsw": HNSWVectorStorage,
    "networkx": NetworkXStorage,
    "sqlite_graph": SQLiteGraphStorage,
    # Optional (import-guarded):
    "neo4j": "nano_graphrag._storage.gdb_neo4j:Neo4jStorage",
    "milvus": "nano_graphrag._storage.vdb_milvus:MilvusVectorStorage",
    "qdrant": "nano_graphrag._storage.vdb_qdrant:QdrantVectorStorage",
}

class GraphRAG(_ConfigFields):
    key_string_value_json_storage_cls: type[BaseKVStorage] | str | None = None
    vector_db_storage_cls: type[BaseVectorStorage] | str | None = None
    graph_storage_cls: type[BaseGraphStorage] | str | None = None
```

In `__post_init__`, resolve strings:
```python
if isinstance(self.graph_storage_cls, str):
    self.graph_storage_cls = _resolve_storage(self.graph_storage_cls)
```

### 4.3 Docstrings on All Public Methods

Priority list (every method gets docstrings with args, returns, examples):

1. `GraphRAG.__init__` — document all params with defaults
2. `GraphRAG.insert` / `ainsert` — document input types, delta detection, InsertResult
3. `GraphRAG.insert_documents` / `ainsert_documents` — document doc ID behavior
4. `GraphRAG.query` / `aquery` — document modes, QueryParam, QueryResult
5. `GraphRAG.astream_query` — document async generator, StreamEvent types
6. `GraphRAG.refine` / `arefine` — document phases, return dict
7. `GraphRAG.export_vault` / `aexport_vault` — document output structure
8. `GraphRAG.rebuild_graph` / `arebuild_graph` — document manifest-based rebuild
9. `GraphRAG.from_config` — document config loading paths
10. `QueryParam` — document every field with mode-specific relevance
11. `GraphRAGConfig` — document `from_env`, `from_yaml`, `from_dict`, `merge`

### 4.4 `__all__` in `__init__.py`

```python
__all__ = [
    "GraphRAG",
    "GraphRAGConfig",
    "GraphRAGSettings",
    "QueryParam",
    "QueryResult",
    "InsertResult",
    "ResponseType",
    "LiteLLMWrapper",
    "ExtractedEntity",
    "ExtractedRelationship",
    "RELATION_VOCABULARY",
    "RELATION_ALIASES",
    # Exceptions
    "GraphRAGError",
    "ConfigError",
    "ExtractionError",
    "QueryError",
    "StorageError",
    "LLMError",
]
```

### 4.5 Examples Refresh

Create new examples covering:

| Example | Demonstrates |
|---------|-------------|
| `example_query_structured.py` | `QueryResult` with sources and tokens |
| `example_insert_with_progress.py` | Callback system with Rich progress |
| `example_config_yaml.py` | `GraphRAGConfig.from_yaml()` + `from_config()` |
| `example_streaming_query.py` | `astream_query` async generator |
| `example_refinement.py` | `arefine` with phase selection |
| `example_vault_export.py` | `aexport_vault` with custom path |
| `example_string_storage.py` | `GraphRAG(graph_storage="sqlite")` |
| `example_cost_tracking.py` | `TokenTrackingCallback` for cost monitoring |

### 4.6 Tests

- Test string-based storage resolution for all registered backends.
- Test `__all__` exports are importable.
- Test deprecated fields raise `DeprecationWarning` (if kept temporarily) or `ConfigError` (if removed).

---

## Phase 5: Query Explainability & Streaming Metadata

> **Goal:** Users can trace which entities/communities informed an answer and get metadata during streaming.
> **Files:** `graphrag_query.py`, `_ops/query.py`, `_schemas.py`
> **Breaking:** Streaming changes are additive; `astream_query` gains a new event type.

### 5.1 Query Tracing in `_ops/query.py`

Each query function (`local_query`, `global_query`, `naive_query`) currently returns `str`. Modify to return a tuple `(answer: str, trace: QueryTrace)` internally, then wrap in `QueryResult` at the `GraphRAG.aquery` level.

```python
@dataclass
class QueryTrace:
    entities_matched: list[str]           # entity IDs that matched the query
    communities_used: list[str]           # community IDs whose reports were used
    chunks_used: list[str]                # chunk IDs that contributed context
    retrieval_scores: dict[str, float]    # entity_id → similarity score
    mode_specific: dict                   # extra info per mode (e.g., global: community ratings)
```

### 5.2 `query(..., explain=True)` Mode

When `QueryParam.explain = True`, the returned `QueryResult.sources` is populated with full `QuerySource` objects. When `False` (default), `sources` is empty (backward compat, no extra LLM calls).

### 5.3 Streaming with Metadata (`StreamEvent`)

Replace raw string yielding with typed events:

```python
@dataclass
class StreamTextChunk:
    text: str

@dataclass
class StreamSourceRef:
    sources: list[QuerySource]

@dataclass
class StreamComplete:
    trace: QueryTrace
    tokens: TokenUsage | None
    latency_ms: float

StreamEvent = StreamTextChunk | StreamSourceRef | StreamComplete
```

`astream_query` yields `StreamEvent`:
```python
async for event in rag.astream_query("...", param):
    match event:
        case StreamTextChunk(text): print(text, end="")
        case StreamSourceRef(sources): display_source_table(sources)
        case StreamComplete(trace, tokens, latency): print_summary(trace, tokens)
```

**Backward compat:** `async for chunk in rag.astream_query(...)` still works because `StreamTextChunk.__str__()` returns `text`. But users who check `isinstance(chunk, StreamEvent)` get structured data.

### 5.4 Tests

- Test `QueryTrace` is populated for each mode.
- Test `explain=True` vs `explain=False` result differences.
- Test `StreamEvent` types in `astream_query`.
- Test backward compat: iterating `astream_query` as strings still works.

---

## Phase 6: Visualization & Interactive Features

> **Goal:** Built-in graph visualization and status dashboard.
> **Files:** New `_visualization.py`, new `_dashboard.py`
> **Breaking:** No — entirely additive.

### 6.1 Graph Visualization (`_visualization.py`)

```python
async def visualize_graph(graph_storage, output: str = "graph.html", **kwargs) -> str:
    """Generate interactive HTML visualization using pyvis/networkx.

    Args:
        output: Output HTML file path.
        kwargs: Passed to pyvis (physics, node_size, etc.)

    Returns:
        Path to generated HTML file.
    """
```

Implementation:
- Use `pyvis` (optional dep) or generate standalone HTML with D3.js embedded.
- Color nodes by entity type.
- Size nodes by degree.
- Click node to see entity description.
- Highlight communities with background shading.

CLI: `nano-graphrag visualize --output graph.html`

### 6.2 Status Dashboard (`_dashboard.py`)

```python
async def status(working_dir: str) -> GraphStatus:
    """Return comprehensive graph status without loading full graph."""

@dataclass
class GraphStatus:
    entity_count: int
    entity_types: dict[str, int]        # {"PERSON": 45, "ORG": 23, ...}
    relationship_count: int
    community_count: int
    community_levels: dict[int, int]    # {0: 12, 1: 5, 2: 2}
    storage_size_bytes: int
    last_insert_timestamp: str | None
    document_count: int
    chunk_count: int
    health: Literal["healthy", "warning", "error"]
    warnings: list[str]                 # e.g., "empty graph", "missing community reports"
```

CLI: `nano-graphrag status` outputs Rich-formatted table.

API: `rag.status()` → `GraphStatus`

### 6.3 Jupyter/Marimo Display Hook

```python
def _repr_html_(self) -> str:
    """Rich display for Jupyter notebooks."""
    status = self.status()
    return f"""
    <div style="...">
        <h3>nano-graphrag: {status.entity_count} entities, {status.relationship_count} relationships</h3>
        ...
    </div>
    """
```

### 6.4 `nano-graphrag query --explain` Rich Output

When CLI `--explain` flag is set:
- Show query answer in Rich `Panel`
- Show sources in Rich `Table` with columns: Type, Name, Relevance, Snippet
- Show token usage and cost in Rich `Columns`

### 6.5 Dependencies

```toml
[project.optional-dependencies]
viz = ["pyvis>=2.0"]
```

### 6.6 Tests

- Test `visualize_graph` generates valid HTML.
- Test `GraphStatus` with empty/populated working dirs.
- Test `_repr_html_` output contains key metrics.

---

## Implementation Order & Dependencies

```
Phase 1 (Exceptions + Structured Results)
  ├── 1.1 Exception hierarchy          [no deps]
  ├── 1.2 QueryResult dataclass        [no deps]
  ├── 1.3 InsertResult dataclass       [no deps]
  ├── 1.4 Eager validation             [depends on 1.1]
  ├── 1.5 Wire exceptions              [depends on 1.1, 1.2, 1.3]
  └── 1.6 Tests                        [depends on all above]

Phase 2 (Callbacks & Progress)          [depends on Phase 1 for types]
  ├── 2.1 Callback protocols           [depends on 1.2, 1.3]
  ├── 2.2 Built-in callbacks           [depends on 2.1]
  ├── 2.3 Wire into GraphRAG           [depends on 2.1]
  ├── 2.4 Token tracking               [depends on 2.1]
  └── 2.5 Tests                        [depends on all above]

Phase 3 (CLI)                           [depends on Phase 2 for progress]
  ├── 3.1 Dependencies                 [no deps]
  ├── 3.2 CLI commands                  [depends on 1.1, 1.2, 1.3]
  ├── 3.3 Rich integration             [depends on 3.2, 2.2]
  ├── 3.4 console_scripts              [depends on 3.2]
  └── 3.5 Tests                        [depends on all above]

Phase 4 (Config + Docs)                 [partially independent]
  ├── 4.1 Remove deprecated fields     [breaking — coordinate with version]
  ├── 4.2 String-based storage         [no deps]
  ├── 4.3 Docstrings                   [depends on 1.2, 1.3 for new types]
  ├── 4.4 __all__                      [depends on 1.1, 1.2, 1.3]
  ├── 4.5 Examples refresh             [depends on all above]
  └── 4.6 Tests                        [depends on all above]

Phase 5 (Explainability)                [depends on Phase 1 for types]
  ├── 5.1 Query tracing                [depends on 1.2]
  ├── 5.2 explain=True mode            [depends on 5.1]
  ├── 5.3 StreamEvent types            [depends on 1.2]
  └── 5.4 Tests                        [depends on all above]

Phase 6 (Visualization)                 [depends on Phase 3 for CLI]
  ├── 6.1 Graph visualization          [no deps]
  ├── 6.2 Status dashboard             [depends on 1.2 for GraphStatus]
  ├── 6.3 Jupyter display hook         [depends on 6.2]
  ├── 6.4 CLI explain output           [depends on 5.2, 3.3]
  ├── 6.5 Dependencies                 [no deps]
  └── 6.6 Tests                        [depends on all above]
```

### Recommended Execution Sequence

```
Sprint 1: Phase 1 (all) + Phase 4.2 (string storage) + Phase 4.4 (__all__)
Sprint 2: Phase 2 (all) + Phase 4.3 (docstrings) — can be parallelized
Sprint 3: Phase 3 (CLI) + Phase 4.5 (examples)
Sprint 4: Phase 5 (explainability)
Sprint 5: Phase 6 (visualization) + Phase 4.1 (deprecation cleanup)
```

---

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| `QueryResult` breaks existing `isinstance(result, str)` | `QueryResult` inherits from `str` or implements `__str__`; add deprecation period |
| CLI dependencies bloat core install | Make `typer`/`rich`/`pyvis` optional extras (`[cli]`, `[viz]`) |
| Callback system adds overhead | Measure overhead in tests; skip callbacks when list is empty (zero-cost abstraction) |
| Streaming API change breaks users | `StreamEvent.__str__` returns text; old `async for chunk in ...` still works |
| Config simplification is breaking | Phase behind version bump (0.1.0); provide migration guide |

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Public methods with docstrings | 0/8 | 8/8 |
| Custom exception types | 0 | 8+ |
| CLI commands | 0 | 8+ |
| Query return type | `str` | `QueryResult` (str-compat) |
| Insert return type | `None` | `InsertResult` |
| Progress feedback | Log lines only | Callbacks + Rich bars |
| Config validation | 1 field | All fields |
| Examples covering features | 4/12 features | 12/12 features |
| `__all__` exports | 0 | All public names |
