# UX Sprint Audit: Next Steps Implementation Plan

**Date:** 2026-05-20
**Status:** Ready for Implementation
**Scope:** 8 steps to fix gaps, bugs, and missing wiring from the UX/DX improvement sprint (Phases 1-4)

## Context

A comprehensive audit was performed on uncommitted changes across:
- 15 modified source files
- 4 new modules (`_exceptions.py`, `_callbacks.py`, `_visualization.py`, `_cli.py`)
- ~20 new test files (3,400+ lines)
- ~3,400 lines added, ~74 lines removed from source

Research was conducted via Google AI Mode and web sources on three topics:
1. **Exception wrapping** — httpx lookup-dict + context manager pattern (`raise ... from exc`)
2. **Breaking API changes** — `FutureWarning` + `__getattr__` proxy (Pydantic v2, PEP 702)
3. **Callback dispatchers** — Null Object pattern (OpenTelemetry `NoOpTracer`, sentry `NonRecordingClient`)

**Key decision:** The user confirmed **no backward compatibility shim** is needed for `QueryResult` (was `str`). The clean breaking change stands — users migrate to `result.answer`.

---

## Execution Order

Steps 1, 4, 5, 6, 8 are independent and can be done in parallel.
Steps 2 is SKIPPED (no backward compat needed).
Step 3 is the largest — phase it A→B→C.
Step 7 should follow Step 3 (tests reference new exceptions).

---

## Step 1: Replace `hasattr` Guards with Null Object Dispatcher

**Priority:** High (foundational)
**Effort:** Small (~20 lines new, 8 lines removed)
**Files:** `_callbacks.py`, `graphrag.py`, `graphrag_insert.py`, `graphrag_query.py`

### What to do

1. In `_callbacks.py`, add a `_NullDispatcher` class that inherits from the same interface but all methods are `pass`:

```python
class _NullDispatcher:
    """Singleton no-op dispatcher. All methods are no-ops."""

    async def extraction_start(self, total_docs: int) -> None: pass
    async def extraction_progress(self, committed: int, total: int) -> None: pass
    async def extraction_complete(self, result) -> None: pass
    async def extraction_error(self, error: Exception) -> None: pass
    async def query_start(self, query: str, mode: str) -> None: pass
    async def query_complete(self, result) -> None: pass
    async def query_sources_found(self, sources: list) -> None: pass
    async def community_report(self, level: int, count: int) -> None: pass
    async def llm_call(self, model: str, prompt_tokens: int, completion_tokens: int, latency_ms: float) -> None: pass
    async def chunk_extracted(self, doc_id: str, chunks: int) -> None: pass
```

2. In `graphrag.py` `__post_init__`, change the callback initialization:

```python
# Replace:
from ._callbacks import _CallbackDispatcher
self._callback_dispatcher = _CallbackDispatcher(self.callbacks)

# With:
from ._callbacks import _CallbackDispatcher, _NullDispatcher
self._callback_dispatcher = (
    _CallbackDispatcher(self.callbacks) if self.callbacks else _NullDispatcher()
)
```

3. Remove all 8 `hasattr(self, "_callback_dispatcher")` guards in:
   - `graphrag_insert.py:358-359` — `extraction_progress`
   - `graphrag_insert.py:371-372` — `extraction_start`
   - `graphrag_insert.py:457-458` — `community_report`
   - `graphrag_insert.py:542-543` — `extraction_complete`
   - `graphrag_query.py:93-94` — `query_start`
   - `graphrag_query.py:145-146` — `query_complete`
   - `graphrag_query.py:181-182` — `query_start` (stream)

Replace each `if hasattr(self, "_callback_dispatcher"):` + indented `await` block with just the `await` call directly.

4. In `_CallbackDispatcher._safe_call`, upgrade error logging from `logger.debug` to `logger.warning`:

```python
# Change:
logger.debug("callback_error", ...)

# To:
logger.warning("callback_error", callback=type(cb).__name__, method=method_name, exc_info=True)
```

### Verification
- `dotenvx run -- uv run pytest tests/test_ux_sprint2.py -x` — callback tests still pass
- `dotenvx run -- uv run pytest tests/ -x -k "callback"` — no regressions

---

## Step 2: SKIPPED — No Backward Compatibility Shim

**Decision:** The `QueryResult` return type change is a clean break. No `FutureWarning`, no `__getattr__` proxy, no deprecation cycle. Users migrate to `result.answer` directly. The `__str__` and `__bool__` methods on `QueryResult` are retained for convenience only.

---

## Step 3: Complete Exception Wiring (httpx Pattern)

**Priority:** High
**Effort:** Large (~43 call sites)
**Confidence:** High

### Phase A: Create LiteLLM Exception Mapping and Context Manager

**File:** `_exceptions.py`

Add at the bottom:

```python
from contextlib import contextmanager
from typing import Generator

_LITELLM_EXCEPTION_MAP: dict[type, type] = {}


def _build_litellm_exception_map() -> dict[type, type]:
    """Lazy-load litellm exception mapping."""
    try:
        import litellm
        mapping = {}
        for attr, target in [
            ("RateLimitError", RateLimitError),
            ("AuthenticationError", AuthError),
            ("APIConnectionError", LLMError),
            ("ServiceUnavailableError", LLMError),
            ("InternalServerError", LLMError),
            ("Timeout", LLMError),
        ]:
            cls = getattr(litellm, attr, None)
            if cls is not None:
                mapping[cls] = target
        return mapping
    except ImportError:
        return {}


@contextmanager
def translate_litellm_errors() -> Generator[None, None, None]:
    """Context manager that maps litellm exceptions to nano-graphrag exceptions.

    Uses the httpx pattern: lookup dict + ``raise MappedExc(...) from exc``.
    Already-wrapped GraphRAGError instances pass through untouched.
    Unmapped exceptions propagate as-is.
    """
    global _LITELLM_EXCEPTION_MAP
    if not _LITELLM_EXCEPTION_MAP:
        _LITELLM_EXCEPTION_MAP = _build_litellm_exception_map()
    try:
        yield
    except GraphRAGError:
        raise
    except Exception as exc:
        for from_exc, to_exc in _LITELLM_EXCEPTION_MAP.items():
            if isinstance(exc, from_exc):
                raise to_exc(str(exc), cause=exc) from exc
        raise
```

### Phase B: Wire into `_llm_litellm.py`

**File:** `_llm_litellm.py`

The retry logic in `litellm_completion()` uses tenacity's `@retry` decorator. After retries exhaust, the final exception is re-raised. Wrap the outer call:

1. Find the main `async def litellm_completion(...)` function.
2. At the point where the tenacity-wrapped call is made, wrap it:

```python
from ._exceptions import translate_litellm_errors

async def litellm_completion(...):
    ...
    with translate_litellm_errors():
        response = await _retry_wrapper(...)  # the existing tenacity-wrapped call
    return response
```

Also wrap `litellm_embedding()` the same way.

### Phase C: Convert Remaining Raw Exceptions

Replace these call sites:

| File | Line | Current | Replace With |
|------|------|---------|-------------|
| `_ops/extraction_writeback.py` | ~293 | `raise RuntimeError(f"Alias extraction failed...")` | `raise ExtractionError(f"Alias extraction failed: {failure_rate:.1%} of batches failed", details={"failed": failed_batches, "total": total_batches, "rate": failure_rate})` |
| `_storage/gdb_sqlite.py` | ~492 | `raise ValueError(f"Clustering algorithm {algorithm} not supported")` | `raise StorageError(f"Clustering algorithm {algorithm} not supported", details={"algorithm": algorithm})` |
| `_storage/gdb_sqlite.py` | ~514 | `raise NotImplementedError("Node embedding is not supported...")` | Leave as-is (ABC method, correct) |
| `_storage/gdb_networkx.py` | ~316 | `raise ValueError(f"Clustering algorithm {algorithm} not supported")` | `raise StorageError(f"Clustering algorithm {algorithm} not supported", details={"algorithm": algorithm})` |
| `_storage/gdb_networkx.py` | ~331 | `raise ValueError(f"Node embedding algorithm {algorithm} not supported")` | `raise StorageError(f"Node embedding algorithm {algorithm} not supported", details={"algorithm": algorithm})` |
| `_storage/gdb_neo4j.py` | ~54 | `raise ValueError("Missing neo4j_url or neo4j_auth...")` | `raise StorageConfigError("Missing neo4j_url or neo4j_auth in addon_params")` |
| `_storage/gdb_neo4j.py` | ~501 | `raise ValueError(...)` | `raise StorageError(...)` |
| `_visualization.py` | ~136 | `raise ValueError("Graph storage has no _graph attribute...")` | `raise StorageError("Graph storage has no _graph attribute (NetworkX backend required)")` |
| `_utils.py` | ~219 | `raise ValueError(f"Unknown tokenizer_type: {self.tokenizer_type}")` | `raise ConfigError(f"Unknown tokenizer_type: {self.tokenizer_type}", details={"tokenizer_type": self.tokenizer_type})` |
| `_utils.py` | ~225-264 | `raise RuntimeError(...)` (3 instances) | `raise ConfigError(...)` with appropriate details |
| `_config.py` | ~93 | `raise ValueError(f"Invalid log_level={v!r}...")` | `raise ConfigError(f"Invalid log_level={v!r}. Must be one of: {sorted(valid)}", details={"log_level": v, "valid": sorted(valid)})` |
| `_ops/chunking.py` | ~75 | `raise ValueError("tokenizer_wrapper is required")` | `raise ConfigError("tokenizer_wrapper is required")` |

**Do NOT change** the ~30 `raise NotImplementedError` in `base.py` ABC methods — those are correct interface stubs.

### Verification
- `dotenvx run -- uv run pytest tests/test_ux_sprint1.py -x` — exception hierarchy tests
- `dotenvx run -- uv run pytest tests/ -x -k "error or exception"` — all exception-related tests
- `uv run ruff check src/nano_graphrag/` — lint clean

---

## Step 4: Fix `InsertResult` Delta Counts

**Priority:** Medium
**Effort:** Small (~15 lines changed)
**File:** `graphrag_insert.py`

### What to do

The current code reports total graph nodes/edges, not the delta from this insert. Fix:

1. Before extraction begins (after `success = False` line, around line ~172), capture pre-counts:

```python
pre_entity_count = 0
pre_relationship_count = 0
graph = self.chunk_entity_relation_graph
if graph is not None and hasattr(graph, "_graph") and graph._graph is not None:
    pre_entity_count = graph._graph.number_of_nodes()
    pre_relationship_count = graph._graph.number_of_edges()
```

2. In the success path (around line ~531), change the `InsertResult` construction:

```python
insert_result = InsertResult(
    documents_processed=len(docs_to_process),
    documents_skipped=len(normalized_docs) - len(docs_to_process),
    entities_created=graph._graph.number_of_nodes() - pre_entity_count if graph and graph._graph else 0,
    relationships_created=graph._graph.number_of_edges() - pre_relationship_count if graph and graph._graph else 0,
    latency_ms=elapsed_ms,
)
```

3. Update the early-return paths too:
   - Line ~193 (no valid docs): `return InsertResult(documents_processed=0)`
   - Line ~272 (all docs unchanged): `return InsertResult(documents_processed=0, documents_skipped=len(normalized_docs))`

### Verification
- Check that `InsertResult.entities_created` equals the number of nodes added, not the total graph size
- `dotenvx run -- uv run pytest tests/test_rag.py -x`

---

## Step 5: Fix `_STORAGE_REGISTRY` Mapping

**Priority:** Medium
**Effort:** Small (~2 lines)
**File:** `graphrag.py`

### What to do

The `"json"` key currently maps to `SQLiteKVStorage`. Fix to map to `JsonKVStorage`:

```python
_STORAGE_REGISTRY: dict[str, str] = {
    "json": "nano_graphrag._storage.kv_json:JsonKVStorage",
    "sqlite": "nano_graphrag._storage.kv_json:SQLiteKVStorage",
    "hnsw": "nano_graphrag._storage.vdb_hnswlib:HNSWVectorStorage",
    "networkx": "nano_graphrag._storage.gdb_networkx:NetworkXStorage",
    "sqlite_graph": "nano_graphrag._storage.gdb_sqlite:SQLiteGraphStorage",
}
```

Verify that `nano_graphrag._storage.kv_json` exports `JsonKVStorage` (check the module).

### Verification
- `dotenvx run -- uv run pytest tests/test_ux_sprint1.py -x -k "storage_registry"`
- `dotenvx run -- uv run pytest tests/ -x -k "resolve_storage"`

---

## Step 6: Add `pyvis` Optional Dependency

**Priority:** Medium
**Effort:** Trivial (~3 lines)
**File:** `pyproject.toml`

### What to do

Add a `viz` extras group:

```toml
[project.optional-dependencies]
viz = [
    "pyvis>=1.3",
]
cli = [
    "typer>=0.12.0",
]
```

Then `uv sync` to update the lockfile.

### Verification
- `uv sync` succeeds
- `uv run pytest tests/test_ux_sprint5.py -x -k "visualize"` — visualization fallback works without pyvis installed

---

## Step 7: Rename Sprint Test Files

**Priority:** Low
**Effort:** Small (just `git mv`)
**Do AFTER Step 3** (tests reference exception names that change)

### What to do

```bash
git mv tests/test_ux_sprint1.py tests/test_exceptions.py
git mv tests/test_ux_sprint2.py tests/test_callbacks.py
git mv tests/test_ux_sprint3.py tests/test_cli.py
git mv tests/test_ux_sprint4.py tests/test_result_types.py
git mv tests/test_ux_sprint5.py tests/test_visualization.py
```

### Verification
- `dotenvx run -- uv run pytest tests/test_exceptions.py tests/test_callbacks.py tests/test_cli.py tests/test_result_types.py tests/test_visualization.py -x`

---

## Step 8: Clean Up Untracked Artifacts

**Priority:** Low
**Effort:** Trivial
**Files:** `.gitignore`, `docs/plan/`

### What to do

1. Add to `.gitignore`:

```
sage_snapshot.txt
```

2. Delete the artifact:

```bash
rm sage_snapshot.txt
```

3. `docs/plan.md` (672 lines, the master UX improvement plan) — decide: commit it or move to `.claude/plans/`. If it's reference material for the project, commit it. If it's session planning, move it.

4. `docs/plan/20260517-sage-improvements.md` and `20260518-test-coverage-improvement.md` — these are already committed. Leave them.

5. `CHANGELOG.md` — commit this (it documents the release).

### Verification
- `git status` shows only intended files

---

## Summary Matrix

| Step | Priority | Effort | Dependencies | Status |
|------|----------|--------|-------------|--------|
| 1. Null Dispatcher | High | Small | None | Ready |
| 2. Backward Compat | — | — | SKIPPED | Not needed |
| 3A. LiteLLM Exception Map | High | Medium | None | Ready |
| 3B. Wire into _llm_litellm | High | Medium | 3A | Ready |
| 3C. Convert raw exceptions | High | Large | 3A | Ready |
| 4. InsertResult delta | Medium | Small | None | Ready |
| 5. Storage registry fix | Medium | Small | None | Ready |
| 6. pyvis optional dep | Medium | Trivial | None | Ready |
| 7. Rename test files | Low | Small | Step 3 | Ready |
| 8. Clean up artifacts | Low | Trivial | None | Ready |
