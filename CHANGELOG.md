# Changelog

All notable changes to nano-graphrag are documented here.

## [0.0.9.0] - 2026-05-18

### Added

- **Exception hierarchy** (`_exceptions.py`): `GraphRAGError` base class with 15 typed subclasses (`ConfigError`, `ModeNotEnabledError`, `GraphIntegrityError`, `AuthError`, `RateLimitError`, `QueryError`, `StorageError`, etc.). All exceptions carry `message` and `details` dict.
- **Result types** (`_schemas.py`): `QueryResult`, `InsertResult`, `TokenUsage`, `QuerySource`, `QueryTrace`, `StreamTextChunk`, `StreamSourceRef`, `StreamComplete` dataclasses.
- **Callback system** (`_callbacks.py`): `ExtractionCallback`, `QueryCallback`, `LLMCallback` protocols. Built-in `LoggingCallback` and `TokenTrackingCallback`. `_CallbackDispatcher` routes events to registered callbacks.
- **CLI** (`_cli.py`): 7 commands via `typer` — `query`, `insert`, `status`, `refine`, `rebuild`, `export`, `config`. Install with `pip install nano-graphrag[cli]`.
- **Visualization** (`_visualization.py`): `GraphStatus` dataclass, `compute_status()`, `visualize_graph()` (pyvis with HTML fallback), Jupyter `_repr_html_()`.
- **GraphRAG convenience methods**: `status()`, `astatus()`, `export_graph_html()`, `aexport_graph_html()`.
- **String-based storage registry**: `_STORAGE_REGISTRY` + `_resolve_storage()` for backend resolution.
- **Eager validation**: `_normalize_settings` now raises `ConfigError` at construction time for invalid configs.
- **`__all__`**: 42 explicit public exports.

### Changed

- **[BREAKING] `aquery()` now returns `QueryResult` instead of `str`.**
  Previously `aquery()` returned a raw answer string. It now returns a `QueryResult` dataclass. Use `result.answer` to get the answer text, or `str(result)` / `bool(result)` for backward-compatible string operations. Code using `isinstance(result, str)` will break.
- **[BREAKING] `astream_query()` now yields `StreamTextChunk` instead of `str`.**
  Previously `astream_query()` yielded raw strings. It now yields `StreamTextChunk` objects. Use `str(chunk)` or `chunk.text` to access the text. Code using `isinstance(chunk, str)` will break.
- `GraphRAGConfig.__init__` raises `ConfigError` for invalid settings (previously silent).
- `_normalize_settings` validates `entity_extraction_quality`, `extraction_batch_size`, `embedding_dim` ranges.

### Migration Guide

```python
# Before (0.0.8.x)
answer = await rag.aquery("What is X?")
print(answer)  # str

# After (0.0.9.0)
result = await rag.aquery("What is X?")
print(result)           # works — QueryResult.__str__ returns answer
print(result.answer)    # explicit
print(result.mode)      # new: "global", "local", etc.
print(result.sources)   # new: list[QuerySource]
print(result.latency_ms)  # new: float

# Before (streaming)
async for chunk in rag.astream_query("What is X?"):
    print(chunk, end="")  # str

# After (streaming) — still works via __str__
async for chunk in rag.astream_query("What is X?"):
    print(chunk, end="")       # works — StreamTextChunk.__str__ returns text
    print(chunk.text, end="")  # explicit
```
