## 1. Workflow
- **Analyze First:** Read relevant files before proposing solutions. Never hallucinate.
- **Approve Changes:** Present a plan for approval before modifying code.
- **Minimal Scope:** Change as little code as possible. No new abstractions.

## 2. Output Style
- High-level summaries only.
- No speculation about code you haven't read.

## 3. Technical Stack
- **Python:**
  - Package manager: `uv`.
  - Execution: Always `uv run <command>`. Never `python`.
  - Sync: `uv sync`.
- **Env vars:** `.env` uses [dotenvx encryption](https://dotenvx.com/encryption). Values are `encrypted:...` format. `python-dotenv` loads raw encrypted strings — **NOT usable by litellm/SDKs**. Must run via `dotenvx run -- uv run <command>` to inject decrypted values. Never use `load_dotenv()` alone for API keys.
- **Frontend:**
  - Verify: Run `npm run check` and `npm test` after changes.
- **Docs:** Update `ARCHITECTURE.md` if structure changes.
- **Files:** Markdown files must follow `YYYYMMDD-filename.md` format.

## 4. Known Issues
- **Empty graph bug:** If embedding API fails during `_ainsert_documents` (rate limit, auth), graph stays empty but `document_index` commits (SQLite auto-commits). Subsequent runs skip via content hash. Fix pending — see `.claude/handoffs/2026-04-16-graph-building-bugfixes.md`.
- **`__pycache__` staleness:** After editing `.py` files, clear caches: `find . -name "__pycache__" -type d -exec rm -rf {} +`. Python 3.9 on this machine sometimes serves stale `.pyc`.

## 5. Project Structure

```
nano_graphrag/          # Main package
  graphrag.py           # GraphRAG class — public API (insert, query, rebuild)
  base.py               # GraphRAGConfig dataclass, base storage ABCs, from_env()
  graphrag_insert.py    # _ainsert_documents — parallel extraction + incremental writes
  graphrag_query.py     # Query interface (local/global/naive/entity_grounded modes)
  graphrag_runtime.py   # Runtime config, logging setup (logger: "nano-graphrag")
  _llm_litellm.py       # Async LLM calls via LiteLLM — retry, structured output
  _schemas.py           # Pydantic v2 models (ExtractedEntity, BatchedEntityExtractionOutput, etc.)
  _splitter.py          # Text chunking
  _utils.py             # Utilities (limit_async_func_call semaphore, hashing)
  _ops/                 # Operation modules
    extraction_structured.py  # Batched entity extraction (current)
    extraction_legacy.py      # Legacy single-chunk extraction
    extraction_gliner.py      # GLiNER-based extraction
    extraction_common.py      # Shared extraction helpers
    extraction_writeback.py   # Write extracted data to storage
    extraction_rebuild.py     # Rebuild from manifests
    community.py              # Community report generation
    chunking.py               # Document chunking strategies
    query.py                  # Query operation implementations
  _storage/             # Storage backends
    kv_json.py           # JsonKVStorage (default KV)
    gdb_networkx.py      # NetworkXStorage (default graph)
    gdb_sqlite.py        # SQLite graph storage
    gdb_neo4j.py         # Neo4j graph storage
    vdb_hnswlib.py       # HNSWVectorStorage (default vector)
    vdb_nanovectordb.py  # Nano vector DB fallback

bench/                  # Benchmark framework
  __main__.py           # CLI: python -m bench
  runner.py             # BenchmarkConfig, ExperimentRunner
  datasets/             # Dataset loaders (MultiHopRAG, 2Wiki, HotpotQA, MuSiQue)
  metrics/              # TokenF1, ExactMatch, NativeContextRecall
  retrievers/           # Retriever implementations
  techniques/           # Advanced techniques (reranker, adaptive router, raptor)

tests/                  # pytest-asyncio (auto mode), "integration" marker for live services
experiments/            # YAML benchmark configs
```

## 6. Key Architecture

- **Config:** `GraphRAGConfig` dataclass in `base.py`. `from_env()` parses env vars with `_parse_int()` helpers. YAML overrides merged in `bench/runner.py`.
- **Extraction pipeline:** `_ainsert_documents()` in `graphrag_insert.py` — delta detection via content hash → parallel doc extraction (`asyncio.gather` + semaphore) → batched chunk extraction (N chunks/LLM call) → incremental flush every N docs.
- **LLM calls:** `_llm_litellm.py` — `litellm_completion()` async with exponential backoff (3 retries). Structured output via Pydantic models with provider-specific fallback (structured → prompt-based → text).
- **Storage pattern:** Namespace-based. `BaseKVStorage` → `JsonKVStorage`. `BaseVectorStorage` → `HNSWVectorStorage`. `BaseGraphStorage` → `NetworkXStorage`. All async.
- **Concurrency:** Two-layer — doc-level semaphore (`doc_extraction_max_async=4`) × chunk-level semaphore (`extraction_max_async=16`) = max 64 concurrent LLM calls.
- **Logging:** Logger `"nano-graphrag"` configured in `graphrag_runtime.py`. Format: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`.

## 7. Commands

```bash
# Run tests
dotenvx run -- uv run pytest tests/ -x

# Run single test
dotenvx run -- uv run pytest tests/test_rag.py -k "test_name"

# Run integration tests (requires API keys)
dotenvx run -- uv run pytest tests/ -m integration

# Run benchmark
dotenvx run -- uv run python -m bench --config experiments/benchmark_multihop_rag_openrouter.yaml

# Lint
uv run ruff check nano_graphrag/

# Format
uv run ruff format nano_graphrag/

# Clear stale caches (run after editing .py files)
find . -name "__pycache__" -type d -exec rm -rf {} +

# Sync dependencies
uv sync
```

## 8. Conventions

- **Python 3.9** target (pyproject + ruff config). No walrus operator, no `str | None` (use `Optional[str]`).
- **Ruff:** line-length 100, select E/F/W/I, ignore E501.
- **Pydantic v2** for all schemas (`_schemas.py`). Use `Field()` with descriptions.
- **Async-first:** All I/O uses `async/await`. No sync wrappers around async code.
- **Concurrency:** `limit_async_func_call` decorator or `asyncio.Semaphore` for rate limiting. `asyncio.gather` for parallelism.
- **Env vars:** Always `dotenvx run -- uv run <command>`. Never `load_dotenv()` alone for API keys — values are encrypted.
- **Imports:** Relative within package (`from .base import ...`). Absolute for external deps.
