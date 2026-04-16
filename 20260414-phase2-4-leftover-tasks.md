# Phase 2-4 Leftover Tasks

This note captures follow-up work that remains after implementing:

- persisted reverse contribution indexing for incremental rebuilds
- `astream_query(...)` streaming support
- opt-in entity linking

These are not blockers for the current implementation, but they are the highest-signal next tasks for a follow-up thread.

## 1. Run and stabilize the broader test surface

The focused suite is green, but the full repo test matrix was not run in this pass.

Remaining work:

- Run the full `uv run pytest` suite.
- Run benchmark-related tests if they are part of normal CI expectations.
- Check whether any provider/integration tests need updated fixtures for streaming or entity-linking behavior.

## 2. Add streaming coverage for custom model wrappers

Current behavior:

- LiteLLM-backed models stream natively.
- custom user-supplied model functions fall back to buffered one-chunk streaming.

Remaining work:

- Decide whether to formalize a streaming callable contract for custom `best_model_func` / `cheap_model_func`.
- Add tests for custom model objects that expose their own async streaming interface.
- Document that contract if adopted.

## 3. Consider a sync streaming helper

Current behavior:

- `astream_query(...)` exists.
- no sync `stream_query(...)` helper was added.

Remaining work:

- Decide whether a synchronous iterator wrapper is worth supporting.
- If yes, implement it carefully around the existing event-loop helpers and add tests for notebook/CLI usage.

## 4. Broaden entity-linking behavior beyond the current conservative path

Current behavior:

- exact registry match
- conservative fuzzy candidate search
- optional LLM disambiguation for ambiguous candidates
- no full historical relinking migration

Remaining work:

- Add richer candidate ranking signals beyond name similarity:
  - entity type compatibility
  - description overlap
  - neighborhood/relationship overlap
- Add explicit provenance metadata for link decisions in manifests or registry metadata.
- Decide whether to support a repair/relink command for existing corpora.
- Decide whether linked entities should trigger more aggressive alias propagation into old manifests.

## 5. Improve reverse-index introspection and maintenance

Current behavior:

- reverse index is persisted and auto-regenerated when missing/stale enough to block rebuild

Remaining work:

- Add explicit health/status tooling for the reverse index.
- Add a manual rebuild command or maintenance helper.
- Consider tracking lightweight stats:
  - last rebuild time
  - indexed document count
  - reverse-index version/migration metadata beyond the current simple marker

## 6. Add examples and user-facing docs

Current behavior:

- README and architecture docs mention the new capabilities

Remaining work:

- Add a minimal streaming example showing `async for chunk in rag.astream_query(...)`.
- Add an entity-linking example with `enable_entity_linking=True`.
- Add a short note explaining the cost/precision tradeoff of entity linking.

## 7. Evaluate performance and quality on real corpora

The implementation is covered by targeted tests, but it has not yet been benchmarked for:

- incremental rebuild latency improvement from the reverse index
- streaming perceived-latency improvement
- entity-linking precision/false-merge rate

Remaining work:

- Benchmark incremental update time before vs after on multi-document corpora.
- Measure time-to-first-token for `astream_query(...)`.
- Create a small entity-linking evaluation set with:
  - true duplicates
  - near-duplicates that should not merge
  - ambiguous short names

## 8. Optional API cleanup

Potential cleanup items if we want to harden the public surface:

- Export/document streaming-related types if a formal stream contract is introduced.
- Decide whether `louvain` should remain an alias to the current Leiden backend or be separated later.
- Consider whether manifest alias metadata should become a more explicit typed schema.

## Suggested next thread prompt

If you want to continue from the highest-value remaining work, a good next thread would be:

"Run the full test suite, fix any regressions, then add a minimal streaming example and benchmark the incremental reverse-index speedup plus entity-linking quality on a small evaluation corpus."
