# Test Coverage Improvement Plan

**Date:** 2026-05-18
**Status:** In Progress
**Scope:** Add ~80-120 new tests across 5 phases using pytest-tagging patterns

## Current State

- **400 tests** across 28 files, flat `tests/` structure
- Good coverage: config, storage backends, entity registry, extraction ops, litellm wrapper
- No property-based, metamorphic, contract, or differential tests
- Missing: hypothesis as dev dependency, `--import-mode=importlib` in addopts

## Coverage Gaps

| Module | Gap |
|--------|-----|
| `_ops/refinement/` (pipeline, merge, enrich, infer) | **0 tests** |
| `_vault/export.py` | **0 tests** |
| `_ops/community.py::generate_community_report` | **Minimal** |
| `_ops/extraction_writeback.py` | **Not directly tested** |
| `_storage/vdb_nanovectordb.py` | **0 tests** |
| `_utils.py` (hashing, IDs, semaphore, truncate) | **Minimal** |
| `_schemas.py` (validation, normalization) | **Minimal** |
| `_ops/query.py` | **Only via integration** |

## Phase 1: Infrastructure

### 1.1 Update `pyproject.toml`
- Add `--import-mode=importlib` to `addopts`
- Register new markers: `property`, `metamorphic`, `contract`, `differential`

### 1.2 Add `hypothesis` dev dependency
```bash
uv add --dev hypothesis
```

### 1.3 Create `tests/strategies.py`
Shared hypothesis strategies for:
- `st_entity_name()` — text strings 1-50 chars
- `st_entity_type()` — sampled from common types
- `st_extracted_entity()` — builds `ExtractedEntity` via `st.builds`
- `st_extracted_relationship()` — builds `ExtractedRelationship`
- `st_text_chunk()` — text with content, tokens, doc_id, order
- `st_embedding_vector(dim=N)` — float arrays
- `st_content_string()` — text for hashing

### 1.4 Update `tests/conftest.py`
Add shared fixtures:
- `fake_embedding_func` — deterministic embedding (identity-like)
- `mock_graph_storage` — async mock with all BaseGraphStorage methods
- `mock_kv_storage` — async mock with all BaseKVStorage methods
- `mock_entity_vdb` — async mock with upsert/query/delete
- `global_config` — standard config dict with fake model funcs

## Phase 2: Property-Based Tests (`tests/test_property.py`)

### 2.1 Hashing & ID Generation (~6 tests)
- `test_mdhash_id_deterministic` — same input → same output
- `test_mdhash_id_idempotent` — `f(f(x)) == f(x)`
- `test_mdhash_id_different_inputs` — different input → different output
- `test_sha256_id_deterministic`
- `test_sha256_id_idempotent`
- `test_stable_entity_id_namespace_aware` — same name + namespace → same ID

### 2.2 Relation Normalization (~4 tests)
- `test_normalize_relation_idempotent` — applying twice gives same result
- `test_normalize_vocabulary_members` — all `RELATION_VOCABULARY` normalize to self
- `test_normalize_aliases_resolve` — all `RELATION_ALIASES` resolve correctly
- `test_normalize_empty_returns_related_to` — empty/None → `"related_to"`

### 2.3 Schema Validation (~4 tests)
- `test_extracted_entity_roundtrip` — model_dump → model_validate round-trip
- `test_extracted_relationship_roundtrip`
- `test_batched_output_roundtrip`
- `test_confidence_bounds` — 0.0 <= confidence <= 1.0

### 2.4 Text Splitting (~3 tests)
- `test_split_tokens_concatenation` — join(chunks) preserves all tokens
- `test_split_tokens_no_empty_chunks` — no empty output chunks
- `test_split_tokens_idempotent_no_separators` — no separators → single chunk

### 2.5 Utility Functions (~3 tests)
- `test_serialize_deserialize_source_ids_roundtrip`
- `test_clean_str_removes_control_chars`
- `test_split_string_by_multi_markers_preserves_content`

## Phase 3: Contract Tests (`tests/test_contract.py`)

### 3.1 GraphRAG Public API Signatures (~5 tests)
- `test_ainsert_signature` — accepts `str | list[str]`
- `test_aquery_signature` — accepts `str, QueryParam`
- `test_arefine_signature` — accepts `list[str] | None`
- `test_aexport_vault_signature` — accepts `str | None, bool | None`
- `test_arebuild_graph_signature` — no required params

### 3.2 Config Round-Trip Contracts (~4 tests)
- `test_graphrag_config_from_dict_roundtrip` — to_dict → from_dict → to_dict
- `test_graphrag_config_from_yaml_roundtrip`
- `test_graphrag_settings_from_dict_roundtrip`
- `test_query_param_from_config` — all fields present

### 3.3 Storage ABC Contracts (~4 tests)
- `test_base_vector_storage_abstract_methods` — query, upsert, delete
- `test_base_kv_storage_abstract_methods` — all_keys, get_by_id, upsert, delete, drop
- `test_base_graph_storage_abstract_methods` — all 20+ methods
- `test_storage_namespace_pattern` — namespace + global_config fields

### 3.4 Export Contracts (~2 tests)
- `test_graphrag_class_has_all_public_methods` — insert, query, refine, export_vault, rebuild
- `test_response_types_are_strings` — ResponseType constants

## Phase 4: Metamorphic Tests (`tests/test_metamorphic.py`)

### 4.1 Entity Extraction Relations (~4 tests)
- `test_more_text_more_entities` — longer text with more entity mentions → >= entities extracted
- `test_duplicate_input_duplicate_output` — same input → same extraction count
- `test_entity_linking_monotonicity` — more shared neighbors → higher IOU
- `test_merge_candidates_type_filtered` — only same-type entities can merge

### 4.2 Embedding Similarity (~3 tests)
- `test_identical_text_high_similarity` — same text → cosine sim ≈ 1.0
- `test_random_texts_lower_similarity` — random pair sim < identical pair sim
- `test_embedding_dimension_preserved` — output dim always matches config

### 4.3 Refinement Monotonicity (~3 tests)
- `test_higher_threshold_fewer_merges` — merge threshold 0.99 < 0.80 in merged count
- `test_enrichment_preserves_entity_name` — enriched description contains entity name
- `test_rejection_cache_pruning` — older entries removed after TTL

## Phase 5: Unit Tests for Uncovered Modules

### 5.1 Refinement Pipeline (`tests/test_refinement.py`, ~20 tests)

**Pipeline module:**
- `test_refinement_journal_add_and_save` — entries persisted to JSONL
- `test_refinement_journal_max_entries` — truncates at `_MAX_JOURNAL_ENTRIES * 2`
- `test_rejection_cache_contains_and_set` — basic CRUD
- `test_rejection_cache_prune_ttl` — expired entries removed
- `test_arefine_disabled_returns_skipped` — enable_refinement=False
- `test_arefine_no_valid_phases` — empty phases list
- `test_get_rejection_ttl_scaling` — larger graph → longer TTL

**Merge module:**
- `test_find_merge_candidates_brute_force` — same-type, above threshold
- `test_find_merge_candidates_type_filtered` — different types excluded
- `test_merge_phase_no_nodes` — < 2 nodes → empty stats
- `test_merge_phase_hub_cap_respected` — no entity merged > hub_cap times

**Enrich module:**
- `test_validate_enrichment_empty_fails` — empty enriched text
- `test_validate_enrichment_missing_name_fails` — entity name not in enriched
- `test_validate_enrichment_low_overlap_fails` — < 40% word overlap
- `test_validate_entity_subject_suspicious_sentence` — proper noun not entity
- `test_enrich_phase_no_llm_returns_empty` — missing llm_func

**Infer module:**
- `test_select_candidates_weighted_top_heavy` — 80% top + 20% explore
- `test_infer_phase_no_llm_returns_empty`
- `test_co_occurrence_building` — shared chunks → co-occurrence count

### 5.2 Vault Export (`tests/test_vault_export.py`, ~10 tests)
- `test_safe_filename_strips_special_chars`
- `test_safe_filename_reserves_index`
- `test_yaml_frontmatter_basic`
- `test_yaml_frontmatter_with_list`
- `test_yaml_frontmatter_quoting`
- `test_write_index_creates_file`
- `test_export_empty_graph_returns_zero_stats`
- `test_export_creates_entity_directories`
- `test_export_sparse_entities_rolled_up`
- `test_export_communities_with_reports`

### 5.3 Community Reports (`tests/test_community.py`, ~8 tests)
- `test_community_report_json_to_str_basic`
- `test_community_report_json_to_str_string_findings`
- `test_pack_single_community_by_sub_communities`
- `test_generate_report_empty_schema_returns`
- `test_generate_report_deletes_stale`
- `test_pack_community_describe_node_ordering`
- `test_pack_community_describe_edge_temporal`
- `test_pack_community_truncation_respects_budget`

### 5.4 Extraction Writeback (`tests/test_extraction_writeback.py`, ~8 tests)
- `test_process_entity_writeback_creates_node`
- `test_process_entity_writeback_registers_entity`
- `test_process_relationship_writeback_creates_edge`
- `test_process_relationship_writeback_temporal_fields`
- `test_write_manifest_empty_entities_returns_none`
- `test_write_manifest_upserts_to_vdb`
- `test_extract_aliases_empty_entities`
- `test_extract_aliases_no_llm_returns_empty`

### 5.5 Extended Utils (`tests/test_utils_extended.py`, ~8 tests)
- `test_limit_async_func_call_limits_concurrency`
- `test_wrap_embedding_func_attrs`
- `test_truncate_list_by_token_size_basic`
- `test_truncate_list_by_token_size_zero_returns_empty`
- `test_compute_args_hash_deterministic`
- `test_is_float_regex`
- `test_enclose_string_with_quotes`
- `test_async_rw_lock_read_write`

### 5.6 NanoVectorDB (`tests/test_nanovectordb.py`, ~6 tests)
- `test_nanovecdb_upsert_and_query`
- `test_nanovecdb_upsert_empty`
- `test_nanovecdb_delete`
- `test_nanovecdb_delete_empty`
- `test_nanovecdb_persistence`
- `test_nanovecdb_cosine_threshold`

## Execution Order

1. Phase 1 (infrastructure) — must be first
2. Phase 2 + 3 (property + contract) — can run in parallel
3. Phase 4 (metamorphic) — depends on Phase 1 strategies
4. Phase 5 (unit tests) — independent, can run in parallel with Phase 2-4

## Files to Create/Modify

| File | Action | ~Tests |
|------|--------|--------|
| `pyproject.toml` | Edit | — |
| `tests/strategies.py` | New | — |
| `tests/conftest.py` | Edit | — |
| `tests/test_property.py` | New | ~20 |
| `tests/test_contract.py` | New | ~15 |
| `tests/test_metamorphic.py` | New | ~10 |
| `tests/test_refinement.py` | New | ~20 |
| `tests/test_vault_export.py` | New | ~10 |
| `tests/test_community.py` | New | ~8 |
| `tests/test_extraction_writeback.py` | New | ~8 |
| `tests/test_utils_extended.py` | New | ~8 |
| `tests/test_nanovectordb.py` | New | ~6 |

**Total: ~113 new tests**

## Verification

```bash
# Run all new tests
dotenvx run -- uv run pytest tests/test_property.py tests/test_contract.py tests/test_metamorphic.py tests/test_refinement.py tests/test_vault_export.py tests/test_community.py tests/test_extraction_writeback.py tests/test_utils_extended.py tests/test_nanovectordb.py -v

# Run by marker
uv run pytest -m property
uv run pytest -m contract
uv run pytest -m metamorphic

# Run full suite
dotenvx run -- uv run pytest tests/ -x

# Lint
uv run ruff check src/nano_graphrag/ tests/
```
