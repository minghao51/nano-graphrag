# Changelog

Nano GraphRAG - Development History since Fork

## Overview

This changelog documents all development work since the branch was forked from the main repository, spanning from March 23, 2026 to April 8, 2026.

---

## April 2026

### 2026-04-08 - Extraction Module Refactoring

**Commits:** `f54a3dc`, `0814661`, `55ad6d4`

**Changes:**
- Completed extraction module split cleanup
- Refactored GraphRAG runtime and extraction modules
- Removed extra blank line in extraction_common for style consistency

**Files Modified:**
- `nano_graphrag/_ops/extraction_common.py`
- `nano_graphrag/_ops/extraction_legacy.py`
- `nano_graphrag/_ops/extraction_rebuild.py`
- `nano_graphrag/_ops/extraction_structured.py`
- `nano_graphrag/_ops/extraction_writeback.py`
- `nano_graphrag/graphrag_runtime.py`

### 2026-04-07 - LiteLLM Schema Instruction Helper

**Commit:** `856ea9a`

**Purpose:** Extract duplicated schema instruction logic into reusable helper

**Details:**
- Added `_add_schema_instruction_to_messages()` helper function
- Extracted `SCHEMA_INSTRUCTION_TEMPLATE` constant
- Replaced duplicate inline logic with calls to the helper
- Removed redundant json imports from within functions

**Problem Solved:** Schema instruction handling is now centralized - future changes only need to be made in one place.

### 2026-04-06 - Structured Output & Storage Improvements

**Commits:** `f610325`, `57f23b9`, `51ce71f`

**Changes:**
1. **Harden LiteLLM structured output handling** (`f610325`)
2. **Stabilize incremental graph extraction updates** (`57f23b9`)
3. **Replace default storage backends** (`51ce71f`)
   - Migrated from JSON file storage to SQLite for key-value storage
   - Migrated from FAISS to HNSWlib for vector storage
   - Improved performance and reduced dependencies

### 2026-04-02 - Qwen Support & Configuration Cleanup

**Commits:** `eea7e92`, `fb7a51f`, `3ffaaf2`, `531c4b4`

**Changes:**
1. **Core features:**
   - Added Qwen model support
   - Implemented timeout handling for LLM calls
   - LiteLLM integration improvements

2. **Benchmark fixes:**
   - Resolved MuSiQue dataset loading issues
   - Fixed registry import paths

3. **Project cleanup:**
   - Removed obsolete experiment and test files
   - Simplified project configuration files

---

## March 2026

### 2026-03-24 - Comprehensive Benchmark Infrastructure

**Commit:** `02bb7d3`

**Major Additions:**

1. **LiteLLM Provider Support:**
   - Provider detection for OpenAI, Anthropic, Google, Cohere, Mistral, Ollama
   - Structured output support with provider-native json_schema format
   - OpenRouter structured output support with require_parameters
   - Fallback handling for providers that reject structured output

2. **Testing:**
   - Added 49 comprehensive tests for LiteLLM integration
   - Improved entity extraction with structured output fallback

3. **Multi-hop Retrieval:**
   - Enhanced multi-hop retriever with better query decomposition

4. **Documentation:**
   - Updated documentation structure (docs/benchmarks, docs/guides)
   - Added experiment validation and setup scripts

5. **Benchmark Configs:**
   - Added OpenRouter benchmark configs for all datasets

### 2026-03-23 - Multi-Hop RAG Implementation (Phase 3)

**Commits:** `7c268e7` through `1e0f71b` (18 commits)

**Overview:** Complete implementation of multi-hop retrieval and benchmark infrastructure

#### Phase 3 Verification (`7c268e7`)
- Documented all implemented features and tests
- Recorded 35/35 unit tests passing
- Verified all M3 checklist items complete
- Created benchmark configs for all 4 datasets
- Note: Actual performance metrics pending benchmark execution with LLM API

#### Benchmark Configs (`ff200d9`)
Created benchmark configurations for:
- MultiHop-RAG dataset
- MuSiQue dataset
- HotpotQA dataset
- 2WikiMHQA dataset

All configs test naive, local, and multihop modes with LLM cache enabled.

#### Query Decomposition Fix (`409e83d`)
- Fixed MultiHopRetriever to use best_model_func/cheap_model_func
- Fixed integration test corpus format (content vs text field)
- Added end-to-end integration test for multi-hop retrieval

#### MultiHopGraphRAG Subclass (`70701b8`)
- Created MultiHopGraphRAG subclass with injected_context parameter
- Updated _create_graphrag to use subclass when multihop mode enabled
- Added multihop query handling in main query loop
- Use MultiHopRetriever for context retrieval in multihop mode

#### MultiHopRetriever Implementation (`ca6434c`, `c9424a0`, `eb18d73`)
- Implemented core logic for multi-hop retrieval
- Added HopState dataclass for multi-hop tracking
- Added retriever infrastructure with protocol and result types
- Registered MultiHopRetriever in plugin registry

#### Context Merging & Token Budget (`cdba89d`, `bf83719`)
- Verified token budget enforcement in context merging
- Verified context merging with deduplication
- Implemented entity state carry-over across hops

#### Testing (`9770f33`)
- Added query decomposition test coverage

#### LLM Cache Integration (`15d7024`, `5bd9fd8`, `1e0f71b`)
- Integrated BenchmarkLLMCache into ExperimentRunner
- Added cache.wrap() method for LLM function decoration
- Improved wrap() method - removed hardcoded defaults, fixed kwargs handling
- Added type hints and @functools.wraps decorator

#### Phase 1 Implementation (`97ac082`)
Complete Phase 1 benchmark infrastructure:
- Added NativeContextRecallMetric for context recall evaluation
- Added HuggingFace datasets dependency for auto-download
- Implemented download() for all 4 multi-hop datasets (MuSiQue, HotpotQA, 2Wiki, MultiHop-RAG)
- Added auto_download flag to config
- Restructured to bench/ module with python -m bench CLI
- Implemented compare functionality for A/B testing
- Support nested YAML config schema (roadmap-compliant)
- Added comprehensive benchmark usage documentation
- Deprecated nano_graphrag._benchmark in favor of bench/

**New Files:**
- bench/ module with datasets/, metrics/, cache.py, runner.py, compare.py
- bench/__main__.py, bench/run.py for CLI
- docs/benchmark-usage.md usage guide
- examples/benchmarks/configs/ with example configs

All 22 benchmark tests passing.

#### Type Safety Improvements (`334cbd7`, `f154845`)
- Added QAPair dataclass with id, question, answer, supporting_facts, and metadata
- Added Passage dataclass with id, title, and text
- Updated BenchmarkDataset protocol to use Iterator[QAPair] and Iterator[Passage]
- Updated all datasets to yield typed objects
- Fixed critical bug where runner would crash using dict-style access on QAPair objects
- Added comprehensive tests for typed returns and graceful field handling

---

## Summary Statistics

**Total Commits:** 30
**Total Files Changed:** 134
**Lines Added:** ~20,082
**Lines Removed:** ~5,320

### Major Feature Areas

1. **Multi-Hop RAG Implementation** - Complete infrastructure for multi-hop question answering
2. **Benchmark Framework** - Comprehensive benchmarking system with 4 datasets
3. **LLM Provider Support** - Multi-provider support with LiteLLM integration
4. **Storage Migration** - SQLite and HNSWlib replacing JSON and FAISS
5. **Extraction Module Refactoring** - Clean separation of extraction strategies
6. **Qwen Model Support** - Added support for Qwen family of models

### Testing

- **49** LiteLLM integration tests
- **35** Multi-hop RAG tests
- **22** Benchmark infrastructure tests
- **12** Cache integration tests
- **Plus** additional tests for storage, retrieval, and type safety

### Documentation

- Restructured documentation hierarchy
- Added benchmark usage guides
- Added Neo4j integration guide
- Created FAQ documentation
- Added contribution guidelines

---

## Contributors

- minghao (Primary developer)
- Claude Sonnet 4.6 (AI pair programmer - credited in multiple commits)

## Dependencies

Major dependency changes:
- Added: HuggingFace datasets, LiteLLM, HNSWlib
- Updated: Multiple Python packages via uv.lock
- Removed: OpenAI-specific test dependencies (migrated to LiteLLM)
