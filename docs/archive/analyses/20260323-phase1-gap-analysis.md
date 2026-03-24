# Phase 1 Implementation Gap Analysis

**Date:** 2026-03-23
**Status:** Phase 1 ~60% Complete

---

## Executive Summary

The repository has a **partial implementation of Phase 1** benchmark infrastructure. Core functionality exists (datasets, metrics, config runner, cache), but several key features from the roadmap are missing or incomplete. The implementation works for basic experiments but lacks the polish, completeness, and developer experience specified in the roadmap.

**Overall Progress:** ~60% of Phase 1 deliverables

---

## Detailed Comparison

### 1. Dataset Loaders (1.1) — 70% Complete

#### ✅ What Exists
- `BenchmarkDataset` Protocol with `questions()` and `corpus()` methods
- Implementations for all 4 required datasets:
  - `MultiHopRAGDataset`
  - `HotpotQADataset`
  - `MuSiQueDataset`
  - `TwoWikiMultiHopQADataset`
- Dataset loading from JSON files works correctly
- `max_samples` limiting works

#### ❌ What's Missing

1. **No `QAPair` and `Passage` dataclasses**
   - **Roadmap spec:** Typed dataclasses with `id`, `question`, `answer`, `supporting_facts`, `metadata`
   - **Current:** Returns generic `Dict[str, Any]`
   - **Impact:** Loss of type safety, harder to document expected fields, no IDE autocomplete

2. **No `download()` method**
   - **Roadmap spec:** `def download(self, cache_dir: str = "~/.cache/nano-bench") -> None`
   - **Current:** Users must manually download datasets and provide file paths
   - **Impact:** Poor DX, friction getting started, not "one-command" ready

3. **No automatic data caching**
   - **Roadmap spec:** "Each loader caches downloaded data locally"
   - **Current:** No caching layer for downloaded datasets
   - **Impact:** Re-downloads on every run if implemented, wasteful

4. **No `supporting_facts` tracking**
   - **Roadmap spec:** `supporting_facts: list[str]` in `QAPair`
   - **Current:** Metadata not preserved in standard format
   - **Impact:** Cannot compute context recall metric (needs gold supporting facts)

#### 📋 Proposed Fixes

```python
# nano_graphrag/_benchmark/datasets.py

@dataclass
class QAPair:
    """Question-answer pair with metadata."""
    id: str
    question: str
    answer: str
    supporting_facts: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

@dataclass
class Passage:
    """Corpus passage/document."""
    id: str
    title: str = ""
    text: str

class BenchmarkDataset(Protocol):
    name: str

    def questions(self, split: str = "validation") -> Iterator[QAPair]: ...
    def corpus(self) -> Iterator[Passage]: ...
    def download(self, cache_dir: str = "~/.cache/nano-bench") -> None: ...
```

**Priority:** P1 (blocks context recall metric and good DX)

---

### 2. LLM Call Cache (1.2) — 50% Complete

#### ✅ What Exists
- `BenchmarkLLMCache` class with get/set operations
- Hash-based cache key generation using `compute_args_hash`
- Persistent storage via `JsonKVStorage`
- Batch operations (`get_batch`, `set_batch`)
- Cache statistics (`stats()` method)
- Cache clearing (`clear()` method)

#### ❌ What's Missing

1. **No hit/miss tracking**
   - **Roadmap spec:** `self.hits = 0`, `self.misses = 0`, `hit_rate` in stats
   - **Current:** `stats()` only returns `total_entries` and `models`
   - **Impact:** Cannot measure cache effectiveness, critical for cost optimization

2. **No integration with experiment runner**
   - **Roadmap spec:** "wrap `GraphRAGConfig.best_model_func` and `cheap_model_func`"
   - **Current:** Cache exists but is never used by `ExperimentRunner`
   - **Impact:** Zero cost savings, cache is dead code

3. **No cache enable/disable flag**
   - **Roadmap spec:** `enabled: bool = True` parameter
   - **Current:** No way to disable cache at runtime
   - **Impact:** Cannot easily test with/without cache

#### 📋 Proposed Fixes

```python
# nano_graphrag/_benchmark/cache.py

@dataclass
class BenchmarkLLMCache:
    storage: BaseKVStorage
    cache_name: str = "benchmark_llm_cache"
    enabled: bool = True
    hits: int = 0
    misses: int = 0

    async def get(self, prompt: str, model: str, system_prompt: Optional[str] = None) -> Optional[str]:
        if not self.enabled:
            return None

        result = await self._get_from_storage(prompt, model, system_prompt)
        if result is not None:
            self.hits += 1
            return result
        self.misses += 1
        return None

    def stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total > 0 else 0.0,
            "total_entries": ...,
        }

# nano_graphrag/_benchmark/runner.py

class ExperimentRunner:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self._cache = create_benchmark_cache(
            working_dir=config.graphrag_config.get("working_dir", "./cache"),
            enabled=config.graphrag_config.get("enable_llm_cache", True),
        )

    def _create_graphrag(self) -> GraphRAG:
        rag_config = GraphRAGConfig.from_dict(self.config.graphrag_config)

        # Wrap LLM functions with cache
        if self._cache.enabled:
            original_best = rag_config.best_model_func
            original_cheap = rag_config.cheap_model_func

            rag_config.best_model_func = self._cache.wrap(original_best)
            rag_config.cheap_model_func = self._cache.wrap(original_cheap)

        return GraphRAG.from_config(rag_config)
```

**Priority:** P0 (core feature for cost-effective experimentation)

---

### 3. Metrics Suite (1.3) — 60% Complete

#### ✅ What Exists
- `Metric` ABC with `compute()` method
- `ExactMatchMetric` with normalization
- `TokenF1Metric` with token overlap
- `MetricSuite` for batch computation
- Optional Ragas integration (`RagasFaithfulnessMetric`, `RagasAnswerRelevanceMetric`)

#### ❌ What's Missing

1. **No `MetricResult` dataclass**
   - **Roadmap spec:** `@dataclass class MetricResult` with typed fields
   - **Current:** Returns generic `Dict[str, float]`
   - **Impact:** Less type-safe, harder to extend with metadata

2. **No native context recall metric**
   - **Roadmap spec:** "Context Recall (LLM-as-judge) — score whether all necessary supporting facts from gold.supporting_facts appear"
   - **Current:** Only Ragas wrapper, no native implementation
   - **Impact:** Cannot evaluate retrieval quality without Ragas dependency

3. **No per-question timing**
   - **Roadmap spec:** Timing in `MetricResult.metadata`
   - **Current:** Only total experiment duration
   - **Impact:** Cannot identify slow questions, no latency profiling

4. **Ragas metrics are separate classes**
   - **Roadmap spec:** Unified in `MetricSuite`
   - **Current:** Separate, not in baseline suite
   - **Impact:** Inconsistent API

#### 📋 Proposed Fixes

```python
# nano_graphrag/_benchmark/metrics.py

@dataclass
class MetricResult:
    """Structured metric result with metadata."""
    exact_match: float
    token_f1: float
    faithfulness: Optional[float] = None
    context_recall: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class NativeContextRecallMetric(Metric):
    """Native context recall without Ragas dependency."""

    async def compute(self, prediction: str, gold: str, question: str = "", context: str = "") -> float:
        # Simple heuristic: check if gold supporting facts appear in context
        gold_facts = gold.get("supporting_facts", [])
        if not gold_facts:
            return 1.0  # No facts to check

        found = sum(1 for fact in gold_facts if fact.lower() in context.lower())
        return found / len(gold_facts)
```

**Priority:** P1 (important for retrieval evaluation)

---

### 4. Config-Driven Runner (1.4) — 70% Complete

#### ✅ What Exists
- YAML config loading (`BenchmarkConfig.from_yaml()`)
- CLI entry point (`examples/benchmarks/run_experiment.py`)
- Dry-run support (`--dry-run`)
- Config overrides via CLI args
- JSON result output

#### ❌ What's Missing

1. **Wrong config schema**
   - **Roadmap spec:** Nested `dataset:`, `graphrag:`, `query:`, `cache:`, `metrics:`, `output:` sections
   - **Current:** Flat structure with inconsistent naming
   - **Impact:** Not compliant with roadmap, harder to read

2. **No `bench` Python module**
   - **Roadmap spec:** `python -m bench.run --config ...`
   - **Current:** Must use `examples/benchmarks/run_experiment.py`
   - **Impact:** Longer command, not "importable" as a module

3. **No compare command**
   - **Roadmap spec:** `python -m bench.compare results/run_abc/ results/run_def/`
   - **Current:** No comparison tool
   - **Impact:** Cannot easily A/B test results

4. **No parallel execution**
   - **Roadmap spec:** `python -m bench.run --config-dir experiments/ --parallel 4`
   - **Current:** Single config only
   - **Impact:** Slower iteration when testing multiple configs

5. **Missing config options**
   - **Roadmap spec:** `cache.enabled`, `cache.backend`, `metrics.llm_judge`
   - **Current:** No cache config in YAML, no LLM judge config
   - **Impact:** Cannot control cache or LLM judge via config

#### 📋 Proposed Fixes

```yaml
# experiments/multihop_musique.yaml (ROADMAP COMPLIANT)
name: multihop_musique_baseline
version: "1.0"
description: "Local-mode GraphRAG baseline on MuSiQue dev set"

dataset:
  name: musique
  split: validation
  max_samples: 200
  auto_download: true  # NEW: automatic dataset download

graphrag:
  working_dir: ./workdirs/musique_baseline
  llm_model: gpt-4o-mini
  embedding_model: text-embedding-3-small
  chunk_func: chunking_by_token_size
  chunk_token_size: 1200
  chunk_overlap_token_size: 100

query:
  modes:
    - local
    - naive
  param_overrides:
    top_k: 60

cache:
  enabled: true
  backend: disk  # disk, redis, memory

metrics:
  exact_match: true
  token_f1: true
  llm_judge:
    enabled: false
    model: gpt-4o-mini

output:
  results_dir: ./results
  save_predictions: true
```

```bash
# NEW: Module-based CLI
python -m bench.run --config experiments/multihop_musique.yaml
python -m bench.compare results/run_abc/ results/run_def/
```

**Priority:** P1 (important for usability and roadmap compliance)

---

## Missing Deliverables

### Completely Missing Features

1. **Dataset Auto-Download** (1.1)
   - No HuggingFace integration
   - No `~/.cache/nano-bench` caching
   - **Estimated effort:** 2-3 days

2. **Cache Integration** (1.2)
   - Cache defined but not used
   - No hit/miss tracking
   - **Estimated effort:** 1-2 days

3. **Context Recall Metric** (1.3)
   - Requires `supporting_facts` in dataset
   - No native implementation
   - **Estimated effort:** 1-2 days

4. **Compare Command** (1.4)
   - No A/B testing diff tool
   - **Estimated effort:** 2-3 days

5. **Module Structure** (1.4)
   - Should be `bench/` not `nano_graphrag/_benchmark/`
   - Should be importable as `bench.run`
   - **Estimated effort:** 1 day (restructure)

---

## Recommendations

### Immediate Actions (Week 1)

1. **Integrate the cache** — Highest ROI for cost savings
   - Add hit/miss tracking to `BenchmarkLLMCache`
   - Wrap LLM functions in `ExperimentRunner._create_graphrag()`
   - Print cache stats in results summary

2. **Fix dataset types** — Critical for context recall
   - Add `QAPair` and `Passage` dataclasses
   - Preserve `supporting_facts` from datasets
   - Implement native `ContextRecallMetric`

3. **Add auto-download** — Huge DX improvement
   - Integrate HuggingFace `datasets` library
   - Implement `download()` method for each dataset
   - Cache in `~/.cache/nano-bench`

### Short-term (Weeks 2-3)

4. **Restructure to `bench/` module**
   - Move `nano_graphrag/_benchmark/` → `bench/`
   - Create `bench/__main__.py` for `python -m bench` CLI
   - Update all imports

5. **Implement compare command**
   - `bench/compare.py` with diff tables
   - Support comparing 2+ result directories
   - Markdown table output

6. **Update config schema**
   - Migrate to nested structure from roadmap
   - Add `cache:` section
   - Add `metrics.llm_judge:` section

### Long-term (Week 4+)

7. **Add parallel execution**
   - `--config-dir` flag
   - `--parallel N` flag
   - Process pool for concurrent runs

8. **Add results backends**
   - MLflow integration (optional)
   - W&B integration (optional)
   - JSON backend (default)

---

## Effort Estimation

| Task | Effort | Priority | Dependencies |
|------|--------|----------|--------------|
| Cache integration + hit/miss | 1-2 days | P0 | None |
| Dataset type safety (QAPair/Passage) | 1 day | P1 | None |
| Auto-download + caching | 2-3 days | P1 | None |
| Native context recall | 1-2 days | P1 | Dataset types |
| Module restructure (bench/) | 1 day | P1 | None |
| Compare command | 2-3 days | P1 | Module restructure |
| Config schema migration | 1 day | P1 | None |
| Parallel execution | 2-3 days | P2 | Module restructure |
| MLflow/W&B backends | 3-5 days | P3 | None |

**Total to complete Phase 1:** ~15-20 days

---

## Conclusion

The current implementation provides a **solid foundation** but is **not yet production-ready** according to the roadmap's standards. The biggest gaps are:

1. **Cache is dead code** — defined but never used
2. **No auto-download** — high friction for new users
3. **Weak typing** — generic Dicts instead of dataclasses
4. **No A/B testing** — missing compare command

**Recommended next step:** Integrate the cache (1-2 days) for immediate cost savings, then tackle dataset auto-download for better DX.

---

*Last updated: 2026-03-23*
