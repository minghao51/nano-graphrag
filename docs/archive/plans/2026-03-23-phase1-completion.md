# Phase 1 Benchmark Infrastructure — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete the Phase 1 benchmark infrastructure to match the technical roadmap specifications, enabling reproducible, cost-effective multi-hop RAG experiments with one-command execution.

**Architecture:**
- Incremental implementation: P0 (cache) → P1 (datasets, module structure, compare) → P2 (polish)
- Keep `nano_graphrag/` core untouched; all changes in `_benchmark/` or new `bench/` module
- Maintain backward compatibility with existing experiment configs
- Test-driven development: write failing test → implement → verify → commit

**Tech Stack:**
- Python 3.9+, async/await throughout
- PyYAML for config, pytest for testing
- HuggingFace `datasets` for auto-download (optional dependency)
- Existing: `JsonKVStorage`, `compute_args_hash`, `GraphRAG`

**Effort Estimate:** ~15-20 days total

---

## Task 1: Cache Integration (P0 - Highest ROI)

**Why first:** Immediate cost savings, unlocks fast iteration, currently dead code.

**Files:**
- Modify: `nano_graphrag/_benchmark/cache.py`
- Modify: `nano_graphrag/_benchmark/runner.py`
- Create: `tests/benchmark/test_cache.py`

### Step 1.1: Add hit/miss tracking to cache

**Step 1.1.1: Write failing test for cache statistics**

```python
# tests/benchmark/test_cache.py
import pytest
from nano_graphrag._benchmark import create_benchmark_cache
import tempfile
import os

@pytest.mark.asyncio
async def test_cache_tracks_hits_and_misses():
    """Cache should track hits and misses correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = create_benchmark_cache(tmpdir, enabled=True)

        # First call should miss
        result1 = await cache.get("test prompt", "gpt-4o-mini")
        assert result1 is None
        assert cache.misses == 1
        assert cache.hits == 0

        # Set cache
        await cache.set("test prompt", "gpt-4o-mini", "cached response")

        # Second call should hit
        result2 = await cache.get("test prompt", "gpt-4o-mini")
        assert result2 == "cached response"
        assert cache.hits == 1
        assert cache.misses == 1

        # Stats should include hit rate
        stats = await cache.stats()
        assert stats["hit_rate"] == 0.5  # 1 hit out of 2 calls
```

**Step 1.1.2: Run test to verify it fails**

```bash
uv run pytest tests/benchmark/test_cache.py::test_cache_tracks_hits_and_misses -v
```

Expected: FAIL with `AttributeError: 'BenchmarkLLMCache' object has no attribute 'misses'`

**Step 1.1.3: Implement hit/miss tracking in cache**

```python
# nano_graphrag/_benchmark/cache.py

@dataclass
class BenchmarkLLMCache:
    """Persistent LLM cache using BaseKVStorage pattern.

    Wraps the existing hashing_kv pattern used in GraphRAG for benchmark experiments.
    """

    storage: BaseKVStorage
    cache_name: str = "benchmark_llm_cache"
    enabled: bool = True  # NEW: enable/disable flag

    # NEW: Hit/miss tracking
    hits: int = 0
    misses: int = 0

    async def get(
        self,
        prompt: str,
        model: str,
        system_prompt: Optional[str] = None,
    ) -> Optional[str]:
        """Get cached response if available."""
        if not self.enabled:
            # Don't track when disabled
            return None

        cache_key = self._make_cache_key(prompt, model, system_prompt)
        result = await self.storage.get_by_id(cache_key)

        if result is not None:
            self.hits += 1  # NEW: track hit
            return result.get("response")

        self.misses += 1  # NEW: track miss
        return None

    async def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        all_keys = await self.storage.all_keys()
        cache_keys = [k for k in all_keys if k.startswith(f"{self.cache_name}:")]
        entries = await self.storage.get_by_ids(cache_keys)
        models = set()

        for entry in entries:
            if entry:
                models.add(entry.get("model", "unknown"))

        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total > 0 else 0.0,  # NEW
            "total_entries": len(cache_keys),
            "models": sorted(list(models)),
        }
```

**Step 1.1.4: Run test to verify it passes**

```bash
uv run pytest tests/benchmark/test_cache.py::test_cache_tracks_hits_and_misses -v
```

Expected: PASS

**Step 1.1.5: Commit**

```bash
git add nano_graphrag/_benchmark/cache.py tests/benchmark/test_cache.py
git commit -m "feat(benchmark): add cache hit/miss tracking and enable/disable flag"
```

---

### Step 1.2: Add cache wrapper function

**Step 1.2.1: Write failing test for wrapper function**

```python
# tests/benchmark/test_cache.py (add to existing file)

@pytest.mark.asyncio
async def test_cache_wrapper_decorates_llm_function():
    """Cache.wrap() should add caching to any LLM function."""
    from nano_graphrag._benchmark import create_benchmark_cache
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = create_benchmark_cache(tmpdir, enabled=True)
        call_count = {"value": 0}

        # Mock LLM function
        async def mock_llm(prompt, model="gpt-4o-mini", system_prompt=None):
            call_count["value"] += 1
            return f"Response to: {prompt}"

        # Wrap the function
        wrapped_llm = cache.wrap(mock_llm)

        # First call should invoke the function
        result1 = await wrapped_llm("test prompt", model="gpt-4o-mini")
        assert result1 == "Response to: test prompt"
        assert call_count["value"] == 1

        # Second call should hit cache
        result2 = await wrapped_llm("test prompt", model="gpt-4o-mini")
        assert result2 == "Response to: test prompt"
        assert call_count["value"] == 1  # No additional call

        # Different prompt should miss
        result3 = await wrapped_llm("different prompt", model="gpt-4o-mini")
        assert result3 == "Response to: different prompt"
        assert call_count["value"] == 2
```

**Step 1.2.2: Run test to verify it fails**

```bash
uv run pytest tests/benchmark/test_cache.py::test_cache_wrapper_decorates_llm_function -v
```

Expected: FAIL with `AttributeError: 'BenchmarkLLMCache' object has no attribute 'wrap'`

**Step 1.2.3: Implement wrap function**

```python
# nano_graphrag/_benchmark/cache.py (add to BenchmarkLLMCache class)

def wrap(self, llm_func):
    """Wrap an LLM function with transparent caching.

    Args:
        llm_func: Async LLM function with signature (prompt, model, system_prompt, **kwargs)

    Returns:
        Wrapped async function that checks cache before calling llm_func
    """
    async def cached_llm(prompt, model=None, system_prompt=None, **kwargs):
        # Normalize model parameter (GraphRAG sometimes passes it via kwargs)
        if model is None:
            model = kwargs.get("model", "gpt-4o-mini")

        # Check cache
        cached_response = await self.get(prompt, model, system_prompt)
        if cached_response is not None:
            return cached_response

        # Cache miss - call the original function
        response = await llm_func(prompt, model=model, system_prompt=system_prompt, **kwargs)

        # Store in cache
        await self.set(prompt, model, response, system_prompt)

        return response

    return cached_llm
```

**Step 1.2.4: Run test to verify it passes**

```bash
uv run pytest tests/benchmark/test_cache.py::test_cache_wrapper_decorates_llm_function -v
```

Expected: PASS

**Step 1.2.5: Commit**

```bash
git add nano_graphrag/_benchmark/cache.py tests/benchmark/test_cache.py
git commit -m "feat(benchmark): add cache.wrap() method for LLM function decoration"
```

---

### Step 1.3: Integrate cache into ExperimentRunner

**Step 1.3.1: Write test for cache integration**

```python
# tests/benchmark/test_runner.py (create new file)

import pytest
import tempfile
from nano_graphrag._benchmark import BenchmarkConfig, ExperimentRunner
from pathlib import Path

@pytest.mark.asyncio
async def test_runner_uses_cache_when_enabled():
    """ExperimentRunner should wrap LLM functions when cache is enabled."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a minimal config
        config_dict = {
            "dataset_name": "multihop_rag",
            "dataset_path": "tests/fixtures/sample_questions.json",
            "corpus_path": "tests/fixtures/sample_corpus.json",
            "max_samples": 2,
            "graphrag_config": {
                "working_dir": tmpdir,
                "llm_model": "gpt-4o-mini",
                "embedding_model": "text-embedding-3-small",
                "enable_llm_cache": True,  # Enable cache
            },
            "query_modes": ["local"],
            "metrics": ["exact_match"],
            "output_dir": tmpdir,
            "experiment_name": "test_cache",
        }

        config = BenchmarkConfig.from_dict(config_dict)
        runner = ExperimentRunner(config)

        # Verify cache was created
        assert runner._cache is not None
        assert runner._cache.enabled is True
```

**Step 1.3.2: Create test fixtures**

```python
# tests/fixtures/sample_questions.json
[
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "What is 2+2?", "answer": "4"}
]

# tests/fixtures/sample_corpus.json
[
    {"content": "France is a country in Europe. Its capital is Paris."},
    {"content": "Basic arithmetic: 2+2 equals 4."}
]
```

**Step 1.3.3: Run test to verify it fails**

```bash
uv run pytest tests/benchmark/test_runner.py::test_runner_uses_cache_when_enabled -v
```

Expected: FAIL with `AttributeError: 'ExperimentRunner' object has no attribute '_cache'`

**Step 1.3.4: Implement cache integration in runner**

```python
# nano_graphrag/_benchmark/runner.py

from .cache import create_benchmark_cache

class ExperimentRunner:
    """Run benchmark experiments from config."""

    def __init__(self, config: BenchmarkConfig):
        """Initialize runner with configuration."""
        self.config = config
        self._dataset: Optional[BenchmarkDataset] = None
        self._rag: Optional[GraphRAG] = None
        self._metric_suite: Optional[MetricSuite] = None
        self._cache = self._create_cache()  # NEW: create cache instance

    def _create_cache(self) -> Optional[BenchmarkLLMCache]:
        """Create cache instance if enabled in config."""
        from .cache import create_benchmark_cache

        # Check if cache is enabled in graphrag config
        cache_enabled = self.config.graphrag_config.get("enable_llm_cache", False)
        if not cache_enabled:
            return None

        working_dir = self.config.graphrag_config.get("working_dir", "./nano_graphrag")
        return create_benchmark_cache(working_dir, enabled=True)

    def _create_graphrag(self) -> GraphRAG:
        """Create GraphRAG instance from config."""
        rag_config = GraphRAGConfig.from_dict(self.config.graphrag_config)

        # NEW: Wrap LLM functions with cache if enabled
        if self._cache is not None and self._cache.enabled:
            original_best = rag_config.best_model_func
            original_cheap = rag_config.cheap_model_func

            rag_config.best_model_func = self._cache.wrap(original_best)
            rag_config.cheap_model_func = self._cache.wrap(original_cheap)

        return GraphRAG.from_config(rag_config)
```

**Step 1.3.5: Update BenchmarkConfig to support cache config**

```python
# nano_graphrag/_benchmark/runner.py (update BenchmarkConfig)

@dataclass
class BenchmarkConfig:
    # ... existing fields ...

    @classmethod
    def from_yaml(cls, path: str) -> "BenchmarkConfig":
        """Load config from YAML file."""
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required to load YAML configs. Install with: uv add pyyaml"
            )

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        # NEW: Normalize enable_llm_cache to graphrag_config
        if "cache" in data:
            cache_config = data.pop("cache")
            if "enabled" in cache_config:
                data.setdefault("graphrag_config", {})["enable_llm_cache"] = cache_config["enabled"]

        return cls(**data)
```

**Step 1.3.6: Run test to verify it passes**

```bash
uv run pytest tests/benchmark/test_runner.py::test_runner_uses_cache_when_enabled -v
```

Expected: PASS

**Step 1.3.7: Commit**

```bash
git add nano_graphrag/_benchmark/runner.py tests/benchmark/test_runner.py tests/fixtures/
git commit -m "feat(benchmark): integrate cache into ExperimentRunner"
```

---

### Step 1.4: Display cache stats in results

**Step 1.4.1: Write test for cache stats in results**

```python
# tests/benchmark/test_runner.py (add to existing file)

@pytest.mark.asyncio
async def test_runner_includes_cache_stats_in_results():
    """ExperimentResult should include cache statistics."""
    # Similar setup as previous test, but verify result contains cache stats
    ...
    result = await runner.run()

    # Should include cache stats if cache was used
    assert "cache_stats" in result.__dict__
    assert result.cache_stats["hits"] >= 0
    assert result.cache_stats["misses"] >= 0
```

**Step 1.4.2: Run test to verify it fails**

```bash
uv run pytest tests/benchmark/test_runner.py::test_runner_includes_cache_stats_in_results -v
```

Expected: FAIL with `AttributeError: 'ExperimentResult' object has no attribute 'cache_stats'`

**Step 1.4.3: Add cache_stats to ExperimentResult**

```python
# nano_graphrag/_benchmark/runner.py

@dataclass
class ExperimentResult:
    """Result with config for full reproducibility."""

    experiment_name: str
    timestamp: str
    config: BenchmarkConfig
    mode_results: Dict[str, Dict[str, float]]
    predictions: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    duration_seconds: float = 0.0
    cache_stats: Optional[Dict[str, Any]] = None  # NEW

    def save(self, output_dir: str) -> str:
        """Save results to JSON file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp_safe = self.timestamp.replace(":", "-").replace(" ", "_")
        filename = f"{self.experiment_name}_{timestamp_safe}.json"
        filepath = output_path / filename

        result_dict = {
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp,
            "config": self.config.to_dict(),
            "mode_results": self.mode_results,
            "predictions": self.predictions,
            "duration_seconds": self.duration_seconds,
            "cache_stats": self.cache_stats,  # NEW
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)

        return str(filepath)
```

**Step 1.4.4: Capture cache stats in runner**

```python
# nano_graphrag/_benchmark/runner.py (update ExperimentRunner.run() method)

async def run(self) -> ExperimentResult:
    """Execute full experiment."""
    start_time = datetime.now()
    timestamp = start_time.isoformat()

    # ... existing dataset loading, insertion, query code ...

    # Compute duration
    end_time = datetime.now()
    duration_seconds = (end_time - start_time).total_seconds()

    # NEW: Capture cache stats
    cache_stats = None
    if self._cache is not None:
        cache_stats = await self._cache.stats()

    # Create result
    result = ExperimentResult(
        experiment_name=self.config.experiment_name,
        timestamp=timestamp,
        config=self.config,
        mode_results=mode_results,
        predictions=all_predictions,
        duration_seconds=duration_seconds,
        cache_stats=cache_stats,  # NEW
    )

    # Save results
    output_path = result.save(self.config.output_dir)
    print(f"\n[Results] Saved to {output_path}")

    # NEW: Print cache stats
    if cache_stats:
        print(f"\n[Cache] Hit rate: {cache_stats['hit_rate']:.2%}")
        print(f"[Cache] {cache_stats['hits']} hits, {cache_stats['misses']} misses")

    return result
```

**Step 1.4.5: Run test to verify it passes**

```bash
uv run pytest tests/benchmark/test_runner.py::test_runner_includes_cache_stats_in_results -v
```

Expected: PASS

**Step 1.4.6: Commit**

```bash
git add nano_graphrag/_benchmark/runner.py tests/benchmark/test_runner.py
git commit -m "feat(benchmark): include cache statistics in experiment results"
```

---

## Task 2: Dataset Type Safety (P1)

**Why second:** Unlocks context recall metric, improves type safety and DX.

**Files:**
- Modify: `nano_graphrag/_benchmark/datasets.py`
- Modify: `nano_graphrag/_benchmark/metrics.py`
- Modify: `nano_graphrag/_benchmark/runner.py`
- Create: `tests/benchmark/test_datasets.py`

### Step 2.1: Add QAPair and Passage dataclasses

**Step 2.1.1: Write failing test for typed datasets**

```python
# tests/benchmark/test_datasets.py

import pytest
from nano_graphrag._benchmark import MultiHopRAGDataset
import tempfile
import json

def test_dataset_returns_typed_qa_pairs():
    """Dataset questions() should return typed QAPair objects."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test data
        questions_path = Path(tmpdir) / "questions.json"
        questions_data = [
            {
                "id": "q1",
                "question": "What is the capital of France?",
                "answer": "Paris",
                "supporting_facts": ["France is in Europe", "Paris is the capital"],
            }
        ]
        questions_path.write_text(json.dumps(questions_data))

        corpus_path = Path(tmpdir) / "corpus.json"
        corpus_data = [{"content": "France is a country."}]
        corpus_path.write_text(json.dumps(corpus_data))

        # Load dataset
        dataset = MultiHopRAGDataset(
            questions_path=str(questions_path),
            corpus_path=str(corpus_path),
        )

        # Verify typed return
        questions = dataset.questions()
        assert len(questions) == 1
        qa = questions[0]

        # Should be a QAPair, not a dict
        assert hasattr(qa, "id")
        assert hasattr(qa, "question")
        assert hasattr(qa, "answer")
        assert hasattr(qa, "supporting_facts")
        assert qa.id == "q1"
        assert qa.supporting_facts == ["France is in Europe", "Paris is the capital"]
```

**Step 2.1.2: Run test to verify it fails**

```bash
uv run pytest tests/benchmark/test_datasets.py::test_dataset_returns_typed_qa_pairs -v
```

Expected: FAIL with `AssertionError` or `AttributeError` (currently returns dict)

**Step 2.1.3: Implement QAPair and Passage dataclasses**

```python
# nano_graphrag/_benchmark/datasets.py

from dataclasses import dataclass, field
from typing import Iterator, List, Protocol

@dataclass
class QAPair:
    """Question-answer pair with metadata."""

    id: str
    question: str
    answer: str
    supporting_facts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Passage:
    """Corpus passage/document."""

    id: str
    title: str = ""
    text: str = ""

class BenchmarkDataset(Protocol):
    """Protocol for benchmark datasets."""

    name: str

    def questions(self, split: str = "test") -> Iterator[QAPair]:
        """Return QAPair objects with questions and answers."""
        ...

    def corpus(self) -> Iterator[Passage]:
        """Return Passage objects for corpus documents."""
        ...

    def download(self, cache_dir: str = "~/.cache/nano-bench") -> None:
        """Download dataset if not available locally."""
        ...
```

**Step 2.1.4: Update MultiHopRAGDataset to use QAPair**

```python
# nano_graphrag/_benchmark/datasets.py

@dataclass
class MultiHopRAGDataset:
    """MultiHop-RAG dataset loader."""

    questions_path: str
    corpus_path: str
    max_samples: int = -1

    def questions(self, split: str = "test") -> Iterator[QAPair]:
        """Load questions from JSON file."""
        with open(self.questions_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Apply max_samples limit
        if self.max_samples > 0:
            data = data[: self.max_samples]

        # Convert to QAPair objects
        for i, item in enumerate(data):
            yield QAPair(
                id=item.get("id", f"q_{i}"),
                question=item["question"],
                answer=item["answer"],
                supporting_facts=item.get("supporting_facts", []),
                metadata={k: v for k, v in item.items() if k not in ["id", "question", "answer", "supporting_facts"]},
            )

    def corpus(self) -> Iterator[Passage]:
        """Load corpus documents from JSON file."""
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for i, doc in enumerate(data):
            if isinstance(doc, dict):
                content = doc.get("content", "")
            elif isinstance(doc, str):
                content = doc
            else:
                continue

            if content.strip():
                yield Passage(
                    id=doc.get("id", f"doc_{i}"),
                    title=doc.get("title", ""),
                    text=content.strip(),
                )

    def download(self, cache_dir: str = "~/.cache/nano-bench") -> None:
        """Download MultiHop-RAG dataset from HuggingFace."""
        from pathlib import Path

        cache_dir = Path(cache_dir).expanduser()
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Implementation will be added in Task 3 (Auto-download)
        raise NotImplementedError(
            "Auto-download not yet implemented. "
            "Please download dataset manually and provide paths."
        )
```

**Step 2.1.5: Update other dataset classes similarly**

Apply same pattern to `HotpotQADataset`, `MuSiQueDataset`, `TwoWikiMultiHopQADataset`.

**Step 2.1.6: Run test to verify it passes**

```bash
uv run pytest tests/benchmark/test_datasets.py::test_dataset_returns_typed_qa_pairs -v
```

Expected: PASS

**Step 2.1.7: Commit**

```bash
git add nano_graphrag/_benchmark/datasets.py tests/benchmark/test_datasets.py
git commit -m "feat(benchmark): add QAPair and Passage dataclasses for type safety"
```

---

### Step 2.2: Implement native context recall metric

**Step 2.2.1: Write failing test for context recall**

```python
# tests/benchmark/test_metrics.py

import pytest
from nano_graphrag._benchmark import NativeContextRecallMetric, QAPair

@pytest.mark.asyncio
async def test_context_recall_with_supporting_facts():
    """Context recall should measure if supporting facts appear in retrieved context."""
    metric = NativeContextRecallMetric()

    qa_pair = QAPair(
        id="q1",
        question="What is the capital of France?",
        answer="Paris",
        supporting_facts=["France is in Europe", "Paris is the capital"],
    )

    context = "France is a country located in Europe. Paris is known as the capital."

    score = await metric.compute(
        prediction="Paris",
        gold=qa_pair,
        question="What is the capital of France?",
        context=context,
    )

    # Both facts should be found
    assert score == 1.0

@pytest.mark.asyncio
async def test_context_recall_partial_match():
    """Context recall should handle partial matches."""
    metric = NativeContextRecallMetric()

    qa_pair = QAPair(
        id="q1",
        question="Test",
        answer="Test",
        supporting_facts=["Fact 1", "Fact 2", "Fact 3"],
    )

    context = "Only Fact 1 is mentioned here."

    score = await metric.compute(
        prediction="Test",
        gold=qa_pair,
        question="Test",
        context=context,
    )

    # Only 1 out of 3 facts found
    assert score == 1.0 / 3.0
```

**Step 2.2.2: Run test to verify it fails**

```bash
uv run pytest tests/benchmark/test_metrics.py::test_context_recall_with_supporting_facts -v
```

Expected: FAIL with `ImportError: cannot import name 'NativeContextRecallMetric'`

**Step 2.2.3: Implement NativeContextRecallMetric**

```python
# nano_graphrag/_benchmark/metrics.py

@dataclass
class NativeContextRecallMetric(Metric):
    """Native context recall metric without Ragas dependency.

    Measures what fraction of gold supporting facts appear in the retrieved context.
    """

    async def compute(
        self,
        prediction: str,
        gold: QAPair,  # Note: changed type hint
        question: str = "",
        context: str = "",
    ) -> float:
        """Compute context recall score."""
        # Extract supporting facts from gold QAPair
        supporting_facts = getattr(gold, "supporting_facts", [])

        if not supporting_facts:
            # No facts to check, return perfect score
            return 1.0

        if not context:
            # No context provided, return 0
            return 0.0

        # Check how many facts appear in context (case-insensitive)
        context_lower = context.lower()
        found_count = sum(
            1 for fact in supporting_facts
            if fact.lower() in context_lower
        )

        return found_count / len(supporting_facts)
```

**Step 2.2.4: Update Metric base class signature**

```python
# nano_graphrag/_benchmark/metrics.py

class Metric(ABC):
    """Abstract base class for evaluation metrics."""

    @abstractmethod
    async def compute(
        self,
        prediction: str,
        gold: Union[str, QAPair],  # NEW: accept both string and QAPair
        question: str = "",
        context: str = "",
    ) -> float:
        """Compute the metric score."""
        ...
```

**Step 2.2.5: Update existing metrics to handle QAPair**

```python
# nano_graphrag/_benchmark/metrics.py

@dataclass
class ExactMatchMetric(Metric):
    """Normalized exact match metric."""

    case_sensitive: bool = False
    remove_articles: bool = True

    async def compute(
        self,
        prediction: str,
        gold: Union[str, QAPair],
        question: str = "",
        context: str = "",
    ) -> float:
        """Compute exact match score."""
        # Extract gold answer if QAPair
        if isinstance(gold, QAPair):
            gold = gold.answer

        # ... rest of existing implementation ...
```

Apply same change to `TokenF1Metric`.

**Step 2.2.6: Run test to verify it passes**

```bash
uv run pytest tests/benchmark/test_metrics.py -v
```

Expected: PASS

**Step 2.2.7: Commit**

```bash
git add nano_graphrag/_benchmark/metrics.py tests/benchmark/test_metrics.py
git commit -m "feat(benchmark): add native context recall metric"
```

---

### Step 2.3: Update runner to pass QAPair to metrics

**Step 2.3.1: Update ExperimentRunner to use new dataset types**

```python
# nano_graphrag/_benchmark/runner.py

async def run(self) -> ExperimentResult:
    """Execute full experiment."""
    # ... existing code ...

    # Load dataset
    self._dataset = self._load_dataset()
    questions_list = list(self._dataset.questions(split=self.config.dataset_split))  # NEW: convert to list
    corpus_list = list(self._dataset.corpus())  # NEW: convert to list

    print(f"[Dataset] Loaded {len(questions_list)} questions and {len(corpus_list)} corpus documents")

    # Insert corpus
    self._rag = self._create_graphrag()
    print(f"[Index] Inserting {len(corpus_list)} documents into GraphRAG...")

    # Convert Passage to dict for insertion
    corpus_dict = {doc.id: doc.text for doc in corpus_list}
    await self._rag.ainsert_documents(corpus_dict)

    # ... rest of existing code, but use questions_list[i] instead of questions[i] ...

    for i, qa in enumerate(questions_list):
        question = qa.question  # NEW: access QAPair field
        gold = qa.answer  # NEW: access QAPair field

        # ... query code ...

        # Compute metrics - pass QAPair instead of dict
        for name, metric in self._metric_suite.metrics.items():
            score = await metric.compute(
                prediction=pred,
                gold=qa,  # NEW: pass entire QAPair
                question=question,
                context="",
            )
            metric_sums[name] += score

    # ... rest of existing code ...
```

**Step 2.3.2: Update predictions storage**

```python
# nano_graphrag/_benchmark/runner.py

# Store predictions
all_predictions[mode] = [
    {
        "question": qa.question,  # NEW: access QAPair field
        "prediction": pred,
        "gold": qa.answer,  # NEW: access QAPair field
    }
    for qa, pred in zip(questions_list, predictions)
]
```

**Step 2.3.3: Commit**

```bash
git add nano_graphrag/_benchmark/runner.py
git commit -m "refactor(benchmark): update runner to use typed QAPair and Passage"
```

---

## Task 3: Dataset Auto-Download (P1)

**Why third:** Huge DX improvement, removes manual download friction.

**Files:**
- Modify: `nano_graphrag/_benchmark/datasets.py`
- Modify: `pyproject.toml`
- Create: `tests/benchmark/test_download.py`

### Step 3.1: Add HuggingFace datasets dependency

**Step 3.1.1: Add optional dependency**

```bash
uv add --optional datasets
```

**Step 3.1.2: Update pyproject.toml**

```toml
# pyproject.toml

[project.optional-dependencies]
benchmark = [
    "datasets",  # For auto-download
    "ragas",     # For LLM-as-judge metrics
]
```

**Step 3.1.3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps(benchmark): add HuggingFace datasets as optional dependency"
```

---

### Step 3.2: Implement download() for each dataset

**Step 3.2.1: Write failing test for auto-download**

```python
# tests/benchmark/test_download.py

import pytest
from nano_graphrag._benchmark import MuSiQueDataset
import tempfile

def test_musique_dataset_download():
    """MuSiQue dataset should download from HuggingFace."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset = MuSiQueDataset(
            data_path="",  # Will be set by download
            split="dev",
            max_samples=10,
        )

        # Download dataset
        dataset.download(cache_dir=tmpdir)

        # Verify files were created
        assert (Path(tmpdir) / "musique" / "dev.json").exists()

        # Now load from downloaded path
        dataset.data_path = str(Path(tmpdir) / "musique" / "dev.json")
        questions = list(dataset.questions())

        assert len(questions) == 10
        assert all(hasattr(q, "id") for q in questions)
```

**Step 3.2.2: Run test to verify it fails**

```bash
uv run pytest tests/benchmark/test_download.py::test_musique_dataset_download -v
```

Expected: FAIL with `NotImplementedError`

**Step 3.2.3: Implement download for MuSiQueDataset**

```python
# nano_graphrag/_benchmark/datasets.py

@dataclass
class MuSiQueDataset:
    """MuSiQue dataset loader (Multi-hop Question Answering)."""

    data_path: str
    split: str = "dev"
    max_samples: int = -1

    def download(self, cache_dir: str = "~/.cache/nano-bench") -> None:
        """Download MuSiQue dataset from HuggingFace."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "HuggingFace datasets library is required for auto-download. "
                "Install with: uv add --optional datasets"
            )

        from pathlib import Path

        cache_dir = Path(cache_dir).expanduser()
        dataset_dir = cache_dir / "musique"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        print(f"[Download] Loading MuSiQue {self.split} split from HuggingFace...")
        hf_dataset = load_dataset("dataset-source/musique", split=self.split)

        # Convert to our format
        questions_data = []
        for item in hf_dataset:
            questions_data.append({
                "id": item.get("id", f"musique_{len(questions_data)}"),
                "question": item["question"],
                "answer": item["answer"],
                "supporting_facts": item.get("question_decomposition", []),
            })

        # Save to JSON
        output_path = dataset_dir / f"{self.split}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(questions_data, f, indent=2, ensure_ascii=False)

        # Update data_path
        self.data_path = str(output_path)

        print(f"[Download] Saved to {output_path}")
```

**Step 3.2.4: Implement download for other datasets**

Apply similar implementation to:
- `HotpotQADataset` → `hotpotqa/hotpotqa` from HuggingFace
- `TwoWikiMultiHopQADataset` → `dataset-source/2wikimultihopqa` from HuggingFace
- `MultiHopRAGDataset` → Need to check actual HuggingFace repo name

**Step 3.2.5: Run test to verify it passes**

```bash
uv run pytest tests/benchmark/test_download.py::test_musique_dataset_download -v
```

Expected: PASS (may take a minute to download)

**Step 3.2.6: Commit**

```bash
git add nano_graphrag/_benchmark/datasets.py tests/benchmark/test_download.py
git commit -m "feat(benchmark): add auto-download for MuSiQue dataset"
```

---

### Step 3.3: Add auto_download flag to config

**Step 3.3.1: Update BenchmarkConfig**

```python
# nano_graphrag/_benchmark/runner.py

@dataclass
class BenchmarkConfig:
    # === Dataset ===
    dataset_name: str
    dataset_path: str
    corpus_path: Optional[str] = None
    dataset_split: str = "test"
    max_samples: int = -1
    auto_download: bool = False  # NEW: enable auto-download

    # ... rest of fields ...
```

**Step 3.3.2: Update ExperimentRunner to auto-download**

```python
# nano_graphrag/_benchmark/runner.py

class ExperimentRunner:
    def _load_dataset(self) -> BenchmarkDataset:
        """Load dataset based on config."""
        dataset_name = self.config.dataset_name.lower()

        if dataset_name == "multihop_rag":
            if not self.config.corpus_path:
                raise ValueError("corpus_path is required for MultiHopRAG dataset")

            dataset = MultiHopRAGDataset(
                questions_path=self.config.dataset_path,
                corpus_path=self.config.corpus_path,
                max_samples=self.config.max_samples,
            )
        elif dataset_name == "hotpotqa":
            from .datasets import HotpotQADataset

            dataset = HotpotQADataset(
                data_path=self.config.dataset_path,
                split=self.config.dataset_split,
                max_samples=self.config.max_samples,
            )
        # ... etc for other datasets ...

        # NEW: Auto-download if enabled
        if self.config.auto_download:
            cache_dir = self.config.graphrag_config.get("working_dir", "./nano_graphrag")
            cache_dir = str(Path(cache_dir) / "datasets")
            dataset.download(cache_dir=cache_dir)

        return dataset
```

**Step 3.3.3: Commit**

```bash
git add nano_graphrag/_benchmark/runner.py
git commit -m "feat(benchmark): add auto_download flag to config"
```

---

## Task 4: Module Restructure to `bench/` (P1)

**Why fourth:** Aligns with roadmap, enables `python -m bench` CLI.

**Files:**
- Create: `bench/` directory structure
- Move: `nano_graphrag/_benchmark/` → `bench/`
- Modify: All imports across codebase

### Step 4.1: Create bench/ directory structure

**Step 4.1.1: Create directory layout**

```bash
mkdir -p bench/datasets bench/metrics bench/retrievers bench/techniques
touch bench/__init__.py bench/__main__.py
touch bench/datasets/__init__.py bench/metrics/__init__.py
touch bench/config.py bench/runner.py bench/cache.py bench/compare.py
```

**Step 4.1.2: Move files from _benchmark to bench/**

```bash
# Copy files first
cp nano_graphrag/_benchmark/datasets.py bench/datasets/
cp nano_graphrag/_benchmark/metrics.py bench/metrics/
cp nano_graphrag/_benchmark/cache.py bench/cache.py
cp nano_graphrag/_benchmark/runner.py bench/runner.py
cp nano_graphrag/_benchmark/__init__.py bench/__init__.py

# Update imports in moved files
# (Will be done in subsequent steps)
```

**Step 4.1.3: Create bench/__main__.py for CLI**

```python
# bench/__main__.py

"""Entry point for 'python -m bench' CLI."""

import sys
from bench.run import main

if __name__ == "__main__":
    sys.exit(main())
```

**Step 4.1.4: Create bench/run.py**

```python
# bench/run.py

"""CLI for running GraphRAG benchmark experiments."""

import sys
from examples.benchmarks.run_experiment import main

# Re-export the existing CLI
if __name__ == "__main__":
    sys.exit(main())
```

**Step 4.1.5: Commit**

```bash
git add bench/
git commit -m "refactor(benchmark): create bench/ module structure"
```

---

### Step 4.2: Update imports in bench/ files

**Step 4.2.1: Update bench/datasets/__init__.py**

```python
# bench/datasets/__init__.py

from .base import BenchmarkDataset, QAPair, Passage
from .loaders import (
    MultiHopRAGDataset,
    HotpotQADataset,
    MuSiQueDataset,
    TwoWikiMultiHopQADataset,
)

__all__ = [
    "BenchmarkDataset",
    "QAPair",
    "Passage",
    "MultiHopRAGDataset",
    "HotpotQADataset",
    "MuSiQueDataset",
    "TwoWikiMultiHopQADataset",
]
```

**Step 4.2.2: Update bench/metrics/__init__.py**

```python
# bench/metrics/__init__.py

from .base import Metric, MetricSuite
from .metrics import ExactMatchMetric, TokenF1Metric, NativeContextRecallMetric

__all__ = [
    "Metric",
    "MetricSuite",
    "ExactMatchMetric",
    "TokenF1Metric",
    "NativeContextRecallMetric",
]
```

**Step 4.2.3: Update bench/__init__.py**

```python
# bench/__init__.py

"""Benchmark infrastructure for nano-graphrag."""

from .config import BenchmarkConfig
from .runner import ExperimentRunner, ExperimentResult
from .cache import create_benchmark_cache
from .datasets import BenchmarkDataset
from .metrics import MetricSuite, get_baseline_suite

__all__ = [
    "BenchmarkConfig",
    "ExperimentRunner",
    "ExperimentResult",
    "create_benchmark_cache",
    "BenchmarkDataset",
    "MetricSuite",
    "get_baseline_suite",
]
```

**Step 4.2.4: Commit**

```bash
git add bench/
git commit -m "refactor(benchmark): update imports in bench module"
```

---

### Step 4.3: Update examples/benchmarks to use bench/ module

**Step 4.3.1: Update run_experiment.py**

```python
# examples/benchmarks/run_experiment.py

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# OLD: from nano_graphrag._benchmark import BenchmarkConfig, ExperimentRunner
# NEW:
from bench import BenchmarkConfig, ExperimentRunner

# ... rest of file unchanged ...
```

**Step 4.3.2: Update tests to use bench/ module**

```python
# tests/benchmark/test_cache.py

# OLD: from nano_graphrag._benchmark import create_benchmark_cache
# NEW:
from bench import create_benchmark_cache

# ... rest of test unchanged ...
```

Apply to all test files.

**Step 4.3.3: Commit**

```bash
git add examples/benchmarks/run_experiment.py tests/benchmark/
git commit -m "refactor(benchmark): update imports to use bench module"
```

---

### Step 4.4: Deprecate nano_graphrag/_benchmark

**Step 4.4.1: Add deprecation notice**

```python
# nano_graphrag/_benchmark/__init__.py

import warnings

warnings.warn(
    "The 'nano_graphrag._benchmark' module is deprecated. "
    "Please use 'bench' module instead. "
    "Example: 'from bench import BenchmarkConfig' instead of "
    "'from nano_graphrag._benchmark import BenchmarkConfig'",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from bench for backward compatibility
from bench import *
```

**Step 4.4.2: Commit**

```bash
git add nano_graphrag/_benchmark/__init__.py
git commit -m "chore(benchmark): deprecate nano_graphrag._benchmark in favor of bench"
```

---

## Task 5: Compare Command (P1)

**Why fifth:** Enables A/B testing, critical for experimentation workflow.

**Files:**
- Create: `bench/compare.py`
- Create: `tests/benchmark/test_compare.py`

### Step 5.1: Implement compare functionality

**Step 5.1.1: Write failing test for compare**

```python
# tests/benchmark/test_compare.py

import pytest
import tempfile
import json
from pathlib import Path
from bench.compare import compare_results, print_diff_table

def test_compare_two_experiments():
    """Compare should compute deltas between two experiment results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create two result files
        result1 = {
            "experiment_name": "exp1",
            "mode_results": {
                "local": {"exact_match": 0.5, "token_f1": 0.6}
            }
        }
        result2 = {
            "experiment_name": "exp2",
            "mode_results": {
                "local": {"exact_match": 0.6, "token_f1": 0.55}
            }
        }

        path1 = Path(tmpdir) / "result1.json"
        path2 = Path(tmpdir) / "result2.json"
        path1.write_text(json.dumps(result1))
        path2.write_text(json.dumps(result2))

        # Compare
        diff = compare_results(str(path1), str(path2))

        # Should compute deltas
        assert diff["local"]["exact_match"]["delta"] == 0.1
        assert diff["local"]["token_f1"]["delta"] == -0.05
```

**Step 5.1.2: Run test to verify it fails**

```bash
uv run pytest tests/benchmark/test_compare.py::test_compare_two_experiments -v
```

Expected: FAIL with `ImportError: cannot import name 'compare_results'`

**Step 5.1.3: Implement compare function**

```python
# bench/compare.py

"""Compare benchmark experiment results."""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class ComparisonResult:
    """Result of comparing two experiments."""

    baseline: str
    challenger: str
    deltas: Dict[str, Dict[str, Dict[str, float]]]


def load_result(result_path: str) -> Dict[str, Any]:
    """Load experiment result from JSON file."""
    with open(result_path, "r") as f:
        return json.load(f)


def compare_results(baseline_path: str, challenger_path: str) -> ComparisonResult:
    """Compare two experiment results and compute deltas.

    Args:
        baseline_path: Path to baseline experiment result JSON
        challenger_path: Path to challenger experiment result JSON

    Returns:
        ComparisonResult with deltas for each mode and metric
    """
    baseline = load_result(baseline_path)
    challenger = load_result(challenger_path)

    deltas = {}

    # Compare each mode
    for mode in baseline["mode_results"]:
        if mode not in challenger["mode_results"]:
            continue

        baseline_scores = baseline["mode_results"][mode]
        challenger_scores = challenger["mode_results"][mode]

        deltas[mode] = {}

        # Compare each metric
        for metric in baseline_scores:
            if metric not in challenger_scores:
                continue

            baseline_val = baseline_scores[metric]
            challenger_val = challenger_scores[metric]

            deltas[mode][metric] = {
                "baseline": baseline_val,
                "challenger": challenger_val,
                "delta": challenger_val - baseline_val,
            }

    return ComparisonResult(
        baseline=baseline_path,
        challenger=challenger_path,
        deltas=deltas,
    )


def print_diff_table(comparison: ComparisonResult) -> str:
    """Print comparison as markdown table.

    Returns:
        Markdown table string
    """
    lines = []
    lines.append("## Benchmark Comparison")
    lines.append("")
    lines.append(f"**Baseline:** `{comparison.baseline}`")
    lines.append(f"**Challenger:** `{comparison.challenger}`")
    lines.append("")
    lines.append("### Results")
    lines.append("")

    # Table header
    lines.append("| Mode | Metric | Baseline | Challenger | Delta |")
    lines.append("|------|--------|----------|-------------|-------|")

    # Table rows
    for mode in comparison.deltas:
        for metric in comparison.deltas[mode]:
            data = comparison.deltas[mode][metric]
            delta = data["delta"]

            # Format delta with sign and indicator
            delta_str = f"{delta:+.3f}"
            if delta > 0:
                delta_str += " ✓"
            elif delta < 0:
                delta_str += " ✗"

            lines.append(
                f"| {mode} | {metric} | {data['baseline']:.3f} | "
                f"{data['challenger']:.3f} | {delta_str} |"
            )

    return "\n".join(lines)


def main():
    """CLI entry point for compare command."""
    parser = argparse.ArgumentParser(
        description="Compare two benchmark experiment results",
    )
    parser.add_argument(
        "baseline",
        help="Path to baseline experiment result JSON",
    )
    parser.add_argument(
        "challenger",
        help="Path to challenger experiment result JSON",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Save comparison to file instead of printing",
    )

    args = parser.parse_args()

    # Compare
    comparison = compare_results(args.baseline, args.challenger)

    # Generate output
    output = print_diff_table(comparison)

    if args.output:
        Path(args.output).write_text(output)
        print(f"[Compare] Saved to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
```

**Step 5.1.4: Run test to verify it passes**

```bash
uv run pytest tests/benchmark/test_compare.py::test_compare_two_experiments -v
```

Expected: PASS

**Step 5.1.5: Test CLI manually**

```bash
# Create test results
echo '{"experiment_name":"test1","mode_results":{"local":{"exact_match":0.5}}}' > /tmp/result1.json
echo '{"experiment_name":"test2","mode_results":{"local":{"exact_match":0.6}}}' > /tmp/result2.json

# Run compare
python -m bench.compare /tmp/result1.json /tmp/result2.json
```

**Step 5.1.6: Commit**

```bash
git add bench/compare.py tests/benchmark/test_compare.py
git commit -m "feat(benchmark): add compare command for A/B testing"
```

---

## Task 6: Config Schema Migration (P1)

**Why sixth:** Aligns with roadmap, improves readability.

**Files:**
- Modify: `bench/runner.py` (BenchmarkConfig)
- Modify: `examples/benchmarks/configs/example.yaml`
- Create: `examples/benchmarks/configs/multihop_musique.yaml`

### Step 6.1: Update BenchmarkConfig for nested schema

**Step 6.1.1: Write test for nested config loading**

```python
# tests/benchmark/test_config.py

import pytest
import tempfile
from pathlib import Path
from bench import BenchmarkConfig

def test_load_nested_config():
    """Should load nested YAML config as specified in roadmap."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        config_yaml = """
name: test_experiment
version: "1.0"
description: "Test experiment"

dataset:
  name: musique
  split: validation
  max_samples: 100
  auto_download: true

graphrag:
  working_dir: ./workdirs/test
  llm_model: gpt-4o-mini
  embedding_model: text-embedding-3-small

query:
  modes:
    - local
    - global
  param_overrides:
    top_k: 20

cache:
  enabled: true
  backend: disk

metrics:
  exact_match: true
  token_f1: true
  llm_judge:
    enabled: false

output:
  results_dir: ./results
  save_predictions: true
"""
        config_path.write_text(config_yaml)

        config = BenchmarkConfig.from_yaml(str(config_path))

        # Verify nested fields were parsed correctly
        assert config.experiment_name == "test_experiment"
        assert config.dataset_name == "musique"
        assert config.max_samples == 100
        assert config.auto_download is True
        assert config.graphrag_config["working_dir"] == "./workdirs/test"
        assert "local" in config.query_modes
        assert config.graphrag_config["enable_llm_cache"] is True
```

**Step 6.1.2: Run test to verify it fails**

```bash
uv run pytest tests/benchmark/test_config.py::test_load_nested_config -v
```

Expected: FAIL (current schema doesn't support nested structure)

**Step 6.1.3: Update BenchmarkConfig to support nested schema**

```python
# bench/runner.py

@dataclass
class BenchmarkConfig:
    """Experiment configuration (YAML-serializable).

    Supports both flat and nested YAML schemas for backward compatibility.
    """

    # === Top-level ===
    experiment_name: str = "experiment"
    version: str = "1.0"
    description: str = ""

    # === Dataset (nested or flat) ===
    dataset_name: str = ""
    dataset_path: str = ""
    corpus_path: Optional[str] = None
    dataset_split: str = "test"
    max_samples: int = -1
    auto_download: bool = False

    # === GraphRAG config ===
    graphrag_config: Dict[str, Any] = field(default_factory=dict)

    # === Query modes ===
    query_modes: List[str] = field(default_factory=lambda: ["local", "global"])
    query_params: Dict[str, Any] = field(default_factory=dict)

    # === Metrics ===
    metrics: List[str] = field(default_factory=lambda: ["exact_match", "token_f1"])

    # === Output ===
    output_dir: str = "./benchmark_results"

    @classmethod
    def from_yaml(cls, path: str) -> "BenchmarkConfig":
        """Load config from YAML file.

        Supports both flat and nested schemas.
        """
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required to load YAML configs. Install with: uv add pyyaml"
            )

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        # Normalize nested schema to flat
        normalized = cls._normalize_config(data)
        return cls(**normalized)

    @classmethod
    def _normalize_config(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize nested config schema to flat structure.

        Handles both:
        - Flat schema (backward compatible)
        - Nested schema (roadmap compliant)
        """
        normalized = {}

        # Top-level fields
        normalized["experiment_name"] = data.get("name", "experiment")
        normalized["version"] = data.get("version", "1.0")
        normalized["description"] = data.get("description", "")

        # Dataset section
        if "dataset" in data:
            dataset = data["dataset"]
            normalized["dataset_name"] = dataset.get("name", "")
            normalized["dataset_path"] = dataset.get("path", "")
            normalized["corpus_path"] = dataset.get("corpus_path")
            normalized["dataset_split"] = dataset.get("split", "test")
            normalized["max_samples"] = dataset.get("max_samples", -1)
            normalized["auto_download"] = dataset.get("auto_download", False)
        else:
            # Flat schema
            normalized["dataset_name"] = data.get("dataset_name", "")
            normalized["dataset_path"] = data.get("dataset_path", "")
            normalized["corpus_path"] = data.get("corpus_path")
            normalized["dataset_split"] = data.get("dataset_split", "test")
            normalized["max_samples"] = data.get("max_samples", -1)
            normalized["auto_download"] = data.get("auto_download", False)

        # GraphRAG section
        if "graphrag" in data:
            graphrag = data["graphrag"]
            normalized["graphrag_config"] = graphrag
        else:
            normalized["graphrag_config"] = data.get("graphrag_config", {})

        # Query section
        if "query" in data:
            query = data["query"]
            normalized["query_modes"] = query.get("modes", ["local", "global"])
            normalized["query_params"] = query.get("param_overrides", {})
        else:
            normalized["query_modes"] = data.get("query_modes", ["local", "global"])
            normalized["query_params"] = data.get("query_params", {})

        # Cache section
        if "cache" in data:
            cache = data["cache"]
            normalized["graphrag_config"]["enable_llm_cache"] = cache.get("enabled", False)

        # Metrics section
        if "metrics" in data:
            metrics = data["metrics"]
            metric_list = []
            if metrics.get("exact_match", False):
                metric_list.append("exact_match")
            if metrics.get("token_f1", False):
                metric_list.append("token_f1")
            if metrics.get("llm_judge", {}).get("enabled", False):
                metric_list.append("faithfulness")
                metric_list.append("answer_relevance")
            normalized["metrics"] = metric_list
        else:
            normalized["metrics"] = data.get("metrics", ["exact_match", "token_f1"])

        # Output section
        if "output" in data:
            output = data["output"]
            normalized["output_dir"] = output.get("results_dir", "./benchmark_results")
        else:
            normalized["output_dir"] = data.get("output_dir", "./benchmark_results")

        return normalized

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "name": self.experiment_name,
            "version": self.version,
            "description": self.description,
            "dataset": {
                "name": self.dataset_name,
                "path": self.dataset_path,
                "corpus_path": self.corpus_path,
                "split": self.dataset_split,
                "max_samples": self.max_samples,
                "auto_download": self.auto_download,
            },
            "graphrag": self.graphrag_config,
            "query": {
                "modes": self.query_modes,
                "param_overrides": self.query_params,
            },
            "metrics": {m: True for m in self.metrics},
            "output": {
                "results_dir": self.output_dir,
            },
        }
```

**Step 6.1.4: Run test to verify it passes**

```bash
uv run pytest tests/benchmark/test_config.py::test_load_nested_config -v
```

Expected: PASS

**Step 6.1.5: Test backward compatibility**

```python
# tests/benchmark/test_config.py (add test)

def test_load_flat_config_still_works():
    """Flat config schema should still work for backward compatibility."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        config_yaml = """
dataset_name: musique
dataset_path: /path/to/data
max_samples: 100
graphrag_config:
  working_dir: ./workdirs
query_modes:
  - local
"""
        config_path.write_text(config_yaml)

        config = BenchmarkConfig.from_yaml(str(config_path))

        assert config.dataset_name == "musique"
        assert config.max_samples == 100
        assert "local" in config.query_modes
```

**Step 6.1.6: Run backward compatibility test**

```bash
uv run pytest tests/benchmark/test_config.py::test_load_flat_config_still_works -v
```

Expected: PASS

**Step 6.1.7: Commit**

```bash
git add bench/runner.py tests/benchmark/test_config.py
git commit -m "feat(benchmark): support nested config schema with backward compatibility"
```

---

### Step 6.2: Create roadmap-compliant example config

**Step 6.2.1: Create multihop_musique.yaml**

```yaml
# examples/benchmarks/configs/multihop_musique.yaml
name: multihop_musique_baseline
version: "1.0"
description: "Local-mode GraphRAG baseline on MuSiQue dev set"

dataset:
  name: musique
  split: validation
  max_samples: 200
  auto_download: true

graphrag:
  working_dir: ./workdirs/musique_baseline
  llm_model: gpt-4o-mini
  embedding_model: text-embedding-3-small
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
  backend: disk

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

**Step 6.2.2: Update example.yaml to nested schema**

```yaml
# examples/benchmarks/configs/example.yaml
name: multihop_rag_example
version: "1.0"
description: "Example benchmark configuration"

dataset:
  name: multihop_rag
  path: ./fixtures/MultiHopRAG.json
  corpus_path: ./fixtures/MultiHopRAG_corpus.json
  split: test
  max_samples: 100

graphrag:
  working_dir: ./nano_graphrag_cache_benchmark
  llm_model: gpt-4o-mini
  embedding_model: text-embedding-3-small
  enable_local: true
  enable_naive_rag: false
  enable_llm_cache: true
  chunk_token_size: 1200
  chunk_overlap_token_size: 100

query:
  modes:
    - local
    - global

cache:
  enabled: true
  backend: disk

metrics:
  exact_match: true
  token_f1: true

output:
  results_dir: ./benchmark_results
```

**Step 6.2.3: Commit**

```bash
git add examples/benchmarks/configs/
git commit -m "docs(benchmark): update example configs to nested schema"
```

---

## Task 7: Documentation & Polish (P2)

**Why last:** Documentation is important but code functionality comes first.

**Files:**
- Update: `README.md`
- Update: `bench/README.md`
- Create: `docs/benchmark-usage.md`

### Step 7.1: Update main README

**Step 7.1.1: Add benchmark section to README**

```markdown
# nano-graphrag

... existing content ...

## Benchmarking

nano-graphrag includes a comprehensive benchmark framework for evaluating multi-hop RAG performance.

### Quick Start

```bash
# Run a benchmark experiment
python -m bench.run --config examples/benchmarks/configs/multihop_musique.yaml

# Compare two experiments
python -m bench.compare results/exp1.json results/exp2.json
```

### Features

- **4 Multi-hop datasets**: MultiHop-RAG, MuSiQue, HotpotQA, 2WikiMultiHopQA
- **Auto-download**: Datasets download automatically from HuggingFace
- **LLM caching**: Transparent caching saves API costs
- **Metrics**: EM, F1, faithfulness, context recall
- **A/B testing**: Built-in comparison tool

See [Benchmark Documentation](docs/benchmark-usage.md) for details.
```

**Step 7.1.2: Commit**

```bash
git add README.md
git commit -m "docs(benchmark): add benchmark section to main README"
```

---

### Step 7.2: Create comprehensive benchmark usage guide

**Step 7.2.1: Create docs/benchmark-usage.md**

```markdown
# Benchmark Usage Guide

Complete guide to running and analyzing GraphRAG benchmarks.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Configuration](#configuration)
4. [Datasets](#datasets)
5. [Metrics](#metrics)
6. [Caching](#caching)
7. [A/B Testing](#ab-testing)
8. [Python API](#python-api)

## Installation

```bash
# Core dependencies
uv sync

# Optional: For LLM-as-judge metrics
uv add --optional ragas

# Optional: For dataset auto-download
uv add --optional datasets
```

## Quick Start

### 1. Create a config file

```yaml
# my_experiment.yaml
name: my_experiment
dataset:
  name: musique
  split: validation
  max_samples: 100
  auto_download: true

graphrag:
  working_dir: ./workdirs/my_experiment
  llm_model: gpt-4o-mini

query:
  modes:
    - local
    - global

cache:
  enabled: true

metrics:
  exact_match: true
  token_f1: true

output:
  results_dir: ./results
```

### 2. Run the experiment

```bash
python -m bench.run --config my_experiment.yaml
```

### 3. View results

Results are saved to `./results/my_experiment_<timestamp>.json`:

```json
{
  "experiment_name": "my_experiment",
  "mode_results": {
    "local": {"exact_match": 0.42, "token_f1": 0.55},
    "global": {"exact_match": 0.38, "token_f1": 0.51}
  },
  "cache_stats": {
    "hits": 150,
    "misses": 50,
    "hit_rate": 0.75
  },
  "duration_seconds": 123.45
}
```

## Configuration

### Nested Schema (Recommended)

```yaml
name: experiment_name
version: "1.0"
description: "Experiment description"

dataset:
  name: musique | hotpotqa | 2wiki | multihop_rag
  path: /path/to/data  # or use auto_download
  corpus_path: /path/to/corpus  # for multihop_rag
  split: validation | dev | test
  max_samples: 100
  auto_download: true

graphrag:
  working_dir: ./workdirs/exp
  llm_model: gpt-4o-mini
  embedding_model: text-embedding-3-small
  chunk_token_size: 1200

query:
  modes:
    - local
    - global
    - naive
  param_overrides:
    top_k: 20

cache:
  enabled: true
  backend: disk

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

### CLI Options

```bash
python -m bench.run --config config.yaml [OPTIONS]

Options:
  --name, -n         Override experiment name
  --max-samples N    Limit number of samples
  --modes MODES      Query modes to run (space-separated)
  --output-dir, -o   Override output directory
  --dry-run          Validate config without running
```

## Datasets

### Supported Datasets

| Dataset | Split Sizes | Hop Depth | Auto-Download |
|---------|-------------|-----------|---------------|
| MultiHop-RAG | 2556 dev | 2 | ✓ |
| MuSiQue | 2417 dev | 2-4 | ✓ |
| HotpotQA | 7405 dev | 2 | ✓ |
| 2WikiMultiHopQA | 12576 dev | 2-5 | ✓ |

### Using Custom Datasets

```python
from bench import BenchmarkDataset, QAPair, Passage
from typing import Iterator

class MyDataset(BenchmarkDataset):
    name = "my_dataset"

    def questions(self, split: str = "test") -> Iterator[QAPair]:
        # Load your questions
        yield QAPair(
            id="q1",
            question="What is...?",
            answer="The answer",
            supporting_facts=["fact1", "fact2"],
        )

    def corpus(self) -> Iterator[Passage]:
        # Load your corpus
        yield Passage(
            id="doc1",
            title="Document 1",
            text="Document content...",
        )

    def download(self, cache_dir: str = "~/.cache/nano-bench") -> None:
        # Optional: Implement auto-download
        pass
```

## Metrics

### Available Metrics

- **Exact Match (EM)**: Normalized string comparison
- **Token F1**: Token overlap F1 score
- **Faithfulness** (Ragas): Is answer supported by context?
- **Context Recall** (Native): Are supporting facts in context?
- **Answer Relevance** (Ragas): Is answer relevant to question?

### Custom Metrics

```python
from bench import Metric

class MyMetric(Metric):
    async def compute(self, prediction, gold, question="", context=""):
        # Your metric logic
        return 0.5

# Add to suite
suite = MetricSuite()
suite.add_metric("my_metric", MyMetric())
```

## Caching

LLM caching is enabled by default to reduce API costs.

### Cache Statistics

After running an experiment, cache stats are printed:

```
[Cache] Hit rate: 75.00%
[Cache] 150 hits, 50 misses
```

### Disable Cache

```yaml
cache:
  enabled: false
```

Or via CLI:

```bash
python -m bench.run --config config.yaml --no-cache
```

## A/B Testing

### Compare Two Experiments

```bash
python -m bench.compare results/exp1.json results/exp2.json
```

Output:

```
## Benchmark Comparison

**Baseline:** `results/exp1.json`
**Challenger:** `results/exp2.json`

### Results

| Mode | Metric | Baseline | Challenger | Delta |
|------|--------|----------|-------------|-------|
| local | exact_match | 0.500 | 0.600 | +0.100 ✓ |
| local | token_f1 | 0.600 | 0.550 | -0.050 ✗ |
```

### Save Comparison

```bash
python -m bench.compare exp1.json exp2.json --output comparison.md
```

## Python API

### Basic Usage

```python
import asyncio
from bench import BenchmarkConfig, ExperimentRunner

async def main():
    # Load config
    config = BenchmarkConfig.from_yaml("config.yaml")

    # Run experiment
    runner = ExperimentRunner(config)
    result = await runner.run()

    # Access results
    print(result.mode_results)
    print(result.cache_stats)

asyncio.run(main())
```

### Custom Workflow

```python
from bench import BenchmarkConfig, ExperimentRunner, create_benchmark_cache

# Create cache
cache = create_benchmark_cache("./cache", enabled=True)

# Create config
config = BenchmarkConfig(
    dataset_name="musique",
    dataset_path="",
    auto_download=True,
    graphrag_config={"working_dir": "./workdirs"},
)

# Run with cache
runner = ExperimentRunner(config)
runner._cache = cache  # Inject custom cache

result = await runner.run()
```

## Tips

1. **Start small**: Use `max_samples: 10` to test your config
2. **Enable cache**: Saves costs on repeated runs
3. **Use auto-download**: Don't manually download datasets
4. **Compare often**: Use `bench.compare` to track improvements
5. **Check predictions**: Set `save_predictions: true` to debug

## Troubleshooting

### Import Error

```
ImportError: cannot import name 'BenchmarkConfig' from 'nano_graphrag._benchmark'
```

**Solution**: Use `from bench import BenchmarkConfig` instead.

### Cache Not Working

**Symptoms**: Hit rate is 0%

**Solution**: Ensure `cache.enabled: true` in config.

### Download Fails

**Symptoms**: `NotImplementedError: Auto-download not yet implemented`

**Solution**: Install datasets library: `uv add --optional datasets`
```

**Step 7.2.2: Commit**

```bash
git add docs/benchmark-usage.md
git commit -m "docs(benchmark): add comprehensive usage guide"
```

---

## Summary

This plan completes Phase 1 of the benchmark infrastructure:

1. ✅ **Cache Integration** (P0) — Immediate cost savings
2. ✅ **Dataset Type Safety** (P1) — Better DX, unlocks context recall
3. ✅ **Auto-Download** (P1) — Removes manual download friction
4. ✅ **Module Structure** (P1) — Roadmap compliant, `python -m bench`
5. ✅ **Compare Command** (P1) — Enables A/B testing
6. ✅ **Config Schema** (P1) — Nested, readable, backward compatible
7. ✅ **Documentation** (P2) — Complete usage guide

**Total Estimated Effort:** ~15-20 days

**Next Steps After Phase 1:**
- Phase 2: Modular A/B Architecture (Plugin Registry)
- Phase 3: Multi-Hop RAG Engine
- Advanced Techniques (DSPy, Rerankers, etc.)

---

*Plan created: 2026-03-23*
