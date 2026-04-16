# nano-graphrag: Experiment & Benchmark Module — Implementation Roadmap

> Historical planning document. Use `experiments/README.md` for the current runnable benchmark workflow.

> **Scope:** Transform `nano-graphrag` into a first-class experimentation platform that supports rapid benchmarking (multi-hop RAG and beyond), clean A/B testing of pipeline components, and integration of advanced retrieval techniques — while keeping the ~1100-line core untouched.

---

## Table of Contents

1. [Guiding Principles](#guiding-principles)
2. [Repository Structure](#repository-structure)
3. [Phase 1 — Benchmark Infrastructure](#phase-1--benchmark-infrastructure)
4. [Phase 2 — Modular A/B Architecture](#phase-2--modular-ab-architecture)
5. [Phase 3 — Multi-Hop RAG Engine](#phase-3--multi-hop-rag-engine)
6. [Advanced Techniques](#advanced-techniques)
7. [Evaluation Strategy](#evaluation-strategy)
8. [Effort & Priority Matrix](#effort--priority-matrix)
9. [Milestone Summary](#milestone-summary)

---

## Guiding Principles

- **Core isolation.** The `nano_graphrag/` package stays untouched. All experiment scaffolding lives in a parallel `bench/` tree and imports from core via public APIs only.
- **Reproducibility first.** Every run is fully reproducible: config pinned in YAML, LLM responses optionally cached, results stored with the config that produced them.
- **One-command experiments.** `python -m bench.run --config experiments/multihop_baseline.yaml` should be the full surface area for running a benchmark.
- **Incremental complexity.** Each phase delivers standalone value before the next phase begins. Phase 1 alone (benchmark harness) is useful without Phase 2 or 3.
- **Cheap ablations.** LLM call caching means iterating on retrieval logic costs zero API calls after the first run.

---

## Repository Structure

```
nano-graphrag/
├── nano_graphrag/          # CORE — do not modify
│   ├── _op.py
│   ├── _llm.py
│   ├── base.py
│   └── ...
│
├── bench/                  # NEW — experiment framework
│   ├── __init__.py
│   ├── run.py              # CLI entry point
│   ├── config.py           # Pydantic schema for experiment YAML
│   ├── cache.py            # LLM call cache layer
│   ├── registry.py         # Plugin registry for swappable components
│   ├── compare.py          # A/B runner and diff reporting
│   │
│   ├── datasets/           # Dataset loaders
│   │   ├── base.py         # BenchmarkDataset protocol
│   │   ├── multihop_rag.py
│   │   ├── musique.py
│   │   ├── hotpotqa.py
│   │   └── twowiki.py
│   │
│   ├── metrics/            # Evaluation metrics
│   │   ├── base.py         # MetricSuite protocol
│   │   ├── exact_match.py
│   │   ├── token_f1.py
│   │   ├── llm_judge.py    # Faithfulness + context recall
│   │   └── suite.py        # Composite runner
│   │
│   ├── retrievers/         # Pluggable retrieval strategies
│   │   ├── base.py
│   │   ├── multihop.py     # Phase 3 iterative retriever
│   │   ├── hipporag_ppr.py # Advanced: PPR-based retrieval
│   │   └── hybrid.py
│   │
│   └── techniques/         # Advanced technique implementations
│       ├── dspy_tuner.py
│       ├── raptor.py
│       ├── reranker.py
│       ├── edge_confidence.py
│       ├── adaptive_router.py
│       └── incremental_community.py
│
├── experiments/            # YAML experiment configs (version-controlled)
│   ├── baseline_naive.yaml
│   ├── baseline_local.yaml
│   ├── multihop_musique.yaml
│   └── ablation_chunker.yaml
│
├── results/                # Auto-generated run outputs (gitignored)
│   └── {run_id}/
│       ├── config.yaml     # Exact config used
│       ├── metrics.json
│       ├── predictions.jsonl
│       └── run.log
│
└── tests/
    ├── bench/              # Tests for the bench framework
    └── ...                 # Existing core tests
```

---

## Phase 1 — Benchmark Infrastructure

**Target completion:** 3–4 weeks
**Deliverable:** A repeatable, cached benchmark harness covering the four main multi-hop datasets and three core metrics.

---

### 1.1 Dataset Loaders

Define a shared `BenchmarkDataset` protocol so the runner is dataset-agnostic.

```python
# bench/datasets/base.py
from typing import Protocol, Iterator
from dataclasses import dataclass

@dataclass
class QAPair:
    id: str
    question: str
    answer: str                    # gold answer string
    supporting_facts: list[str]    # passage IDs used in reasoning chain
    metadata: dict

@dataclass
class Passage:
    id: str
    title: str
    text: str

class BenchmarkDataset(Protocol):
    name: str

    def questions(self, split: str = "validation") -> Iterator[QAPair]: ...
    def corpus(self) -> Iterator[Passage]: ...
    def download(self, cache_dir: str = "~/.cache/nano-bench") -> None: ...
```

Implement loaders for each dataset:

| Dataset | Split sizes | Hop depth | Download source |
|---|---|---|---|
| **MultiHop-RAG** | 2556 dev | 2 | HuggingFace |
| **MuSiQue** | 2417 dev | 2–4 | HuggingFace |
| **HotpotQA** | 7405 dev (distractor) | 2 | HuggingFace |
| **2WikiMultiHopQA** | 12576 dev | 2–5 | HuggingFace |

Each loader caches downloaded data locally and returns typed iterators. Corpus ingestion respects nano-graphrag's `insert()` API — no special handling needed.

---

### 1.2 LLM Call Cache

The most important efficiency feature. Every prompt → response pair is hashed and stored. Re-running the same experiment after a code change uses cached responses unless `--no-cache` is passed.

```python
# bench/cache.py
import hashlib, json
from pathlib import Path
from nano_graphrag.base import BaseKVStorage

class LLMCallCache:
    """Wraps any nano-graphrag LLM function with transparent prompt caching."""

    def __init__(self, storage: BaseKVStorage, enabled: bool = True):
        self._store = storage
        self.enabled = enabled
        self.hits = 0
        self.misses = 0

    def wrap(self, llm_func):
        async def cached_llm(prompt, system_prompt=None, history_messages=None, **kwargs):
            if not self.enabled:
                return await llm_func(prompt, system_prompt, history_messages, **kwargs)

            key = self._hash(prompt, system_prompt, history_messages, kwargs)
            cached = await self._store.get_by_id(key)
            if cached:
                self.hits += 1
                return cached["response"]

            self.misses += 1
            response = await llm_func(prompt, system_prompt, history_messages, **kwargs)
            await self._store.upsert({key: {"response": response, "prompt": prompt[:200]}})
            return response

        return cached_llm

    def _hash(self, *args) -> str:
        payload = json.dumps(args, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total else 0.0,
        }
```

**Integration:** In `bench/run.py`, wrap `GraphRAGConfig.best_model_func` and `cheap_model_func` through `LLMCallCache.wrap()` before constructing the `GraphRAG` instance. The core is unmodified.

---

### 1.3 Metrics Suite

```python
# bench/metrics/suite.py
from dataclasses import dataclass, field

@dataclass
class MetricResult:
    exact_match: float
    token_f1: float
    faithfulness: float | None = None   # LLM-as-judge, optional
    context_recall: float | None = None # LLM-as-judge, optional
    metadata: dict = field(default_factory=dict)

class MetricSuite:
    def __init__(self, use_llm_judge: bool = False, judge_model: str = "gpt-4o-mini"):
        self.use_llm_judge = use_llm_judge
        self.judge_model = judge_model

    def evaluate(self, prediction: str, gold: QAPair) -> MetricResult:
        em = exact_match(prediction, gold.answer)
        f1 = token_f1(prediction, gold.answer)
        faith = context_recall = None

        if self.use_llm_judge:
            faith = self._llm_faithfulness(prediction, gold)
            context_recall = self._llm_context_recall(prediction, gold)

        return MetricResult(exact_match=em, token_f1=f1,
                            faithfulness=faith, context_recall=context_recall)
```

**Metric definitions:**

- **Exact Match** — normalised string equality after lowercasing, stripping articles and punctuation. Standard SQuAD normalisation.
- **Token F1** — overlap of unigram token bags between prediction and gold, standard for open-domain QA.
- **Faithfulness (LLM-as-judge)** — prompt a judge LLM with `(question, retrieved_context, answer)` and ask it to score whether the answer is fully supported by the context. Returns 0–1.
- **Context Recall (LLM-as-judge)** — score whether all necessary supporting facts from `gold.supporting_facts` appear in the retrieved context. Returns 0–1.

---

### 1.4 Config-Driven Experiment Runner

**YAML schema:**

```yaml
# experiments/multihop_musique.yaml
name: multihop_musique_baseline
version: "1.0"
description: "Local-mode GraphRAG baseline on MuSiQue dev set"

dataset:
  name: musique
  split: validation
  max_samples: 200        # set null for full split

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
    - multihop       # Phase 3 — skip until implemented
  param_overrides:
    top_k: 60

cache:
  enabled: true
  backend: disk      # or: redis, memory

metrics:
  exact_match: true
  token_f1: true
  llm_judge:
    enabled: false   # set true to run faithfulness + context recall
    model: gpt-4o-mini

output:
  results_dir: ./results
  save_predictions: true
```

**CLI:**

```bash
# Run a single experiment
python -m bench.run --config experiments/multihop_musique.yaml

# Dry run — print resolved pipeline without any LLM calls
python -m bench.run --config experiments/multihop_musique.yaml --dry-run

# Compare two experiments side by side
python -m bench.compare results/run_abc/ results/run_def/

# Run all experiments in a directory
python -m bench.run --config-dir experiments/ --parallel 4
```

---

## Phase 2 — Modular A/B Architecture

**Target completion:** 2–3 weeks after Phase 1
**Deliverable:** A plugin registry and A/B runner that makes swapping any pipeline stage a one-line YAML change.

---

### 2.1 Plugin Registry

Define `Protocol` classes for each swappable pipeline stage. Implementations register themselves by name and are resolved at runtime from the YAML config.

```python
# bench/registry.py
from typing import Protocol, Callable, Any

# ── Protocols ─────────────────────────────────────────────────────────────────

class Chunker(Protocol):
    """Splits text into chunks. Matches nano-graphrag's chunk_func signature."""
    def __call__(self, content: str, token_size: int, **kwargs) -> list[dict]: ...

class EntityExtractor(Protocol):
    """Extracts entities and relations from a chunk. Wraps best_model_func."""
    async def __call__(self, chunk: str, **kwargs) -> dict: ...

class Retriever(Protocol):
    """Retrieves context from the graph given a query."""
    async def __call__(self, query: str, graph_rag, param) -> str: ...

class Reranker(Protocol):
    """Re-scores a list of retrieved passages."""
    def __call__(self, query: str, passages: list[str]) -> list[tuple[str, float]]: ...

class Generator(Protocol):
    """Generates a final answer given query + context."""
    async def __call__(self, query: str, context: str, **kwargs) -> str: ...

# ── Registry ──────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, dict[str, Any]] = {
    "chunker": {},
    "retriever": {},
    "reranker": {},
    "generator": {},
}

def register(stage: str, name: str):
    """Decorator: @register('chunker', 'separator') registers a chunker."""
    def decorator(cls):
        _REGISTRY[stage][name] = cls
        return cls
    return decorator

def resolve(stage: str, name: str) -> Any:
    if name not in _REGISTRY[stage]:
        raise KeyError(f"No {stage} registered as '{name}'. "
                       f"Available: {list(_REGISTRY[stage])}")
    return _REGISTRY[stage][name]
```

**Built-in registrations** (thin wrappers around existing core functions):

```python
from nano_graphrag._op import chunking_by_token_size, chunking_by_seperators

@register("chunker", "token_size")
class TokenSizeChunker: ...     # wraps chunking_by_token_size

@register("chunker", "separator")
class SeparatorChunker: ...     # wraps chunking_by_seperators

@register("retriever", "local")
class LocalRetriever: ...       # wraps QueryParam(mode="local")

@register("retriever", "global")
class GlobalRetriever: ...

@register("retriever", "naive")
class NaiveRetriever: ...

@register("retriever", "multihop")   # Phase 3
class MultiHopRetriever: ...
```

---

### 2.2 A/B Experiment Config

Extend the YAML schema to support side-by-side variant comparison:

```yaml
# experiments/ablation_chunker.yaml
name: ablation_chunker_size
ab_test: true

shared:
  dataset:
    name: musique
    split: validation
    max_samples: 200
  metrics:
    exact_match: true
    token_f1: true

variant_a:
  label: "chunk_1200"
  graphrag:
    working_dir: ./workdirs/chunk_1200
    chunk_func: token_size
    chunk_token_size: 1200

variant_b:
  label: "chunk_400"
  graphrag:
    working_dir: ./workdirs/chunk_400
    chunk_func: token_size
    chunk_token_size: 400
```

**A/B runner** executes both variants against the same question set (same cache, same random seed) and produces a diff report:

```
┌──────────────────────┬──────────────┬──────────────┬──────────┐
│ Metric               │ chunk_1200   │ chunk_400    │ Delta    │
├──────────────────────┼──────────────┼──────────────┼──────────┤
│ Exact Match          │ 0.421        │ 0.387        │ +0.034 ✓ │
│ Token F1             │ 0.558        │ 0.521        │ +0.037 ✓ │
│ Avg latency (s)      │ 3.2          │ 2.6          │ −0.6     │
│ Avg tokens/query     │ 3840         │ 2910         │ −930     │
└──────────────────────┴──────────────┴──────────────┴──────────┘
```

---

### 2.3 Results Storage

```python
# bench/results.py
@dataclass
class RunResult:
    run_id: str                           # timestamp + short hash
    config: dict                          # full resolved config
    predictions: list[PredictionRecord]   # per-question output
    aggregate_metrics: dict               # mean/std across questions
    cache_stats: dict                     # hit rate, misses
    timing: dict                          # total, per-question mean

    def save(self, output_dir: Path): ...
    def to_markdown_table(self) -> str: ...
```

Backends supported via a `ResultsBackend` protocol: local JSON (default), MLflow (optional import), Weights & Biases (optional import). The backend is declared in the YAML under `output.backend`.

---

## Phase 3 — Multi-Hop RAG Engine

**Target completion:** 3–4 weeks after Phase 2
**Deliverable:** A new `QueryParam(mode="multihop")` that iteratively traverses the knowledge graph to answer bridging questions — and benchmark results proving it beats single-pass retrieval on MuSiQue/HotpotQA.

---

### 3.1 Design

Multi-hop retrieval in GraphRAG has a structural advantage over plain vector RAG: entities and relationships are explicit graph nodes and edges, making cross-document bridging discoverable without full document retrieval.

The engine follows four steps per query:

```
Question
  │
  ▼
[1] Query decomposer          ← LLM call: break into N sub-questions
  │
  ├─► Sub-Q 1 ─► Local retrieve ─► Entities₁, Context₁
  ├─► Sub-Q 2 ─► Graph expand(Entities₁) ─► Entities₂, Context₂
  └─► Sub-Q N ─► Graph expand(Entities_{N-1}) ─► Entities_N, Context_N
  │
  ▼
[4] Context merger             ← dedup, rank by relevance, trim to token budget
  │
  ▼
Final answer generation
```

---

### 3.2 Implementation

```python
# bench/retrievers/multihop.py
from dataclasses import dataclass, field
from nano_graphrag import GraphRAG, QueryParam

@dataclass
class HopState:
    sub_question: str
    retrieved_entities: list[str] = field(default_factory=list)
    context_chunks: list[str] = field(default_factory=list)
    answer_fragment: str = ""

class MultiHopRetriever:
    """
    Iterative graph retriever for multi-hop questions.

    Algorithm:
      1. Decompose the original question into sub-questions via LLM.
      2. For each sub-question, run local GraphRAG retrieval seeded with
         entities discovered in previous hops.
      3. Accumulate entity state across hops (bridging chain).
      4. Merge all retrieved context, deduplicate, and rank.
    """

    def __init__(
        self,
        max_hops: int = 4,
        entities_per_hop: int = 10,
        context_token_budget: int = 8000,
        decompose_model: str = "gpt-4o-mini",
    ):
        self.max_hops = max_hops
        self.entities_per_hop = entities_per_hop
        self.context_token_budget = context_token_budget
        self.decompose_model = decompose_model

    async def retrieve(self, question: str, graph_rag: GraphRAG) -> str:
        # Step 1: Decompose
        sub_questions = await self._decompose(question, graph_rag)

        # Step 2 & 3: Iterative hop retrieval
        hop_states: list[HopState] = []
        carry_entities: list[str] = []

        for sub_q in sub_questions[:self.max_hops]:
            state = HopState(sub_question=sub_q)
            context = await self._retrieve_hop(
                sub_q, graph_rag, seed_entities=carry_entities
            )
            state.context_chunks = context["chunks"]
            state.retrieved_entities = context["entities"]
            carry_entities = state.retrieved_entities   # bridge to next hop
            hop_states.append(state)

        # Step 4: Merge contexts
        return self._merge_contexts(hop_states, self.context_token_budget)

    async def _decompose(self, question: str, graph_rag: GraphRAG) -> list[str]:
        prompt = DECOMPOSE_PROMPT.format(question=question, max_hops=self.max_hops)
        raw = await graph_rag._llm(prompt)
        return self._parse_sub_questions(raw)

    async def _retrieve_hop(
        self, sub_q: str, graph_rag: GraphRAG, seed_entities: list[str]
    ) -> dict:
        # Use only_need_context=True to get raw retrieved context
        param = QueryParam(
            mode="local",
            only_need_context=True,
            top_k=self.entities_per_hop,
        )
        # Inject seed entity filter if we have carry-over from previous hop
        if seed_entities:
            param = self._bias_param_to_entities(param, seed_entities)

        context_str = await graph_rag.aquery(sub_q, param=param)
        return self._parse_context(context_str)

    def _merge_contexts(self, hop_states: list[HopState], budget: int) -> str:
        # Deduplicate chunks by content hash, rank by hop recency + relevance
        seen, merged = set(), []
        for state in reversed(hop_states):   # later hops are more specific
            for chunk in state.context_chunks:
                h = hash(chunk[:80])
                if h not in seen:
                    seen.add(h)
                    merged.append(chunk)
        # Trim to token budget (approximate at 4 chars/token)
        out, total = [], 0
        for chunk in merged:
            tokens = len(chunk) // 4
            if total + tokens > budget:
                break
            out.append(chunk)
            total += tokens
        return "\n\n".join(out)
```

**Decomposition prompt:**

```python
DECOMPOSE_PROMPT = """\
You are decomposing a complex question into simpler sub-questions that, \
when answered in sequence, build toward answering the original question.

Original question: {question}

Generate at most {max_hops} sub-questions. Each sub-question should be \
independently answerable and together they should bridge to the final answer.
Output only a JSON array of strings.
"""
```

---

### 3.3 Integration

The multi-hop retriever plugs into the existing `GraphRAG.aquery` flow without modifying core. A new `mode="multihop"` value is handled by an overriding `aquery` wrapper in the bench layer:

```python
# bench/run.py
async def run_query(question: str, mode: str, graph_rag: GraphRAG) -> str:
    if mode == "multihop":
        retriever = MultiHopRetriever(**config.multihop_params)
        context = await retriever.retrieve(question, graph_rag)
        return await graph_rag.aquery(
            question,
            param=QueryParam(mode="local", only_need_context=False),
            injected_context=context,   # patch point — see note below
        )
    else:
        return await graph_rag.aquery(question, param=QueryParam(mode=mode))
```

> **Patch point note:** `injected_context` requires a one-line addition to `GraphRAG.aquery` — the only permissible core modification. If preferred, this can be avoided by subclassing `GraphRAG` and overriding the generation step in the subclass.

---

## Advanced Techniques

### A. DSPy Prompt Tuning

**Rationale:** The `entity_extraction` prompt is the highest-leverage single prompt in the pipeline. Errors here cascade into graph construction. DSPy's `BootstrapFewShot` can optimise it for smaller/cheaper models (Llama 3.1 8B, Qwen 2.5 7B) without manual prompt engineering.

**Implementation plan:**

1. Create a labelled dataset of 50–100 `(chunk, expected_entities_json)` pairs from existing benchmark data.
2. Define a DSPy `Signature` that mirrors the entity extraction task.
3. Use `BootstrapFewShot` with the labelled set to compile an optimised few-shot prompt.
4. Inject the compiled prompt by replacing `PROMPTS["entity_extraction"]` before instantiating `GraphRAG`.

```python
# bench/techniques/dspy_tuner.py
import dspy
from nano_graphrag.prompt import PROMPTS

class EntityExtractionSignature(dspy.Signature):
    """Extract entities and relations from a text chunk as JSON."""
    chunk: str = dspy.InputField()
    entities_json: str = dspy.OutputField(desc="JSON object with entities and relations")

def tune_entity_extraction(train_examples: list[dspy.Example], model: str) -> str:
    """Returns an optimised prompt string for PROMPTS['entity_extraction']."""
    lm = dspy.LM(model)
    dspy.configure(lm=lm)
    module = dspy.Predict(EntityExtractionSignature)
    teleprompter = dspy.BootstrapFewShot(metric=entity_extraction_metric, max_bootstrapped_demos=4)
    compiled = teleprompter.compile(module, trainset=train_examples)
    return compiled.extended_signature.instructions
```

**Expected outcome:** Comparable extraction quality with models 10–30× cheaper to run than GPT-4o.

---

### B. Adaptive Mode Router

**Rationale:** Requiring callers to specify `mode=` is a usability barrier and risks misclassification. An automatic router removes this decision.

**Implementation plan:**

1. Define routing rules as a 3-class classification: `local` (entity-specific, factual), `global` (thematic, summarisation), `multihop` (bridging, relational across entities).
2. Implement a two-stage router: fast heuristic first (keyword signals), LLM classifier as fallback.

```python
# bench/techniques/adaptive_router.py
import re

MULTIHOP_SIGNALS = [
    r"\bwho.*also\b", r"\bboth.*and\b", r"\bconnect(ion|ed)\b",
    r"\brelationship between\b", r"\bcompared to\b", r"\bin common\b",
]
GLOBAL_SIGNALS = [
    r"\bthemes?\b", r"\boverall\b", r"\bin general\b",
    r"\bsummariz\b", r"\bacross\b", r"\bmain ideas?\b",
]

def route_query(question: str, use_llm_fallback: bool = False) -> str:
    q = question.lower()
    multihop_score = sum(1 for p in MULTIHOP_SIGNALS if re.search(p, q))
    global_score   = sum(1 for p in GLOBAL_SIGNALS   if re.search(p, q))

    if multihop_score >= 2:
        return "multihop"
    if global_score >= 2:
        return "global"
    if multihop_score == 1 and use_llm_fallback:
        return _llm_route(question)   # single LLM call to disambiguate
    return "local"
```

---

### C. Edge Confidence Weighting

**Rationale:** Not all extracted relationships are equally reliable. High-confidence edges (from clear, explicit text) should be preferred in graph traversal over low-confidence ones (inferred, ambiguous). This directly improves precision in multi-hop path search.

**Implementation plan:**

1. During entity extraction, prompt the LLM to include a `confidence` score (0.0–1.0) per relation triple.
2. Store confidence as an edge attribute in the `networkx` (or neo4j) graph.
3. Modify the local retrieval path-scoring to weight edge traversal by confidence.

```python
# bench/techniques/edge_confidence.py
from nano_graphrag.base import BaseGraphStorage

async def score_edges_by_confidence(graph: BaseGraphStorage) -> None:
    """Post-process step: normalise and store edge confidence weights."""
    for src, dst, data in graph.edges(data=True):
        raw_conf = data.get("confidence", 1.0)
        freq = data.get("occurrence_count", 1)
        # Combine extraction confidence with corpus frequency signal
        weight = 0.7 * float(raw_conf) + 0.3 * min(freq / 10, 1.0)
        await graph.upsert_edge(src, dst, {"weight": weight, **data})
```

Expose as a post-insert hook in the experiment config:

```yaml
graphrag:
  post_insert_hooks:
    - edge_confidence_scoring
```

---

### D. Cross-Encoder Reranker

**Rationale:** The local retriever returns top-K candidates by embedding similarity. A cross-encoder re-scores these against the full query, capturing lexical and semantic overlap that bi-encoders miss. Plugs cleanly into the `only_need_context=True` interface.

**Implementation plan:**

1. Retrieve with `QueryParam(only_need_context=True, top_k=100)` to get a wide candidate set.
2. Run a cross-encoder model (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`, ~22 MB) over `(query, passage)` pairs.
3. Return top-K reranked passages as the final context.

```python
# bench/techniques/reranker.py
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", top_k: int = 20):
        self.model = CrossEncoder(model)
        self.top_k = top_k

    def rerank(self, query: str, passages: list[str]) -> list[str]:
        pairs = [(query, p) for p in passages]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)
        return [p for p, _ in ranked[:self.top_k]]
```

---

### E. HippoRAG — Personalised PageRank Retrieval

**Rationale:** Personalised PageRank (PPR) over the knowledge graph can discover multi-hop bridging paths in a single graph operation — no iterative LLM calls required. This is how [HippoRAG (2024)](https://arxiv.org/abs/2405.14831) achieves state-of-the-art multi-hop recall.

**Implementation plan:**

1. After corpus ingestion, build a `networkx` graph with edge weights (or use the existing graph).
2. For a query, extract seed entities by vector similarity.
3. Run PPR from those seed nodes. High-scoring nodes at convergence are the bridging entities.
4. Retrieve the chunks associated with top-scoring nodes as context.

```python
# bench/retrievers/hipporag_ppr.py
import networkx as nx
import numpy as np

class HippoRAGRetriever:
    """
    Personalised PageRank retrieval over the nano-graphrag knowledge graph.
    Based on: HippoRAG (Gutierrez et al., 2024).
    """

    def __init__(self, alpha: float = 0.85, top_k_seed: int = 5, top_k_result: int = 20):
        self.alpha = alpha             # PPR damping factor
        self.top_k_seed = top_k_seed   # seed entities from query embedding
        self.top_k_result = top_k_result

    async def retrieve(self, query: str, graph_rag) -> str:
        # 1. Get the networkx graph from storage
        G = graph_rag.chunk_entity_relation_graph._graph

        # 2. Identify seed nodes via entity embedding similarity
        seed_entities = await self._find_seed_entities(query, graph_rag)

        # 3. Build personalisation vector
        nodes = list(G.nodes())
        personalisation = {n: (1.0 / len(seed_entities) if n in seed_entities else 0.0)
                           for n in nodes}

        # 4. Run PPR
        scores = nx.pagerank(G, alpha=self.alpha, personalization=personalisation,
                             weight="weight")

        # 5. Retrieve chunks for top-scoring nodes
        top_nodes = sorted(scores, key=scores.get, reverse=True)[:self.top_k_result]
        return await self._nodes_to_context(top_nodes, graph_rag)

    async def _find_seed_entities(self, query: str, graph_rag) -> list[str]:
        query_emb = await graph_rag.embedding_func([query])
        # Query the entity vector store for closest entities
        results = await graph_rag.entities_vdb.query(query_emb[0], top_k=self.top_k_seed)
        return [r["entity_name"] for r in results]
```

---

### F. RAPTOR Hierarchical Summaries

**Rationale:** GraphRAG's community reports already provide a coarse hierarchy. RAPTOR extends this with a recursive summarisation tree over raw chunks, enabling retrieval at the right granularity level: chunk → cluster → community → global.

**Implementation plan:**

1. After standard chunking and before entity extraction, cluster chunks by embedding similarity (Gaussian Mixture Model, as in the original paper).
2. Summarise each cluster into a synthetic "meta-chunk" using `cheap_model_func`.
3. Repeat recursively until a single root summary is produced.
4. Index all levels of the tree into the vector store alongside original chunks.
5. At query time, retrieve from all levels and let the reranker surface the most relevant granularity.

```yaml
# In experiment config
graphrag:
  post_chunk_hooks:
    - raptor_tree_build
  raptor:
    max_levels: 3
    cluster_model: gmm
    summary_model: cheap   # uses cheap_model_func
```

---

### G. Incremental Community Updates

**Rationale:** Currently every `insert()` call triggers full community re-computation. This is O(n) in corpus size and blocks streaming/continuous ingestion. The `20260318-incremental-community-update-research.md` doc identifies this gap.

**Implementation plan:**

1. Track which graph nodes changed during the last insert (new nodes, modified edge weights).
2. Identify the minimal subgraph containing changed nodes and their 2-hop neighbours.
3. Re-run Leiden clustering only over this subgraph; merge results into the existing community assignments.
4. Invalidate and regenerate only the community reports for affected communities.

This is the most architecturally invasive change (requires modifying the community compute step in core). Recommended approach: implement in a `GraphRAGIncremental` subclass that overrides `_run_community_detection`.

---

## Evaluation Strategy

### Benchmark Targets

| Dataset | Baseline (naive) | Target (multi-hop) | Stretch target |
|---|---|---|---|
| MuSiQue F1 | ~0.25 | 0.42+ | 0.50+ |
| HotpotQA F1 | ~0.45 | 0.62+ | 0.68+ |
| 2WikiMHQA F1 | ~0.38 | 0.55+ | 0.62+ |
| MultiHop-RAG F1 | ~0.40 | 0.57+ | 0.64+ |

*Baselines estimated from the existing MultiHop-RAG evaluation notebook in the repo. Targets informed by HippoRAG and HybridRAG reported numbers.*

### Ablation Checkpoints

Run the benchmark suite after each major component lands to isolate contribution:

```
1. naive mode          → baseline
2. local mode          → +graph retrieval
3. local + reranker    → +reranking
4. multihop (Phase 3)  → +iterative decomposition
5. multihop + PPR      → +graph-native traversal
6. multihop + PPR + RAPTOR → +hierarchical context
```

---

## Effort & Priority Matrix

| Technique | Effort | Expected Gain | Priority |
|---|---|---|---|
| Phase 1: Benchmark harness | Medium | Foundational | **P0** |
| Phase 1: LLM cache | Low | High (cost & speed) | **P0** |
| Phase 2: Plugin registry | Medium | Foundational | **P1** |
| Phase 2: A/B runner | Low | High (iteration speed) | **P1** |
| DSPy prompt tuning | Low | High (small model compat) | **P1** |
| Adaptive mode router | Low | Medium | **P1** |
| Phase 3: Multi-hop engine | High | High | **P2** |
| Edge confidence weighting | Medium | Medium–High | **P2** |
| Cross-encoder reranker | Low | Medium | **P2** |
| HippoRAG PPR | Medium | High | **P2** |
| RAPTOR tree | High | Medium–High | **P3** |
| Incremental communities | High | Medium (scale) | **P3** |
| Temporal graph edges | Very High | Domain-specific | **P4** |

---

## Milestone Summary

### M1 — Benchmark Harness (Weeks 1–4)
- [ ] `BenchmarkDataset` protocol + loaders for MultiHop-RAG and MuSiQue
- [ ] `MetricSuite` (EM, F1, LLM-as-judge)
- [ ] `LLMCallCache` with disk backend
- [ ] YAML experiment config schema + CLI runner
- [ ] Baseline results for `naive` and `local` modes on MuSiQue dev 200

### M2 — Modular Pipeline (Weeks 5–7)
- [ ] `Plugin registry` with Protocol definitions for all stages
- [ ] A/B runner with diff report output
- [ ] HotpotQA and 2WikiMHQA loaders
- [ ] MLflow / JSON results backend
- [ ] DSPy entity extraction tuning (Llama 3.1 8B target)

### M3 — Multi-Hop Engine (Weeks 8–11)
- [ ] `MultiHopRetriever` with query decomposition
- [ ] Entity state carry-over across hops
- [ ] Context merger with dedup and token budget management
- [ ] `mode="multihop"` integration via bench wrapper
- [ ] Full benchmark results across all 4 datasets, all modes

### M4 — Advanced Retrieval (Weeks 12–16)
- [ ] Adaptive mode router (heuristic + LLM fallback)
- [ ] Edge confidence scoring post-insert hook
- [ ] Cross-encoder reranker stage
- [ ] HippoRAG PPR retriever
- [ ] Ablation study across all Phase 3 + M4 components

### M5 — Scale & Research (Weeks 17+)
- [ ] RAPTOR hierarchical summary tree
- [ ] Incremental community updates (subclass approach)
- [ ] Full benchmark suite on complete dataset splits
- [ ] Results write-up + comparison table vs HippoRAG, HybridRAG, LightRAG

---

*Last updated: 2026-03-23*
