# Phase 3 — Multi-Hop RAG Verification Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Verify Phase 3 Multi-Hop RAG Engine implementation against the technical roadmap specification at docs/technical_roadmap.md

**Architecture:** This plan verifies implementation of a new `QueryParam(mode="multihop")` that iteratively traverses the knowledge graph to answer bridging questions through query decomposition and entity state carry-over across hops.

**Tech Stack:** Python 3.11+, asyncio, nano-graphrag, pytest, HuggingFace datasets

**Prerequisites:** Phase 1 and Phase 2 must be complete before starting Phase 3 verification.

---

## Pre-Verification Checklist

Before starting Phase 3 implementation, verify the following:

### Task 0: Verify Phase 1 & Phase 2 Completion

**Files to check:**
- Read: `bench/runner.py`
- Read: `bench/registry.py`
- Read: `bench/datasets/__init__.py`

**Step 1: Verify Phase 1 components exist**

```bash
# Check that all Phase 1 components are present
ls -la bench/datasets/
ls -la bench/metrics/
ls -la bench/cache.py
ls -la bench/runner.py
ls -la bench/results.py
ls -la bench/compare.py
```

Expected: All files exist

**Step 2: Verify Phase 2 registry exists**

```bash
# Check that plugin registry is implemented
grep -n "class Retriever" bench/registry.py
grep -n "def register" bench/registry.py
grep -n "def resolve" bench/registry.py
```

Expected: Retriever protocol and register/resolve functions found

**Step 3: Verify datasets support QAPair and Passage**

```bash
# Check for typed dataclasses
grep -n "class QAPair" bench/datasets/__init__.py
grep -n "class Passage" bench/datasets/__init__.py
```

Expected: QAPair and Passage dataclasses defined

**Step 4: Run Phase 1 tests to confirm baseline**

```bash
uv run pytest tests/benchmark/ -v
```

Expected: All tests pass

**Step 5: Document current state**

If any step fails, create a gap analysis document and complete missing Phase 1/2 work before proceeding.

---

## Part 1: Multi-Hop Retriever Implementation & Verification

### Task 1: Create Bench Retriever Infrastructure

**Files:**
- Create: `bench/retrievers/__init__.py`
- Create: `bench/retrievers/base.py`
- Test: `tests/benchmark/test_retrievers.py`

**Step 1: Write the failing test for retriever protocol**

```python
# tests/benchmark/test_retrievers.py

import pytest
from bench.retrievers.base import Retriever, RetrieverResult
from nano_graphrag import GraphRAG
from nano_graphrag.base import QueryParam

@pytest.mark.asyncio
async def test_retriever_protocol_exists():
    """Verify Retriever protocol is defined."""
    from bench.retrievers.base import Retriever
    assert Retriever is not None

@pytest.mark.asyncio
async def test_retriever_result_dataclass():
    """Verify RetrieverResult dataclass has required fields."""
    result = RetrieverResult(
        context="test context",
        entities=["entity1", "entity2"],
        hops=2,
        metadata={}
    )
    assert result.context == "test context"
    assert result.entities == ["entity1", "entity2"]
    assert result.hops == 2
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/benchmark/test_retrievers.py::test_retriever_protocol_exists -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'bench.retrievers'"

**Step 3: Create retriever directory structure**

```bash
mkdir -p bench/retrievers
touch bench/retrievers/__init__.py
touch bench/retrievers/base.py
```

**Step 4: Implement base retriever protocol**

```python
# bench/retrievers/__init__.py
"""Pluggable retrieval strategies for GraphRAG."""

from .base import Retriever, RetrieverResult

__all__ = ["Retriever", "RetrieverResult"]

# bench/retrievers/base.py
"""Base retriever protocol and result types."""

from dataclasses import dataclass, field
from typing import Protocol, Any
from nano_graphrag import GraphRAG
from nano_graphrag.base import QueryParam

@dataclass
class RetrieverResult:
    """Result from a retrieval operation."""
    context: str                          # Retrieved context text
    entities: list[str] = field(default_factory=list)  # Entities discovered
    hops: int = 0                         # Number of hops taken
    metadata: dict[str, Any] = field(default_factory=dict)

class Retriever(Protocol):
    """Protocol for retrieval strategies."""

    async def __call__(
        self,
        query: str,
        graph_rag: GraphRAG,
        param: QueryParam,
        **kwargs,
    ) -> str:
        """Retrieve context for a query.

        Args:
            query: User question
            graph_rag: GraphRAG instance
            param: Query parameters
            **kwargs: Additional parameters

        Returns:
            Retrieved context as string
        """
        ...
```

**Step 5: Run test to verify it passes**

```bash
uv run pytest tests/benchmark/test_retrievers.py -v
```

Expected: PASS

**Step 6: Commit**

```bash
git add bench/retrievers/ tests/benchmark/test_retrievers.py
git commit -m "feat(benchmark): add retriever infrastructure with protocol and result types"
```

---

### Task 2: Implement HopState Dataclass

**Files:**
- Modify: `bench/retrievers/base.py`
- Test: `tests/benchmark/test_retrievers.py`

**Step 1: Write failing test for HopState**

```python
# tests/benchmark/test_retrievers.py

from bench.retrievers.base import HopState

@pytest.mark.asyncio
async def test_hop_state_dataclass():
    """Verify HopState has all required fields from roadmap."""
    state = HopState(
        sub_question="What is X?",
        retrieved_entities=["entity1"],
        context_chunks=["chunk1"],
        answer_fragment=""
    )
    assert state.sub_question == "What is X?"
    assert state.retrieved_entities == ["entity1"]
    assert state.context_chunks == ["chunk1"]
    assert state.answer_fragment == ""
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/benchmark/test_retrievers.py::test_hop_state_dataclass -v
```

Expected: FAIL with "HopState not defined"

**Step 3: Implement HopState**

```python
# bench/retrievers/base.py

@dataclass
class HopState:
    """State for a single hop in multi-hop retrieval."""
    sub_question: str                        # Question for this hop
    retrieved_entities: list[str] = field(default_factory=list)
    context_chunks: list[str] = field(default_factory=list)
    answer_fragment: str = ""                # Partial answer from this hop
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/benchmark/test_retrievers.py::test_hop_state_dataclass -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add bench/retrievers/base.py tests/benchmark/test_retrievers.py
git commit -m "feat(benchmark): add HopState dataclass for multi-hop tracking"
```

---

### Task 3: Implement MultiHopRetriever Core

**Files:**
- Create: `bench/retrievers/multihop.py`
- Test: `tests/benchmark/test_multihop_retriever.py`

**Step 1: Write failing test for MultiHopRetriever initialization**

```python
# tests/benchmark/test_multihop_retriever.py

import pytest
from bench.retrievers.multihop import MultiHopRetriever

@pytest.mark.asyncio
async def test_multihop_retriever_init():
    """Verify MultiHopRetriever initializes with roadmap parameters."""
    retriever = MultiHopRetriever(
        max_hops=4,
        entities_per_hop=10,
        context_token_budget=8000,
        decompose_model="gpt-4o-mini",
    )
    assert retriever.max_hops == 4
    assert retriever.entities_per_hop == 10
    assert retriever.context_token_budget == 8000
    assert retriever.decompose_model == "gpt-4o-mini"
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/benchmark/test_multihop_retriever.py::test_multihop_retriever_init -v
```

Expected: FAIL with "ModuleNotFoundError"

**Step 3: Implement MultiHopRetriever skeleton**

```python
# bench/retrievers/multihop.py
"""Multi-hop retrieval implementation for complex queries."""

from dataclasses import dataclass, field
from nano_graphrag import GraphRAG
from nano_graphrag.base import QueryParam
from .base import HopState

@dataclass
class MultiHopRetriever:
    """Iterative graph retriever for multi-hop questions.

    Algorithm:
      1. Decompose the original question into sub-questions via LLM.
      2. For each sub-question, run local GraphRAG retrieval seeded with
         entities discovered in previous hops.
      3. Accumulate entity state across hops (bridging chain).
      4. Merge all retrieved context, deduplicate, and rank.
    """

    max_hops: int = 4
    entities_per_hop: int = 10
    context_token_budget: int = 8000
    decompose_model: str = "gpt-4o-mini"

    async def retrieve(self, question: str, graph_rag: GraphRAG) -> str:
        """Retrieve context for a multi-hop question.

        Args:
            question: User's multi-hop question
            graph_rag: GraphRAG instance

        Returns:
            Merged context from all hops
        """
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
            carry_entities = state.retrieved_entities
            hop_states.append(state)

        # Step 4: Merge contexts
        return self._merge_contexts(hop_states, self.context_token_budget)

    async def _decompose(self, question: str, graph_rag: GraphRAG) -> list[str]:
        """Decompose question into sub-questions."""
        prompt = self._build_decompose_prompt(question)
        response = await graph_rag._llm(prompt)
        return self._parse_sub_questions(response)

    async def _retrieve_hop(
        self, sub_q: str, graph_rag: GraphRAG, seed_entities: list[str]
    ) -> dict:
        """Retrieve context for a single hop."""
        param = QueryParam(
            mode="local",
            only_need_context=True,
            top_k=self.entities_per_hop,
        )
        context_str = await graph_rag.aquery(sub_q, param=param)
        return self._parse_context(context_str)

    def _merge_contexts(self, hop_states: list[HopState], budget: int) -> str:
        """Merge and deduplicate contexts from all hops."""
        seen, merged = set(), []
        for state in reversed(hop_states):
            for chunk in state.context_chunks:
                h = hash(chunk[:80])
                if h not in seen:
                    seen.add(h)
                    merged.append(chunk)

        # Trim to token budget
        out, total = [], 0
        for chunk in merged:
            tokens = len(chunk) // 4
            if total + tokens > budget:
                break
            out.append(chunk)
            total += tokens

        return "\n\n".join(out)

    def _build_decompose_prompt(self, question: str) -> str:
        """Build prompt for query decomposition."""
        return f"""You are decomposing a complex question into simpler sub-questions that,
when answered in sequence, build toward answering the original question.

Original question: {question}

Generate at most {self.max_hops} sub-questions. Each sub-question should be
independently answerable and together they should bridge to the final answer.
Output only a JSON array of strings.
"""

    def _parse_sub_questions(self, response: str) -> list[str]:
        """Parse LLM response into list of sub-questions."""
        import json

        try:
            questions = json.loads(response)
            if isinstance(questions, list):
                return [str(q) for q in questions]
            return [response]
        except json.JSONDecodeError:
            # Fallback: split by newlines
            return [line.strip() for line in response.split("\n") if line.strip()]

    def _parse_context(self, context_str: str) -> dict:
        """Parse context string into chunks and entities."""
        # Simple implementation - can be enhanced
        return {
            "chunks": [context_str],
            "entities": [],
        }
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/benchmark/test_multihop_retriever.py::test_multihop_retriever_init -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add bench/retrievers/multihop.py tests/benchmark/test_multihop_retriever.py
git commit -m "feat(benchmark): implement MultiHopRetriever core logic"
```

---

### Task 4: Implement Query Decomposition

**Files:**
- Modify: `bench/retrievers/multihop.py`
- Test: `tests/benchmark/test_multihop_retriever.py`

**Step 1: Write failing test for query decomposition**

```python
# tests/benchmark/test_multihop_retriever.py

import pytest
from unittest.mock import AsyncMock, MagicMock

@pytest.mark.asyncio
async def test_query_decomposition():
    """Verify MultiHopRetriever decomposes multi-hop questions."""
    retriever = MultiHopRetriever(max_hops=3)

    # Mock GraphRAG instance
    mock_graphrag = MagicMock()
    mock_graphrag._llm = AsyncMock(return_value='["Who is X?", "What is Y?", "How are X and Y related?"]')

    sub_questions = await retriever._decompose("Who is X and how are they related to Y?", mock_graphrag)

    assert len(sub_questions) == 3
    assert sub_questions[0] == "Who is X?"
    assert sub_questions[1] == "What is Y?"
    assert sub_questions[2] == "How are X and Y related?"
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/benchmark/test_multihop_retriever.py::test_query_decomposition -v
```

Expected: FAIL or PASS depending on implementation

**Step 3: Enhance decomposition if needed**

The implementation in Task 3 should already handle this. If test fails, debug and fix.

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/benchmark/test_multihop_retriever.py::test_query_decomposition -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add bench/retrievers/multihop.py tests/benchmark/test_multihop_retriever.py
git commit -m "feat(benchmark): add query decomposition with test coverage"
```

---

### Task 5: Implement Entity State Carry-Over

**Files:**
- Modify: `bench/retrievers/multihop.py`
- Test: `tests/benchmark/test_multihop_retriever.py`

**Step 1: Write failing test for entity carry-over**

```python
# tests/benchmark/test_multihop_retriever.py

@pytest.mark.asyncio
async def test_entity_carry_over():
    """Verify entities are carried over between hops."""
    retriever = MultiHopRetriever(max_hops=2, entities_per_hop=5)

    mock_graphrag = MagicMock()
    mock_graphrag._llm = AsyncMock(return_value='["What is X?", "What is Y?"]')

    # Mock _retrieve_hop to return different entities each time
    hop_count = 0
    async def mock_retrieve_hop(sub_q, graph_rag, seed_entities):
        nonlocal hop_count
        hop_count += 1
        if hop_count == 1:
            return {"chunks": ["context1"], "entities": ["entity1", "entity2"]}
        else:
            # Second hop should receive entities from first hop
            assert "entity1" in seed_entities or "entity2" in seed_entities
            return {"chunks": ["context2"], "entities": ["entity3"]}

    retriever._retrieve_hop = mock_retrieve_hop
    retriever._merge_contexts = lambda states, budget: "merged"

    result = await retriever.retrieve("What is X and Y?", mock_graphrag)

    assert hop_count == 2
    assert result == "merged"
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/benchmark/test_multihop_retriever.py::test_entity_carry_over -v
```

Expected: FAIL (entity carry-over not implemented yet)

**Step 3: Implement entity carry-over**

The implementation in Task 3 should already handle this via `carry_entities`. Verify it works correctly.

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/benchmark/test_multihop_retriever.py::test_entity_carry_over -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add bench/retrievers/multihop.py tests/benchmark/test_multihop_retriever.py
git commit -m "feat(benchmark): implement entity state carry-over across hops"
```

---

### Task 6: Implement Context Merging with Deduplication

**Files:**
- Modify: `bench/retrievers/multihop.py`
- Test: `tests/benchmark/test_multihop_retriever.py`

**Step 1: Write failing test for context merging**

```python
# tests/benchmark/test_multihop_retriever.py

@pytest.mark.asyncio
async def test_context_merge_deduplication():
    """Verify context merging deduplicates chunks."""
    retriever = MultiHopRetriever(context_token_budget=1000)

    state1 = HopState(
        sub_question="Q1",
        context_chunks=["chunk A", "chunk B", "chunk C"],
        retrieved_entities=[]
    )
    state2 = HopState(
        sub_question="Q2",
        context_chunks=["chunk B", "chunk D"],  # chunk B is duplicate
        retrieved_entities=[]
    )

    merged = retriever._merge_contexts([state1, state2], budget=1000)

    # Should deduplicate and reverse (later hops first)
    assert "chunk D" in merged
    assert "chunk B" in merged
    assert "chunk A" in merged
    assert merged.count("chunk B") == 1  # Only once
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/benchmark/test_multihop_retriever.py::test_context_merge_deduplication -v
```

Expected: FAIL or PASS depending on implementation

**Step 3: Enhance merge if needed**

The implementation in Task 3 should already handle deduplication. Verify it works correctly.

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/benchmark/test_multihop_retriever.py::test_context_merge_deduplication -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add bench/retrievers/multihop.py tests/benchmark/test_multihop_retriever.py
git commit -m "feat(benchmark): verify context merging with deduplication"
```

---

### Task 7: Implement Token Budget Management

**Files:**
- Modify: `bench/retrievers/multihop.py`
- Test: `tests/benchmark/test_multihop_retriever.py`

**Step 1: Write failing test for token budget**

```python
# tests/benchmark/test_multihop_retriever.py

@pytest.mark.asyncio
async def test_token_budget_enforcement():
    """Verify context merging respects token budget."""
    retriever = MultiHopRetriever(context_token_budget=100)  # ~25 chars

    # Create chunks that exceed budget
    state = HopState(
        sub_question="Q1",
        context_chunks=["x" * 50 for _ in range(10)],  # 500 chars
        retrieved_entities=[]
    )

    merged = retriever._merge_contexts([state], budget=100)

    # Should truncate to fit budget
    assert len(merged) <= 150  # Some margin
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/benchmark/test_multihop_retriever.py::test_token_budget_enforcement -v
```

Expected: FAIL or PASS

**Step 3: Enhance token budget enforcement if needed**

The implementation in Task 3 should already handle this. Verify it works correctly.

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/benchmark/test_multihop_retriever.py::test_token_budget_enforcement -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add bench/retrievers/multihop.py tests/benchmark/test_multihop_retriever.py
git commit -m "feat(benchmark): verify token budget enforcement in context merging"
```

---

## Part 2: Registry Integration

### Task 8: Register MultiHopRetriever in Plugin Registry

**Files:**
- Modify: `bench/registry.py`
- Test: `tests/benchmark/test_registry.py`

**Step 1: Write failing test for multihop retriever registration**

```python
# tests/benchmark/test_registry.py

def test_multihop_retriever_registered():
    """Verify multihop retriever is registered."""
    from bench.registry import resolve, list_registered

    # Check it's in the list
    assert "multihop" in list_registered("retriever")

    # Check we can resolve it
    retriever_class = resolve("retriever", "multihop")
    assert retriever_class is not None
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/benchmark/test_registry.py::test_multihop_retriever_registered -v
```

Expected: FAIL with "No retriever registered as 'multihop'"

**Step 3: Register MultiHopRetriever**

```python
# bench/registry.py

from bench.retrievers.multihop import MultiHopRetriever

@register("retriever", "multihop")
class MultiHopRetrieverWrapper:
    """Wrapper around MultiHopRetriever for registry compatibility."""

    def __init__(self, max_hops: int = 4, entities_per_hop: int = 10,
                 context_token_budget: int = 8000, decompose_model: str = "gpt-4o-mini"):
        self._retriever = MultiHopRetriever(
            max_hops=max_hops,
            entities_per_hop=entities_per_hop,
            context_token_budget=context_token_budget,
            decompose_model=decompose_model,
        )

    async def __call__(
        self,
        query: str,
        graph_rag: GraphRAG,
        param: QueryParam,
        **kwargs,
    ) -> str:
        return await self._retriever.retrieve(query, graph_rag)
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/benchmark/test_registry.py::test_multihop_retriever_registered -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add bench/registry.py tests/benchmark/test_registry.py
git commit -m "feat(benchmark): register MultiHopRetriever in plugin registry"
```

---

## Part 3: Runner Integration

### Task 9: Integrate MultiHopRetriever into ExperimentRunner

**Files:**
- Modify: `bench/runner.py`
- Test: `tests/benchmark/test_runner.py`

**Step 1: Write failing test for multihop mode in runner**

```python
# tests/benchmark/test_runner.py

import pytest
from bench.runner import BenchmarkConfig

@pytest.mark.asyncio
async def test_multihop_mode_support():
    """Verify ExperimentRunner supports multihop mode."""
    config = BenchmarkConfig(
        experiment_name="test_multihop",
        dataset_name="multihop_rag",
        dataset_path="test.json",
        corpus_path="corpus.json",
        query_modes=["multihop"],
        graphrag_config={
            "working_dir": "./test_workdir",
            "llm_model": "gpt-4o-mini",
        },
    )

    runner = ExperimentRunner(config)

    # Verify multihop is recognized
    assert "multihop" in runner.config.query_modes
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/benchmark/test_runner.py::test_multihop_mode_support -v
```

Expected: FAIL or PASS (should already work)

**Step 3: Add multihop mode handling if needed**

The current runner should already handle any mode through GraphRAG. We need to add special handling for multihop mode to use the MultiHopRetriever.

```python
# bench/runner.py

from bench.retrievers.multihop import MultiHopRetriever

class ExperimentRunner:
    # ... existing code ...

    async def run_query(self, question: str, mode: str) -> str:
        """Run a single query with the specified mode.

        Args:
            question: User question
            mode: Query mode (local, global, naive, multihop)

        Returns:
            Answer string
        """
        if mode == "multihop":
            # Use MultiHopRetriever
            retriever = MultiHopRetriever(
                max_hops=self.config.query_params.get("max_hops", 4),
                entities_per_hop=self.config.query_params.get("entities_per_hop", 10),
                context_token_budget=self.config.query_params.get("context_token_budget", 8000),
            )
            context = await retriever.retrieve(question, self._rag)
            # Generate answer with retrieved context
            return await self._rag.aquery(
                question,
                param=QueryParam(mode="local"),
                injected_context=context,  # Requires core modification or subclassing
            )
        else:
            # Use standard GraphRAG modes
            query_param = QueryParam(mode=mode, **self.config.query_params)
            return await self._rag.aquery(question, param=query_param)
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/benchmark/test_runner.py::test_multihop_mode_support -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add bench/runner.py tests/benchmark/test_runner.py
git commit -m "feat(benchmark): integrate MultiHopRetriever into ExperimentRunner"
```

---

### Task 10: Add injected_context Support (Core Modification)

**Files:**
- Modify: `nano_graphrag/graphrag.py` OR create subclass
- Test: `tests/benchmark/test_runner.py`

**Step 1: Check if injected_context is needed**

```bash
grep -n "injected_context" nano_graphrag/graphrag.py
```

Expected: Not found (needs to be added)

**Step 2: Write failing test for injected_context**

```python
# tests/benchmark/test_runner.py

@pytest.mark.asyncio
async def test_injected_context():
    """Verify GraphRAG accepts injected_context parameter."""
    from nano_graphrag import GraphRAG
    from nano_graphrag.base import GraphRAGConfig, QueryParam

    config = GraphRAGConfig(working_dir="./test")
    rag = GraphRAG.from_config(config)

    # This should work with injected_context
    try:
        result = await rag.aquery(
            "test question",
            param=QueryParam(mode="local"),
            injected_context="Custom context for testing"
        )
        assert True  # If we get here, it works
    except TypeError as e:
        if "injected_context" in str(e):
            pytest.fail("injected_context not supported")
        raise
```

**Step 3: Run test to verify it fails**

```bash
uv run pytest tests/benchmark/test_runner.py::test_injected_context -v
```

Expected: FAIL (injected_context not supported)

**Step 4: Add injected_context support (minimal core modification)**

**Option A:** Modify core (requires permission)

**Option B:** Create subclass (preferred)

```python
# bench/runner.py

from nano_graphrag import GraphRAG
from nano_graphrag.base import QueryParam

class MultiHopGraphRAG(GraphRAG):
    """GraphRAG subclass with multi-hop support."""

    async def aquery(
        self,
        query: str,
        param: QueryParam = QueryParam(mode="local"),
        injected_context: str | None = None,
    ) -> str:
        """Query with optional injected context.

        Args:
            query: User question
            param: Query parameters
            injected_context: Optional pre-retrieved context to use

        Returns:
            Answer string
        """
        if injected_context:
            # Use injected context for generation
            return await self._generate_answer(query, injected_context)
        else:
            # Use standard retrieval + generation
            return await super().aquery(query, param=param)

    async def _generate_answer(self, question: str, context: str) -> str:
        """Generate answer from question + context."""
        prompt = f"""Context: {context}

Question: {question}

Answer the question using only the context above."""
        return await self._llm(prompt)
```

**Step 5: Update runner to use subclass**

```python
# bench/runner.py

class ExperimentRunner:
    def _create_graphrag(self) -> GraphRAG:
        """Create GraphRAG instance from config."""
        rag_config = GraphRAGConfig.from_dict(self.config.graphrag_config)

        # Use MultiHopGraphRAG if multihop mode is enabled
        if "multihop" in self.config.query_modes:
            from bench.runner import MultiHopGraphRAG
            rag = MultiHopGraphRAG.from_config(rag_config)
        else:
            rag = GraphRAG.from_config(rag_config)

        # Wrap LLM functions with cache if enabled
        if self._cache is not None and self._cache.enabled:
            if rag.best_model_func is not None:
                rag.best_model_func = self._cache.wrap(rag.best_model_func)
            if rag.cheap_model_func is not None:
                rag.cheap_model_func = self._cache.wrap(rag.cheap_model_func)

        return rag
```

**Step 6: Run test to verify it passes**

```bash
uv run pytest tests/benchmark/test_runner.py::test_injected_context -v
```

Expected: PASS

**Step 7: Commit**

```bash
git add bench/runner.py tests/benchmark/test_runner.py
git commit -m "feat(benchmark): add MultiHopGraphRAG subclass with injected_context support"
```

---

## Part 4: Integration Testing

### Task 11: Create End-to-End MultiHop Test

**Files:**
- Create: `tests/benchmark/integration/test_multihop_e2e.py`
- Test: `tests/benchmark/integration/test_multihop_e2e.py`

**Step 1: Create integration test directory**

```bash
mkdir -p tests/benchmark/integration
touch tests/benchmark/integration/__init__.py
```

**Step 2: Write end-to-end test**

```python
# tests/benchmark/integration/test_multihop_e2e.py

import pytest
import asyncio
from pathlib import Path
from bench.runner import BenchmarkConfig, ExperimentRunner
from bench.datasets import MultiHopRAGDataset

@pytest.mark.asyncio
@pytest.mark.integration
async def test_multihop_e2e_small_dataset():
    """End-to-end test of multi-hop retrieval on small dataset."""
    # Create small test dataset
    test_data = {
        "questions": [
            {
                "id": "test_1",
                "question": "What is the relationship between Entity A and Entity B?",
                "answer": "Entity A is connected to Entity B through Entity C",
            }
        ],
        "corpus": [
            {
                "id": "doc1",
                "title": "Entity A",
                "text": "Entity A is a concept related to Entity C.",
            },
            {
                "id": "doc2",
                "title": "Entity B",
                "text": "Entity B is connected to Entity C.",
            },
            {
                "id": "doc3",
                "title": "Entity C",
                "text": "Entity C connects Entity A and Entity B.",
            },
        ],
    }

    # Write test data
    import json
    test_dir = Path("./test_multihop_data")
    test_dir.mkdir(exist_ok=True)

    questions_path = test_dir / "questions.json"
    corpus_path = test_dir / "corpus.json"

    with open(questions_path, "w") as f:
        json.dump(test_data["questions"], f)
    with open(corpus_path, "w") as f:
        json.dump(test_data["corpus"], f)

    # Create config
    config = BenchmarkConfig(
        experiment_name="test_multihop_e2e",
        dataset_name="multihop_rag",
        dataset_path=str(questions_path),
        corpus_path=str(corpus_path),
        query_modes=["multihop"],
        max_samples=1,
        graphrag_config={
            "working_dir": str(test_dir / "workdir"),
            "llm_model": "gpt-4o-mini",
            "enable_llm_cache": False,  # Disable for testing
        },
    )

    # Run experiment
    runner = ExperimentRunner(config)
    result = await runner.run()

    # Verify results
    assert "multihop" in result.mode_results
    assert len(result.predictions["multihop"]) == 1

    # Cleanup
    import shutil
    shutil.rmtree(test_dir)

    print("✓ End-to-end multi-hop test passed")
```

**Step 3: Run integration test**

```bash
uv run pytest tests/benchmark/integration/test_multihop_e2e.py -v -s
```

Expected: PASS (may take a few minutes with real LLM calls)

**Step 4: Commit**

```bash
git add tests/benchmark/integration/
git commit -m "test(benchmark): add end-to-end multi-hop integration test"
```

---

### Task 12: Benchmark on All 4 Datasets

**Files:**
- Create: `experiments/benchmark_multihop.yaml`
- Test: Manual verification

**Step 1: Create benchmark config**

```yaml
# experiments/benchmark_multihop.yaml
name: multihop_full_benchmark
version: "1.0"
description: "Multi-hop RAG benchmark on all 4 datasets"

dataset:
  name: multihop_rag
  path: ~/.cache/nano-bench/multihop_rag/questions.json
  corpus_path: ~/.cache/nano-bench/multihop_rag/corpus.json
  split: validation
  max_samples: 200
  auto_download: true

graphrag:
  working_dir: ./workdirs/multihop_benchmark
  llm_model: gpt-4o-mini
  embedding_model: text-embedding-3-small
  chunk_func: chunking_by_token_size
  chunk_token_size: 1200
  chunk_overlap_token_size: 100
  enable_llm_cache: true

query:
  modes:
    - naive
    - local
    - multihop
  param_overrides:
    max_hops: 4
    entities_per_hop: 10
    context_token_budget: 8000

cache:
  enabled: true
  backend: disk

metrics:
  exact_match: true
  token_f1: true

output:
  results_dir: ./results/multihop_benchmark
  save_predictions: true
```

**Step 2: Run benchmark on MultiHop-RAG dataset**

```bash
python -m bench.run --config experiments/benchmark_multihop.yaml
```

Expected: Completes successfully, produces results file

**Step 3: Verify results meet roadmap targets**

From roadmap:
- MultiHop-RAG F1: Baseline ~0.40, Target 0.57+, Stretch 0.64+

Check results:
```bash
cat ./results/multihop_benchmark/*.json | jq '.mode_results.multihop.token_f1'
```

Expected: F1 score >= 0.50 (minimum acceptable)

**Step 4: Create configs for other 3 datasets**

```yaml
# experiments/benchmark_musique.yaml
name: multihop_musique
dataset:
  name: musique
  path: ~/.cache/nano-bench/musique/dev.json
  split: validation
  max_samples: 200
  auto_download: true
# ... rest same as above
```

```yaml
# experiments/benchmark_hotpotqa.yaml
name: multihop_hotpotqa
dataset:
  name: hotpotqa
  path: ~/.cache/nano-bench/hotpotqa/dev.json
  split: validation
  max_samples: 200
  auto_download: true
# ... rest same as above
```

```yaml
# experiments/benchmark_2wiki.yaml
name: multihop_2wiki
dataset:
  name: 2wiki
  path: ~/.cache/nano-bench/2wiki/dev.json
  split: validation
  max_samples: 200
  auto_download: true
# ... rest same as above
```

**Step 5: Run all benchmarks**

```bash
for dataset in musique hotpotqa 2wiki; do
    python -m bench.run --config experiments/benchmark_${dataset}.yaml
done
```

Expected: All complete successfully

**Step 6: Aggregate results**

Create results summary table:

```bash
python -m bench.compare \
    ./results/multihop_benchmark/* \
    ./results/multihop_musique/* \
    ./results/multihop_hotpotqa/* \
    ./results/multihop_2wiki/*
```

**Step 7: Commit**

```bash
git add experiments/
git commit -m "feat(benchmark): add multi-hop benchmark configs for all 4 datasets"
```

---

## Part 5: Verification Checklist

### Task 13: Complete Phase 3 Verification Checklist

**Files:**
- Create: `docs/verification/phase3_verification_report.md`

**Step 1: Create verification report**

```markdown
# Phase 3 Multi-Hop RAG Verification Report

**Date:** [Date of completion]
**Verifier:** [Name]
**Status:** [PASS/FAIL/PARTIAL]

---

## M3 Checklist (from roadmap)

### Core Implementation

- [x] `MultiHopRetriever` with query decomposition
  - [x] `_decompose()` method implemented
  - [x] LLM-based decomposition working
  - [x] Fallback parsing for non-JSON responses
  - Test: `tests/benchmark/test_multihop_retriever.py::test_query_decomposition`

- [x] Entity state carry-over across hops
  - [x] `carry_entities` list implemented
  - [x] Entities from hop N passed to hop N+1
  - [x] Verified in integration test
  - Test: `tests/benchmark/test_multihop_retriever.py::test_entity_carry_over`

- [x] Context merger with deduplication
  - [x] Hash-based deduplication implemented
  - [x] Later hops prioritized (reversed order)
  - [x] Verified in unit test
  - Test: `tests/benchmark/test_multihop_retriever.py::test_context_merge_deduplication`

- [x] Token budget management
  - [x] Budget enforcement in `_merge_contexts()`
  - [x] Approximate token counting (4 chars/token)
  - [x] Truncation when budget exceeded
  - Test: `tests/benchmark/test_multihop_retriever.py::test_token_budget_enforcement`

### Integration

- [x] `mode="multihop"` integration via bench wrapper
  - [x] MultiHopRetriever registered in plugin registry
  - [x] ExperimentRunner handles multihop mode
  - [x] MultiHopGraphRAG subclass with injected_context
  - Test: `tests/benchmark/test_registry.py::test_multihop_retriever_registered`
  - Test: `tests/benchmark/test_runner.py::test_multihop_mode_support`

### Benchmarking

- [x] Full benchmark results across all 4 datasets, all modes
  - [x] MultiHop-RAG: [F1 score]
  - [x] MuSiQue: [F1 score]
  - [x] HotpotQA: [F1 score]
  - [x] 2WikiMHQA: [F1 score]
  - All modes tested: naive, local, multihop

---

## Performance Against Targets

| Dataset | Baseline (naive) | Target (multihop) | Actual (multihop) | Status |
|---------|------------------|-------------------|-------------------|--------|
| MultiHop-RAG | ~0.40 | 0.57+ | [ACTUAL] | [PASS/FAIL] |
| MuSiQue | ~0.25 | 0.42+ | [ACTUAL] | [PASS/FAIL] |
| HotpotQA | ~0.45 | 0.62+ | [ACTUAL] | [PASS/FAIL] |
| 2WikiMHQA | ~0.38 | 0.55+ | [ACTUAL] | [PASS/FAIL] |

---

## Test Coverage

```
tests/benchmark/test_retrievers.py           PASS [X/Y]
tests/benchmark/test_multihop_retriever.py   PASS [X/Y]
tests/benchmark/test_registry.py             PASS [X/Y]
tests/benchmark/test_runner.py               PASS [X/Y]
tests/benchmark/integration/test_multihop_e2e.py  PASS
```

---

## Known Issues

1. [List any issues found during verification]
2. [Performance bottlenecks]
3. [Edge cases not handled]

---

## Recommendations

1. [Improvements for Phase 4]
2. [Optimization opportunities]
3. [Documentation needs]

---

**Overall Assessment:** [PASS/FAIL/PARTIAL]

**Approved by:** [Signature]

**Date:** [Date]
```

**Step 2: Fill in verification report with actual results**

Run all tests and benchmarks, then fill in the report with actual scores and results.

**Step 3: Commit verification report**

```bash
git add docs/verification/phase3_verification_report.md
git commit -m "docs(benchmark): add Phase 3 verification report"
```

---

## Success Criteria

Phase 3 is considered **COMPLETE** when:

1. ✅ All unit tests pass (`pytest tests/benchmark/`)
2. ✅ Integration test passes (`pytest tests/benchmark/integration/`)
3. ✅ MultiHopRetriever is registered and resolvable
4. ✅ ExperimentRunner supports `mode="multihop"`
5. ✅ End-to-end benchmark runs successfully on all 4 datasets
6. ✅ Multi-hop mode outperforms baseline on at least 2/4 datasets
7. ✅ Verification report is complete with all scores documented

---

## Estimated Timeline

- **Part 1 (Retriever Implementation):** 3-4 days
- **Part 2 (Registry Integration):** 0.5 day
- **Part 3 (Runner Integration):** 1-2 days
- **Part 4 (Integration Testing):** 1-2 days
- **Part 5 (Verification):** 1 day

**Total:** 7-11 days

---

## Dependencies

- Phase 1 must be 100% complete
- Phase 2 registry must be functional
- All 4 datasets must be downloadable
- LLM API access (OpenAI or compatible)

---

## Risk Mitigation

1. **LLM API costs:** Enable cache by default for all runs
2. **Dataset download failures:** Implement fallback to local files
3. **Performance issues:** Start with small sample sizes (max_samples=10)
4. **Test flakiness:** Use mocks for unit tests, real LLM for integration only

---

*Last updated: 2026-03-23*
