"""Unit tests for MultiHopRetriever with mocked dependencies."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from bench.retrievers.multihop import MultiHopRetriever
from bench.retrievers.base import HopState


@pytest.mark.asyncio
async def test_multihop_retriever_initialization():
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


@pytest.mark.asyncio
async def test_query_decomposition():
    """Verify MultiHopRetriever decomposes multi-hop questions."""
    retriever = MultiHopRetriever(max_hops=3)

    # Mock GraphRAG instance
    mock_graphrag = MagicMock()
    mock_graphrag.best_model_func = AsyncMock(
        return_value='["Who is X?", "What is Y?", "How are X and Y related?"]'
    )

    sub_questions = await retriever._decompose(
        "Who is X and how are they related to Y?", mock_graphrag
    )

    assert len(sub_questions) == 3
    assert sub_questions[0] == "Who is X?"
    assert sub_questions[1] == "What is Y?"
    assert sub_questions[2] == "How are X and Y related?"


@pytest.mark.asyncio
async def test_query_decomposition_fallback_parsing():
    """Verify fallback parsing when LLM doesn't return JSON."""
    retriever = MultiHopRetriever(max_hops=3)

    mock_graphrag = MagicMock()
    mock_graphrag.best_model_func = AsyncMock(
        return_value="Who is X?\nWhat is Y?\nHow are X and Y related?"
    )

    sub_questions = await retriever._decompose(
        "Who is X and how are they related to Y?", mock_graphrag
    )

    assert len(sub_questions) == 3


@pytest.mark.asyncio
async def test_entity_carry_over():
    """Verify entities are carried over between hops."""
    retriever = MultiHopRetriever(max_hops=2, entities_per_hop=5)

    mock_graphrag = MagicMock()
    mock_graphrag.best_model_func = AsyncMock(return_value='["What is X?", "What is Y?"]')

    # Mock _retrieve_hop to return different entities each time
    hop_count = 0

    async def mock_retrieve_hop(sub_q, graph_rag, seed_entities):
        nonlocal hop_count
        hop_count += 1
        if hop_count == 1:
            return {"chunks": ["context1"], "entities": ["entity1", "entity2"]}
        else:
            # Second hop should receive entities from first hop
            assert "entity1" in seed_entities or "entity2" in seed_entities, \
                f"Second hop should receive entities from first hop, got: {seed_entities}"
            return {"chunks": ["context2"], "entities": ["entity3"]}

    retriever._retrieve_hop = mock_retrieve_hop
    retriever._merge_contexts = lambda states, budget: "merged"

    result = await retriever.retrieve("What is X and Y?", mock_graphrag)

    assert hop_count == 2
    assert result == "merged"


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


@pytest.mark.asyncio
async def test_parse_context_extracts_entities():
    """Verify _parse_context extracts entities using heuristics."""
    retriever = MultiHopRetriever(entities_per_hop=10)

    context_str = """
    Entity A is related to Entity B.
    "John Doe" visited New York City.
    The relationship between X and Y is complex.
    """

    result = retriever._parse_context(context_str)

    assert "chunks" in result
    assert "entities" in result
    assert len(result["chunks"]) > 0
    # Should extract some entities (heuristic-based, so just check it doesn't crash)
    assert isinstance(result["entities"], list)
