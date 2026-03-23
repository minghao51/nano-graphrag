"""Tests for MultiHopRetriever implementation."""

import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.mark.asyncio
async def test_multihop_retriever_init():
    """Verify MultiHopRetriever initializes with roadmap parameters."""
    from bench.retrievers.multihop import MultiHopRetriever

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
    from bench.retrievers.multihop import MultiHopRetriever

    retriever = MultiHopRetriever(max_hops=3)

    # Mock GraphRAG instance
    mock_graphrag = MagicMock()
    mock_graphrag._llm = AsyncMock(return_value='["Who is X?", "What is Y?", "How are X and Y related?"]')

    sub_questions = await retriever._decompose("Who is X and how are they related to Y?", mock_graphrag)

    assert len(sub_questions) == 3
    assert sub_questions[0] == "Who is X?"
    assert sub_questions[1] == "What is Y?"
    assert sub_questions[2] == "How are X and Y related?"
