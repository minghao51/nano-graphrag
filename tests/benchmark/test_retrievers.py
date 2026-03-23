"""Tests for retriever infrastructure."""

import pytest

# These imports will fail until we create the module
# from bench.retrievers.base import Retriever, RetrieverResult


@pytest.mark.asyncio
async def test_retriever_protocol_exists():
    """Verify Retriever protocol is defined."""
    from bench.retrievers.base import Retriever
    assert Retriever is not None


@pytest.mark.asyncio
async def test_retriever_result_dataclass():
    """Verify RetrieverResult dataclass has required fields."""
    from bench.retrievers.base import RetrieverResult

    result = RetrieverResult(
        context="test context",
        entities=["entity1", "entity2"],
        hops=2,
        metadata={}
    )
    assert result.context == "test context"
    assert result.entities == ["entity1", "entity2"]
    assert result.hops == 2


@pytest.mark.asyncio
async def test_hop_state_dataclass():
    """Verify HopState has all required fields from roadmap."""
    from bench.retrievers.base import HopState

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
