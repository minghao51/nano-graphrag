"""Tests for MultiHopRetriever implementation."""

import pytest


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
