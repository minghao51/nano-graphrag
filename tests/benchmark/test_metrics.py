"""Tests for benchmark metrics."""

import pytest
from bench import NativeContextRecallMetric, QAPair


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

    context = "France is in Europe. Paris is the capital."

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
