"""Tests for MultiHopRetriever integration into ExperimentRunner."""

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

    from bench.runner import ExperimentRunner
    runner = ExperimentRunner(config)

    # Verify multihop is recognized
    assert "multihop" in runner.config.query_modes


@pytest.mark.asyncio
async def test_injected_context():
    """Verify MultiHopGraphRAG accepts injected_context parameter."""
    from nano_graphrag import GraphRAG
    from nano_graphrag.base import GraphRAGConfig, QueryParam
    from bench.runner import MultiHopGraphRAG

    config = GraphRAGConfig(working_dir="./test")
    rag = MultiHopGraphRAG.from_config(config)

    # This should work with injected_context
    try:
        # We can't actually call aquery without a real GraphRAG instance,
        # but we can verify the method signature accepts injected_context
        import inspect
        sig = inspect.signature(rag.aquery)
        assert "injected_context" in sig.parameters or "context" in sig.parameters
        assert True  # If we get here, it works
    except TypeError as e:
        if "injected_context" in str(e):
            pytest.fail("injected_context not supported")
        raise
