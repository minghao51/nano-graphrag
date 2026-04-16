"""End-to-end integration tests for multi-hop retrieval."""

import json
import os
import pytest
import shutil
from pathlib import Path


def _has_live_llm_key() -> bool:
    """Check if any LLM provider has a valid API key."""
    # Check OpenRouter
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if openrouter_key and openrouter_key.startswith("sk-or-v1-"):
        return True

    # Check OpenAI
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if openai_key and openai_key.startswith("sk-"):
        return True

    return False


def _get_default_model() -> tuple[str, str]:
    """Return (model, provider) based on available keys."""
    if os.environ.get("OPENROUTER_API_KEY", "").strip().startswith("sk-or-v1-"):
        return ("openrouter/openai/gpt-4o-mini", "openrouter")

    if os.environ.get("OPENAI_API_KEY", "").strip().startswith("sk-"):
        return ("gpt-4o-mini", "openai")

    return ("gpt-4o-mini", "openai")  # Default fallback


@pytest.mark.asyncio
@pytest.mark.integration
async def test_multihop_e2e_small_dataset():
    """End-to-end test of multi-hop retrieval on small dataset."""
    if not _has_live_llm_key():
        pytest.skip("A live LLM API key (OpenRouter or OpenAI) is required for the integration test")

    from bench.runner import BenchmarkConfig, ExperimentRunner

    # Get model based on available API key
    llm_model, provider = _get_default_model()

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
                "content": "Entity A is a concept related to Entity C.",
            },
            {
                "id": "doc2",
                "title": "Entity B",
                "content": "Entity B is connected to Entity C.",
            },
            {
                "id": "doc3",
                "title": "Entity C",
                "content": "Entity C connects Entity A and Entity B.",
            },
        ],
    }

    # Write test data
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
            "llm_model": llm_model,
            "enable_llm_cache": False,  # Disable for testing
        },
    )

    try:
        # Run experiment
        runner = ExperimentRunner(config)
        result = await runner.run()

        # Verify results
        assert "multihop" in result.mode_results
        assert len(result.predictions["multihop"]) == 1

        print("✓ End-to-end multi-hop test passed")
    finally:
        # Cleanup
        if test_dir.exists():
            shutil.rmtree(test_dir)
