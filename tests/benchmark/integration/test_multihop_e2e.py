"""End-to-end integration tests for multi-hop retrieval."""

import json
import pytest
import shutil
from pathlib import Path


@pytest.mark.asyncio
@pytest.mark.integration
async def test_multihop_e2e_small_dataset():
    """End-to-end test of multi-hop retrieval on small dataset."""
    from bench.runner import BenchmarkConfig, ExperimentRunner

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
            "llm_model": "gpt-4o-mini",
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
