"""Tests for benchmark results backends."""

from bench.results import JSONResultsBackend, PredictionRecord, RunResult


async def test_json_results_backend_load_rehydrates_prediction_records(tmp_path):
    """Loading a saved run should reconstruct PredictionRecord dataclasses."""
    backend = JSONResultsBackend(str(tmp_path))
    result = RunResult(
        run_id="run-1",
        experiment_name="benchmark",
        timestamp="2026-03-23T19:30:00",
        config={"dataset": {"name": "musique"}},
        variant_label="baseline",
        mode_results={"local": {"exact_match": 1.0}},
        predictions=[
            PredictionRecord(
                question_id="q1",
                question="Where was Ada Lovelace born?",
                gold_answer="London",
                prediction="London",
                metrics={"exact_match": 1.0},
                latency_seconds=0.1,
            )
        ],
        aggregate_metrics={"local": {"exact_match": 1.0}},
        cache_stats=None,
        timing={"query_seconds": 0.1},
        duration_seconds=0.1,
    )

    await backend.save(result)
    loaded = await backend.load("run-1")

    assert loaded is not None
    assert isinstance(loaded.predictions[0], PredictionRecord)
    assert loaded.predictions[0].question_id == "q1"
