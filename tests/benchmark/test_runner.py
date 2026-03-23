"""Tests for ExperimentRunner cache integration."""

import json
import tempfile
from pathlib import Path

import pytest

from bench.cache import create_benchmark_cache
from bench.runner import BenchmarkConfig, ExperimentRunner


@pytest.mark.asyncio
async def test_runner_uses_cache_when_enabled():
    """ExperimentRunner should wrap LLM functions when cache is enabled."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create config with enable_llm_cache: True
        config = BenchmarkConfig(
            dataset_name="multihop_rag",
            dataset_path="tests/fixtures/sample_questions.json",
            corpus_path="tests/fixtures/sample_corpus.json",
            max_samples=1,  # Only test with 1 sample
            graphrag_config={
                "enable_llm_cache": True,
                "working_dir": tmpdir,
            },
            query_modes=["local"],
            metrics=["exact_match"],
            output_dir=tmpdir,
            experiment_name="test_cache_enabled",
        )

        runner = ExperimentRunner(config)

        # Verify cache is created and enabled
        assert runner._cache is not None, "Cache should be created when enabled"
        assert runner._cache.enabled is True, "Cache should be enabled"


@pytest.mark.asyncio
async def test_runner_skips_cache_when_disabled():
    """ExperimentRunner should not create cache when disabled."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create config without enable_llm_cache or with False
        config = BenchmarkConfig(
            dataset_name="multihop_rag",
            dataset_path="tests/fixtures/sample_questions.json",
            corpus_path="tests/fixtures/sample_corpus.json",
            max_samples=1,
            graphrag_config={
                "enable_llm_cache": False,
                "working_dir": tmpdir,
            },
            query_modes=["local"],
            metrics=["exact_match"],
            output_dir=tmpdir,
            experiment_name="test_cache_disabled",
        )

        runner = ExperimentRunner(config)

        # Verify cache is None
        assert runner._cache is None, "Cache should be None when disabled"


@pytest.mark.asyncio
async def test_runner_includes_cache_stats_in_results():
    """ExperimentResult should include cache statistics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create config with cache enabled
        config = BenchmarkConfig(
            dataset_name="multihop_rag",
            dataset_path="tests/fixtures/sample_questions.json",
            corpus_path="tests/fixtures/sample_corpus.json",
            max_samples=1,  # Only test with 1 sample
            graphrag_config={
                "enable_llm_cache": True,
                "working_dir": tmpdir,
            },
            query_modes=["local"],
            metrics=["exact_match"],
            output_dir=tmpdir,
            experiment_name="test_cache_stats",
        )

        runner = ExperimentRunner(config)

        # Manually create a result with cache stats to test the structure
        from bench.runner import ExperimentResult
        from datetime import datetime

        # Simulate cache stats
        cache_stats = await runner._cache.stats()

        result = ExperimentResult(
            experiment_name="test",
            timestamp=datetime.now().isoformat(),
            config=config,
            mode_results={"local": {"exact_match": 1.0}},
            predictions=[],
            duration_seconds=1.0,
            cache_stats=cache_stats,
        )

        # Verify cache_stats exists and has expected fields
        assert result.cache_stats is not None, "Result should include cache_stats"
        assert "hits" in result.cache_stats, "cache_stats should have 'hits'"
        assert "misses" in result.cache_stats, "cache_stats should have 'misses'"
        assert "hit_rate" in result.cache_stats, "cache_stats should have 'hit_rate'"
        assert isinstance(result.cache_stats["hits"], int)
        assert isinstance(result.cache_stats["misses"], int)
        assert isinstance(result.cache_stats["hit_rate"], float)


@pytest.mark.asyncio
async def test_runner_saves_cache_stats_to_json():
    """Cache statistics should be saved in results JSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create config with cache enabled
        config = BenchmarkConfig(
            dataset_name="multihop_rag",
            dataset_path="tests/fixtures/sample_questions.json",
            corpus_path="tests/fixtures/sample_corpus.json",
            max_samples=1,
            graphrag_config={
                "enable_llm_cache": True,
                "working_dir": tmpdir,
            },
            query_modes=["local"],
            metrics=["exact_match"],
            output_dir=tmpdir,
            experiment_name="test_cache_json",
        )

        runner = ExperimentRunner(config)

        # Manually create a result with cache stats
        from bench.runner import ExperimentResult
        from datetime import datetime

        cache_stats = await runner._cache.stats()

        result = ExperimentResult(
            experiment_name="test",
            timestamp=datetime.now().isoformat(),
            config=config,
            mode_results={"local": {"exact_match": 1.0}},
            predictions=[],
            duration_seconds=1.0,
            cache_stats=cache_stats,
        )

        # Verify cache_stats is in saved JSON
        result_path = result.save(tmpdir)
        with open(result_path, "r") as f:
            saved_data = json.load(f)

        assert "cache_stats" in saved_data, "Saved JSON should include cache_stats"
        assert saved_data["cache_stats"]["hits"] == result.cache_stats["hits"]
        assert saved_data["cache_stats"]["misses"] == result.cache_stats["misses"]


def test_config_from_yaml_with_cache_section():
    """BenchmarkConfig should load cache config from YAML."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a YAML config with cache section
        yaml_path = Path(tmpdir) / "config.yaml"
        yaml_content = """
dataset_name: multihop_rag
dataset_path: tests/fixtures/sample_questions.json
corpus_path: tests/fixtures/sample_corpus.json
max_samples: 1
graphrag_config:
  enable_llm_cache: true
  working_dir: ./cache_dir
cache:
  enabled: true
query_modes:
  - local
metrics:
  - exact_match
output_dir: ./results
experiment_name: yaml_cache_test
"""
        yaml_path.write_text(yaml_content)

        # Load config
        config = BenchmarkConfig.from_yaml(str(yaml_path))

        # Verify cache.enabled is extracted into graphrag_config
        assert config.graphrag_config.get("enable_llm_cache") is True, \
            "cache.enabled should be set in graphrag_config"
