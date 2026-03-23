"""Tests for BenchmarkConfig."""

import tempfile
from pathlib import Path

import pytest

from bench import BenchmarkConfig


def test_load_nested_config():
    """Should load nested YAML config as specified in roadmap."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        config_yaml = """
name: test_experiment
version: "1.0"
description: "Test experiment"

dataset:
  name: musique
  split: validation
  max_samples: 100
  auto_download: true

graphrag:
  working_dir: ./workdirs/test
  llm_model: gpt-4o-mini
  embedding_model: text-embedding-3-small

query:
  modes:
    - local
    - global
  param_overrides:
    top_k: 20

cache:
  enabled: true
  backend: disk

metrics:
  exact_match: true
  token_f1: true
  llm_judge:
    enabled: false

output:
  results_dir: ./results
  save_predictions: true
"""
        config_path.write_text(config_yaml)

        config = BenchmarkConfig.from_yaml(str(config_path))

        assert config.experiment_name == "test_experiment"
        assert config.dataset_name == "musique"
        assert config.max_samples == 100
        assert config.auto_download is True
        assert config.graphrag_config["working_dir"] == "./workdirs/test"
        assert "local" in config.query_modes
        assert "global" in config.query_modes
        assert config.graphrag_config["enable_llm_cache"] is True


def test_load_flat_config():
    """Should still load flat schema for backward compatibility."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        config_yaml = """
dataset_name: multihop_rag
dataset_path: data/questions.json
corpus_path: data/corpus.json
max_samples: 50
graphrag_config:
  working_dir: ./cache
  enable_llm_cache: true
query_modes:
  - local
metrics:
  - exact_match
  - token_f1
output_dir: ./results
experiment_name: flat_schema_test
"""
        config_path.write_text(config_yaml)

        config = BenchmarkConfig.from_yaml(str(config_path))

        assert config.experiment_name == "flat_schema_test"
        assert config.dataset_name == "multihop_rag"
        assert config.max_samples == 50
        assert config.graphrag_config["working_dir"] == "./cache"
        assert config.graphrag_config["enable_llm_cache"] is True


def test_to_dict():
    """to_dict should produce roadmap-compliant nested schema."""
    config = BenchmarkConfig(
        experiment_name="test",
        dataset_name="musique",
        dataset_path="/path/to/data",
        max_samples=100,
        auto_download=True,
        graphrag_config={"working_dir": "./work"},
        query_modes=["local", "global"],
        metrics=["exact_match", "token_f1"],
        output_dir="./results",
    )

    result = config.to_dict()

    assert result["name"] == "test"
    assert result["dataset"]["name"] == "musique"
    assert result["dataset"]["max_samples"] == 100
    assert result["graphrag"]["working_dir"] == "./work"
