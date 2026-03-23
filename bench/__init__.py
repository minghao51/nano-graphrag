"""
Benchmark infrastructure for nano-graphrag.

This package provides a minimal framework for running reproducible experiments
on GraphRAG with multi-hop RAG benchmarks.
"""

from .datasets import BenchmarkDataset, MultiHopRAGDataset, QAPair
from .metrics import (
    ExactMatchMetric,
    Metric,
    MetricSuite,
    NativeContextRecallMetric,
    TokenF1Metric,
    get_baseline_suite,
)
from .runner import BenchmarkConfig, ExperimentResult, ExperimentRunner

__all__ = [
    # Datasets
    "BenchmarkDataset",
    "MultiHopRAGDataset",
    "QAPair",
    # Metrics
    "Metric",
    "ExactMatchMetric",
    "NativeContextRecallMetric",
    "TokenF1Metric",
    "MetricSuite",
    "get_baseline_suite",
    # Runner
    "BenchmarkConfig",
    "ExperimentRunner",
    "ExperimentResult",
]
