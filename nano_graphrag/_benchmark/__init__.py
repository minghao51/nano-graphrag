"""
Benchmark infrastructure for nano-graphrag.

This package provides a minimal framework for running reproducible experiments
on GraphRAG with multi-hop RAG benchmarks.

.. deprecated::
    Use ``bench`` module instead. The ``nano_graphrag._benchmark`` module
    will be removed in a future version.
"""

import warnings

warnings.warn(
    "nano_graphrag._benchmark is deprecated, use 'bench' module instead",
    DeprecationWarning,
    stacklevel=2,
)

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
