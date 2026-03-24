"""
Benchmark infrastructure for nano-graphrag.

This package provides a minimal framework for running reproducible experiments
on GraphRAG with multi-hop RAG benchmarks.
"""

from .compare import compare_results, format_delta_table
from .datasets import BenchmarkDataset, MultiHopRAGDataset, QAPair
from .metrics import (
    ExactMatchMetric,
    Metric,
    MetricSuite,
    NativeContextRecallMetric,
    TokenF1Metric,
    get_baseline_suite,
)
from .registry import (
    GlobalRetriever,
    LocalRetriever,
    NaiveRetriever,
    SeparatorChunker,
    TokenSizeChunker,
    clear_registry,
    list_registered,
    register,
    resolve,
)
from .results import (
    JSONResultsBackend,
    PredictionRecord,
    ResultsBackend,
    RunResult,
)
from .runner import (
    ABConfig,
    ABExperimentRunner,
    BenchmarkConfig,
    ExperimentResult,
    ExperimentRunner,
)

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
    # A/B
    "ABConfig",
    "ABExperimentRunner",
    # Registry
    "register",
    "resolve",
    "list_registered",
    "clear_registry",
    "TokenSizeChunker",
    "SeparatorChunker",
    "LocalRetriever",
    "GlobalRetriever",
    "NaiveRetriever",
    # Results
    "ResultsBackend",
    "JSONResultsBackend",
    "RunResult",
    "PredictionRecord",
    # Compare
    "compare_results",
    "format_delta_table",
]
