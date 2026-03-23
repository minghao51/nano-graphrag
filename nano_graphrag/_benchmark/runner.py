"""Experiment runner for GraphRAG benchmarks."""

import asyncio
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..base import GraphRAGConfig, QueryParam
from ..graphrag import GraphRAG
from .cache import create_benchmark_cache
from .datasets import BenchmarkDataset, MultiHopRAGDataset
from .metrics import MetricSuite, get_baseline_suite


@dataclass
class BenchmarkConfig:
    """Experiment configuration (YAML-serializable)."""

    # === Dataset ===
    dataset_name: str  # "multihop_rag", "hotpotqa", "musique", "2wiki"
    dataset_path: str  # Path to questions JSON file
    corpus_path: Optional[str] = None  # Path to corpus JSON file (for MultiHopRAG)
    dataset_split: str = "test"
    max_samples: int = -1  # -1 means all samples

    # === GraphRAG config ===
    graphrag_config: Dict[str, Any] = field(default_factory=dict)

    # === Query modes ===
    query_modes: List[str] = field(default_factory=lambda: ["local", "global"])
    query_params: Dict[str, Any] = field(default_factory=dict)

    # === Metrics ===
    metrics: List[str] = field(default_factory=lambda: ["exact_match", "token_f1"])

    # === Output ===
    output_dir: str = "./benchmark_results"
    experiment_name: str = "experiment"

    @classmethod
    def from_yaml(cls, path: str) -> "BenchmarkConfig":
        """Load config from YAML file."""
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required to load YAML configs. Install with: uv add pyyaml"
            )

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        # Handle nested cache section
        if "cache" in data and isinstance(data["cache"], dict):
            cache_config = data["cache"]
            if "enabled" in cache_config:
                # Ensure graphrag_config exists
                if "graphrag_config" not in data:
                    data["graphrag_config"] = {}
                data["graphrag_config"]["enable_llm_cache"] = cache_config["enabled"]
            # Remove cache section to avoid dataclass error
            del data["cache"]

        return cls(**data)

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "BenchmarkConfig":
        """Create config from dictionary."""
        # Filter out None values
        filtered = {k: v for k, v in config.items() if v is not None}
        return cls(**filtered)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "dataset_name": self.dataset_name,
            "dataset_path": self.dataset_path,
            "corpus_path": self.corpus_path,
            "dataset_split": self.dataset_split,
            "max_samples": self.max_samples,
            "graphrag_config": self.graphrag_config,
            "query_modes": self.query_modes,
            "query_params": self.query_params,
            "metrics": self.metrics,
            "output_dir": self.output_dir,
            "experiment_name": self.experiment_name,
        }

    def to_yaml(self, path: str) -> None:
        """Save config to YAML file."""
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required to save YAML configs. Install with: uv add pyyaml"
            )

        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


@dataclass
class ExperimentResult:
    """Result with config for full reproducibility."""

    experiment_name: str
    timestamp: str
    config: BenchmarkConfig
    mode_results: Dict[str, Dict[str, float]]  # mode -> metric_scores
    predictions: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)  # mode -> list of {question, prediction, gold}
    duration_seconds: float = 0.0
    cache_stats: Optional[Dict[str, Any]] = None

    def save(self, output_dir: str) -> str:
        """Save results to JSON file.

        Returns:
            Path to saved results file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp_safe = self.timestamp.replace(":", "-").replace(" ", "_")
        filename = f"{self.experiment_name}_{timestamp_safe}.json"
        filepath = output_path / filename

        # Convert to dict for JSON serialization
        result_dict = {
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp,
            "config": self.config.to_dict(),
            "mode_results": self.mode_results,
            "predictions": self.predictions,
            "duration_seconds": self.duration_seconds,
            "cache_stats": self.cache_stats,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)

        return str(filepath)


class ExperimentRunner:
    """Run benchmark experiments from config."""

    def __init__(self, config: BenchmarkConfig):
        """Initialize runner with configuration.

        Args:
            config: Benchmark configuration
        """
        self.config = config
        self._dataset: Optional[BenchmarkDataset] = None
        self._rag: Optional[GraphRAG] = None
        self._metric_suite: Optional[MetricSuite] = None
        self._cache = self._create_cache()

    def _create_cache(self):
        """Create LLM cache if enabled in config.

        Returns:
            BenchmarkLLMCache instance or None if disabled
        """
        enable_cache = self.config.graphrag_config.get("enable_llm_cache", False)
        if not enable_cache:
            return None

        # Get working directory for cache storage
        working_dir = self.config.graphrag_config.get("working_dir", "./cache")

        return create_benchmark_cache(working_dir, enabled=True)

    def _load_dataset(self) -> BenchmarkDataset:
        """Load dataset based on config."""
        dataset_name = self.config.dataset_name.lower()

        if dataset_name == "multihop_rag":
            if not self.config.corpus_path:
                raise ValueError("corpus_path is required for MultiHopRAG dataset")
            return MultiHopRAGDataset(
                questions_path=self.config.dataset_path,
                corpus_path=self.config.corpus_path,
                max_samples=self.config.max_samples,
            )
        elif dataset_name == "hotpotqa":
            from .datasets import HotpotQADataset

            return HotpotQADataset(
                data_path=self.config.dataset_path,
                split=self.config.dataset_split,
                max_samples=self.config.max_samples,
            )
        elif dataset_name == "musique":
            from .datasets import MuSiQueDataset

            return MuSiQueDataset(
                data_path=self.config.dataset_path,
                split=self.config.dataset_split,
                max_samples=self.config.max_samples,
            )
        elif dataset_name == "2wiki":
            from .datasets import TwoWikiMultiHopQADataset

            return TwoWikiMultiHopQADataset(
                data_path=self.config.dataset_path,
                split=self.config.dataset_split,
                max_samples=self.config.max_samples,
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    def _create_graphrag(self) -> GraphRAG:
        """Create GraphRAG instance from config."""
        rag_config = GraphRAGConfig.from_dict(self.config.graphrag_config)
        rag = GraphRAG.from_config(rag_config)

        # Wrap LLM functions with cache if enabled
        if self._cache is not None and self._cache.enabled:
            if rag.best_model_func is not None:
                rag.best_model_func = self._cache.wrap(rag.best_model_func)
            if rag.cheap_model_func is not None:
                rag.cheap_model_func = self._cache.wrap(rag.cheap_model_func)

        return rag

    def _create_metric_suite(self) -> MetricSuite:
        """Create metric suite from config."""
        from .metrics import ExactMatchMetric, TokenF1Metric

        suite = MetricSuite()

        for metric_name in self.config.metrics:
            if metric_name == "exact_match":
                suite.add_metric("exact_match", ExactMatchMetric())
            elif metric_name == "token_f1":
                suite.add_metric("token_f1", TokenF1Metric())
            else:
                raise ValueError(f"Unknown metric: {metric_name}")

        return suite

    async def run(self) -> ExperimentResult:
        """Execute full experiment.

        Process:
        1. Load dataset
        2. Insert corpus into GraphRAG
        3. Run queries for each mode
        4. Compute metrics
        5. Save results

        Returns:
            ExperimentResult with scores and predictions
        """
        start_time = datetime.now()
        timestamp = start_time.isoformat()

        # Load dataset
        self._dataset = self._load_dataset()
        questions = self._dataset.questions(split=self.config.dataset_split)
        corpus = self._dataset.corpus()

        print(f"[Dataset] Loaded {len(questions)} questions and {len(corpus)} corpus documents")

        # Create GraphRAG instance
        self._rag = self._create_graphrag()

        # Insert corpus
        print(f"[Index] Inserting {len(corpus)} documents into GraphRAG...")
        await self._rag.ainsert_documents({f"doc_{i}": doc for i, doc in enumerate(corpus)})
        print("[Index] Insertion complete")

        # Create metric suite
        self._metric_suite = self._create_metric_suite()

        # Run queries for each mode
        mode_results = {}
        all_predictions = {}

        for mode in self.config.query_modes:
            print(f"\n[Query] Running {mode} queries...")
            predictions = []
            golds = []

            for i, qa in enumerate(questions):
                question = qa["question"]
                gold = qa.get("answer", "")

                # Build query params
                query_param = QueryParam(mode=mode, **self.config.query_params)  # type: ignore[arg-type]

                # Run query
                prediction = await self._rag.aquery(question, query_param)

                predictions.append(prediction)
                golds.append(gold)

                if (i + 1) % 10 == 0:
                    print(f"  Processed {i + 1}/{len(questions)} queries")

            # Compute metrics
            scores = await self._metric_suite.compute_batch(predictions, golds)
            mode_results[mode] = scores

            # Store predictions
            all_predictions[mode] = [
                {"question": qa["question"], "prediction": pred, "gold": qa.get("answer", "")}
                for qa, pred in zip(questions, predictions)
            ]

            print(f"[Query] {mode} results: {scores}")

        # Compute duration
        end_time = datetime.now()
        duration_seconds = (end_time - start_time).total_seconds()

        # Get cache statistics if cache was used
        cache_stats = None
        if self._cache is not None:
            cache_stats = await self._cache.stats()
            print(f"\n[Cache] Hits: {cache_stats['hits']}, Misses: {cache_stats['misses']}, Hit Rate: {cache_stats['hit_rate']:.2%}")

        # Create result
        result = ExperimentResult(
            experiment_name=self.config.experiment_name,
            timestamp=timestamp,
            config=self.config,
            mode_results=mode_results,
            predictions=all_predictions,
            duration_seconds=duration_seconds,
            cache_stats=cache_stats,
        )

        # Save results
        output_path = result.save(self.config.output_dir)
        print(f"\n[Results] Saved to {output_path}")

        return result

    def run_sync(self) -> ExperimentResult:
        """Synchronous wrapper for run()."""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.run())
