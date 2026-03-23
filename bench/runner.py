"""Experiment runner for GraphRAG benchmarks."""

import asyncio
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from nano_graphrag.base import GraphRAGConfig, QueryParam
from nano_graphrag.graphrag import GraphRAG
from .cache import create_benchmark_cache
from .datasets import BenchmarkDataset, MultiHopRAGDataset
from .metrics import MetricSuite, get_baseline_suite


@dataclass
class BenchmarkConfig:
    """Experiment configuration (YAML-serializable)."""

    # === Top-level ===
    experiment_name: str = "experiment"
    version: str = "1.0"
    description: str = ""

    # === Dataset ===
    dataset_name: str = ""
    dataset_path: str = ""
    corpus_path: Optional[str] = None
    dataset_split: str = "test"
    max_samples: int = -1
    auto_download: bool = False

    # === GraphRAG config ===
    graphrag_config: Dict[str, Any] = field(default_factory=dict)

    # === Query modes ===
    query_modes: List[str] = field(default_factory=lambda: ["local", "global"])
    query_params: Dict[str, Any] = field(default_factory=dict)

    # === Metrics ===
    metrics: List[str] = field(default_factory=lambda: ["exact_match", "token_f1"])

    # === Output ===
    output_dir: str = "./benchmark_results"

    @classmethod
    def from_yaml(cls, path: str) -> "BenchmarkConfig":
        """Load config from YAML file.

        Supports both flat and nested schemas.
        """
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required to load YAML configs. Install with: uv add pyyaml"
            )

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        normalized = cls._normalize_config(data)
        return cls(**normalized)

    @classmethod
    def _normalize_config(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize nested config schema to flat structure.

        Handles both:
        - Flat schema (backward compatible)
        - Nested schema (roadmap compliant)
        """
        normalized = {}

        normalized["experiment_name"] = data.get("name", data.get("experiment_name", "experiment"))
        normalized["version"] = data.get("version", "1.0")
        normalized["description"] = data.get("description", "")

        if "dataset" in data:
            dataset = data["dataset"]
            normalized["dataset_name"] = dataset.get("name", "")
            normalized["dataset_path"] = dataset.get("path", "")
            normalized["corpus_path"] = dataset.get("corpus_path")
            normalized["dataset_split"] = dataset.get("split", "test")
            normalized["max_samples"] = dataset.get("max_samples", -1)
            normalized["auto_download"] = dataset.get("auto_download", False)
        else:
            normalized["dataset_name"] = data.get("dataset_name", "")
            normalized["dataset_path"] = data.get("dataset_path", "")
            normalized["corpus_path"] = data.get("corpus_path")
            normalized["dataset_split"] = data.get("dataset_split", "test")
            normalized["max_samples"] = data.get("max_samples", -1)
            normalized["auto_download"] = data.get("auto_download", False)

        if "graphrag" in data:
            normalized["graphrag_config"] = data["graphrag"]
        else:
            normalized["graphrag_config"] = data.get("graphrag_config", {})

        if "query" in data:
            query = data["query"]
            normalized["query_modes"] = query.get("modes", ["local", "global"])
            normalized["query_params"] = query.get("param_overrides", {})
        else:
            normalized["query_modes"] = data.get("query_modes", ["local", "global"])
            normalized["query_params"] = data.get("query_params", {})

        if "cache" in data:
            cache = data["cache"]
            if isinstance(cache, dict):
                normalized["graphrag_config"]["enable_llm_cache"] = cache.get("enabled", False)

        if "metrics" in data:
            metrics = data["metrics"]
            if isinstance(metrics, dict):
                metric_list = []
                if metrics.get("exact_match", False):
                    metric_list.append("exact_match")
                if metrics.get("token_f1", False):
                    metric_list.append("token_f1")
                if metrics.get("llm_judge", {}).get("enabled", False):
                    metric_list.append("faithfulness")
                    metric_list.append("answer_relevance")
                normalized["metrics"] = metric_list
            else:
                normalized["metrics"] = metrics
        else:
            normalized["metrics"] = data.get("metrics", ["exact_match", "token_f1"])

        if "output" in data:
            output = data["output"]
            normalized["output_dir"] = output.get("results_dir", "./benchmark_results")
        else:
            normalized["output_dir"] = data.get("output_dir", "./benchmark_results")

        return normalized

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "BenchmarkConfig":
        """Create config from dictionary."""
        normalized = cls._normalize_config(config)
        filtered = {k: v for k, v in normalized.items() if v is not None}
        return cls(**filtered)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary (nested schema)."""
        return {
            "name": self.experiment_name,
            "version": "1.0",
            "description": "",
            "dataset": {
                "name": self.dataset_name,
                "path": self.dataset_path,
                "corpus_path": self.corpus_path,
                "split": self.dataset_split,
                "max_samples": self.max_samples,
                "auto_download": self.auto_download,
            },
            "graphrag": self.graphrag_config,
            "query": {
                "modes": self.query_modes,
                "param_overrides": self.query_params,
            },
            "metrics": self.metrics,
            "output": {
                "results_dir": self.output_dir,
            },
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
            dataset = MultiHopRAGDataset(
                questions_path=self.config.dataset_path,
                corpus_path=self.config.corpus_path,
                max_samples=self.config.max_samples,
            )
        elif dataset_name == "hotpotqa":
            from .datasets import HotpotQADataset

            dataset = HotpotQADataset(
                data_path=self.config.dataset_path,
                split=self.config.dataset_split,
                max_samples=self.config.max_samples,
            )
        elif dataset_name == "musique":
            from .datasets import MuSiQueDataset

            dataset = MuSiQueDataset(
                data_path=self.config.dataset_path,
                split=self.config.dataset_split,
                max_samples=self.config.max_samples,
            )
        elif dataset_name == "2wiki":
            from .datasets import TwoWikiMultiHopQADataset

            dataset = TwoWikiMultiHopQADataset(
                data_path=self.config.dataset_path,
                split=self.config.dataset_split,
                max_samples=self.config.max_samples,
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        if self.config.auto_download:
            cache_dir = self.config.graphrag_config.get("working_dir", "./nano_graphrag")
            cache_dir = str(Path(cache_dir) / "datasets")
            print(f"[Download] Auto-download enabled, downloading to {cache_dir}...")
            dataset.download(cache_dir=cache_dir)

        return dataset

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
        questions_list = list(self._dataset.questions(split=self.config.dataset_split))
        corpus_list = list(self._dataset.corpus())

        print(f"[Dataset] Loaded {len(questions_list)} questions and {len(corpus_list)} corpus documents")

        # Create GraphRAG instance
        self._rag = self._create_graphrag()

        # Insert corpus
        print(f"[Index] Inserting {len(corpus_list)} documents into GraphRAG...")
        await self._rag.ainsert_documents({doc.id: doc.text for doc in corpus_list})
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

            for i, qa in enumerate(questions_list):
                question = qa.question
                gold = qa.answer

                # Build query params
                query_param = QueryParam(mode=mode, **self.config.query_params)  # type: ignore[arg-type]

                # Run query
                prediction = await self._rag.aquery(question, query_param)

                predictions.append(prediction)
                golds.append(gold)

                if (i + 1) % 10 == 0:
                    print(f"  Processed {i + 1}/{len(questions_list)} queries")

            # Compute metrics
            scores = await self._metric_suite.compute_batch(predictions, golds)
            mode_results[mode] = scores

            # Store predictions
            all_predictions[mode] = [
                {"question": qa.question, "prediction": pred, "gold": qa.answer}
                for qa, pred in zip(questions_list, predictions)
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
