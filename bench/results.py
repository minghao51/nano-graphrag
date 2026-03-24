"""Results storage backends for benchmark experiments."""

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class PredictionRecord:
    """Single prediction result."""

    question_id: str
    question: str
    gold_answer: str
    prediction: str
    metrics: Dict[str, float]
    latency_seconds: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    """Complete run result for storage."""

    run_id: str
    experiment_name: str
    timestamp: str
    config: Dict[str, Any]
    variant_label: str
    mode_results: Dict[str, Dict[str, float]]
    predictions: List[PredictionRecord]
    aggregate_metrics: Dict[str, Dict[str, float]]
    cache_stats: Optional[Dict[str, Any]]
    timing: Dict[str, float]
    duration_seconds: float

    def to_markdown_table(self) -> str:
        """Format as markdown table."""
        lines = [f"## {self.experiment_name} ({self.variant_label})"]
        lines.append(f"**Timestamp:** {self.timestamp}")
        lines.append("")
        lines.append("### Aggregate Metrics")
        lines.append("")
        lines.append("| Mode | Metric | Score |")
        lines.append("|------|--------|-------|")
        for mode, metrics in self.mode_results.items():
            for metric, score in metrics.items():
                lines.append(f"| {mode} | {metric} | {score:.3f} |")
        return "\n".join(lines)


class ResultsBackend(ABC):
    """Protocol for results storage backends."""

    @abstractmethod
    async def save(self, result: RunResult) -> str:
        """Save result and return path/URI."""
        ...

    @abstractmethod
    async def load(self, run_id: str) -> Optional[RunResult]:
        """Load result by run_id."""
        ...

    @abstractmethod
    async def list_runs(self, experiment_name: Optional[str] = None) -> List[str]:
        """List all run_ids, optionally filtered by experiment."""
        ...


class JSONResultsBackend(ResultsBackend):
    """JSON file-based results storage."""

    def __init__(self, results_dir: str = "./benchmark_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    async def save(self, result: RunResult) -> str:
        output_path = self.results_dir / f"{result.run_id}.json"
        result_dict = asdict(result)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        return str(output_path)

    async def load(self, run_id: str) -> Optional[RunResult]:
        filepath = self.results_dir / f"{run_id}.json"
        if not filepath.exists():
            return None
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["predictions"] = [
            PredictionRecord(**prediction) for prediction in data.get("predictions", [])
        ]
        return RunResult(**data)

    async def list_runs(self, experiment_name: Optional[str] = None) -> List[str]:
        runs = []
        for f in self.results_dir.glob("*.json"):
            if experiment_name:
                with open(f) as fp:
                    data = json.load(fp)
                    if data.get("experiment_name") != experiment_name:
                        continue
            runs.append(f.stem)
        return sorted(runs)
