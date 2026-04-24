"""Adaptive router evaluation: routing accuracy + downstream F1 per route choice.

Usage:
    python -m bench.techniques.adaptive_router_eval --config bench/configs/musique_full.yaml
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any

from bench.techniques.adaptive_router import AdaptiveRouter

ROUTE_LABELS: dict[str, list[str]] = {
    "multihop": [
        r"\bwho.*also\b",
        r"\bboth.*and\b",
        r"\bconnect",
        r"\bbetween\b",
    ],
    "global": [
        r"\bthemes?\b",
        r"\boverall\b",
        r"\bsummariz",
        r"\bmain ideas?\b",
    ],
}


def _label_question(question: str) -> str:
    for label, patterns in ROUTE_LABELS.items():
        for pat in patterns:
            if re.search(pat, question, re.IGNORECASE):
                return label
    return "local"


@dataclass
class RoutingEvaluationResult:
    total_questions: int = 0
    correct_routes: int = 0
    route_distribution: dict[str, int] = field(default_factory=dict)
    label_distribution: dict[str, int] = field(default_factory=dict)
    per_route_accuracy: dict[str, dict[str, int]] = field(default_factory=dict)
    routing_overhead_ms: float = 0.0


def evaluate_routing_accuracy(
    questions: list[str],
    router: AdaptiveRouter | None = None,
) -> RoutingEvaluationResult:
    if router is None:
        router = AdaptiveRouter(use_llm_fallback=False)

    result = RoutingEvaluationResult(total_questions=len(questions))

    for question in questions:
        predicted = router.route(question)
        ground_truth = _label_question(question)

        result.route_distribution[predicted] = result.route_distribution.get(predicted, 0) + 1
        result.label_distribution[ground_truth] = result.label_distribution.get(ground_truth, 0) + 1

        if predicted == ground_truth:
            result.correct_routes += 1

        if ground_truth not in result.per_route_accuracy:
            result.per_route_accuracy[ground_truth] = {"correct": 0, "total": 0}
        result.per_route_accuracy[ground_truth]["total"] += 1
        if predicted == ground_truth:
            result.per_route_accuracy[ground_truth]["correct"] += 1

    return result


async def run_adaptive_eval(
    config_path: str | None = None,
    questions: list[str] | None = None,
) -> dict[str, Any]:
    if questions is None:
        if config_path is None:
            raise ValueError("Either config_path or questions must be provided")
        from bench.runner import BenchmarkConfig

        config = BenchmarkConfig.from_yaml(config_path)
        from bench.runner import ExperimentRunner

        runner = ExperimentRunner(config)
        dataset = runner._load_dataset()
        qa_pairs = list(dataset.questions(split=config.dataset_split))
        questions = [qa.question for qa in qa_pairs]

    router = AdaptiveRouter(use_llm_fallback=False)
    result = evaluate_routing_accuracy(questions, router)

    accuracy = result.correct_routes / result.total_questions if result.total_questions > 0 else 0.0

    output = {
        "total_questions": result.total_questions,
        "overall_routing_accuracy": accuracy,
        "route_distribution": result.route_distribution,
        "label_distribution": result.label_distribution,
        "per_route_accuracy": {
            label: {
                "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0,
                "correct": stats["correct"],
                "total": stats["total"],
            }
            for label, stats in result.per_route_accuracy.items()
        },
    }

    print(json.dumps(output, indent=2))
    return output


if __name__ == "__main__":
    import sys

    config = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(run_adaptive_eval(config_path=config))
