"""Metrics for evaluating RAG responses."""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from ..datasets import QAPair


class Metric(ABC):
    """Abstract base class for evaluation metrics."""

    @abstractmethod
    async def compute(
        self,
        prediction: str,
        gold: Union[str, QAPair],
        question: str = "",
        context: str = "",
    ) -> float:
        """Compute the metric score.

        Args:
            prediction: The predicted answer
            gold: The ground truth answer (string or QAPair object)
            question: The original question (optional)
            context: The retrieved context (optional)

        Returns:
            Score between 0.0 and 1.0
        """
        ...


def normalize_answer(s: str) -> str:
    """Normalize answer string for comparison.

    Removes articles, punctuation, and extra whitespace.
    """
    # Remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # Remove punctuation
    s = re.sub(r"[^\w\s]", " ", s)
    # Normalize whitespace
    s = " ".join(s.split())
    return s.lower().strip()


@dataclass
class ExactMatchMetric(Metric):
    """Normalized exact match metric.

    Compares normalized strings (case-insensitive, articles removed).
    """

    case_sensitive: bool = False
    remove_articles: bool = True

    async def compute(
        self,
        prediction: str,
        gold: Union[str, QAPair],
        question: str = "",
        context: str = "",
    ) -> float:
        """Compute exact match score."""
        # Extract gold answer string
        if isinstance(gold, QAPair):
            gold_str = gold.answer
        else:
            gold_str = gold

        if not self.case_sensitive:
            prediction = prediction.lower()
            gold_str = gold_str.lower()

        if self.remove_articles:
            prediction = normalize_answer(prediction)
            gold_str = normalize_answer(gold_str)
        else:
            prediction = prediction.strip()
            gold_str = gold_str.strip()

        return 1.0 if prediction == gold_str else 0.0


@dataclass
class TokenF1Metric(Metric):
    """Token-level F1 score metric.

    Computes F1 score based on token overlap between prediction and gold.
    """

    async def compute(
        self,
        prediction: str,
        gold: Union[str, QAPair],
        question: str = "",
        context: str = "",
    ) -> float:
        """Compute token F1 score."""
        # Extract gold answer string
        if isinstance(gold, QAPair):
            gold_str = gold.answer
        else:
            gold_str = gold

        pred_tokens = normalize_answer(prediction).split()
        gold_tokens = normalize_answer(gold_str).split()

        if not gold_tokens:
            # If gold is empty, score is 1.0 if prediction is also empty
            return 1.0 if not pred_tokens else 0.0

        if not pred_tokens:
            return 0.0

        # Compute token overlap
        gold_token_set = set(gold_tokens)
        pred_token_set = set(pred_tokens)

        common = gold_token_set & pred_token_set

        precision = len(common) / len(pred_token_set)
        recall = len(common) / len(gold_token_set)

        if precision + recall == 0:
            return 0.0

        f1 = 2 * precision * recall / (precision + recall)
        return f1


@dataclass
class MetricSuite:
    """Collection of metrics with batch computation support."""

    metrics: Dict[str, Metric] = field(default_factory=dict)

    async def compute(
        self,
        prediction: str,
        gold: Union[str, QAPair],
        question: str = "",
        context: str = "",
    ) -> Dict[str, float]:
        """Compute all metrics for a single prediction."""
        results = {}
        for name, metric in self.metrics.items():
            score = await metric.compute(
                prediction=prediction,
                gold=gold,
                question=question,
                context=context,
            )
            results[name] = score
        return results

    async def compute_batch(
        self,
        predictions: List[str],
        golds: List[Union[str, QAPair]],
        questions: Optional[List[str]] = None,
        contexts: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Compute all metrics for a batch of predictions.

        Returns average scores for each metric.
        """
        if len(predictions) != len(golds):
            raise ValueError(
                f"Length mismatch: {len(predictions)} predictions vs {len(golds)} golds"
            )

        if questions is not None and len(questions) != len(predictions):
            raise ValueError(
                f"Length mismatch: {len(questions)} questions vs {len(predictions)} predictions"
            )

        if contexts is not None and len(contexts) != len(predictions):
            raise ValueError(
                f"Length mismatch: {len(contexts)} contexts vs {len(predictions)} predictions"
            )

        # Initialize accumulators
        metric_sums = {name: 0.0 for name in self.metrics}
        count = len(predictions)

        # Compute metrics for each prediction
        for i, (pred, gold) in enumerate(zip(predictions, golds)):
            question = questions[i] if questions else ""
            context = contexts[i] if contexts else ""

            for name, metric in self.metrics.items():
                score = await metric.compute(
                    prediction=pred,
                    gold=gold,
                    question=question,
                    context=context,
                )
                metric_sums[name] += score

        # Compute averages
        return {name: total / count for name, total in metric_sums.items()}

    def add_metric(self, name: str, metric: Metric) -> None:
        """Add a metric to the suite."""
        self.metrics[name] = metric

    def remove_metric(self, name: str) -> None:
        """Remove a metric from the suite."""
        self.metrics.pop(name, None)


def get_baseline_suite() -> MetricSuite:
    """Get the baseline metric suite with EM and F1."""
    return MetricSuite(
        metrics={
            "exact_match": ExactMatchMetric(),
            "token_f1": TokenF1Metric(),
        }
    )


# Optional Ragas integration (requires ragas package)
@dataclass
class NativeContextRecallMetric(Metric):
    """Native context recall metric without Ragas dependency.

    Measures what fraction of gold supporting facts appear in the retrieved context.
    """

    async def compute(
        self,
        prediction: str,
        gold: Union[str, QAPair],
        question: str = "",
        context: str = "",
    ) -> float:
        """Compute context recall score."""
        supporting_facts = getattr(gold, "supporting_facts", [])

        if not supporting_facts:
            return 1.0

        if not context:
            return 0.0

        context_lower = context.lower()
        found_count = sum(
            1 for fact in supporting_facts
            if fact.lower() in context_lower
        )

        return found_count / len(supporting_facts)


@dataclass
class RagasFaithfulnessMetric(Metric):
    """LLM-as-judge faithfulness metric using Ragas.

    Requires: ragas package (install with `uv add ragas`)
    """

    async def compute(
        self,
        prediction: str,
        gold: Union[str, QAPair],
        question: str = "",
        context: str = "",
    ) -> float:
        """Compute faithfulness using Ragas."""
        try:
            from ragas import evaluate
            from ragas.metrics import faithfulness
            from datasets import Dataset
        except ImportError:
            raise ImportError(
                "Ragas is required for faithfulness metric. "
                "Install with: uv add ragas"
            )

        # Extract gold answer string if needed
        if isinstance(gold, QAPair):
            gold_str = gold.answer
            question = gold.question  # Use QAPair question if not provided
        else:
            gold_str = gold

        # Create dataset for Ragas
        data = {
            "question": [question],
            "answer": [prediction],
            "contexts": [[context] if context else []],
        }
        dataset = Dataset.from_dict(data)

        # Run evaluation
        result = evaluate(dataset, metrics=[faithfulness])

        # Extract score
        return result.to_pandas()["faithfulness"].iloc[0]


@dataclass
class RagasAnswerRelevanceMetric(Metric):
    """LLM-as-judge answer relevance metric using Ragas.

    Requires: ragas package (install with `uv add ragas`)
    """

    async def compute(
        self,
        prediction: str,
        gold: Union[str, QAPair],
        question: str = "",
        context: str = "",
    ) -> float:
        """Compute answer relevance using Ragas."""
        try:
            from ragas import evaluate
            from ragas.metrics import answer_relevance
            from datasets import Dataset
        except ImportError:
            raise ImportError(
                "Ragas is required for answer relevance metric. "
                "Install with: uv add ragas"
            )

        # Extract gold answer string if needed
        if isinstance(gold, QAPair):
            gold_str = gold.answer
            question = gold.question  # Use QAPair question if not provided
        else:
            gold_str = gold

        # Create dataset for Ragas
        data = {
            "question": [question],
            "answer": [prediction],
            "contexts": [[context] if context else []],
        }
        dataset = Dataset.from_dict(data)

        # Run evaluation
        result = evaluate(dataset, metrics=[answer_relevance])

        # Extract score
        return result.to_pandas()["answer_relevance"].iloc[0]


def get_ragas_suite() -> MetricSuite:
    """Get metric suite with Ragas LLM-as-judge metrics.

    Requires: ragas package
    """
    return MetricSuite(
        {
            "exact_match": ExactMatchMetric(),
            "token_f1": TokenF1Metric(),
            "faithfulness": RagasFaithfulnessMetric(),
            "answer_relevance": RagasAnswerRelevanceMetric(),
        }
    )
