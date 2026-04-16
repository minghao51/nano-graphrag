"""Cross-encoder reranker for retrieved passages.

Uses a cross-encoder model to re-score retrieved passages based on their
relevance to the query. This captures query-passage interactions that
bi-encoders (embedding similarity) miss.

Example:
    reranker = CrossEncoderReranker(model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_k=20)
    ranked = reranker("What is the capital of France?", ["Paris is in France.", "Berlin is in Germany."])
    # Returns: [("Paris is in France.", 0.95), ("Berlin is in Germany.", 0.12)]
"""

from __future__ import annotations

from typing import Any, Optional, Sequence


class CrossEncoderReranker:
    """Re-score retrieved passages using a cross-encoder model.

    Cross-encoders take both the query and passage as input and output
    a relevance score. This is more accurate than bi-encoder similarity
    but slower since it requires a model call per query-passage pair.

    Args:
        model: Model name or path. Defaults to "cross-encoder/ms-marco-MiniLM-L-6-v2".
            Other options:
            - "cross-encoder/ms-marco-electro-small-base" (~50MB, faster)
            - "cross-encoder/quora-roberta-base" (~270MB, Q&A focused)
        top_k: Number of top passages to return after reranking.
        device: Device to run model on ("cpu", "cuda", "mps"). Auto-detected by default.
        batch_size: Batch size for inference. Larger is faster but uses more memory.

    Attributes:
        model: Loaded cross-encoder model.
        top_k: Number of passages to return.
    """

    def __init__(
        self,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 20,
        device: Optional[str] = None,
        batch_size: int = 32,
    ) -> None:
        self._model_name = model
        self._top_k = top_k
        self._device = device
        self._batch_size = batch_size
        self._model: Any = None

    def __call__(self, query: str, passages: Sequence[str]) -> list[tuple[str, float]]:
        """Rerank passages by relevance to the query.

        Args:
            query: User question or search query.
            passages: List of retrieved passages to rerank.

        Returns:
            List of (passage, score) tuples sorted by score in descending order,
            truncated to top_k passages.

        Raises:
            ImportError: If sentence-transformers is not installed.
            RuntimeError: If model fails to load.
        """
        if self._model is None:
            self._load_model()

        if not passages:
            return []

        # Create query-passage pairs
        pairs = [[query, passage] for passage in passages]

        # Get scores from cross-encoder
        scores = self._model.predict(pairs, batch_size=self._batch_size)

        # Sort passages by score
        ranked = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)

        # Return top_k
        return ranked[: self._top_k]

    def _load_model(self) -> None:
        """Load the cross-encoder model.

        Raises:
            ImportError: If sentence-transformers is not installed.
        """
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for CrossEncoderReranker. "
                "Install with: uv add sentence-transformers"
            ) from exc

        device = self._device
        if device is None:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._model = CrossEncoder(
            self._model_name, device=device, default_activation_function=torch.nn.Sigmoid()
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "CrossEncoderReranker":
        """Create reranker from configuration dict.

        Args:
            config: Configuration dict with keys:
                - model (str): Model name
                - top_k (int): Number of top passages
                - device (str | None): Device to use
                - batch_size (int): Batch size

        Returns:
            Configured CrossEncoderReranker instance.
        """
        return cls(
            model=config.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            top_k=config.get("top_k", 20),
            device=config.get("device"),
            batch_size=config.get("batch_size", 32),
        )
