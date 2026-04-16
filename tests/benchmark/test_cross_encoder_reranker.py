"""Tests for CrossEncoderReranker."""

import sys
from unittest.mock import MagicMock, patch

import pytest

# Mock the imports before importing the module
sys.modules["sentence_transformers"] = MagicMock()
sys.modules["torch"] = MagicMock()


class TestCrossEncoderReranker:
    """Test CrossEncoderReranker functionality."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        from bench.techniques.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()

        assert reranker._model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert reranker._top_k == 20
        assert reranker._batch_size == 32
        assert reranker._model is None

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        from bench.techniques.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(
            model="cross-encoder/ms-marco-electro-small-base",
            top_k=10,
            device="cpu",
            batch_size=16,
        )

        assert reranker._model_name == "cross-encoder/ms-marco-electro-small-base"
        assert reranker._top_k == 10
        assert reranker._device == "cpu"
        assert reranker._batch_size == 16

    def test_call_with_empty_passages(self):
        """Test reranking with empty passage list."""
        from bench.techniques.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()
        result = reranker("test query", [])

        assert result == []

    @patch("bench.techniques.reranker.CrossEncoderReranker._load_model")
    def test_call_with_mock_model(self, mock_load):
        """Test reranking with a mock model."""
        from bench.techniques.reranker import CrossEncoderReranker

        # Create a mock model that returns fixed scores
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.95, 0.12, 0.78]
        mock_load.return_value = None

        reranker = CrossEncoderReranker(top_k=2)
        reranker._model = mock_model

        passages = [
            "Paris is the capital of France.",
            "Berlin is in Germany.",
            "London is in the UK.",
        ]
        result = reranker("What is the capital of France?", passages)

        # Should return top 2 by score
        assert len(result) == 2
        assert result[0][0] == "Paris is the capital of France."
        assert result[0][1] == 0.95
        assert result[1][0] == "London is in the UK."
        assert result[1][1] == 0.78

    def test_from_config(self):
        """Test creating reranker from configuration dict."""
        from bench.techniques.reranker import CrossEncoderReranker

        config = {
            "model": "cross-encoder/ms-marco-electro-small-base",
            "top_k": 15,
            "device": "cuda",
            "batch_size": 64,
        }

        reranker = CrossEncoderReranker.from_config(config)

        assert reranker._model_name == "cross-encoder/ms-marco-electro-small-base"
        assert reranker._top_k == 15
        assert reranker._device == "cuda"
        assert reranker._batch_size == 64

    def test_from_config_with_defaults(self):
        """Test creating reranker from configuration dict with defaults."""
        from bench.techniques.reranker import CrossEncoderReranker

        config = {"top_k": 10}

        reranker = CrossEncoderReranker.from_config(config)

        assert reranker._model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert reranker._top_k == 10

    @patch("torch.cuda.is_available", return_value=False)
    def test_load_model_with_mocked_dependency(self, mock_cuda):
        """Test that model loading works when sentence-transformers is available."""
        from bench.techniques.reranker import CrossEncoderReranker

        # Mock the sentence_transformers module
        mock_encoder_instance = MagicMock()

        # Patch the import inside the _load_model method
        with patch.dict(sys.modules, {"sentence_transformers": MagicMock(CrossEncoder=MagicMock(return_value=mock_encoder_instance))}):
            reranker = CrossEncoderReranker()
            reranker._load_model()

            # Verify model was loaded
            assert reranker._model == mock_encoder_instance
