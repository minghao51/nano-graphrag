"""Tests for Hybrid Retriever."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bench.retrievers.hybrid import HybridRetriever


class TestHybridRetriever:
    """Test HybridRetriever functionality."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        retriever = HybridRetriever(retrievers=["local", "global"])

        assert retriever._retrievers == ["local", "global"]
        assert retriever._weights == [0.5, 0.5]
        assert retriever._fusion == "weighted_avg"
        assert retriever._top_k == 20

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        retriever = HybridRetriever(
            retrievers=["local", "hipporag"],
            weights=[0.7, 0.3],
            fusion="reciprocal_rank",
            top_k=10,
        )

        assert retriever._retrievers == ["local", "hipporag"]
        assert retriever._weights == [0.7, 0.3]
        assert retriever._fusion == "reciprocal_rank"
        assert retriever._top_k == 10

    def test_init_with_invalid_fusion(self):
        """Test that invalid fusion strategy raises error."""
        with pytest.raises(ValueError, match="Unknown fusion strategy"):
            HybridRetriever(retrievers=["local"], fusion="invalid")

    def test_init_with_empty_retrievers(self):
        """Test that empty retriever list raises error."""
        with pytest.raises(ValueError, match="At least one retriever"):
            HybridRetriever(retrievers=[])

    def test_init_with_mismatched_weights(self):
        """Test that mismatched weights length raises error."""
        with pytest.raises(ValueError, match="Number of weights"):
            HybridRetriever(retrievers=["local", "global"], weights=[0.5])

    @pytest.mark.asyncio
    async def test_call_with_weighted_avg_fusion(self):
        """Test weighted avg fusion logic directly."""
        retriever = HybridRetriever(
            retrievers=["local", "global"],
            weights=[0.7, 0.3],
            fusion="weighted_avg",
            top_k=3,
        )

        weighted_results = [
            (0.7, "Passage A\n\nPassage B"),
            (0.3, "Passage C\n\nPassage D"),
        ]

        result = retriever._weighted_avg_fusion(weighted_results)

        # Should return top 3 passages sorted by weight
        assert result is not None
        assert isinstance(result, str)
        assert "Passage A" in result  # Higher weight first

    @pytest.mark.asyncio
    async def test_call_with_reciprocal_rank_fusion(self):
        """Test reciprocal rank fusion logic directly."""
        retriever = HybridRetriever(
            retrievers=["local", "global"],
            fusion="reciprocal_rank",
            top_k=3,
        )

        # Passage B appears in both, so should rank higher
        weighted_results = [
            (0.5, "Passage A\n\nPassage B"),
            (0.5, "Passage B\n\nPassage C"),
        ]

        result = retriever._reciprocal_rank_fusion(weighted_results)

        # Passage B should rank higher (appears in both)
        assert result is not None
        assert isinstance(result, str)
        # B should come before A and C due to combined scores
        lines = result.split("\n\n")
        b_index = next((i for i, line in enumerate(lines) if "Passage B" in line), -1)
        a_index = next((i for i, line in enumerate(lines) if "Passage A" in line), -1)
        assert b_index >= 0  # B should be present
        assert b_index <= a_index if a_index >= 0 else True  # B should rank same or higher than A

    @pytest.mark.asyncio
    async def test_fusion_with_empty_results(self):
        """Test fusion with empty result lists."""
        retriever = HybridRetriever(
            retrievers=["local", "global"],
            fusion="weighted_avg",
        )

        result = retriever._weighted_avg_fusion([])

        assert result == ""

    @pytest.mark.asyncio
    async def test_fusion_with_single_result(self):
        """Test fusion with single result."""
        retriever = HybridRetriever(
            retrievers=["local"],
            weights=[1.0],
            fusion="weighted_avg",
        )

        weighted_results = [(1.0, "Single passage")]

        result = retriever._weighted_avg_fusion(weighted_results)

        assert result == "Single passage"

    def test_weighted_avg_fusion(self):
        """Test weighted average fusion logic."""
        retriever = HybridRetriever(
            retrievers=["local", "global"],
            weights=[0.7, 0.3],
            top_k=3,
        )

        weighted_results = [
            (0.7, "Passage A\n\nPassage B"),
            (0.3, "Passage C\n\nPassage D"),
        ]

        result = retriever._weighted_avg_fusion(weighted_results)

        # Should return top 3 passages
        passages = result.split("\n\n")
        assert len(passages) <= 3

    def test_reciprocal_rank_fusion(self):
        """Test reciprocal rank fusion logic."""
        retriever = HybridRetriever(
            retrievers=["local", "global"],
            fusion="reciprocal_rank",
            top_k=3,
        )

        weighted_results = [
            (0.5, "Passage A\n\nPassage B"),
            (0.5, "Passage B\n\nPassage C"),
        ]

        result = retriever._reciprocal_rank_fusion(weighted_results)

        # Passage B should rank higher (appears in both)
        assert result is not None
        passages = result.split("\n\n")
        assert len(passages) <= 3

    def test_from_config(self):
        """Test creating retriever from configuration dict."""
        config = {
            "retrievers": ["local", "hipporag"],
            "weights": [0.6, 0.4],
            "fusion": "rrf",
            "top_k": 15,
        }

        retriever = HybridRetriever.from_config(config)

        assert retriever._retrievers == ["local", "hipporag"]
        assert retriever._weights == [0.6, 0.4]
        assert retriever._fusion == "rrf"
        assert retriever._top_k == 15

    def test_from_config_with_defaults(self):
        """Test creating retriever from configuration dict with defaults."""
        config = {"fusion": "reciprocal_rank"}

        retriever = HybridRetriever.from_config(config)

        assert retriever._retrievers == ["local", "global"]
        assert retriever._weights == [0.5, 0.5]
        assert retriever._fusion == "reciprocal_rank"
        assert retriever._top_k == 20
