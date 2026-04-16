"""Tests for edge confidence weighting."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import networkx as nx
import pytest

from bench.techniques.edge_confidence import (
    create_edge_confidence_hook,
    get_edge_weight,
    score_edges_by_confidence,
)


class TestEdgeConfidence:
    """Test edge confidence weighting functionality."""

    @pytest.fixture
    def mock_graph_storage(self):
        """Create a mock graph storage for testing."""
        storage = MagicMock()

        # Create a NetworkX graph with test edges
        graph = nx.Graph()
        graph.add_edge("A", "B", confidence="0.9", occurrence_count="5")
        graph.add_edge("B", "C", confidence="0.5", occurrence_count="2")
        graph.add_edge("C", "D", occurrence_count="1")  # No confidence
        graph.add_edge("D", "E")  # No confidence, no count

        # Mock the upsert_edge method
        storage._graph = graph
        storage.upsert_edge = AsyncMock()

        return storage

    @pytest.mark.asyncio
    async def test_score_edges_with_confidence(self, mock_graph_storage):
        """Test edge scoring with confidence values."""
        await score_edges_by_confidence(mock_graph_storage)

        # Verify that upsert_edge was called for each edge
        assert mock_graph_storage.upsert_edge.call_count == 4

        # Check the first edge (high confidence, high frequency)
        call_args = mock_graph_storage.upsert_edge.call_args_list[0]
        src, dst, data = call_args[0]
        weight = float(data["weight"])
        # 0.7 * 0.9 + 0.3 * (5/10) = 0.63 + 0.15 = 0.78
        assert 0.77 < weight < 0.79

    @pytest.mark.asyncio
    async def test_score_edges_without_confidence(self, mock_graph_storage):
        """Test edge scoring when confidence is not provided."""
        await score_edges_by_confidence(mock_graph_storage)

        # Check edge C-D (no confidence, low frequency)
        call_args = mock_graph_storage.upsert_edge.call_args_list[2]
        src, dst, data = call_args[0]
        weight = float(data["weight"])
        # 0.7 * 1.0 + 0.3 * (1/10) = 0.7 + 0.03 = 0.73
        assert 0.72 < weight < 0.74

    @pytest.mark.asyncio
    async def test_score_edges_custom_weights(self, mock_graph_storage):
        """Test edge scoring with custom weights."""
        await score_edges_by_confidence(
            mock_graph_storage,
            confidence_weight=0.5,
            frequency_weight=0.5,
            max_frequency_cap=5,
        )

        # Check the first edge with custom weights
        call_args = mock_graph_storage.upsert_edge.call_args_list[0]
        src, dst, data = call_args[0]
        weight = float(data["weight"])
        # 0.5 * 0.9 + 0.5 * (5/5) = 0.45 + 0.5 = 0.95
        assert 0.94 < weight < 0.96

    @pytest.mark.asyncio
    async def test_get_edge_weight(self, mock_graph_storage):
        """Test getting edge weight."""
        # First score the edges
        await score_edges_by_confidence(mock_graph_storage)

        # Mock the get_edge method to return the updated edge data
        async def mock_get_edge(src, dst):
            for call in mock_graph_storage.upsert_edge.call_args_list:
                if call[0][0] == src and call[0][1] == dst:
                    return call[0][2]
            return None

        mock_graph_storage.get_edge = mock_get_edge

        # Get the weight
        weight = await get_edge_weight(mock_graph_storage, "A", "B")
        assert weight is not None
        assert 0.7 < weight < 0.8

    @pytest.mark.asyncio
    async def test_get_edge_weight_nonexistent(self, mock_graph_storage):
        """Test getting weight for non-existent edge."""
        mock_graph_storage.get_edge = AsyncMock(return_value=None)

        weight = await get_edge_weight(mock_graph_storage, "X", "Y")
        assert weight is None

    def test_create_edge_confidence_hook(self):
        """Test creating an edge confidence hook."""
        hook = create_edge_confidence_hook(
            confidence_weight=0.8,
            frequency_weight=0.2,
            max_frequency_cap=20,
        )

        # Verify the hook is callable
        assert callable(hook)

    @pytest.mark.asyncio
    async def test_edge_confidence_hook(self, mock_graph_storage):
        """Test using the created hook."""
        hook = create_edge_confidence_hook()

        await hook(mock_graph_storage)

        # Verify that edges were scored
        assert mock_graph_storage.upsert_edge.call_count == 4

    @pytest.mark.asyncio
    async def test_score_edges_frequency_cap(self, mock_graph_storage):
        """Test that frequency is properly capped."""
        # Add an edge with very high frequency
        mock_graph_storage._graph.add_edge("E", "F", confidence="0.8", occurrence_count="100")

        await score_edges_by_confidence(mock_graph_storage, max_frequency_cap=10)

        # Find the call for edge E-F
        for call in mock_graph_storage.upsert_edge.call_args_list:
            if call[0][0] == "E" and call[0][1] == "F":
                weight = float(call[0][2]["weight"])
                # 0.7 * 0.8 + 0.3 * (100/100) = 0.56 + 0.3 = 0.86
                # (capped at max_frequency_cap, so 100/10 = 10, but min(10, 1.0) = 1.0)
                # Actually: 0.7 * 0.8 + 0.3 * min(100/10, 1.0) = 0.56 + 0.3 = 0.86
                assert 0.85 < weight < 0.87
                break
        else:
            pytest.fail("Edge E-F not found in calls")
