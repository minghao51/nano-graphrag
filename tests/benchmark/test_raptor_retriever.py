"""Tests for RAPTOR Hierarchical Summaries."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bench.techniques.raptor import RaptorNode, RaptorRetriever


class TestRaptorNode:
    """Test RaptorNode dataclass."""

    def test_init(self):
        """Test RaptorNode initialization."""
        node = RaptorNode(content="Test content", level=0)

        assert node.content == "Test content"
        assert node.level == 0
        assert node.children is None
        assert node.parent is None
        assert node.embedding is None
        assert node.node_id == ""

    def test_init_with_optional_fields(self):
        """Test RaptorNode with optional fields."""
        node = RaptorNode(
            content="Test",
            level=1,
            node_id="test_1",
            embedding=[0.1, 0.2, 0.3],
        )

        assert node.node_id == "test_1"
        assert node.embedding == [0.1, 0.2, 0.3]


class TestRaptorRetriever:
    """Test RaptorRetriever functionality."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        retriever = RaptorRetriever()

        assert retriever._max_levels == 3
        assert retriever._cluster_model == "gmm"
        assert retriever._summary_model == "cheap"
        assert retriever._chunk_size == 200
        assert retriever._top_k == 10
        assert retriever._root is None

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        retriever = RaptorRetriever(
            max_levels=5,
            cluster_model="kmeans",
            summary_model="best",
            chunk_size=300,
            top_k=20,
        )

        assert retriever._max_levels == 5
        assert retriever._cluster_model == "kmeans"
        assert retriever._summary_model == "best"
        assert retriever._chunk_size == 300
        assert retriever._top_k == 20

    def test_init_with_invalid_cluster_model(self):
        """Test that invalid cluster model raises error."""
        with pytest.raises(ValueError, match="Unknown cluster model"):
            RaptorRetriever(cluster_model="invalid")

    @pytest.mark.asyncio
    async def test_call_with_no_tree(self):
        """Test retrieval when tree hasn't been built."""
        retriever = RaptorRetriever()

        result = await retriever("test query", MagicMock(), MagicMock())

        assert result == ""

    @pytest.mark.asyncio
    async def test_summarize_cluster(self):
        """Test cluster summarization."""
        retriever = RaptorRetriever()

        nodes = [
            RaptorNode(content="Content 1", level=0),
            RaptorNode(content="Content 2", level=0),
        ]

        mock_graph_rag = MagicMock()
        mock_graph_rag.cheap_model_func = AsyncMock(return_value="Summary content")

        summary_node = await retriever._summarize_cluster(nodes, mock_graph_rag)

        assert summary_node.content == "Summary content"
        assert summary_node.level == 0

    @pytest.mark.asyncio
    async def test_summarize_cluster_with_best_model(self):
        """Test cluster summarization with best model."""
        retriever = RaptorRetriever(summary_model="best")

        nodes = [RaptorNode(content="Content", level=0)]

        mock_graph_rag = MagicMock()
        mock_graph_rag.best_model_func = AsyncMock(return_value="Best summary")
        mock_graph_rag.cheap_model_func = AsyncMock(return_value="Cheap summary")

        summary_node = await retriever._summarize_cluster(nodes, mock_graph_rag)

        assert summary_node.content == "Best summary"

    @pytest.mark.asyncio
    async def test_collect_nodes(self):
        """Test collecting all nodes from tree."""
        retriever = RaptorRetriever()

        # Create a simple tree structure
        leaf1 = RaptorNode(content="Leaf 1", level=0)
        leaf2 = RaptorNode(content="Leaf 2", level=0)
        parent = RaptorNode(content="Parent", level=1, children=[leaf1, leaf2])
        leaf1.parent = parent
        leaf2.parent = parent

        nodes = retriever._collect_nodes(parent)

        assert len(nodes) == 3
        assert parent in nodes
        assert leaf1 in nodes
        assert leaf2 in nodes

    def test_collect_nodes_with_none(self):
        """Test collecting nodes from None root."""
        retriever = RaptorRetriever()

        nodes = retriever._collect_nodes(None)

        assert nodes == []

    def test_from_config(self):
        """Test creating retriever from configuration dict."""
        config = {
            "max_levels": 5,
            "cluster_model": "kmeans",
            "summary_model": "best",
            "chunk_size": 300,
            "top_k": 20,
        }

        retriever = RaptorRetriever.from_config(config)

        assert retriever._max_levels == 5
        assert retriever._cluster_model == "kmeans"
        assert retriever._summary_model == "best"
        assert retriever._chunk_size == 300
        assert retriever._top_k == 20

    def test_from_config_with_defaults(self):
        """Test creating retriever from configuration dict with defaults."""
        config = {"top_k": 15}

        retriever = RaptorRetriever.from_config(config)

        assert retriever._max_levels == 3
        assert retriever._cluster_model == "gmm"
        assert retriever._summary_model == "cheap"
        assert retriever._chunk_size == 200
        assert retriever._top_k == 15
