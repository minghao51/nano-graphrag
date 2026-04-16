"""Tests for HippoRAG PPR Retriever."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import networkx as nx
import pytest

from bench.retrievers.hipporag_ppr import HippoRAGRetriever


class TestHippoRAGRetriever:
    """Test HippoRAGRetriever functionality."""

    @pytest.fixture
    def mock_graph_rag(self):
        """Create a mock GraphRAG instance for testing."""
        graph_rag = MagicMock()

        # Create a NetworkX graph with test edges
        graph = nx.Graph()
        graph.add_edge("Entity1", "Entity2", weight="0.8")
        graph.add_edge("Entity2", "Entity3", weight="0.6")
        graph.add_edge("Entity3", "Entity4", weight="0.9")
        graph.add_node("Entity1", description="First entity", source_id="doc1", chunk_id="chunk1")
        graph.add_node("Entity2", description="Second entity", source_id="doc1")
        graph.add_node("Entity3", description="Third entity", source_id="doc2")
        graph.add_node("Entity4", description="Fourth entity", source_id="doc2")

        # Mock the graph storage
        graph_storage = MagicMock()
        graph_storage._graph = graph
        graph_storage.get_node = AsyncMock(side_effect=lambda node: {
            "description": graph.nodes[node].get("description", node),
            "source_id": graph.nodes[node].get("source_id", ""),
            "chunk_id": graph.nodes[node].get("chunk_id", ""),
        })
        graph_rag.chunk_entity_relation_graph = graph_storage

        # Mock the vector DB (empty for now)
        graph_rag.entities_vdb = None

        # Mock the text chunks storage
        graph_rag.text_chunks = MagicMock()
        graph_rag.text_chunks.get_by_id = AsyncMock(return_value={
            "content": "Sample chunk content for testing purposes."
        })

        # Mock the aquery method for fallback
        graph_rag.aquery = AsyncMock(return_value="Fallback answer")

        return graph_rag

    @pytest.mark.asyncio
    async def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        retriever = HippoRAGRetriever()

        assert retriever._alpha == 0.85
        assert retriever._top_k_seed == 5
        assert retriever._top_k_result == 20

    @pytest.mark.asyncio
    async def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        retriever = HippoRAGRetriever(alpha=0.9, top_k_seed=10, top_k_result=50)

        assert retriever._alpha == 0.9
        assert retriever._top_k_seed == 10
        assert retriever._top_k_result == 50

    @pytest.mark.asyncio
    async def test_call_with_empty_graph(self):
        """Test retrieval with an empty graph."""
        retriever = HippoRAGRetriever()
        mock_graph_rag = MagicMock()

        # Empty graph
        graph = nx.Graph()
        graph_storage = MagicMock()
        graph_storage._graph = graph
        mock_graph_rag.chunk_entity_relation_graph = graph_storage
        mock_graph_rag.aquery = AsyncMock(return_value="Fallback answer")

        result = await retriever("test query", mock_graph_rag, MagicMock())

        # Should fallback to local mode
        mock_graph_rag.aquery.assert_called_once()
        assert result == "Fallback answer"

    @pytest.mark.asyncio
    async def test_call_with_graph(self, mock_graph_rag):
        """Test retrieval with a populated graph."""
        retriever = HippoRAGRetriever(alpha=0.85, top_k_result=3)

        result = await retriever("What connects Entity1 and Entity4?", mock_graph_rag, MagicMock())

        # Should return a result
        assert result is not None
        assert isinstance(result, str)
        # Should not fallback to aquery
        mock_graph_rag.aquery.assert_not_called()

    @pytest.mark.asyncio
    async def test_find_seed_entities_with_vector_db(self):
        """Test finding seed entities via vector DB."""
        retriever = HippoRAGRetriever()
        mock_graph_rag = MagicMock()

        # Mock vector DB results
        mock_graph_rag.entities_vdb = MagicMock()
        mock_graph_rag.entities_vdb.query = AsyncMock(return_value=[
            {"entity_name": "Entity1", "id": "entity1"},
            {"entity_name": "Entity2", "id": "entity2"},
        ])

        seeds = await retriever._find_seed_entities("test query", mock_graph_rag)

        assert len(seeds) == 2
        assert "Entity1" in seeds
        assert "Entity2" in seeds

    @pytest.mark.asyncio
    async def test_find_seed_entities_without_vector_db(self):
        """Test finding seed entities when vector DB is not available."""
        retriever = HippoRAGRetriever()
        mock_graph_rag = MagicMock()
        mock_graph_rag.entities_vdb = None

        seeds = await retriever._find_seed_entities("test query", mock_graph_rag)

        assert seeds == []

    @pytest.mark.asyncio
    async def test_nodes_to_context(self, mock_graph_rag):
        """Test converting nodes to context."""
        retriever = HippoRAGRetriever()

        context = await retriever._nodes_to_context(["Entity1", "Entity2"], mock_graph_rag)

        assert context is not None
        assert isinstance(context, str)
        assert "Entity1" in context or "entity1" in context.lower()

    @pytest.mark.asyncio
    async def test_nodes_to_context_with_chunks(self, mock_graph_rag):
        """Test that chunks are included in context."""
        retriever = HippoRAGRetriever()

        context = await retriever._nodes_to_context(["Entity1"], mock_graph_rag)

        # Should include chunk content
        assert "Sample chunk content" in context

    def test_from_config(self):
        """Test creating retriever from configuration dict."""
        config = {
            "alpha": 0.9,
            "top_k_seed": 10,
            "top_k_result": 50,
        }

        retriever = HippoRAGRetriever.from_config(config)

        assert retriever._alpha == 0.9
        assert retriever._top_k_seed == 10
        assert retriever._top_k_result == 50

    def test_from_config_with_defaults(self):
        """Test creating retriever from configuration dict with defaults."""
        config = {"top_k_result": 15}

        retriever = HippoRAGRetriever.from_config(config)

        assert retriever._alpha == 0.85
        assert retriever._top_k_seed == 5
        assert retriever._top_k_result == 15
