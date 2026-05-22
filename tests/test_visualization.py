from __future__ import annotations

import os

import pytest

from nano_graphrag._visualization import (
    GraphStatus,
    _generate_simple_html,
    _repr_html_,
    compute_status,
)


class TestGraphStatus:
    def test_defaults(self):
        s = GraphStatus()
        assert s.entity_count == 0
        assert s.entity_types == {}
        assert s.relationship_count == 0
        assert s.community_count == 0
        assert s.health == "healthy"
        assert s.warnings == []

    def test_with_data(self):
        s = GraphStatus(
            entity_count=100,
            entity_types={"PERSON": 45, "ORG": 30, "GPE": 25},
            relationship_count=200,
            community_count=15,
            document_count=10,
            chunk_count=50,
        )
        assert s.entity_count == 100
        assert s.entity_types["PERSON"] == 45

    def test_health_states(self):
        healthy = GraphStatus(health="healthy")
        assert healthy.health == "healthy"

        warning = GraphStatus(health="warning", warnings=["Empty graph"])
        assert warning.health == "warning"
        assert len(warning.warnings) == 1


class TestComputeStatus:
    @pytest.mark.asyncio
    async def test_nonexistent_dir(self, tmp_path):
        status = await compute_status(str(tmp_path / "nonexistent"))
        assert status.health == "error"
        assert "does not exist" in status.warnings[0]

    @pytest.mark.asyncio
    async def test_empty_working_dir(self, tmp_path):
        workdir = str(tmp_path / "empty_graph")
        os.makedirs(workdir)
        status = await compute_status(workdir)
        assert status.storage_size_bytes == 0
        assert status.health == "warning"

    @pytest.mark.asyncio
    async def test_with_rag_instance(self, clean_working_dir):
        from nano_graphrag import GraphRAG

        rag = GraphRAG(working_dir=clean_working_dir)
        status = await compute_status(clean_working_dir, rag=rag)
        assert isinstance(status, GraphStatus)
        assert status.document_count == 0
        assert status.entity_count == 0


class TestReprHtml:
    def test_basic_html(self):
        s = GraphStatus(
            entity_count=50,
            relationship_count=100,
            community_count=10,
            document_count=5,
            chunk_count=20,
            entity_types={"PERSON": 30, "ORG": 20},
        )
        html = _repr_html_(s)
        assert "nano-graphrag" in html
        assert "50 entities" in html
        assert "PERSON" in html

    def test_with_warnings(self):
        s = GraphStatus(warnings=["Empty graph", "No community reports"])
        html = _repr_html_(s)
        assert "Warnings" in html
        assert "Empty graph" in html


class TestSimpleHtmlVisualization:
    def test_empty_graph(self, tmp_path):
        import networkx as nx

        output = str(tmp_path / "empty_graph.html")
        result = _generate_simple_html(nx.DiGraph(), output, 100)
        assert result == output
        assert os.path.exists(output)

    def test_simple_graph(self, tmp_path):
        import networkx as nx

        output = str(tmp_path / "test_graph.html")
        g = nx.DiGraph()
        g.add_node("e1", entity_name="Entity1", entity_type="PERSON", description="A person")
        g.add_node("e2", entity_name="Entity2", entity_type="ORG", description="An org")
        g.add_edge("e1", "e2", description="works_for", weight=1.0)
        result = _generate_simple_html(g, output, 100)
        assert result == output
        content = open(output).read()
        assert "Entity1" in content
        assert "Entity2" in content
        assert "PERSON" in content
        assert "works_for" in content


class TestGraphRAGStatusMethod:
    @pytest.mark.asyncio
    async def test_astatus(self, clean_working_dir):
        from nano_graphrag import GraphRAG

        rag = GraphRAG(working_dir=clean_working_dir)
        status = await rag.astatus()
        assert isinstance(status, GraphStatus)
        assert status.document_count == 0

    def test_status_sync(self, clean_working_dir):
        from nano_graphrag import GraphRAG

        rag = GraphRAG(working_dir=clean_working_dir)
        status = rag.status()
        assert isinstance(status, GraphStatus)

    @pytest.mark.asyncio
    async def test_aexport_graph_html(self, clean_working_dir, tmp_path):
        from nano_graphrag import GraphRAG

        rag = GraphRAG(working_dir=clean_working_dir)
        output = str(tmp_path / "test_viz.html")
        result = await rag.aexport_graph_html(output=output)
        assert result == output
        assert os.path.exists(output)
