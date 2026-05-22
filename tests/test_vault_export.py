from __future__ import annotations

import os
import tempfile
from unittest.mock import AsyncMock

import pytest

from nano_graphrag._vault.export import (
    _needs_yaml_quoting,
    _safe_filename,
    _write_index,
    _yaml_frontmatter,
    aexport_vault,
)

pytestmark = pytest.mark.unit


def _make_graph_mock(nodes, edges=None):
    mock = AsyncMock()
    mock.get_all_nodes = AsyncMock(return_value=nodes)
    edge_data = edges or {}

    node_edges: dict[str, list[tuple[str, str]]] = {}
    for src, tgt in edge_data:
        node_edges.setdefault(src, []).append((src, tgt))
        node_edges.setdefault(tgt, []).append((src, tgt))

    async def _get_node_edges(nid):
        return node_edges.get(nid, [])

    mock.get_node_edges = AsyncMock(side_effect=_get_node_edges)

    async def _get_edge(src, tgt):
        for (s, t), data in edge_data.items():
            if (s == src and t == tgt) or (s == tgt and t == src):
                return data
        return None

    mock.get_edge = AsyncMock(side_effect=_get_edge)

    async def _get_node(nid):
        return nodes.get(nid)

    mock.get_node = AsyncMock(side_effect=_get_node)
    return mock


def test_safe_filename_strips_special_chars():
    result = _safe_filename("Hello/World:test")
    assert ":" not in result
    assert "/" not in result


def test_safe_filename_reserves_index():
    result = _safe_filename("_index")
    assert result.startswith("_")


def test_safe_filename_empty_input():
    assert _safe_filename("") == "_unnamed_"


def test_yaml_frontmatter_basic():
    result = _yaml_frontmatter({"key": "value"})
    assert result.startswith("---")
    assert result.endswith("---")


def test_yaml_frontmatter_with_list():
    result = _yaml_frontmatter({"items": ["a", "b"]})
    assert "- a" in result
    assert "- b" in result


def test_yaml_frontmatter_float_formatting():
    result = _yaml_frontmatter({"score": 1.23456})
    assert "score: 1.23" in result


def test_needs_yaml_quoting_special_chars():
    assert _needs_yaml_quoting("hello: world") is True
    assert _needs_yaml_quoting("simple") is False


def test_write_index_creates_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_index(tmpdir, ["- [[Item1]]"])
        index_path = os.path.join(tmpdir, "_index.md")
        assert os.path.exists(index_path)
        content = open(index_path).read()
        assert "- [[Item1]]" in content


@pytest.mark.asyncio
async def test_export_empty_graph_returns_zero():
    mock_graph = AsyncMock()
    mock_graph.get_all_nodes = AsyncMock(return_value={})
    with tempfile.TemporaryDirectory() as tmpdir:
        stats = await aexport_vault(mock_graph, None, {"working_dir": tmpdir}, path=tmpdir)
        assert stats["entities"] == 0


@pytest.mark.asyncio
async def test_export_creates_entity_files():
    nodes = {
        "n1": {
            "entity_name": "Alice",
            "entity_type": "PERSON",
            "description": "A person who lives in the city and works hard every day.",
        },
        "n2": {
            "entity_name": "Bob",
            "entity_type": "PERSON",
            "description": "Another person who travels the world and writes about it.",
        },
    }
    mock_graph = _make_graph_mock(nodes)
    with tempfile.TemporaryDirectory() as tmpdir:
        stats = await aexport_vault(mock_graph, None, {"working_dir": tmpdir}, path=tmpdir)
        assert stats["entities"] == 2
        entities_dir = os.path.join(tmpdir, "entities", "PERSON")
        assert os.path.isdir(entities_dir)
        files = os.listdir(entities_dir)
        assert any("Alice" in f for f in files)
        assert any("Bob" in f for f in files)


@pytest.mark.asyncio
async def test_export_sparse_entities_rolled_up():
    nodes = {
        "n1": {
            "entity_name": "X",
            "entity_type": "THING",
            "description": "short",
        },
        "n2": {
            "entity_name": "Y",
            "entity_type": "THING",
            "description": "tiny",
        },
    }
    mock_graph = _make_graph_mock(nodes)
    with tempfile.TemporaryDirectory() as tmpdir:
        stats = await aexport_vault(mock_graph, None, {"working_dir": tmpdir}, path=tmpdir)
        assert stats["entities"] == 0
        assert stats["sparse_rolled_up"] == 2
        sparse_path = os.path.join(tmpdir, "entities", "sparse_entities.md")
        assert os.path.exists(sparse_path)


@pytest.mark.asyncio
async def test_export_communities_with_reports():
    nodes = {
        "n1": {
            "entity_name": "Alpha",
            "entity_type": "ORG",
            "description": "An organization with many members across the globe.",
        },
    }
    mock_graph = _make_graph_mock(nodes)
    mock_kv = AsyncMock()
    mock_kv.all_keys = AsyncMock(return_value=["c1"])
    mock_kv.get_by_ids = AsyncMock(
        return_value=[
            {
                "report_json": {"title": "Community One", "rating": 5},
                "report_string": "This is a community report.",
            }
        ]
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        stats = await aexport_vault(
            mock_graph, mock_kv, {"working_dir": tmpdir}, path=tmpdir, include_communities=True
        )
        assert stats["communities"] == 1
        communities_dir = os.path.join(tmpdir, "communities")
        assert os.path.isdir(communities_dir)
        assert len(os.listdir(communities_dir)) >= 1


@pytest.mark.asyncio
async def test_export_entity_with_edges_shows_connections():
    nodes = {
        "n1": {
            "entity_name": "Alice",
            "entity_type": "PERSON",
            "description": "A person who works at a major research institution.",
        },
        "n2": {
            "entity_name": "MIT",
            "entity_type": "ORG",
            "description": "A major research university located in Cambridge.",
        },
    }
    edge_data = {
        ("n1", "n2"): {"relation_type": "works_for", "confidence": 0.9},
    }
    mock_graph = _make_graph_mock(nodes, edges=edge_data)
    with tempfile.TemporaryDirectory() as tmpdir:
        await aexport_vault(mock_graph, None, {"working_dir": tmpdir}, path=tmpdir)
        alice_path = os.path.join(tmpdir, "entities", "PERSON", "Alice.md")
        assert os.path.exists(alice_path)
        content = open(alice_path).read()
        assert "works_for" in content
        assert "[[MIT]]" in content


@pytest.mark.asyncio
async def test_export_filename_collision_dedup():
    nodes = {
        "n1": {
            "entity_name": "Test",
            "entity_type": "PERSON",
            "description": "First test entity with enough content to export.",
        },
        "n2": {
            "entity_name": "Test",
            "entity_type": "PERSON",
            "description": "Second test entity with enough content to export.",
        },
    }
    mock_graph = _make_graph_mock(nodes)
    with tempfile.TemporaryDirectory() as tmpdir:
        stats = await aexport_vault(mock_graph, None, {"working_dir": tmpdir}, path=tmpdir)
        assert stats["entities"] == 2
        entities_dir = os.path.join(tmpdir, "entities", "PERSON")
        files = os.listdir(entities_dir)
        assert "Test.md" in files
        assert "Test_1.md" in files


@pytest.mark.asyncio
async def test_export_communities_empty_keys_returns_zero():
    nodes = {
        "n1": {
            "entity_name": "Alpha",
            "entity_type": "ORG",
            "description": "An organization with many members across the globe.",
        },
    }
    mock_graph = _make_graph_mock(nodes)
    mock_kv = AsyncMock()
    mock_kv.all_keys = AsyncMock(return_value=[])
    with tempfile.TemporaryDirectory() as tmpdir:
        stats = await aexport_vault(
            mock_graph, mock_kv, {"working_dir": tmpdir}, path=tmpdir, include_communities=True
        )
        assert stats["communities"] == 0


@pytest.mark.asyncio
async def test_export_communities_skips_none_report():
    nodes = {
        "n1": {
            "entity_name": "Alpha",
            "entity_type": "ORG",
            "description": "An organization with many members across the globe.",
        },
    }
    mock_graph = _make_graph_mock(nodes)
    mock_kv = AsyncMock()
    mock_kv.all_keys = AsyncMock(return_value=["c1"])
    mock_kv.get_by_ids = AsyncMock(return_value=[None])
    with tempfile.TemporaryDirectory() as tmpdir:
        stats = await aexport_vault(
            mock_graph, mock_kv, {"working_dir": tmpdir}, path=tmpdir, include_communities=True
        )
        assert stats["communities"] == 0


def test_yaml_frontmatter_special_chars_quoted():
    result = _yaml_frontmatter({"name": "foo:bar", "desc": "has # hash"})
    assert '"foo:bar"' in result
    assert '"has # hash"' in result
