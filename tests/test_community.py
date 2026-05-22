import pytest

from nano_graphrag._ops.community import (
    _community_report_json_to_str,
    _pack_single_community_by_sub_communities,
)
from nano_graphrag.base import CommunitySchema, SingleCommunitySchema

pytestmark = pytest.mark.unit


class MockTokenizer:
    def encode(self, text):
        return [0] * len(text.split())


def test_community_report_json_to_str_basic():
    data = {
        "title": "Test",
        "summary": "Summary",
        "findings": [{"summary": "Finding 1", "explanation": "Details"}],
    }
    result = _community_report_json_to_str(data)
    assert result.startswith("# Test")
    assert "Summary" in result
    assert "## Finding 1" in result
    assert "Details" in result


def test_community_report_json_to_str_string_findings():
    data = {"title": "T", "summary": "S", "findings": ["Just a string"]}
    result = _community_report_json_to_str(data)
    assert "## Just a string" in result


def test_community_report_json_to_str_empty_findings():
    data = {"title": "T", "summary": "S", "findings": []}
    result = _community_report_json_to_str(data)
    assert result.startswith("# T")
    assert "S" in result


def test_pack_single_community_by_sub_communities():
    community: SingleCommunitySchema = {
        "level": 1,
        "title": "c1",
        "edges": [("a", "b")],
        "nodes": ["a", "b"],
        "chunk_ids": [],
        "occurrence": 1.0,
        "sub_communities": ["sub1"],
    }
    sub_report: CommunitySchema = {
        "level": 0,
        "title": "sub1",
        "edges": [],
        "nodes": ["x"],
        "chunk_ids": [],
        "occurrence": 1.0,
        "sub_communities": [],
        "report_string": "report content for sub1",
        "report_json": {"rating": 5},
    }
    already_reports = {"sub1": sub_report}
    tokenizer = MockTokenizer()
    result = _pack_single_community_by_sub_communities(community, 12000, already_reports, tokenizer)
    assert isinstance(result, tuple)
    assert len(result) == 4
    report_str, token_count, nodes, edges = result
    assert isinstance(report_str, str)
    assert isinstance(token_count, int)
    assert isinstance(nodes, set)
    assert isinstance(edges, set)
    assert "report content for sub1" in report_str
    assert "x" in nodes


def test_pack_single_community_empty_sub_communities():
    community: SingleCommunitySchema = {
        "level": 1,
        "title": "c1",
        "edges": [],
        "nodes": [],
        "chunk_ids": [],
        "occurrence": 0.0,
        "sub_communities": [],
    }
    tokenizer = MockTokenizer()
    result = _pack_single_community_by_sub_communities(community, 12000, {}, tokenizer)
    report_str, token_count, nodes, edges = result
    assert "id" in report_str
    assert nodes == set()
    assert edges == set()


def test_community_report_json_to_str_nested_findings():
    data = {
        "title": "Title",
        "summary": "Sum",
        "findings": [
            {"summary": "First", "explanation": "Exp1"},
            {"summary": "Second", "explanation": "Exp2"},
            {"summary": "Third", "explanation": "Exp3"},
        ],
    }
    result = _community_report_json_to_str(data)
    assert "## First" in result
    assert "Exp1" in result
    assert "## Second" in result
    assert "Exp2" in result
    assert "## Third" in result
    assert "Exp3" in result
