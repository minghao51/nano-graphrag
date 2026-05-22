from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from nano_graphrag._entity_grounded_query import EntityGroundedQuery
from nano_graphrag._schemas import QueryResult

pytestmark = pytest.mark.unit


def _make_egq(llm_func=None, llm_stream_func=None):
    registry = MagicMock()
    graph = AsyncMock()
    vdb = AsyncMock()
    llm = llm_func or AsyncMock(return_value="Test answer")
    return EntityGroundedQuery(registry, graph, vdb, llm, llm_stream_func)


_ENTITY_CONTEXT = {
    "e1": {
        "canonical_name": "Alice",
        "aliases": ["Al"],
        "description": "A person",
        "relationships": [],
    },
    "e2": {
        "canonical_name": "Bob",
        "aliases": ["Bobby"],
        "description": "Another person",
        "relationships": [],
    },
}


class TestCalculateConfidence:
    def test_empty_used_returns_zero(self):
        egq = _make_egq()
        assert egq._calculate_confidence("some answer", [], ["e1", "e2"]) == 0.0

    def test_all_top_entities(self):
        egq = _make_egq()
        result = egq._calculate_confidence("answer", ["e1", "e2"], ["e1", "e2", "e3"])
        assert result > 0.0

    def test_entity_outside_top10(self):
        egq = _make_egq()
        retrieved = [f"e{i}" for i in range(20)]
        high = egq._calculate_confidence("answer", ["e0"], retrieved)
        low = egq._calculate_confidence("answer", ["e19"], retrieved)
        assert low < high

    def test_confidence_capped_at_one(self):
        egq = _make_egq()
        result = egq._calculate_confidence("answer", ["e1"], ["e1"])
        assert result <= 1.0


class TestValidateAndNormalize:
    @staticmethod
    def _meta(result: QueryResult, key: str, default=None):
        return result.metadata.get(key, default)

    def test_matches_canonical_name(self):
        egq = _make_egq()
        result = egq._validate_and_normalize("Alice works here", ["e1", "e2"], _ENTITY_CONTEXT)
        assert self._meta(result, "entity_ids") == ["e1"]
        assert self._meta(result, "canonical_entities") == ["Alice"]
        assert self._meta(result, "confidence", 0.0) > 0.0

    def test_matches_alias(self):
        egq = _make_egq()
        result = egq._validate_and_normalize("Al is here", ["e1", "e2"], _ENTITY_CONTEXT)
        assert self._meta(result, "entity_ids") == ["e1"]
        assert self._meta(result, "canonical_entities") == ["Alice"]

    def test_no_match_returns_fallback(self):
        egq = _make_egq()
        egq.registry.resolve_entities_from_text = MagicMock(return_value=[])
        result = egq._validate_and_normalize("xyz unknown", ["e1", "e2"], _ENTITY_CONTEXT)
        assert result.answer == egq.fallback_message
        assert self._meta(result, "confidence") == 0.0

    def test_multiple_entities(self):
        egq = _make_egq()
        result = egq._validate_and_normalize("Alice and Bob", ["e1", "e2"], _ENTITY_CONTEXT)
        assert self._meta(result, "canonical_entities") == ["Alice", "Bob"]
        assert result.answer == "Alice and Bob"

    def test_resolved_from_registry(self):
        egq = _make_egq()
        egq.registry.resolve_entities_from_text = MagicMock(return_value=[("e3", "Charlie")])
        result = egq._validate_and_normalize("xyz unknown", ["e1", "e2"], _ENTITY_CONTEXT)
        assert self._meta(result, "entity_ids") == ["e3"]
        assert self._meta(result, "canonical_entities") == ["Charlie"]


class TestRetrievalMethods:
    @pytest.mark.asyncio
    async def test_local_retrieval(self):
        egq = _make_egq()
        egq.entities_vdb.query = AsyncMock(return_value=[{"id": "e1"}, {"id": "e2"}])
        result = await egq._local_retrieval("question", top_k=10)
        assert result == ["e1", "e2"]

    @pytest.mark.asyncio
    async def test_global_uses_local_retrieval(self):
        egq = _make_egq()
        egq.entities_vdb.query = AsyncMock(return_value=[{"id": "e1"}, {"id": "e2"}])
        result = await egq._retrieve_entities("question", top_k=10, mode="global")
        assert result == ["e1", "e2"]

    @pytest.mark.asyncio
    async def test_naive_retrieval(self):
        egq = _make_egq()
        egq.registry.resolve_entities_from_text = MagicMock(return_value=[("e1", "Alice")])
        result = await egq._naive_retrieval("question", top_k=10)
        assert result == ["e1"]

    @pytest.mark.asyncio
    async def test_multihop_retrieval(self):
        egq = _make_egq()
        egq.entities_vdb.query = AsyncMock(return_value=[{"id": "e1"}])
        egq.graph.get_nodes_edges_batch = AsyncMock(return_value=[[("e1", "e3")]])
        result = await egq._multihop_retrieval("question", top_k=10)
        assert "e3" in result
        assert "e1" in result


class TestBuildEntityContext:
    @pytest.mark.asyncio
    async def test_builds_context_dict(self):
        egq = _make_egq()
        egq.graph.get_nodes_batch = AsyncMock(return_value=[{"description": "A person"}])
        egq.graph.get_nodes_edges_batch = AsyncMock(return_value=[[("e1", "e2")]])
        egq.registry.get_entity_record = MagicMock(
            return_value=MagicMock(canonical_name="Alice", aliases=["Al"])
        )
        result = await egq._build_entity_context(["e1"])
        assert "e1" in result
        assert result["e1"]["canonical_name"] == "Alice"
        assert result["e1"]["aliases"] == ["Al"]
        assert result["e1"]["description"] == "A person"
        assert result["e1"]["relationships"] == [("e1", "e2")]


class TestGenerateAnswer:
    @pytest.mark.asyncio
    async def test_calls_llm_with_prompt(self):
        llm = AsyncMock(return_value="  Alice  ")
        egq = _make_egq(llm_func=llm)
        context = {
            "e1": {
                "canonical_name": "Alice",
                "aliases": [],
                "description": "A person",
                "relationships": [],
            }
        }
        await egq._generate_answer("Who?", context)
        llm.assert_called_once()
        call_arg = llm.call_args[0][0]
        assert "Alice" in call_arg

    @pytest.mark.asyncio
    async def test_returns_stripped_response(self):
        llm = AsyncMock(return_value="  Alice  ")
        egq = _make_egq(llm_func=llm)
        context = {
            "e1": {
                "canonical_name": "Alice",
                "aliases": [],
                "description": "A person",
                "relationships": [],
            }
        }
        result = await egq._generate_answer("Who?", context)
        assert result == "Alice"


class TestFullQuery:
    @pytest.mark.asyncio
    async def test_no_entities_returns_fallback(self):
        egq = _make_egq()
        egq._retrieve_entities = AsyncMock(return_value=[])
        result = await egq.query("anything")
        assert result.answer == egq.fallback_message
        assert result.metadata["confidence"] == 0.0
        assert result.metadata["entity_ids"] == []

    @pytest.mark.asyncio
    async def test_happy_path(self):
        egq = _make_egq()
        egq._retrieve_entities = AsyncMock(return_value=["e1"])
        egq._build_entity_context = AsyncMock(
            return_value={
                "e1": {
                    "canonical_name": "Alice",
                    "aliases": [],
                    "description": "A person",
                    "relationships": [],
                }
            }
        )
        egq._generate_answer = AsyncMock(return_value="Alice")
        result = await egq.query("Who is Alice?")
        assert isinstance(result, QueryResult)
        assert result.metadata["entity_ids"] == ["e1"]
        assert result.metadata["canonical_entities"] == ["Alice"]
        assert result.metadata["confidence"] > 0.0
