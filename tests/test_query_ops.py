from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nano_graphrag._ops.query import (
    _build_naive_query_context,
    _edge_matches_time_range,
    _normalize_date_str,
    naive_query,
)
from nano_graphrag.base import QueryParam
from nano_graphrag.prompt import PROMPTS

pytestmark = pytest.mark.unit


class TestNormalizeDateStr:
    def test_full_date_unchanged(self):
        assert _normalize_date_str("2020-06-15") == "2020-06-15"

    def test_year_only(self):
        assert _normalize_date_str("2020") == "2020-01-01"

    def test_year_month(self):
        assert _normalize_date_str("2020-06") == "2020-06-01"

    def test_none_returns_empty(self):
        assert _normalize_date_str(None) == ""

    def test_empty_returns_empty(self):
        assert _normalize_date_str("") == ""


class TestEdgeMatchesTimeRange:
    def test_no_time_range_always_true(self):
        qp = QueryParam(time_range=None)
        assert (
            _edge_matches_time_range({"valid_from": "2020-01-01", "valid_to": "2021-01-01"}, qp)
            is True
        )

    def test_no_temporal_data_in_edge(self):
        qp = QueryParam(time_range=("2020-01-01", "2023-12-31"))
        assert _edge_matches_time_range({}, qp) is True

    def test_at_point_overlapping(self):
        qp = QueryParam(time_range=("2021-06-01", "2021-06-01"), temporal_mode="at_point")
        edge = {"valid_from": "2021-01-01", "valid_to": "2022-01-01"}
        assert _edge_matches_time_range(edge, qp) is True

    def test_at_point_before_range(self):
        qp = QueryParam(time_range=("2023-01-01", "2023-01-01"), temporal_mode="at_point")
        edge = {"valid_from": "2020-01-01", "valid_to": "2021-01-01"}
        assert _edge_matches_time_range(edge, qp) is False

    def test_any_mode_overlapping(self):
        qp = QueryParam(time_range=("2020-01-01", "2023-12-31"), temporal_mode="any")
        edge = {"valid_from": "2021-01-01", "valid_to": "2022-06-15"}
        assert _edge_matches_time_range(edge, qp) is True

    def test_any_mode_outside_range(self):
        qp = QueryParam(time_range=("2025-01-01", "2026-12-31"), temporal_mode="any")
        edge = {"valid_from": "2020-01-01", "valid_to": "2021-12-31"}
        assert _edge_matches_time_range(edge, qp) is False


class TestNaiveQueryContext:
    @pytest.mark.asyncio
    async def test_returns_joined_content(self):
        chunks_vdb = AsyncMock()
        chunks_vdb.query.return_value = [{"id": "c1"}, {"id": "c2"}]
        text_chunks_db = AsyncMock()
        text_chunks_db.get_by_ids.return_value = [
            {"content": "chunk one"},
            {"content": "chunk two"},
        ]
        qp = QueryParam()
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3]
        result = await _build_naive_query_context(
            "test query", chunks_vdb, text_chunks_db, qp, tokenizer
        )
        assert result == "chunk one--New Chunk--\nchunk two"

    @pytest.mark.asyncio
    async def test_no_vdb_results(self):
        chunks_vdb = AsyncMock()
        chunks_vdb.query.return_value = []
        text_chunks_db = AsyncMock()
        qp = QueryParam()
        result = await _build_naive_query_context(
            "test query", chunks_vdb, text_chunks_db, qp, None
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_token_truncation(self):
        chunks_vdb = AsyncMock()
        chunks_vdb.query.return_value = [{"id": f"c{i}"} for i in range(50)]
        text_chunks_db = AsyncMock()
        text_chunks_db.get_by_ids.return_value = [
            {"content": f"chunk {i} content"} for i in range(50)
        ]
        qp = QueryParam(naive_max_token_for_text_unit=10)
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1] * 100
        with patch(
            "nano_graphrag._ops.query.truncate_list_by_token_size",
            side_effect=lambda items, **kw: items[:2],
        ):
            result = await _build_naive_query_context(
                "test query", chunks_vdb, text_chunks_db, qp, tokenizer
            )
            assert result is not None
            assert result.count("--New Chunk--") == 1


class TestNaiveQuery:
    @pytest.mark.asyncio
    async def test_only_need_context_returns_context(self):
        qp = QueryParam(only_need_context=True)
        chunks_vdb = AsyncMock()
        text_chunks_db = AsyncMock()
        tokenizer = MagicMock()
        global_config = {}
        with patch(
            "nano_graphrag._ops.query._build_naive_query_context", new_callable=AsyncMock
        ) as mock_ctx:
            mock_ctx.return_value = "retrieved context data"
            result = await naive_query(
                "test query", chunks_vdb, text_chunks_db, qp, tokenizer, global_config
            )
            assert result == "retrieved context data"

    @pytest.mark.asyncio
    async def test_no_context_returns_fail(self):
        qp = QueryParam()
        chunks_vdb = AsyncMock()
        text_chunks_db = AsyncMock()
        tokenizer = MagicMock()
        global_config = {}
        with patch(
            "nano_graphrag._ops.query._build_naive_query_context", new_callable=AsyncMock
        ) as mock_ctx:
            mock_ctx.return_value = None
            result = await naive_query(
                "test query", chunks_vdb, text_chunks_db, qp, tokenizer, global_config
            )
            assert result == PROMPTS["fail_response"]

    @pytest.mark.asyncio
    async def test_normal_returns_llm_answer(self):
        qp = QueryParam()
        chunks_vdb = AsyncMock()
        text_chunks_db = AsyncMock()
        tokenizer = MagicMock()
        llm_func = AsyncMock(return_value="LLM generated answer")
        global_config = {"best_model_func": llm_func}
        with patch(
            "nano_graphrag._ops.query._build_naive_query_context", new_callable=AsyncMock
        ) as mock_ctx:
            mock_ctx.return_value = "retrieved context data"
            result = await naive_query(
                "test query", chunks_vdb, text_chunks_db, qp, tokenizer, global_config
            )
            assert result == "LLM generated answer"
            llm_func.assert_awaited_once()
