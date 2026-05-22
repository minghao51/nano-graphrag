from __future__ import annotations

import pytest

from nano_graphrag._schemas import (
    QueryResult,
    QuerySource,
    QueryTrace,
    StreamComplete,
    StreamSourceRef,
    StreamTextChunk,
    TokenUsage,
)


class TestQueryTrace:
    def test_defaults(self):
        qt = QueryTrace()
        assert qt.entities_matched == []
        assert qt.communities_used == []
        assert qt.chunks_used == []
        assert qt.retrieval_scores == {}
        assert qt.mode_specific == {}

    def test_with_data(self):
        qt = QueryTrace(
            entities_matched=["e1", "e2"],
            communities_used=["c1"],
            retrieval_scores={"e1": 0.95, "e2": 0.85},
        )
        assert len(qt.entities_matched) == 2
        assert qt.retrieval_scores["e1"] == 0.95


class TestStreamTextChunk:
    def test_str_returns_text(self):
        chunk = StreamTextChunk(text="hello")
        assert str(chunk) == "hello"

    def test_empty(self):
        chunk = StreamTextChunk(text="")
        assert str(chunk) == ""

    def test_multiline(self):
        chunk = StreamTextChunk(text="line1\nline2")
        assert str(chunk) == "line1\nline2"


class TestStreamSourceRef:
    def test_with_sources(self):
        src = QuerySource(source_type="entity", id="e1", name="Entity1")
        ref = StreamSourceRef(sources=[src])
        assert len(ref.sources) == 1
        assert ref.sources[0].source_type == "entity"

    def test_empty(self):
        ref = StreamSourceRef()
        assert ref.sources == []


class TestStreamComplete:
    def test_with_trace(self):
        trace = QueryTrace(entities_matched=["e1"])
        sc = StreamComplete(trace=trace, latency_ms=500.0)
        assert sc.trace.entities_matched == ["e1"]
        assert sc.latency_ms == 500.0

    def test_with_tokens(self):
        tu = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        sc = StreamComplete(tokens=tu)
        assert sc.tokens.total_tokens == 150

    def test_defaults(self):
        sc = StreamComplete()
        assert sc.trace is None
        assert sc.tokens is None
        assert sc.latency_ms == 0.0


class TestQueryResultExplainability:
    def test_with_trace_in_metadata(self):
        trace = QueryTrace(entities_matched=["e1", "e2"], communities_used=["c1"])
        qr = QueryResult(
            answer="test answer",
            mode="local",
            metadata={"trace": trace},
            latency_ms=100.0,
        )
        assert qr.metadata["trace"].entities_matched == ["e1", "e2"]
        assert str(qr) == "test answer"

    def test_result_from_query_is_str_compat(self):
        qr = QueryResult(answer="hello", mode="global")
        assert isinstance(str(qr), str)
        assert str(qr) == "hello"


class TestBackwardCompat:
    @pytest.mark.asyncio
    async def test_aquery_returns_query_result(self, clean_working_dir):
        from nano_graphrag import GraphRAG, QueryParam

        rag = GraphRAG(working_dir=clean_working_dir, enable_local=False, enable_naive_rag=False)
        result = await rag.aquery("test", QueryParam(mode="global"))
        assert isinstance(result, QueryResult)
        assert str(result) == result.answer

    @pytest.mark.asyncio
    async def test_astream_query_yields_stream_text_chunk(self, clean_working_dir):
        from nano_graphrag import GraphRAG, QueryParam

        rag = GraphRAG(working_dir=clean_working_dir, enable_local=False, enable_naive_rag=False)
        chunks = []
        async for chunk in rag.astream_query("test", QueryParam(mode="global")):
            chunks.append(chunk)
        assert len(chunks) > 0
        text_chunks = [c for c in chunks if isinstance(c, StreamTextChunk)]
        complete_chunks = [c for c in chunks if isinstance(c, StreamComplete)]
        assert len(text_chunks) >= 0
        assert len(complete_chunks) == 1
        assert complete_chunks[0].latency_ms >= 0
        assert all(str(c) == c.text for c in text_chunks)
