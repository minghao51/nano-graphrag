from __future__ import annotations

import pytest

from nano_graphrag._callbacks import (
    LoggingCallback,
    TokenTrackingCallback,
    _CallbackDispatcher,
)
from nano_graphrag._schemas import InsertResult, QueryResult


class TestTokenTrackingCallback:
    @pytest.mark.asyncio
    async def test_accumulates_calls(self):
        tracker = TokenTrackingCallback()
        await tracker.on_call("gpt-4o", prompt_tokens=100, completion_tokens=50, latency_ms=500)
        await tracker.on_call("gpt-4o", prompt_tokens=200, completion_tokens=100, latency_ms=800)

        assert tracker.prompt_tokens == 300
        assert tracker.completion_tokens == 150
        assert tracker.total_tokens == 450
        assert tracker.call_count == 2

    @pytest.mark.asyncio
    async def test_summary(self):
        tracker = TokenTrackingCallback()
        await tracker.on_call("gpt-4o", 100, 50, 500)
        summary = tracker.summary()

        assert summary["prompt_tokens"] == 100
        assert summary["completion_tokens"] == 50
        assert summary["total_tokens"] == 150
        assert summary["call_count"] == 1


class TestLoggingCallback:
    @pytest.mark.asyncio
    async def test_on_start(self):
        cb = LoggingCallback()
        await cb.on_start(total_docs=10)

    @pytest.mark.asyncio
    async def test_on_complete(self):
        cb = LoggingCallback()
        result = InsertResult(documents_processed=5, entities_created=20, latency_ms=1000)
        await cb.on_complete(result)

    @pytest.mark.asyncio
    async def test_on_error(self):
        cb = LoggingCallback()
        await cb.on_error(RuntimeError("test error"))


class TestCallbackDispatcher:
    @pytest.mark.asyncio
    async def test_dispatches_to_extraction_callbacks(self):
        tracker = TokenTrackingCallback()
        logging_cb = LoggingCallback()
        dispatcher = _CallbackDispatcher([tracker, logging_cb])

        await dispatcher.extraction_start(total_docs=5)
        await dispatcher.extraction_progress(committed=3, total=5)
        result = InsertResult(documents_processed=5, latency_ms=500)
        await dispatcher.extraction_complete(result)

    @pytest.mark.asyncio
    async def test_dispatches_to_llm_callbacks(self):
        tracker = TokenTrackingCallback()
        dispatcher = _CallbackDispatcher([tracker])

        await dispatcher.llm_call("gpt-4o", 100, 50, 500)
        assert tracker.call_count == 1

    @pytest.mark.asyncio
    async def test_dispatches_to_query_callbacks(self):
        logging_cb = LoggingCallback()
        dispatcher = _CallbackDispatcher([logging_cb])

        await dispatcher.query_start("test query", "local")
        result = QueryResult(answer="test answer", mode="local")
        await dispatcher.query_complete(result)

    @pytest.mark.asyncio
    async def test_swallows_callback_errors(self):
        class BrokenCallback:
            async def on_start(self, total_docs: int) -> None:
                raise RuntimeError("broken")

            async def on_call(self, model, prompt_tokens, completion_tokens, latency_ms):
                raise RuntimeError("broken")

        dispatcher = _CallbackDispatcher([BrokenCallback()])
        await dispatcher.extraction_start(5)
        await dispatcher.llm_call("gpt-4o", 100, 50, 500)

    @pytest.mark.asyncio
    async def test_empty_callbacks_noop(self):
        dispatcher = _CallbackDispatcher([])
        await dispatcher.extraction_start(5)
        await dispatcher.extraction_progress(3, 5)
        await dispatcher.llm_call("gpt-4o", 100, 50, 500)


class TestGraphRAGCallbacks:
    def test_callbacks_accepted(self, clean_working_dir):
        from nano_graphrag import GraphRAG

        tracker = TokenTrackingCallback()
        rag = GraphRAG(working_dir=clean_working_dir, callbacks=[tracker])
        assert rag._callback_dispatcher is not None

    def test_multiple_callbacks(self, clean_working_dir):
        from nano_graphrag import GraphRAG

        tracker = TokenTrackingCallback()
        logger_cb = LoggingCallback()
        rag = GraphRAG(working_dir=clean_working_dir, callbacks=[tracker, logger_cb])
        assert rag._callback_dispatcher is not None

    @pytest.mark.asyncio
    async def test_extraction_error_callback_fired(
        self, clean_working_dir, deterministic_embedding, monkeypatch
    ):
        from nano_graphrag import GraphRAG

        class ErrorCapture:
            def __init__(self):
                self.errors = []

            async def on_start(self, total_docs: int) -> None: ...
            async def on_doc_progress(self, committed: int, total: int) -> None: ...
            async def on_chunk_extracted(self, doc_id: str, chunks: int) -> None: ...
            async def on_community_report(self, level: int, count: int) -> None: ...
            async def on_complete(self, result) -> None: ...
            async def on_error(self, error: Exception) -> None:
                self.errors.append(type(error).__name__)

        async def fake_model(prompt, system_prompt=None, history_messages=None, **kwargs):
            if system_prompt is not None:
                return (
                    '{"title":"T","summary":"S","rating":1,"rating_explanation":"R","findings":[]}'
                )
            return '("entity"<|>ALICE<|>PERSON<|>A person.)<|COMPLETE|>'

        capture = ErrorCapture()
        rag = GraphRAG(
            working_dir=clean_working_dir,
            best_model_func=fake_model,
            cheap_model_func=fake_model,
            embedding_func=deterministic_embedding,
            callbacks=[capture],
        )

        async def fail_rebuild(*args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(
            "nano_graphrag.graphrag_insert.rebuild_knowledge_graph_for_documents", fail_rebuild
        )
        with pytest.raises(RuntimeError, match="boom"):
            await rag.ainsert_documents({"doc-1": "Alice works here."}, force_rebuild=True)

        assert "RuntimeError" in capture.errors
