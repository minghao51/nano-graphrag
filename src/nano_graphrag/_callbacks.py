from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ._schemas import InsertResult, QueryResult, QuerySource


@runtime_checkable
class ExtractionCallback(Protocol):
    async def on_start(self, total_docs: int) -> None: ...
    async def on_doc_progress(self, committed: int, total: int) -> None: ...
    async def on_chunk_extracted(self, doc_id: str, chunks: int) -> None: ...
    async def on_community_report(self, level: int, count: int) -> None: ...
    async def on_complete(self, result: InsertResult) -> None: ...
    async def on_error(self, error: Exception) -> None: ...


@runtime_checkable
class QueryCallback(Protocol):
    async def on_start(self, query: str, mode: str) -> None: ...
    async def on_sources_found(self, sources: list[QuerySource]) -> None: ...
    async def on_complete(self, result: QueryResult) -> None: ...


@runtime_checkable
class LLMCallback(Protocol):
    async def on_call(
        self, model: str, prompt_tokens: int, completion_tokens: int, latency_ms: float
    ) -> None: ...


Callback = ExtractionCallback | QueryCallback | LLMCallback


class LoggingCallback:
    """Default callback: emits structlog events (current behavior, no change)."""

    def __init__(self):
        from ._utils import logger

        self._logger = logger

    async def on_start(self, total_docs: int) -> None:
        self._logger.info("extraction_start", total_docs=total_docs)

    async def on_doc_progress(self, committed: int, total: int) -> None:
        self._logger.info(
            "extraction_progress",
            committed=committed,
            total=total,
            pct=committed * 100 // max(total, 1),
        )

    async def on_chunk_extracted(self, doc_id: str, chunks: int) -> None:
        self._logger.debug("chunk_extracted", doc_id=doc_id, chunks=chunks)

    async def on_community_report(self, level: int, count: int) -> None:
        self._logger.info("community_report", level=level, count=count)

    async def on_complete(self, result: InsertResult) -> None:
        self._logger.info(
            "extraction_complete",
            docs_processed=result.documents_processed,
            entities=result.entities_created,
            latency_ms=round(result.latency_ms, 1),
        )

    async def on_error(self, error: Exception) -> None:
        self._logger.error("extraction_error", error=str(error), error_type=type(error).__name__)


@dataclass
class TokenTrackingCallback:
    """Accumulates token usage across all LLM calls."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    call_count: int = 0
    _call_log: list[dict[str, Any]] = field(default_factory=list)

    async def on_call(
        self, model: str, prompt_tokens: int, completion_tokens: int, latency_ms: float
    ) -> None:
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += prompt_tokens + completion_tokens
        self.call_count += 1
        self._call_log.append(
            {
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "latency_ms": round(latency_ms, 1),
            }
        )

    def summary(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "call_count": self.call_count,
        }


class _NullDispatcher:
    """No-op dispatcher. All methods are no-ops. Used when no callbacks are registered."""

    async def extraction_start(self, total_docs: int) -> None:
        pass

    async def extraction_progress(self, committed: int, total: int) -> None:
        pass

    async def extraction_complete(self, result) -> None:
        pass

    async def extraction_error(self, error: Exception) -> None:
        pass

    async def query_start(self, query: str, mode: str) -> None:
        pass

    async def query_complete(self, result) -> None:
        pass

    async def query_sources_found(self, sources: list) -> None:
        pass

    async def community_report(self, level: int, count: int) -> None:
        pass

    async def llm_call(
        self, model: str, prompt_tokens: int, completion_tokens: int, latency_ms: float
    ) -> None:
        pass

    async def chunk_extracted(self, doc_id: str, chunks: int) -> None:
        pass


class _CallbackDispatcher:
    """Dispatches events to all registered callbacks, swallowing errors."""

    def __init__(self, callbacks: list[Callback]):
        self._extraction: list[ExtractionCallback] = []
        self._query: list[QueryCallback] = []
        self._llm: list[LLMCallback] = []
        for cb in callbacks:
            if isinstance(cb, ExtractionCallback):
                self._extraction.append(cb)
            if isinstance(cb, QueryCallback):
                self._query.append(cb)
            if isinstance(cb, LLMCallback):
                self._llm.append(cb)

    async def _safe_call(self, callbacks, method_name, *args, **kwargs):
        for cb in callbacks:
            try:
                await getattr(cb, method_name)(*args, **kwargs)
            except Exception:
                from ._utils import logger

                logger.warning(
                    "callback_error",
                    callback=type(cb).__name__,
                    method=method_name,
                    exc_info=True,
                )

    async def extraction_start(self, total_docs: int) -> None:
        await self._safe_call(self._extraction, "on_start", total_docs)

    async def extraction_progress(self, committed: int, total: int) -> None:
        await self._safe_call(self._extraction, "on_doc_progress", committed, total)

    async def chunk_extracted(self, doc_id: str, chunks: int) -> None:
        await self._safe_call(self._extraction, "on_chunk_extracted", doc_id, chunks)

    async def community_report(self, level: int, count: int) -> None:
        await self._safe_call(self._extraction, "on_community_report", level, count)

    async def extraction_complete(self, result: InsertResult) -> None:
        await self._safe_call(self._extraction, "on_complete", result)

    async def extraction_error(self, error: Exception) -> None:
        await self._safe_call(self._extraction, "on_error", error)

    async def query_start(self, query: str, mode: str) -> None:
        await self._safe_call(self._query, "on_start", query, mode)

    async def query_sources_found(self, sources: list) -> None:
        await self._safe_call(self._query, "on_sources_found", sources)

    async def query_complete(self, result: QueryResult) -> None:
        await self._safe_call(self._query, "on_complete", result)

    async def llm_call(
        self, model: str, prompt_tokens: int, completion_tokens: int, latency_ms: float
    ) -> None:
        await self._safe_call(
            self._llm, "on_call", model, prompt_tokens, completion_tokens, latency_ms
        )
