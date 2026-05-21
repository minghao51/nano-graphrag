from __future__ import annotations

import time
from hashlib import sha256

from ._entity_grounded_query import EntityGroundedQuery
from ._exceptions import ModeNotEnabledError, QueryError
from ._ops.query import (
    global_query,
    global_query_stream,
    local_query,
    local_query_stream,
    naive_query,
    naive_query_stream,
)
from ._schemas import QueryResult, QuerySource, StreamComplete, StreamSourceRef, StreamTextChunk
from ._utils import bind_run_context, logger


def _check_mode_permissions(mode: str, enable_local: bool, enable_naive_rag: bool):
    if mode == "local" and not enable_local:
        raise ModeNotEnabledError(
            "enable_local is False, cannot query in local mode",
            details={"mode": mode, "enable_local": enable_local},
        )
    if mode == "naive" and not enable_naive_rag:
        raise ModeNotEnabledError(
            "enable_naive_rag is False, cannot query in naive mode",
            details={"mode": mode, "enable_naive_rag": enable_naive_rag},
        )
    if mode == "entity_grounded" and not enable_local:
        raise ModeNotEnabledError(
            "enable_local is False, cannot query in entity_grounded mode",
            details={"mode": mode, "enable_local": enable_local},
        )


def _build_entity_grounded_query(self, param, runtime):
    entity_query = EntityGroundedQuery(
        entity_registry=self.entity_registry,
        graph_store=self.chunk_entity_relation_graph,
        entities_vdb=self.entities_vdb,
        llm_func=lambda p: runtime["cheap_model_func"](
            p, max_tokens=param.entity_grounded_max_answer_length
        ),
        llm_stream_func=lambda p: runtime["cheap_model_stream_func"](
            p, max_tokens=param.entity_grounded_max_answer_length
        ),
    )
    entity_query.max_answer_length = param.entity_grounded_max_answer_length
    entity_query.require_entity_match = param.entity_grounded_require_entity_match
    entity_query.fuzzy_match_threshold = param.entity_grounded_fuzzy_threshold
    return entity_query


def _query_log_payload(query: str, runtime: dict) -> dict:
    if runtime.get("log_query_text", False):
        return {"query": query[:100] if query else ""}
    query_hash = sha256((query or "").encode("utf-8")).hexdigest()[:16]
    return {"query_hash": query_hash, "query_chars": len(query or "")}


async def _collect_feedback_context(self, query: str):
    context_chunks = []
    doc_ids_retrieved = []
    if self.chunks_vdb is None:
        return None, []
    chunk_results = await self.chunks_vdb.query(query, top_k=20)
    chunk_ids = [r["id"] for r in chunk_results]
    if not chunk_ids:
        return None, []
    chunk_datas = await self.text_chunks.get_by_ids(chunk_ids)
    for c in chunk_datas:
        if c is None:
            continue
        content = c.get("content", "")
        if content:
            context_chunks.append(content)
        fid = c.get("full_doc_id")
        if fid:
            doc_ids_retrieved.append(fid)
    if not context_chunks:
        return None, doc_ids_retrieved
    return "\n\n".join(context_chunks), doc_ids_retrieved


async def aquery(self, query, param):
    bind_run_context(mode=param.mode)
    _check_mode_permissions(param.mode, self.enable_local, self.enable_naive_rag)
    runtime = self._runtime_config()
    logger.info("query_start", **_query_log_payload(query, runtime))

    await self._callback_dispatcher.query_start(query, param.mode)
    await self._callback_dispatcher.query_sources_found([])

    start_time = time.monotonic()
    response = ""
    metadata = {}
    try:
        if param.mode == "local":
            response = await local_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.community_reports,
                self.text_chunks,
                param,
                self.tokenizer_wrapper,
                runtime,
            )
        elif param.mode == "global":
            response = await global_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.community_reports,
                self.text_chunks,
                param,
                self.tokenizer_wrapper,
                runtime,
            )
        elif param.mode == "naive":
            response = await naive_query(
                query,
                self.chunks_vdb,
                self.text_chunks,
                param,
                self.tokenizer_wrapper,
                runtime,
            )
        elif param.mode == "entity_grounded":
            entity_query = _build_entity_grounded_query(self, param, runtime)
            result = await entity_query.query(query, top_k=param.top_k, mode="local")
            response = result.answer
            metadata = result.metadata or {}
            entity_ids = result.metadata.get("entity_ids", [])
            await self._callback_dispatcher.query_sources_found(
                [QuerySource(source_type="entity", id=eid) for eid in entity_ids]
            )
        else:
            raise QueryError(f"Unknown mode {param.mode}", details={"mode": param.mode})
        elapsed = (time.monotonic() - start_time) * 1000
        logger.info("query_complete", latency_ms=round(elapsed, 1), answer_chars=len(response))

        result = QueryResult(
            answer=response,
            mode=param.mode,
            latency_ms=round(elapsed, 1),
            metadata=metadata,
        )

        await self._callback_dispatcher.query_complete(result)

        if self.enable_retrieval_feedback and param.mode in ("local", "naive"):
            try:
                from ._ops.retrieval_feedback import (
                    compute_retrieval_feedback,
                    log_retrieval_feedback,
                )

                context_text, doc_ids_retrieved = await _collect_feedback_context(self, query)
                if not context_text:
                    logger.debug("retrieval_feedback_skipped", reason="no_retrieved_context")
                    return result
                feedback = await compute_retrieval_feedback(
                    query=query,
                    context_text=context_text,
                    global_config=runtime,
                    doc_ids_retrieved=doc_ids_retrieved,
                )
                await log_retrieval_feedback(feedback, self.document_index, runtime)
            except Exception:
                logger.debug("retrieval_feedback_failed", exc_info=True)

        return result
    finally:
        await self._query_done()


async def astream_query(self, query, param):
    bind_run_context(mode=param.mode)
    _check_mode_permissions(param.mode, self.enable_local, self.enable_naive_rag)
    runtime = self._runtime_config()
    logger.info("query_start", **_query_log_payload(query, runtime))
    start_time = time.monotonic()

    await self._callback_dispatcher.query_start(query, param.mode)
    await self._callback_dispatcher.query_sources_found([])

    if param.mode == "local":
        stream = local_query_stream(
            query,
            self.chunk_entity_relation_graph,
            self.entities_vdb,
            self.community_reports,
            self.text_chunks,
            param,
            self.tokenizer_wrapper,
            runtime,
        )
    elif param.mode == "global":
        stream = global_query_stream(
            query,
            self.chunk_entity_relation_graph,
            self.entities_vdb,
            self.community_reports,
            self.text_chunks,
            param,
            self.tokenizer_wrapper,
            runtime,
        )
    elif param.mode == "naive":
        stream = naive_query_stream(
            query,
            self.chunks_vdb,
            self.text_chunks,
            param,
            self.tokenizer_wrapper,
            runtime,
        )
    elif param.mode == "entity_grounded":
        entity_query = _build_entity_grounded_query(self, param, runtime)

        async def _entity_stream():
            entity_ids = await entity_query._retrieve_entities(query, param.top_k, "local")
            if not entity_ids:
                yield entity_query.fallback_message
                return
            sources = [QuerySource(source_type="entity", id=eid) for eid in entity_ids]
            await self._callback_dispatcher.query_sources_found(sources)
            yield StreamSourceRef(sources=sources)
            entity_context = await entity_query._build_entity_context(entity_ids)
            async for chunk in entity_query.generate_answer_stream(query, entity_context):
                yield chunk

        stream = _entity_stream()
    else:
        raise QueryError(f"Unknown mode {param.mode}", details={"mode": param.mode})

    try:
        async for chunk in stream:
            if isinstance(chunk, str):
                yield StreamTextChunk(text=chunk)
            else:
                yield chunk
    finally:
        elapsed = (time.monotonic() - start_time) * 1000
        logger.info("query_stream_complete", latency_ms=round(elapsed, 1))
        yield StreamComplete(latency_ms=round(elapsed, 1))
        await self._query_done()


async def _query_done(self):
    if self.llm_response_cache is not None:
        await self.llm_response_cache.index_done_callback()
