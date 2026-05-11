from __future__ import annotations

import time

from ._entity_grounded_query import EntityGroundedQuery
from ._ops.query import (
    global_query,
    global_query_stream,
    local_query,
    local_query_stream,
    naive_query,
    naive_query_stream,
)
from ._utils import bind_run_context, logger


def _check_mode_permissions(mode: str, enable_local: bool, enable_naive_rag: bool):
    if mode == "local" and not enable_local:
        raise ValueError("enable_local is False, cannot query in local mode")
    if mode == "naive" and not enable_naive_rag:
        raise ValueError("enable_naive_rag is False, cannot query in naive mode")
    if mode == "entity_grounded" and not enable_local:
        raise ValueError("enable_local is False, cannot query in entity_grounded mode")


async def aquery(self, query, param):
    bind_run_context(mode=param.mode)
    _check_mode_permissions(param.mode, self.enable_local, self.enable_naive_rag)
    runtime = self._runtime_config()
    logger.info("query_start", query=query[:100] if query else "")

    start_time = time.monotonic()

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
        entity_query = EntityGroundedQuery(
            entity_registry=self.entity_registry,
            graph_store=self.chunk_entity_relation_graph,
            entities_vdb=self.entities_vdb,
            llm_func=lambda p: runtime["cheap_model_func"](
                p, max_tokens=param.entity_grounded_max_answer_length
            ),
        )
        entity_query.max_answer_length = param.entity_grounded_max_answer_length
        entity_query.require_entity_match = param.entity_grounded_require_entity_match
        entity_query.fuzzy_match_threshold = param.entity_grounded_fuzzy_threshold
        result = await entity_query.query(query, top_k=param.top_k, mode="local")
        response = result.answer
    else:
        raise ValueError(f"Unknown mode {param.mode}")
    await self._query_done()
    elapsed = (time.monotonic() - start_time) * 1000
    logger.info("query_complete", latency_ms=round(elapsed, 1), answer_chars=len(response))
    return response


async def astream_query(self, query, param):
    bind_run_context(mode=param.mode)
    _check_mode_permissions(param.mode, self.enable_local, self.enable_naive_rag)
    runtime = self._runtime_config()
    logger.info("query_start", query=query[:100] if query else "")
    start_time = time.monotonic()

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

        async def _entity_stream():
            entity_ids = await entity_query._retrieve_entities(query, param.top_k, "local")
            if not entity_ids:
                yield entity_query.fallback_message
                return
            entity_context = await entity_query._build_entity_context(entity_ids)
            async for chunk in entity_query.generate_answer_stream(query, entity_context):
                yield chunk

        stream = _entity_stream()
    else:
        raise ValueError(f"Unknown mode {param.mode}")

    try:
        async for chunk in stream:
            yield chunk
    finally:
        elapsed = (time.monotonic() - start_time) * 1000
        logger.info("query_stream_complete", latency_ms=round(elapsed, 1))
        await self._query_done()


async def _query_done(self):
    if self.llm_response_cache is not None:
        await self.llm_response_cache.index_done_callback()
