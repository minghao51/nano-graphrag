from ._ops import global_query, local_query, naive_query


async def aquery(self, query, param):
    if param.mode == "local" and not self.enable_local:
        raise ValueError("enable_local is False, cannot query in local mode")
    if param.mode == "naive" and not self.enable_naive_rag:
        raise ValueError("enable_naive_rag is False, cannot query in naive mode")

    runtime = self._runtime_config()
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
    else:
        raise ValueError(f"Unknown mode {param.mode}")
    await self._query_done()
    return response


async def _query_done(self):
    if self.llm_response_cache is not None:
        await self.llm_response_cache.index_done_callback()
