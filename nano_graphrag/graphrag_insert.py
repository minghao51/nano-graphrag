import asyncio

from ._ops import (
    extract_document_entity_relationships,
    extract_entities,
    generate_community_report,
    get_chunks,
    rebuild_knowledge_graph_for_documents,
)
from ._utils import compute_sha256_id, logger


async def _legacy_custom_ainsert(self, documents: dict[str, str]):
    await self._insert_start()
    try:
        new_docs = {
            doc_id: {
                "content": content.strip(),
                "content_hash": compute_sha256_id(content.strip()),
            }
            for doc_id, content in documents.items()
            if content.strip()
        }
        _add_doc_keys = await self.full_docs.filter_keys(list(new_docs.keys()))
        new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
        if not new_docs:
            logger.warning("All docs are already in the storage")
            return
        logger.info(f"[New Docs] inserting {len(new_docs)} docs")

        inserting_chunks = get_chunks(
            new_docs=new_docs,
            chunk_func=self.chunk_func,
            overlap_token_size=self.chunk_overlap_token_size,
            max_token_size=self.chunk_token_size,
            tokenizer_wrapper=self.tokenizer_wrapper,
        )

        _add_chunk_keys = await self.text_chunks.filter_keys(list(inserting_chunks.keys()))
        inserting_chunks = {k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys}
        if not inserting_chunks:
            logger.warning("All chunks are already in the storage")
            return
        logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")
        if self.enable_naive_rag:
            await self.chunks_vdb.upsert(inserting_chunks)

        logger.info("[Entity Extraction]...")
        maybe_new_kg = await self.entity_extraction_func(
            inserting_chunks,
            knowledge_graph_inst=self.chunk_entity_relation_graph,
            entity_vdb=self.entities_vdb,
            tokenizer_wrapper=self.tokenizer_wrapper,
            global_config=self._runtime_config(),
        )
        if maybe_new_kg is None:
            logger.warning("No new entities found")
            return
        self.chunk_entity_relation_graph = maybe_new_kg

        logger.info("[Community Report]...")
        await self.community_reports.drop()
        await self.chunk_entity_relation_graph.clustering(
            self.graph_cluster_algorithm, affected_node_ids=None
        )
        await generate_community_report(
            self.community_reports,
            self.chunk_entity_relation_graph,
            self.tokenizer_wrapper,
            self._runtime_config(),
        )

        await self.full_docs.upsert(new_docs)
        await self.text_chunks.upsert(inserting_chunks)
    finally:
        await self._insert_done()


async def _ainsert_documents(self, documents: dict[str, str], allow_legacy_custom: bool):
    if self.entity_extraction_func is not extract_entities:
        if allow_legacy_custom:
            return await self._legacy_custom_ainsert(documents)
        raise NotImplementedError(
            "insert_documents requires the built-in extract_entities pipeline."
        )

    await self._insert_start()
    try:
        normalized_docs = {
            doc_id: {
                "content": content.strip(),
                "content_hash": compute_sha256_id(content.strip()),
            }
            for doc_id, content in documents.items()
            if content.strip()
        }
        if not normalized_docs:
            logger.warning("No valid docs to insert")
            return

        existing_docs = await self.full_docs.get_by_ids(list(normalized_docs.keys()))
        docs_to_process = {}
        changed_doc_ids = []
        for doc_id, new_doc, existing_doc in zip(
            normalized_docs.keys(), normalized_docs.values(), existing_docs
        ):
            if existing_doc is None or existing_doc.get("content_hash") != new_doc["content_hash"]:
                docs_to_process[doc_id] = new_doc
                if existing_doc is not None:
                    changed_doc_ids.append(doc_id)
        if not docs_to_process:
            logger.warning("All docs are unchanged")
            return

        logger.info(
            f"[Delta Detection] processing {len(docs_to_process)} docs ({len(changed_doc_ids)} changed)"
        )

        old_manifests = await self.document_index.get_by_ids(changed_doc_ids)
        old_manifest_lookup = {
            doc_id: manifest
            for doc_id, manifest in zip(changed_doc_ids, old_manifests)
            if manifest is not None
        }

        inserting_chunks = get_chunks(
            new_docs=docs_to_process,
            chunk_func=self.chunk_func,
            overlap_token_size=self.chunk_overlap_token_size,
            max_token_size=self.chunk_token_size,
            tokenizer_wrapper=self.tokenizer_wrapper,
        )
        chunks_by_doc: dict[str, dict[str, dict]] = {}
        for chunk_id, chunk in inserting_chunks.items():
            chunks_by_doc.setdefault(chunk["full_doc_id"], {})[chunk_id] = chunk

        new_document_index_entries = {}
        docs_with_entities = 0
        for doc_id, doc in docs_to_process.items():
            chunk_subset = chunks_by_doc.get(doc_id, {})
            manifest = await extract_document_entity_relationships(
                chunk_subset,
                self.tokenizer_wrapper,
                self._runtime_config(),
            )
            manifest["content_hash"] = doc["content_hash"]
            new_document_index_entries[doc_id] = manifest
            if manifest["entities"]:
                docs_with_entities += 1

        old_chunk_ids = sorted(
            {
                chunk_id
                for manifest in old_manifest_lookup.values()
                for chunk_id in manifest.get("chunk_ids", [])
            }
        )
        if old_chunk_ids:
            await self.text_chunks.delete(old_chunk_ids)
            if self.enable_naive_rag:
                await self.chunks_vdb.delete(old_chunk_ids)

        if inserting_chunks:
            await self.text_chunks.upsert(inserting_chunks)
            if self.enable_naive_rag:
                await self.chunks_vdb.upsert(inserting_chunks)

        await self.full_docs.upsert(normalized_docs)
        await self.document_index.upsert(new_document_index_entries)

        logger.info("[Graph Rebuild] refreshing affected entities and relationships")
        self.chunk_entity_relation_graph = await rebuild_knowledge_graph_for_documents(
            self.document_index,
            self.chunk_entity_relation_graph,
            self.entities_vdb,
            self.tokenizer_wrapper,
            self._runtime_config(),
            old_manifest_lookup,
            new_document_index_entries,
        )
        affected_entity_ids = {
            entity_id
            for manifest in list(old_manifest_lookup.values()) + list(new_document_index_entries.values())
            if manifest is not None
            for entity_id in manifest.get("entities", {}).keys()
        }

        all_doc_keys = await self.document_index.all_keys()
        all_manifests = await self.document_index.get_by_ids(all_doc_keys)
        has_graph_data = any(m and m.get("entities") for m in all_manifests)
        if has_graph_data:
            logger.info("[Community Report]...")
            await self.chunk_entity_relation_graph.clustering(
                self.graph_cluster_algorithm,
                affected_node_ids=affected_entity_ids,
            )
            affected_community_ids = getattr(
                self.chunk_entity_relation_graph, "_last_affected_community_ids", None
            )
            await generate_community_report(
                self.community_reports,
                self.chunk_entity_relation_graph,
                self.tokenizer_wrapper,
                self._runtime_config(),
                only_community_ids=affected_community_ids,
            )
        elif docs_with_entities == 0:
            await self.community_reports.drop()
            logger.warning("No entities found in processed documents")
    finally:
        await self._insert_done()


async def _insert_start(self):
    if self.chunk_entity_relation_graph is not None:
        await self.chunk_entity_relation_graph.index_start_callback()


async def _insert_done(self):
    tasks = []
    for storage_inst in [
        self.full_docs,
        self.text_chunks,
        self.document_index,
        self.llm_response_cache,
        self.community_reports,
        self.entities_vdb,
        self.chunks_vdb,
        self.chunk_entity_relation_graph,
    ]:
        if storage_inst is not None:
            tasks.append(storage_inst.index_done_callback())
    await asyncio.gather(*tasks)
