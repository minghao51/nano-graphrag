import asyncio
import os
from hashlib import sha256

from ._ops import (
    extract_document_entity_relationships,
    extract_entities,
    generate_community_report,
    get_chunks,
    rebuild_knowledge_graph_for_documents,
)
from ._utils import compute_sha256_id, logger

BUILTIN_EXTRACTORS = (extract_entities,)


def _compute_extraction_hash(global_config: dict, extraction_func) -> str:
    parts = [
        getattr(extraction_func, "__name__", str(extraction_func)),
        global_config.get("llm_model", ""),
        global_config.get("extraction_backend", ""),
        global_config.get("entity_extraction_quality", ""),
    ]
    return sha256("|".join(parts).encode()).hexdigest()


def _is_builtin_extractor(func) -> bool:
    if func in BUILTIN_EXTRACTORS:
        return True
    if hasattr(func, "__module__") and "nano_graphrag" in func.__module__:
        return True
    return False


async def _flush_doc_progress(
    self,
    doc_id: str,
    doc: dict,
    manifest: dict,
    all_chunks: dict,
    all_docs: dict,
):
    """Flush a single doc's data to storage for progress visibility."""
    doc_chunks = {cid: c for cid, c in all_chunks.items() if c.get("full_doc_id") == doc_id}
    if doc_chunks:
        await self.text_chunks.upsert(doc_chunks)
        if self.enable_naive_rag:
            await self.chunks_vdb.upsert(doc_chunks)
    await self.full_docs.upsert({doc_id: all_docs[doc_id]})
    await self.document_index.upsert({doc_id: manifest})


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


async def _ainsert_documents(
    self, documents: dict[str, str], allow_legacy_custom: bool, force_rebuild: bool = False
):
    if not _is_builtin_extractor(self.entity_extraction_func):
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
        if force_rebuild:
            docs_to_process = normalized_docs
            changed_doc_ids = [
                doc_id
                for doc_id, existing_doc in zip(normalized_docs.keys(), existing_docs)
                if existing_doc is not None
            ]
        else:
            for doc_id, new_doc, existing_doc in zip(
                normalized_docs.keys(), normalized_docs.values(), existing_docs
            ):
                if (
                    existing_doc is None
                    or existing_doc.get("content_hash") != new_doc["content_hash"]
                ):
                    docs_to_process[doc_id] = new_doc
                    if existing_doc is not None:
                        changed_doc_ids.append(doc_id)

            # Check if extraction config changed for unchanged existing docs
            current_hash = _compute_extraction_hash(
                self._runtime_config(), self.entity_extraction_func
            )
            unchanged_ids = [d for d in normalized_docs if d not in docs_to_process]
            if unchanged_ids:
                existing_manifests = await self.document_index.get_by_ids(unchanged_ids)
                for doc_id, manifest in zip(unchanged_ids, existing_manifests):
                    if manifest is not None and manifest.get("extraction_hash") != current_hash:
                        docs_to_process[doc_id] = normalized_docs[doc_id]
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

        runtime_config = self._runtime_config()
        extraction_hash = _compute_extraction_hash(runtime_config, self.entity_extraction_func)

        async def _extract_doc(doc_id: str, doc: dict) -> tuple[str, dict]:
            chunk_subset = chunks_by_doc.get(doc_id, {})
            if self.entity_extraction_func is extract_entities:
                manifest = await extract_document_entity_relationships(
                    chunk_subset,
                    self.tokenizer_wrapper,
                    runtime_config,
                )
            elif self.entity_extraction_func.__name__ == "extract_entities_gliner":
                from ._ops.extraction_gliner import extract_document_entity_relationships_gliner

                manifest = await extract_document_entity_relationships_gliner(
                    chunk_subset,
                    self.tokenizer_wrapper,
                    runtime_config,
                )
            else:
                manifest = await self.entity_extraction_func(
                    chunk_subset,
                    self.chunk_entity_relation_graph,
                    self.entities_vdb,
                    self.tokenizer_wrapper,
                    runtime_config,
                )
            manifest["content_hash"] = doc["content_hash"]
            manifest["extraction_hash"] = extraction_hash
            return doc_id, manifest

        max_doc_concurrency = runtime_config.get("doc_extraction_max_async", 4)
        doc_semaphore = asyncio.Semaphore(max_doc_concurrency)
        doc_write_lock = asyncio.Lock()
        flush_batch_size = runtime_config.get("doc_flush_batch_size", 50)
        _completed_count = 0

        async def _extract_doc_limited(doc_id: str, doc: dict) -> tuple[str, dict]:
            nonlocal _completed_count
            async with doc_semaphore:
                doc_id, manifest = await _extract_doc(doc_id, doc)
            _completed_count += 1
            if _completed_count % flush_batch_size == 0:
                async with doc_write_lock:
                    await self._flush_doc_progress(
                        doc_id,
                        doc,
                        manifest,
                        inserting_chunks,
                        normalized_docs,
                    )
                    logger.info(
                        f"[Extraction] {_completed_count}/{len(docs_to_process)} docs "
                        f"committed ({_completed_count * 100 // len(docs_to_process)}%)"
                    )
            return doc_id, manifest

        doc_items = list(docs_to_process.items())
        logger.info(
            f"[Extraction] processing {len(doc_items)} docs "
            f"(concurrency={max_doc_concurrency}, flush_every={flush_batch_size})"
        )
        results = await asyncio.gather(
            *[_extract_doc_limited(doc_id, doc) for doc_id, doc in doc_items]
        )

        new_document_index_entries = {}
        docs_with_entities = 0
        for doc_id, manifest in results:
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
        snapshot_path = None
        if self.chunk_entity_relation_graph is not None:
            snapshot_path = await self.chunk_entity_relation_graph._snapshot_graph()
        try:
            self.chunk_entity_relation_graph = await rebuild_knowledge_graph_for_documents(
                self.document_index,
                self.graph_contribution_index,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.tokenizer_wrapper,
                self._runtime_config(),
                old_manifest_lookup,
                new_document_index_entries,
            )
            affected_entity_ids = {
                entity_id
                for manifest in list(old_manifest_lookup.values())
                + list(new_document_index_entries.values())
                if manifest is not None
                for entity_id in manifest.get("entities", {}).keys()
            }

            has_graph_data = bool(affected_entity_ids) or docs_with_entities > 0
            if has_graph_data:
                if self.enable_community_reports:
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
                else:
                    logger.info("[Skipping Community Report] - enable_community_reports=False")
                    await self.chunk_entity_relation_graph.clustering(
                        self.graph_cluster_algorithm,
                        affected_node_ids=affected_entity_ids,
                    )
            elif docs_with_entities == 0:
                await self.community_reports.drop()
                logger.warning("No entities found in processed documents")

            # Integrity check: verify graph has expected nodes for new manifests
            manifest_entity_count = sum(
                len(m.get("entities", {})) for m in new_document_index_entries.values()
            )
            if manifest_entity_count > 0 and self.chunk_entity_relation_graph is not None:
                entity_ids_to_check = [
                    eid
                    for m in new_document_index_entries.values()
                    for eid in m.get("entities", {})
                ]
                results = await asyncio.gather(
                    *[self.chunk_entity_relation_graph.has_node(eid) for eid in entity_ids_to_check]
                )
                found = sum(1 for r in results if r)
                if found == 0:
                    logger.warning(
                        f"Integrity check: {manifest_entity_count} entities in manifests "
                        f"but 0 found in graph. Graph rebuild may have silently failed. "
                        f"Consider re-running with force_rebuild=True."
                    )
                elif found < manifest_entity_count:
                    logger.info(
                        f"Integrity check: {found}/{manifest_entity_count} "
                        f"manifest entities found in graph."
                    )
            # Clean up snapshot on success
            if snapshot_path and os.path.exists(snapshot_path):
                os.unlink(snapshot_path)
        except Exception as e:
            if snapshot_path and self.chunk_entity_relation_graph is not None:
                await self.chunk_entity_relation_graph._restore_graph(snapshot_path)
            new_doc_ids = list(new_document_index_entries.keys())
            if new_doc_ids:
                await self.document_index.delete(new_doc_ids)
                logger.warning(
                    f"Rebuild failed, rolled back {len(new_doc_ids)} document manifests. "
                    f"These documents will be re-extracted on the next insert run."
                )
            raise e
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

    if self.entity_registry is not None:
        entity_registry_path = os.path.join(self.working_dir, "entity_registry.json")
        self.entity_registry.save_to_file(entity_registry_path)


async def _rebuild_graph_from_manifests(self):
    """Rebuild entire knowledge graph from existing document manifests without re-extraction."""
    await self._insert_start()
    try:
        all_doc_keys = await self.document_index.all_keys()
        if not all_doc_keys:
            logger.warning("No documents found in document_index, nothing to rebuild")
            return
        all_manifests = await self.document_index.get_by_ids(all_doc_keys)
        manifest_dict = {
            doc_id: manifest
            for doc_id, manifest in zip(all_doc_keys, all_manifests)
            if manifest is not None
        }
        if not manifest_dict:
            logger.warning("No valid manifests found, nothing to rebuild")
            return

        logger.info(f"[Graph Rebuild] rebuilding from {len(manifest_dict)} manifests")
        snapshot_path = None
        if self.chunk_entity_relation_graph is not None:
            snapshot_path = await self.chunk_entity_relation_graph._snapshot_graph()
        try:
            self.chunk_entity_relation_graph = await rebuild_knowledge_graph_for_documents(
                self.document_index,
                self.graph_contribution_index,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.tokenizer_wrapper,
                self._runtime_config(),
                {},  # No old manifests — treat all as new
                manifest_dict,
            )
            if self.enable_community_reports:
                logger.info("[Community Report] regenerating after rebuild...")
                await self.chunk_entity_relation_graph.clustering(
                    self.graph_cluster_algorithm,
                    affected_node_ids=None,
                )
                await generate_community_report(
                    self.community_reports,
                    self.chunk_entity_relation_graph,
                    self.tokenizer_wrapper,
                    self._runtime_config(),
                )
            else:
                await self.chunk_entity_relation_graph.clustering(
                    self.graph_cluster_algorithm,
                    affected_node_ids=None,
                )
            # Clean up snapshot on success
            if snapshot_path and os.path.exists(snapshot_path):
                os.unlink(snapshot_path)
        except Exception as e:
            if snapshot_path and self.chunk_entity_relation_graph is not None:
                await self.chunk_entity_relation_graph._restore_graph(snapshot_path)
            raise e
    finally:
        await self._insert_done()
