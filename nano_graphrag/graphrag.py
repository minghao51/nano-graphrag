import asyncio
import os
import warnings
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Type, Union

from ._ops import (
    chunking_by_token_size,
    extract_document_entity_relationships,
    extract_entities,
    generate_community_report,
    get_chunks,
    global_query,
    local_query,
    naive_query,
    rebuild_knowledge_graph_for_documents,
)
from ._storage import (
    HNSWVectorStorage,
    JsonKVStorage,
    NetworkXStorage,
)
from ._utils import (
    EmbeddingFunc,
    TokenizerWrapper,
    always_get_an_event_loop,
    compute_mdhash_id,
    compute_sha256_id,
    convert_response_to_json,
    limit_async_func_call,
    logger,
)
from .base import (
    DEFAULT_CHEAP_MODEL,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_MODEL,
    SUPPORTED_GRAPH_CLUSTERING,
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    GraphRAGConfig,
    QueryParam,
)


@dataclass
class GraphRAG:
    # === Core ===
    working_dir: str = field(
        default_factory=lambda: (
            f"./nano_graphrag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
        )
    )
    log_level: str = "INFO"
    log_file: Optional[str] = None

    # === Feature Flags ===
    enable_local: bool = True
    enable_naive_rag: bool = False

    # === Text Chunking ===
    tokenizer_type: str = "tiktoken"
    tiktoken_model_name: str = "gpt-4o"
    huggingface_model_name: str = "bert-base-uncased"
    chunk_func: Callable[
        [
            list[list[int]],
            List[str],
            TokenizerWrapper,
            Optional[int],
            Optional[int],
        ],
        List[Dict[str, Union[str, int]]],
    ] = chunking_by_token_size
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100

    # === Entity Extraction ===
    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_tokens: int = 500
    extraction_max_async: int = 16

    # === Graph Clustering ===
    graph_cluster_algorithm: str = "leiden"
    max_graph_cluster_size: int = 10
    graph_cluster_seed: int = 0xDEADBEEF
    leiden_resolutions: list = field(default_factory=lambda: [2.0, 1.0, 0.5])

    # === Node Embedding ===
    node_embedding_algorithm: str = "node2vec"
    node2vec_params: dict = field(
        default_factory=lambda: {
            "dimensions": 1536,
            "num_walks": 10,
            "walk_length": 40,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
        }
    )
    enable_node_embedding: bool = False

    # === Community Reports ===
    special_community_report_llm_kwargs: dict = field(
        default_factory=lambda: {"response_format": {"type": "json_object"}}
    )

    # === Embedding ===
    embedding_func: Optional[EmbeddingFunc] = None
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_api_base: Optional[str] = None
    embedding_api_key: Optional[str] = None
    embedding_dim: int = DEFAULT_EMBEDDING_DIM
    embedding_max_async: Optional[int] = None
    embedding_batch_size: Optional[int] = None
    embedding_batch_num: int = 32
    embedding_func_max_async: int = 16

    # === LLM ===
    best_model_func: Optional[Callable[..., Any]] = None
    best_model_max_token_size: int = 32768
    best_model_max_async: int = 16
    cheap_model_func: Optional[Callable[..., Any]] = None
    cheap_model_max_token_size: int = 32768
    cheap_model_max_async: int = 16

    # === LiteLLM Runtime ===
    llm_model: str = DEFAULT_LLM_MODEL
    llm_cheap_model: str = DEFAULT_CHEAP_MODEL
    llm_api_base: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_max_async: Optional[int] = None
    llm_max_tokens: int = 32768
    llm_timeout: int = 120

    structured_output: bool = True
    use_pydantic_structured_output: bool = True
    fallback_to_parsing: bool = True

    # === Entity Extraction ===
    entity_extraction_func: Callable[..., Any] = extract_entities
    entity_extraction_quality: str = "balanced"

    # === Storage ===
    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage
    vector_db_storage_cls: Type[BaseVectorStorage] = HNSWVectorStorage
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)
    graph_storage_cls: Type[BaseGraphStorage] = NetworkXStorage
    enable_llm_cache: bool = True

    # === Extension ===
    always_create_working_dir: bool = True
    addon_params: dict = field(default_factory=dict)
    convert_response_to_json_func: Callable[..., Any] = convert_response_to_json

    @classmethod
    def from_config(cls, config: GraphRAGConfig) -> "GraphRAG":
        """Create GraphRAG from GraphRAGConfig object."""
        config_dict = config.to_dict()
        valid_fields = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in config_dict.items() if k in valid_fields})

    def __post_init__(self):
        self._normalize_settings()
        self._configure_logging()
        self._build_tokenizer()
        self._configure_runtime()
        self._build_storages()

    def _normalize_settings(self):
        if self.graph_cluster_algorithm not in SUPPORTED_GRAPH_CLUSTERING:
            raise ValueError(
                f"Unsupported graph_cluster_algorithm={self.graph_cluster_algorithm!r}. "
                f"Supported: {', '.join(SUPPORTED_GRAPH_CLUSTERING)}"
            )

        if self.embedding_batch_size is not None:
            self.embedding_batch_num = self.embedding_batch_size
        elif self.embedding_batch_num != 32:
            warnings.warn(
                "`embedding_batch_num` is deprecated; use `embedding_batch_size`.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.embedding_batch_size = self.embedding_batch_num
        else:
            self.embedding_batch_size = self.embedding_batch_num

        if self.llm_max_async is not None:
            self.best_model_max_async = self.llm_max_async
            self.cheap_model_max_async = self.llm_max_async
        if self.embedding_max_async is not None:
            self.embedding_func_max_async = self.embedding_max_async

    def _configure_logging(self):
        import logging

        numeric_level = getattr(logging, self.log_level.upper(), logging.INFO)
        logger.setLevel(numeric_level)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        if not any(getattr(h, "_nano_graphrag_console", False) for h in logger.handlers):
            console_handler = logging.StreamHandler()
            console_handler.setLevel(numeric_level)
            console_handler.setFormatter(formatter)
            console_handler._nano_graphrag_console = True
            logger.addHandler(console_handler)

        if self.log_file:
            if not any(
                getattr(h, "_nano_graphrag_file", None) == self.log_file for h in logger.handlers
            ):
                file_handler = logging.FileHandler(self.log_file)
                file_handler.setLevel(numeric_level)
                file_handler.setFormatter(formatter)
                file_handler._nano_graphrag_file = self.log_file
                logger.addHandler(file_handler)

        params_str = ",\n  ".join(f"{k} = {v}" for k, v in asdict(self).items())
        logger.debug(f"GraphRAG init with param:\n  {params_str}")

    def _build_tokenizer(self):
        model_name = (
            self.tiktoken_model_name
            if self.tokenizer_type == "tiktoken"
            else self.huggingface_model_name
        )
        self.tokenizer_wrapper = TokenizerWrapper(
            tokenizer_type=self.tokenizer_type,
            model_name=model_name,
        )

    def _configure_runtime(self):
        from ._llm_litellm import (
            LiteLLMWrapper,
            litellm_embedding,
            supports_structured_output,
        )

        if not os.path.exists(self.working_dir) and self.always_create_working_dir:
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache",
                global_config=asdict(self),
            )
            if self.enable_llm_cache
            else None
        )

        # If custom model/embedding funcs are provided, use them directly
        use_custom_models = self.best_model_func is not None or self.cheap_model_func is not None
        use_custom_embedding = self.embedding_func is not None

        if use_custom_models or use_custom_embedding:
            # Wrap custom functions with async limiter if needed
            if self.best_model_func is not None:
                self.best_model_func = limit_async_func_call(self.best_model_max_async)(
                    self.best_model_func
                )
            if self.cheap_model_func is not None:
                self.cheap_model_func = limit_async_func_call(self.cheap_model_max_async)(
                    self.cheap_model_func
                )
            if self.embedding_func is not None:
                if not isinstance(self.embedding_func, EmbeddingFunc):
                    self.embedding_func = EmbeddingFunc(
                        embedding_dim=self.embedding_dim,
                        max_token_size=8192,
                        func=limit_async_func_call(self.embedding_func_max_async)(
                            self.embedding_func
                        ),
                    )
            self._use_structured_extraction = False
            return

        structured = self.structured_output and supports_structured_output(self.llm_model)
        if not structured:
            logger.info(
                f"Model {self.llm_model} does not support structured output, "
                "will use text mode with fallback parsing"
            )

        self.best_model_func = limit_async_func_call(self.best_model_max_async)(
            LiteLLMWrapper(
                model=self.llm_model,
                structured_output=self.structured_output,
                use_native_structured_output=self.use_pydantic_structured_output,
                hashing_kv=self.llm_response_cache,
                api_base=self.llm_api_base,
                api_key=self.llm_api_key,
                timeout=self.llm_timeout,
            )
        )
        self.cheap_model_func = limit_async_func_call(self.cheap_model_max_async)(
            LiteLLMWrapper(
                model=self.llm_cheap_model,
                structured_output=self.structured_output,
                use_native_structured_output=self.use_pydantic_structured_output,
                hashing_kv=self.llm_response_cache,
                api_base=self.llm_api_base,
                api_key=self.llm_api_key,
                timeout=self.llm_timeout,
            )
        )

        limited_embedding = limit_async_func_call(self.embedding_func_max_async)(
            partial(
                litellm_embedding,
                model=self.embedding_model,
                api_base=self.embedding_api_base,
                api_key=self.embedding_api_key,
            )
        )
        self.embedding_func = EmbeddingFunc(
            embedding_dim=self.embedding_dim,
            max_token_size=8192,
            func=limited_embedding,
        )

        logger.info(
            f"Using LiteLLM: model={self.llm_model}, "
            f"api_base={self.llm_api_base}, "
            f"structured_output={self.structured_output}"
        )

        self._use_structured_extraction = True

    def _build_storages(self):
        self.full_docs = self.key_string_value_json_storage_cls(
            namespace="full_docs", global_config=asdict(self)
        )
        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks", global_config=asdict(self)
        )
        self.community_reports = self.key_string_value_json_storage_cls(
            namespace="community_reports", global_config=asdict(self)
        )
        self.document_index = self.key_string_value_json_storage_cls(
            namespace="document_index", global_config=asdict(self)
        )
        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="chunk_entity_relation", global_config=asdict(self)
        )

        self.entities_vdb = (
            self.vector_db_storage_cls(
                namespace="entities",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
                meta_fields={"entity_name"},
            )
            if self.enable_local
            else None
        )
        self.chunks_vdb = (
            self.vector_db_storage_cls(
                namespace="chunks",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
            )
            if self.enable_naive_rag
            else None
        )

    def _runtime_config(self) -> dict:
        runtime_config = asdict(self)
        runtime_config["_use_structured_extraction"] = self._use_structured_extraction
        return runtime_config

    def insert(self, string_or_strings):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert(string_or_strings))

    def insert_documents(self, documents: dict[str, str]):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert_documents(documents))

    def query(self, query: str, param: QueryParam = QueryParam()):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, param))

    async def aquery(self, query: str, param: QueryParam = QueryParam()):
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

    async def ainsert(self, string_or_strings):
        if isinstance(string_or_strings, str):
            string_or_strings = [string_or_strings]
        normalized = [c.strip() for c in string_or_strings if c.strip()]
        doc_ids = [compute_mdhash_id(c, prefix="doc-") for c in normalized]
        existing = await self.full_docs.get_by_ids(doc_ids)
        documents = {
            (doc_id if doc else compute_sha256_id(c, prefix="doc-")): c
            for c, doc_id, doc in zip(normalized, doc_ids, existing)
        }
        return await self._ainsert_documents(documents, allow_legacy_custom=True)

    async def ainsert_documents(self, documents: dict[str, str]):
        return await self._ainsert_documents(documents, allow_legacy_custom=False)

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
                if (
                    existing_doc is None
                    or existing_doc.get("content_hash") != new_doc["content_hash"]
                ):
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
                for manifest in list(old_manifest_lookup.values())
                + list(new_document_index_entries.values())
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

    async def _query_done(self):
        if self.llm_response_cache is not None:
            await self.llm_response_cache.index_done_callback()
