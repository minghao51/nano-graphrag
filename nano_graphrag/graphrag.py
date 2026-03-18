import asyncio
import os
import warnings
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Type, Union, cast

from ._llm import (
    amazon_bedrock_embedding,
    azure_gpt_4o_complete,
    azure_gpt_4o_mini_complete,
    azure_openai_embedding,
    create_amazon_bedrock_complete_function,
    gpt_4o_complete,
    gpt_4o_mini_complete,
    openai_embedding,
)
from ._op import (
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
    JsonKVStorage,
    NanoVectorDBStorage,
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
    StorageNameSpace,
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
    tokenizer_type: str = "tiktoken"  # or 'huggingface'
    tiktoken_model_name: str = "gpt-4o"
    huggingface_model_name: str = "bert-base-uncased"  # default HF model
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
    extraction_max_async: int = 16  # NEW: control entity extraction concurrency

    # === Graph Clustering ===
    graph_cluster_algorithm: str = "leiden"
    max_graph_cluster_size: int = 10
    graph_cluster_seed: int = 0xDEADBEEF

    # === Node Embedding (disabled by default for local) ===
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
    enable_node_embedding: bool = False  # NEW: disabled by default

    # === Community Reports ===
    special_community_report_llm_kwargs: dict = field(
        default_factory=lambda: {"response_format": {"type": "json_object"}}
    )

    # === Embedding (NEW CONFIG) ===
    embedding_func: EmbeddingFunc = field(default_factory=lambda: openai_embedding)
    embedding_model: str = DEFAULT_EMBEDDING_MODEL  # NEW
    embedding_api_base: Optional[str] = None  # NEW: for Ollama, vLLM, custom
    embedding_api_key: Optional[str] = None  # NEW
    embedding_dim: int = DEFAULT_EMBEDDING_DIM  # NEW
    embedding_max_async: Optional[int] = None
    embedding_batch_size: Optional[int] = None
    embedding_batch_num: int = 32
    embedding_func_max_async: int = 16

    # === LLM (NEW CONFIG) ===
    using_azure_openai: bool = False
    using_amazon_bedrock: bool = False
    best_model_id: str = "us.anthropic.claude-3-sonnet-20240229-v1:0"
    cheap_model_id: str = "us.anthropic.claude-3-haiku-20240307-v1:0"
    best_model_func: Callable[..., Any] = gpt_4o_complete
    best_model_max_token_size: int = 32768
    best_model_max_async: int = 16
    cheap_model_func: Callable[..., Any] = gpt_4o_mini_complete
    cheap_model_max_token_size: int = 32768
    cheap_model_max_async: int = 16

    # === LiteLLM (Default Runtime) ===
    llm_model: str = DEFAULT_LLM_MODEL  # NEW: e.g., "ollama/llama3.2", "gpt-4o-mini"
    llm_cheap_model: str = DEFAULT_CHEAP_MODEL  # NEW
    llm_api_base: Optional[str] = None  # NEW: e.g., "http://localhost:11434"
    llm_api_key: Optional[str] = None  # NEW
    llm_max_async: Optional[int] = None
    llm_max_tokens: int = 32768  # NEW
    llm_timeout: int = 120  # NEW

    structured_output: bool = True
    use_pydantic_structured_output: bool = True
    fallback_to_parsing: bool = True

    # === Entity Extraction ===
    entity_extraction_func: Callable[..., Any] = extract_entities
    entity_extraction_quality: str = "balanced"  # NEW: "fast" | "balanced" | "thorough"

    # === Storage ===
    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage
    vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBStorage
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)
    graph_storage_cls: Type[BaseGraphStorage] = NetworkXStorage
    enable_llm_cache: bool = True

    # === Extension ===
    always_create_working_dir: bool = True
    addon_params: dict = field(default_factory=dict)
    convert_response_to_json_func: Callable[..., Any] = convert_response_to_json

    @classmethod
    def from_config(cls, config: GraphRAGConfig) -> "GraphRAG":
        """Create GraphRAG from GraphRAGConfig object.

        This is the recommended way to create a GraphRAG instance.

        Example:
            config = GraphRAGConfig.from_env()
            rag = GraphRAG.from_config(config)
        """
        config_dict = config.to_dict()
        valid_fields = {field_def.name for field_def in fields(cls)}
        return cls(**{key: value for key, value in config_dict.items() if key in valid_fields})

    def __post_init__(self):
        self._normalize_runtime_settings()
        self._configure_logging()
        self._build_tokenizer()
        self._configure_runtime()
        self._build_storages()

    def _normalize_runtime_settings(self):
        if self.graph_cluster_algorithm not in SUPPORTED_GRAPH_CLUSTERING:
            supported = ", ".join(SUPPORTED_GRAPH_CLUSTERING)
            raise ValueError(
                f"Unsupported graph_cluster_algorithm={self.graph_cluster_algorithm!r}. "
                f"Supported values: {supported}"
            )

        if self.embedding_batch_size is not None:
            self.embedding_batch_num = self.embedding_batch_size
        elif self.embedding_batch_num != 32:
            warnings.warn(
                "`embedding_batch_num` is deprecated; use `embedding_batch_size` instead.",
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

        if self.using_azure_openai:
            warnings.warn(
                "`using_azure_openai` is deprecated; prefer LiteLLM with `llm_model`, "
                "`llm_api_base`, and `llm_api_key`.",
                DeprecationWarning,
                stacklevel=2,
            )
        if self.using_amazon_bedrock:
            warnings.warn(
                "`using_amazon_bedrock` is deprecated; prefer LiteLLM-compatible "
                "model configuration unless you need the legacy Bedrock path.",
                DeprecationWarning,
                stacklevel=2,
            )

    def _configure_logging(self):
        # === Configure logging ===
        import logging

        numeric_level = getattr(logging, self.log_level.upper(), logging.INFO)
        logger.setLevel(numeric_level)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        if not any(
            getattr(handler, "_nano_graphrag_console", False) for handler in logger.handlers
        ):
            console_handler = logging.StreamHandler()
            console_handler.setLevel(numeric_level)
            console_handler.setFormatter(formatter)
            console_handler._nano_graphrag_console = True
            logger.addHandler(console_handler)

        # File handler (optional)
        if self.log_file:
            existing_file_handler = any(
                getattr(handler, "_nano_graphrag_file", None) == self.log_file
                for handler in logger.handlers
            )
            if not existing_file_handler:
                file_handler = logging.FileHandler(self.log_file)
                file_handler.setLevel(numeric_level)
                file_handler.setFormatter(formatter)
                file_handler._nano_graphrag_file = self.log_file
                logger.addHandler(file_handler)

        # === Print config ===
        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self).items()])
        logger.debug(f"GraphRAG init with param:\n\n  {_print_config}\n")

    def _build_tokenizer(self):
        self.tokenizer_wrapper = TokenizerWrapper(
            tokenizer_type=self.tokenizer_type,
            model_name=self.tiktoken_model_name
            if self.tokenizer_type == "tiktoken"
            else self.huggingface_model_name,
        )

    def _configure_runtime(self):
        if self.using_azure_openai:
            # If there's no OpenAI API key, use Azure OpenAI
            if self.best_model_func == gpt_4o_complete:
                self.best_model_func = azure_gpt_4o_complete
            if self.cheap_model_func == gpt_4o_mini_complete:
                self.cheap_model_func = azure_gpt_4o_mini_complete
            if self.embedding_func == openai_embedding:
                self.embedding_func = azure_openai_embedding
            logger.info(
                "Switched the default openai funcs to Azure OpenAI if you didn't set any of it"
            )

        if self.using_amazon_bedrock:
            self.best_model_func = create_amazon_bedrock_complete_function(self.best_model_id)
            self.cheap_model_func = create_amazon_bedrock_complete_function(self.cheap_model_id)
            self.embedding_func = amazon_bedrock_embedding
            logger.info("Switched the default openai funcs to Amazon Bedrock")

        uses_custom_llm = self.best_model_func not in (
            gpt_4o_complete,
            azure_gpt_4o_complete,
        ) or self.cheap_model_func not in (gpt_4o_mini_complete, azure_gpt_4o_mini_complete)
        uses_custom_embedding = self.embedding_func not in (
            openai_embedding,
            azure_openai_embedding,
            amazon_bedrock_embedding,
        )
        use_litellm_runtime = not (
            self.using_azure_openai
            or self.using_amazon_bedrock
            or uses_custom_llm
            or uses_custom_embedding
        )
        self._use_structured_extraction = use_litellm_runtime

        if not os.path.exists(self.working_dir) and self.always_create_working_dir:
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)
        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache", global_config=asdict(self)
            )
            if self.enable_llm_cache
            else None
        )

        if use_litellm_runtime:
            try:
                from ._llm_litellm import (
                    LiteLLMWrapper,
                    litellm_embedding,
                    supports_structured_output,
                )
            except ImportError as e:
                raise ImportError(
                    "LiteLLM is not installed. Install with: pip install litellm\n"
                    f"Original error: {e}"
                ) from e

            # Determine effective model names (handle ollama/ prefix for provider detection)
            effective_best_model = self.llm_model
            effective_cheap_model = self.llm_cheap_model

            structured = self.structured_output and supports_structured_output(effective_best_model)
            if not structured:
                logger.info(
                    f"LiteLLM model {effective_best_model} does not support structured output, "
                    "will use text mode with fallback to parsing"
                )

            # Create LiteLLM wrappers with api_base and api_key support
            self.best_model_func = limit_async_func_call(self.best_model_max_async)(
                LiteLLMWrapper(
                    model=effective_best_model,
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
                    model=effective_cheap_model,
                    structured_output=self.structured_output,
                    use_native_structured_output=self.use_pydantic_structured_output,
                    hashing_kv=self.llm_response_cache,
                    api_base=self.llm_api_base,
                    api_key=self.llm_api_key,
                    timeout=self.llm_timeout,
                )
            )

            # Create embedding function with api_base and api_key support
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
                f"Using LiteLLM with model {effective_best_model}, "
                f"api_base={self.llm_api_base}, "
                f"structured_output={self.structured_output}, "
                f"use_pydantic_structured_output={self.use_pydantic_structured_output}"
            )

            # Validate models
            try:
                import litellm

                valid_models = set(litellm.get_valid_models())

                for model_name in [effective_best_model, effective_cheap_model]:
                    # Check if exact match
                    if model_name in valid_models:
                        continue

                    # Check if provider prefix is valid
                    provider = model_name.split("/")[0] if "/" in model_name else None
                    if provider:
                        provider_models = [m for m in valid_models if m.startswith(f"{provider}/")]
                        if provider_models:
                            logger.info(
                                f"✓ Model '{model_name}' uses provider '{provider}' "
                                f"({len(provider_models)} models available)"
                            )
                            continue

                    # Warning if not found
                    logger.warning(
                        f"Model '{model_name}' not found in LiteLLM's model list. "
                        f"This may be a typo or a newly added model. "
                        f"Provider '{provider}' detected. "
                        f"Check: https://docs.litellm.ai/docs/supported_models"
                    )
            except Exception as e:
                logger.debug(f"Could not validate models against LiteLLM list: {e}")
        else:
            self.embedding_func = EmbeddingFunc(
                embedding_dim=self.embedding_func.embedding_dim,
                max_token_size=self.embedding_func.max_token_size,
                func=limit_async_func_call(self.embedding_func_max_async)(self.embedding_func),
            )
            self.best_model_func = limit_async_func_call(self.best_model_max_async)(
                partial(self.best_model_func, hashing_kv=self.llm_response_cache)
            )
            self.cheap_model_func = limit_async_func_call(self.cheap_model_max_async)(
                partial(self.cheap_model_func, hashing_kv=self.llm_response_cache)
            )

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
        if param.mode == "local":
            response = await local_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.community_reports,
                self.text_chunks,
                param,
                self.tokenizer_wrapper,
                self._runtime_config(),
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
                self._runtime_config(),
            )
        elif param.mode == "naive":
            response = await naive_query(
                query,
                self.chunks_vdb,
                self.text_chunks,
                param,
                self.tokenizer_wrapper,
                self._runtime_config(),
            )
        else:
            raise ValueError(f"Unknown mode {param.mode}")
        await self._query_done()
        return response

    async def ainsert(self, string_or_strings):
        if isinstance(string_or_strings, str):
            string_or_strings = [string_or_strings]
        documents = {
            compute_sha256_id(content.strip(), prefix="doc-"): content
            for content in string_or_strings
            if content.strip()
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
            if not len(new_docs):
                logger.warning("All docs are already in the storage")
                return
            logger.info(f"[New Docs] inserting {len(new_docs)} docs")

            # ---------- chunking

            inserting_chunks = get_chunks(
                new_docs=new_docs,
                chunk_func=self.chunk_func,
                overlap_token_size=self.chunk_overlap_token_size,
                max_token_size=self.chunk_token_size,
                tokenizer_wrapper=self.tokenizer_wrapper,
            )

            _add_chunk_keys = await self.text_chunks.filter_keys(list(inserting_chunks.keys()))
            inserting_chunks = {k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys}
            if not len(inserting_chunks):
                logger.warning("All chunks are already in the storage")
                return
            logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")
            if self.enable_naive_rag:
                logger.info("Insert chunks for naive RAG")
                await self.chunks_vdb.upsert(inserting_chunks)

            # ---------- extract/summary entity and upsert to graph
            logger.info("[Entity Extraction]...")
            maybe_new_kg = await self.entity_extraction_func(
                inserting_chunks,
                knwoledge_graph_inst=self.chunk_entity_relation_graph,
                entity_vdb=self.entities_vdb,
                tokenizer_wrapper=self.tokenizer_wrapper,
                global_config=self._runtime_config(),
                using_amazon_bedrock=self.using_amazon_bedrock,
            )
            if maybe_new_kg is None:
                logger.warning("No new entities found")
                return
            self.chunk_entity_relation_graph = maybe_new_kg
            # ---------- update clusterings of graph
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

            # ---------- commit upsertings and indexing
            await self.full_docs.upsert(new_docs)
            await self.text_chunks.upsert(inserting_chunks)
        finally:
            await self._insert_done()

    async def _ainsert_documents(self, documents: dict[str, str], allow_legacy_custom: bool):
        if self.entity_extraction_func is not extract_entities:
            if allow_legacy_custom:
                return await self._legacy_custom_ainsert(documents)
            raise NotImplementedError(
                "insert_documents currently requires the built-in extract_entities pipeline."
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
                    using_amazon_bedrock=self.using_amazon_bedrock,
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
                    logger.info("Insert chunks for naive RAG")
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
            affected_entity_ids_for_clustering = {
                entity_id
                for manifest in list(old_manifest_lookup.values()) + list(new_document_index_entries.values())
                if manifest is not None
                for entity_id in manifest.get("entities", {}).keys()
            }

            all_doc_keys = await self.document_index.all_keys()
            all_document_manifests = await self.document_index.get_by_ids(all_doc_keys)
            has_graph_data = any(
                manifest and manifest.get("entities") for manifest in all_document_manifests
            )
            if has_graph_data:
                logger.info("[Community Report]...")
                await self.chunk_entity_relation_graph.clustering(
                    self.graph_cluster_algorithm,
                    affected_node_ids=affected_entity_ids_for_clustering,
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
        tasks = []
        for storage_inst in [
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_start_callback())
        await asyncio.gather(*tasks)

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
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    async def _query_done(self):
        tasks = []
        for storage_inst in [self.llm_response_cache]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)
