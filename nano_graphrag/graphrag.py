from dataclasses import dataclass, field, fields
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Type, Union

from ._ops import chunking_by_token_size, extract_entities
from ._schemas import CommunityReportOutput
from ._utils import (
    EmbeddingFunc,
    always_get_an_event_loop,
    compute_mdhash_id,
    compute_sha256_id,
    convert_response_to_json,
)
from .base import (
    DEFAULT_CHEAP_MODEL,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_MODEL,
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    GraphRAGConfig,
    QueryParam,
)
from .graphrag_insert import (
    _ainsert_documents,
    _flush_doc_progress,
    _insert_done,
    _insert_start,
    _legacy_custom_ainsert,
    _rebuild_graph_from_manifests,
)
from .graphrag_query import _query_done, aquery, astream_query
from .graphrag_runtime import (
    _build_storages,
    _build_tokenizer,
    _configure_logging,
    _configure_runtime,
    _normalize_settings,
    _runtime_config,
)

_SECRET_KEYS = {"llm_api_key", "embedding_api_key"}
_CALLABLE_KEYS = {
    "embedding_func",
    "best_model_func",
    "cheap_model_func",
    "chunk_func",
    "entity_extraction_func",
    "convert_response_to_json_func",
}


@dataclass
class GraphRAG:
    working_dir: str = field(
        default_factory=lambda: (
            f"./nano_graphrag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
        )
    )
    log_level: str = "INFO"
    log_file: Optional[str] = None

    enable_local: bool = True
    enable_naive_rag: bool = False

    tokenizer_type: str = "tiktoken"
    tiktoken_model_name: str = "gpt-4o"
    huggingface_model_name: str = "bert-base-uncased"
    chunk_func: Callable[
        [
            list[list[int]],
            List[str],
            Any,
            Optional[int],
            Optional[int],
        ],
        List[Dict[str, Union[str, int]]],
    ] = chunking_by_token_size
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100

    entity_extract_max_gleaning: int = 0
    entity_summary_to_max_tokens: int = 500
    extraction_max_async: int = 16
    extraction_batch_size: int = 5
    doc_extraction_max_async: int = 4
    doc_flush_batch_size: int = 50

    graph_cluster_algorithm: str = "leiden"
    max_graph_cluster_size: int = 10
    graph_cluster_seed: int = 0xDEADBEEF
    leiden_resolutions: list = field(default_factory=lambda: [2.0, 1.0, 0.5])
    max_incremental_updates_before_full: int = 10
    alias_batch_size: int = 20

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

    special_community_report_llm_kwargs: dict = field(
        default_factory=lambda: {"response_format": CommunityReportOutput}
    )

    embedding_func: Optional[EmbeddingFunc] = None
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_api_base: Optional[str] = None
    embedding_api_key: Optional[str] = None
    embedding_dim: int = DEFAULT_EMBEDDING_DIM
    embedding_max_async: Optional[int] = None
    embedding_batch_size: Optional[int] = None
    embedding_batch_num: int = 32
    embedding_func_max_async: int = 16

    best_model_func: Optional[Callable[..., Any]] = None
    best_model_max_token_size: int = 32768
    best_model_max_async: int = 16
    cheap_model_func: Optional[Callable[..., Any]] = None
    cheap_model_max_token_size: int = 32768
    cheap_model_max_async: int = 16

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

    entity_extraction_func: Callable[..., Any] = extract_entities
    entity_extraction_quality: str = "balanced"
    extraction_backend: str = "llm"

    key_string_value_json_storage_cls: Type[BaseKVStorage] = None
    vector_db_storage_cls: Type[BaseVectorStorage] = None
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)
    graph_storage_cls: Type[BaseGraphStorage] = None
    enable_llm_cache: bool = True
    enable_entity_linking: bool = False
    enable_community_reports: bool = True
    entity_linking_similarity_threshold: float = 0.92
    entity_linking_max_candidates: int = 3

    always_create_working_dir: bool = True
    addon_params: dict = field(default_factory=dict)
    convert_response_to_json_func: Callable[..., Any] = convert_response_to_json

    @classmethod
    def from_config(cls, config: GraphRAGConfig) -> "GraphRAG":
        config_dict = config.to_dict()
        valid_fields = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in config_dict.items() if k in valid_fields}

        if config.extraction_backend == "gliner":
            from ._ops.extraction_gliner import extract_entities_gliner

            kwargs["entity_extraction_func"] = extract_entities_gliner

        return cls(**kwargs)

    def __post_init__(self):
        if self.key_string_value_json_storage_cls is None:
            from ._storage import JsonKVStorage

            self.key_string_value_json_storage_cls = JsonKVStorage
        if self.vector_db_storage_cls is None:
            from ._storage import HNSWVectorStorage

            self.vector_db_storage_cls = HNSWVectorStorage
        if self.graph_storage_cls is None:
            from ._storage import NetworkXStorage

            self.graph_storage_cls = NetworkXStorage

        self._normalize_settings()
        self._configure_logging()
        self._build_tokenizer()
        self._configure_runtime()
        self._build_storages()

    def _to_config_dict(self) -> Dict[str, Any]:
        """Serialize all fields to a dict for use as global_config.

        Includes callables and secrets since downstream consumers need them.
        Use _to_safe_log_dict() for logging instead.
        """
        from dataclasses import asdict

        return asdict(self)

    def _to_safe_log_dict(self) -> Dict[str, Any]:
        """Serialize fields for safe logging — redacts secrets, omits callables."""
        result = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if f.name in _SECRET_KEYS:
                result[f.name] = "***" if val else None
            elif f.name in _CALLABLE_KEYS:
                result[f.name] = f"<{type(val).__name__}>"
            else:
                result[f.name] = val
        return result

    _normalize_settings = _normalize_settings
    _configure_logging = _configure_logging
    _build_tokenizer = _build_tokenizer
    _configure_runtime = _configure_runtime
    _build_storages = _build_storages
    _runtime_config = _runtime_config
    _legacy_custom_ainsert = _legacy_custom_ainsert
    _ainsert_documents = _ainsert_documents
    _flush_doc_progress = _flush_doc_progress
    _rebuild_graph_from_manifests = _rebuild_graph_from_manifests
    _insert_start = _insert_start
    _insert_done = _insert_done
    aquery = aquery
    _query_done = _query_done
    astream_query = astream_query

    def insert(self, string_or_strings):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert(string_or_strings))

    def insert_documents(self, documents: dict[str, str], force_rebuild: bool = False):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.ainsert_documents(documents, force_rebuild=force_rebuild)
        )

    def query(self, query: str, param: QueryParam = QueryParam()):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, param))

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

    async def ainsert_documents(self, documents: dict[str, str], force_rebuild: bool = False):
        return await self._ainsert_documents(
            documents, allow_legacy_custom=False, force_rebuild=force_rebuild
        )

    async def arebuild_graph(self):
        return await self._rebuild_graph_from_manifests()

    def rebuild_graph(self):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.arebuild_graph())
