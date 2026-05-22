from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, fields
from datetime import datetime
from typing import Any

from ._exceptions import StorageConfigError
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
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    GraphRAGConfig,
    QueryParam,
    _ConfigFields,
)
from .graphrag_insert import _InsertMixin
from .graphrag_query import _QueryMixin
from .graphrag_runtime import _ConfigMixin

_STORAGE_REGISTRY: dict[str, str] = {
    "json": "nano_graphrag._storage.kv_json:JsonKVStorage",
    "sqlite": "nano_graphrag._storage.kv_json:SQLiteKVStorage",
    "hnsw": "nano_graphrag._storage.vdb_hnswlib:HNSWVectorStorage",
    "networkx": "nano_graphrag._storage.gdb_networkx:NetworkXStorage",
    "sqlite_graph": "nano_graphrag._storage.gdb_sqlite:SQLiteGraphStorage",
}


def _resolve_storage(name_or_cls):
    if name_or_cls is None or not isinstance(name_or_cls, str):
        return name_or_cls
    key = name_or_cls.strip().lower()
    if key not in _STORAGE_REGISTRY:
        raise StorageConfigError(
            f"Unknown storage backend {name_or_cls!r}. Available: {sorted(_STORAGE_REGISTRY.keys())}",
            details={"requested": name_or_cls, "available": sorted(_STORAGE_REGISTRY.keys())},
        )
    module_path, class_name = _STORAGE_REGISTRY[key].rsplit(":", 1)
    import importlib

    try:
        mod = importlib.import_module(module_path)
        return getattr(mod, class_name)
    except (ImportError, AttributeError) as e:
        raise StorageConfigError(
            f"Failed to load storage backend {name_or_cls!r}: {e}",
            details={"backend": name_or_cls, "module": module_path, "class": class_name},
        ) from e


_SECRET_KEYS = {"api_key", "llm_api_key", "embedding_api_key"}
_CALLABLE_KEYS = {
    "embedding_func",
    "best_model_func",
    "cheap_model_func",
    "chunk_func",
    "entity_extraction_func",
    "convert_response_to_json_func",
}


@dataclass
class GraphRAG(_ConfigFields, _ConfigMixin, _InsertMixin, _QueryMixin):
    working_dir: str = field(  # type: ignore[assignment]
        default_factory=lambda: (
            f"./nano_graphrag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
        )
    )
    llm_max_async: int | None = None  # type: ignore[assignment]
    embedding_max_async: int | None = None  # type: ignore[assignment]
    embedding_batch_size: int | None = None  # type: ignore[assignment]

    tokenizer_type: str = "tiktoken"
    tiktoken_model_name: str = "gpt-4o"
    huggingface_model_name: str = "bert-base-uncased"
    chunk_func: Callable[
        [
            list[list[int]],
            list[str],
            Any,
            int | None,
            int | None,
        ],
        list[dict[str, str | int]],
    ] = chunking_by_token_size
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100

    entity_extract_max_gleaning: int = 0
    entity_summary_to_max_tokens: int = 500

    max_graph_cluster_size: int = 10
    graph_cluster_seed: int = 0xDEADBEEF
    leiden_resolutions: list = field(default_factory=lambda: [2.0, 1.0, 0.5])

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

    special_community_report_llm_kwargs: dict = field(
        default_factory=lambda: {"response_format": CommunityReportOutput}
    )

    embedding_func: EmbeddingFunc | None = None
    embedding_batch_num: int = 32  # deprecated — use embedding_batch_size
    embedding_func_max_async: int = 16

    best_model_func: Callable[..., Any] | None = None
    best_model_max_token_size: int = 32768
    best_model_max_async: int = 16
    cheap_model_func: Callable[..., Any] | None = None
    cheap_model_max_token_size: int = 32768
    cheap_model_max_async: int = 16

    structured_output: bool = True
    use_pydantic_structured_output: bool = True
    fallback_to_parsing: bool = True

    entity_extraction_func: Callable[..., Any] = extract_entities

    key_string_value_json_storage_cls: type[BaseKVStorage] | None = None
    vector_db_storage_cls: type[BaseVectorStorage] | None = None
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)
    graph_storage_cls: type[BaseGraphStorage] | None = None

    always_create_working_dir: bool = True
    addon_params: dict = field(default_factory=dict)
    convert_response_to_json_func: Callable[..., Any] = convert_response_to_json
    callbacks: list = field(default_factory=list)

    # Runtime-attributed storage instances, set by _build_storages in __post_init__
    chunk_entity_relation_graph: BaseGraphStorage | None = field(init=False, default=None)
    entities_vdb: BaseVectorStorage | None = field(init=False, default=None)
    text_chunks: BaseKVStorage | None = field(init=False, default=None)
    community_reports: BaseKVStorage | None = field(init=False, default=None)

    @classmethod
    def from_config(cls, config: GraphRAGConfig) -> GraphRAG:
        config_dict = config.to_dict()
        valid_fields = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in config_dict.items() if k in valid_fields}

        if config.extraction_backend == "gliner":
            from ._ops.extraction_gliner import extract_entities_gliner

            kwargs["entity_extraction_func"] = extract_entities_gliner

        return cls(**kwargs)

    def __post_init__(self):
        if (
            isinstance(self.key_string_value_json_storage_cls, str)
            or self.key_string_value_json_storage_cls is None
        ):
            resolved = _resolve_storage(self.key_string_value_json_storage_cls)
            if resolved is None:
                from ._storage import JsonKVStorage

                resolved = JsonKVStorage
            self.key_string_value_json_storage_cls = resolved
        if isinstance(self.vector_db_storage_cls, str) or self.vector_db_storage_cls is None:
            resolved = _resolve_storage(self.vector_db_storage_cls)
            if resolved is None:
                from ._storage import HNSWVectorStorage

                resolved = HNSWVectorStorage
            self.vector_db_storage_cls = resolved
        if isinstance(self.graph_storage_cls, str) or self.graph_storage_cls is None:
            resolved = _resolve_storage(self.graph_storage_cls)
            if resolved is None:
                from ._storage import NetworkXStorage

                resolved = NetworkXStorage
            self.graph_storage_cls = resolved

        self._normalize_settings()
        self._configure_logging()
        self._build_tokenizer()
        from ._callbacks import _CallbackDispatcher, _NullDispatcher

        self._callback_dispatcher = (
            _CallbackDispatcher(self.callbacks) if self.callbacks else _NullDispatcher()
        )
        self._configure_runtime()
        self._build_storages()

    def _to_config_dict(self) -> dict[str, Any]:
        """Serialize all fields to a dict for use as global_config.

        Includes callables and secrets since downstream consumers need them.
        Use _to_safe_log_dict() for logging instead.
        """
        result = {}
        for f in fields(self):
            if f.name in _RUNTIME_ATTRS:
                continue
            result[f.name] = getattr(self, f.name)
        return result

    def _to_safe_log_dict(self) -> dict[str, Any]:
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

    def insert(self, string_or_strings):
        """Insert one or more text documents into the knowledge graph.

        Args:
            string_or_strings: A single text string or list of text strings to insert.
                Documents are deduplicated via content hash — unchanged documents are skipped.

        Returns:
            None (use ``ainsert`` for async version).
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert(string_or_strings))

    def insert_documents(self, documents: dict[str, str], force_rebuild: bool = False):
        """Insert documents with explicit IDs into the knowledge graph.

        Args:
            documents: Mapping of ``{doc_id: text_content}``.
            force_rebuild: If True, re-extract even unchanged documents.

        Returns:
            None (use ``ainsert_documents`` for async version).
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.ainsert_documents(documents, force_rebuild=force_rebuild)
        )

    def query(self, query: str, param: QueryParam | None = None):
        """Query the knowledge graph.

        Args:
            query: The question or search string.
            param: Query parameters controlling mode, top-k, token limits, etc.
                Defaults to ``QueryParam()`` with ``mode="global"``.

        Returns:
            QueryResult: Structured query output containing ``answer``, ``mode``,
            ``sources``, timing, and metadata.
        """
        if param is None:
            param = QueryParam.from_config(self)
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, param))

    async def ainsert(self, string_or_strings):
        """Async insert one or more text documents.

        Args:
            string_or_strings: A single text string or list of strings.

        Documents are deduplicated by content hash. Delta detection skips
        unchanged documents automatically.
        """
        if isinstance(string_or_strings, str):
            string_or_strings = [string_or_strings]
        normalized = [c.strip() for c in string_or_strings if c.strip()]
        doc_ids = [compute_mdhash_id(c, prefix="doc-") for c in normalized]
        existing = await self.full_docs.get_by_ids(doc_ids)
        documents = {
            (doc_id if doc else compute_sha256_id(c, prefix="doc-")): c
            for c, doc_id, doc in zip(normalized, doc_ids, existing, strict=False)
        }
        return await self._ainsert_documents(documents, allow_legacy_custom=True)

    async def ainsert_documents(self, documents: dict[str, str], force_rebuild: bool = False):
        """Async insert documents with explicit IDs.

        Args:
            documents: Mapping of ``{doc_id: text_content}``.
            force_rebuild: If True, re-extract even unchanged documents.
        """
        return await self._ainsert_documents(
            documents, allow_legacy_custom=False, force_rebuild=force_rebuild
        )

    async def arebuild_graph(self):
        """Rebuild the entire knowledge graph from existing document manifests.

        Re-reads all manifests from ``document_index`` and reconstructs the graph
        without re-running LLM extraction. Useful after storage corruption or
        when switching graph backends.
        """
        return await self._rebuild_graph_from_manifests()

    def rebuild_graph(self):
        """Synchronous wrapper for ``arebuild_graph``."""
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.arebuild_graph())

    async def arefine(self, phases: list[str] | None = None) -> dict:
        """Run the knowledge graph refinement pipeline.

        Args:
            phases: List of phases to run. Options: ``"merge"``, ``"enrich"``, ``"infer"``.
                Defaults to all phases if None.

        Returns:
            dict with per-phase statistics (merged, enriched, inferred counts).
        """
        from ._ops.refinement import arefine

        return await arefine(
            self.chunk_entity_relation_graph,
            self.entities_vdb,
            self.text_chunks,
            self._runtime_config(),
            phases=phases,
        )

    def refine(self, phases: list[str] | None = None) -> dict:
        """Synchronous wrapper for ``arefine``."""
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.arefine(phases))

    async def aexport_vault(
        self, path: str | None = None, include_communities: bool | None = None
    ) -> dict:
        """Export the knowledge graph to an Obsidian-compatible vault.

        Args:
            path: Output directory path. Defaults to ``self.vault_path``.
            include_communities: Whether to include community report files.

        Returns:
            dict with export statistics (entity files, relationship files, etc.).
        """
        from ._vault import aexport_vault

        if include_communities is None:
            include_communities = self.vault_export_communities
        return await aexport_vault(
            self.chunk_entity_relation_graph,
            self.community_reports,
            self._runtime_config(),
            path=path,
            include_communities=include_communities,
        )

    def export_vault(
        self, path: str | None = None, include_communities: bool | None = None
    ) -> dict:
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aexport_vault(path, include_communities))

    async def astatus(self):
        """Return comprehensive graph status without loading the full graph.

        Returns:
            GraphStatus with entity count, community info, health assessment, etc.
        """
        from ._visualization import compute_status

        return await compute_status(self.working_dir, rag=self)

    def status(self):
        """Synchronous wrapper for ``astatus``."""
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.astatus())

    async def aexport_graph_html(
        self, output: str = "graph.html", max_nodes: int = 200, **kwargs
    ) -> str:
        """Generate interactive HTML visualization of the knowledge graph.

        Args:
            output: Output HTML file path.
            max_nodes: Maximum nodes to render (performance).
            kwargs: Passed to pyvis Network configuration.

        Returns:
            Path to the generated HTML file.
        """
        from ._visualization import visualize_graph

        return await visualize_graph(
            self.chunk_entity_relation_graph, output=output, max_nodes=max_nodes, **kwargs
        )

    def export_graph_html(self, output: str = "graph.html", max_nodes: int = 200, **kwargs) -> str:
        """Synchronous wrapper for ``aexport_graph_html``."""
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aexport_graph_html(output, max_nodes, **kwargs))


# Runtime-attributed storage instances, set by _build_storages in __post_init__.
# Declared as dataclass fields (init=False) so mypy sees them as instance attributes.
_RUNTIME_ATTRS = frozenset(
    {
        "chunk_entity_relation_graph",
        "entities_vdb",
        "text_chunks",
        "community_reports",
    }
)
