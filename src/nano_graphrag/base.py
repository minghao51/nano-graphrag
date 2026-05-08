from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Generic, Literal, TypedDict, TypeVar

import numpy as np

from ._config import (
    DEFAULT_CHEAP_MODEL,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_MODEL,
    SUPPORTED_GRAPH_CLUSTERING,  # noqa: F401 - re-exported for graphrag_runtime.py
    GraphRAGSettings,
    _parse_bool_env,
    _parse_int_env,
)
from ._utils import EmbeddingFunc


@dataclass
class QueryParam:
    mode: Literal["local", "global", "naive", "entity_grounded"] = "global"
    only_need_context: bool = False
    response_type: str = "Concise Answer"
    level: int = 2
    top_k: int = 20
    # naive search
    naive_max_token_for_text_unit: int = 12000
    # local search
    local_max_token_for_text_unit: int = 4000  # 12000 * 0.33
    local_max_token_for_local_context: int = 4800  # 12000 * 0.4
    local_max_token_for_community_report: int = 3200  # 12000 * 0.27
    local_community_single_one: bool = False
    # global search
    global_min_community_rating: float = 0
    global_max_consider_community: int = 512
    global_max_token_for_community_report: int = 16384
    global_special_community_map_llm_kwargs: dict = field(
        default_factory=lambda: {"response_format": {"type": "json_object"}}
    )
    # entity-grounded search
    entity_grounded_max_answer_length: int = 50  # tokens
    entity_grounded_require_entity_match: bool = True
    entity_grounded_fuzzy_threshold: float = 0.85
    # temporal search
    time_range: tuple[str, str] | None = None
    temporal_mode: str = "any"


class ResponseType:
    """Predefined response types for QueryParam."""

    CONCISE = "Concise Answer"  # Direct, brief answers (new default)
    SHORT = "Short Answer"  # 1-2 sentences
    SINGLE_PARAGRAPH = "Single Paragraph"  # Brief explanation
    MULTIPLE_PARAGRAPHS = "Multiple Paragraphs"  # Detailed explanation (old default)
    BULLET_POINTS = "Bullet Points"  # Structured list format


class TextChunkSchema(TypedDict):
    tokens: int
    content: str
    full_doc_id: str
    chunk_order_index: int


class SingleCommunitySchema(TypedDict):
    level: int
    title: str
    edges: list[tuple[str, str]]
    nodes: list[str]
    chunk_ids: list[str]
    occurrence: float
    sub_communities: list[str]


class CommunitySchema(SingleCommunitySchema):
    report_string: str
    report_json: dict


T = TypeVar("T")


@dataclass
class StorageNameSpace:
    namespace: str
    global_config: dict

    async def index_start_callback(self):
        """commit the storage operations after indexing"""
        pass

    async def index_done_callback(self):
        """commit the storage operations after indexing"""
        pass

    async def query_done_callback(self):
        """commit the storage operations after querying"""
        pass


@dataclass
class BaseVectorStorage(StorageNameSpace):
    embedding_func: EmbeddingFunc
    meta_fields: set = field(default_factory=set)

    async def query(self, query: str, top_k: int) -> list[dict]:
        raise NotImplementedError

    async def upsert(self, data: dict[str, dict]):
        """Use 'content' field from value for embedding, use key as id.
        If embedding_func is None, use 'embedding' field from value
        """
        raise NotImplementedError

    async def delete(self, ids: list[str]):
        raise NotImplementedError


@dataclass
class BaseKVStorage(Generic[T], StorageNameSpace):
    async def all_keys(self) -> list[str]:
        raise NotImplementedError

    async def get_by_id(self, id: str) -> T | None:
        raise NotImplementedError

    async def get_by_ids(self, ids: list[str], fields: set[str] | None = None) -> list[T | None]:
        raise NotImplementedError

    async def filter_keys(self, data: list[str]) -> set[str]:
        """return un-exist keys"""
        raise NotImplementedError

    async def upsert(self, data: dict[str, T]):
        raise NotImplementedError

    async def delete(self, ids: list[str]):
        raise NotImplementedError

    async def drop(self):
        raise NotImplementedError


@dataclass
class BaseGraphStorage(StorageNameSpace):
    async def has_node(self, node_id: str) -> bool:
        raise NotImplementedError

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        raise NotImplementedError

    async def node_degree(self, node_id: str) -> int:
        raise NotImplementedError

    async def node_degrees_batch(self, node_ids: list[str]) -> list[int]:
        raise NotImplementedError

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        raise NotImplementedError

    async def edge_degrees_batch(self, edge_pairs: list[tuple[str, str]]) -> list[int]:
        raise NotImplementedError

    async def get_node(self, node_id: str) -> dict | None:
        raise NotImplementedError

    async def get_nodes_batch(self, node_ids: list[str]) -> list[dict | None]:
        raise NotImplementedError

    async def get_edge(self, source_node_id: str, target_node_id: str) -> dict | None:
        raise NotImplementedError

    async def get_edges_batch(self, edge_pairs: list[tuple[str, str]]) -> list[dict | None]:
        raise NotImplementedError

    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        raise NotImplementedError

    async def get_nodes_edges_batch(self, node_ids: list[str]) -> list[list[tuple[str, str]]]:
        raise NotImplementedError

    async def upsert_node(self, node_id: str, node_data: dict[str, Any]):
        raise NotImplementedError

    async def upsert_nodes_batch(self, nodes_data: list[tuple[str, dict[str, Any]]]):
        raise NotImplementedError

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, Any]
    ):
        raise NotImplementedError

    async def upsert_edges_batch(self, edges_data: list[tuple[str, str, dict[str, Any]]]):
        raise NotImplementedError

    async def delete_node(self, node_id: str):
        raise NotImplementedError

    async def delete_nodes_batch(self, node_ids: list[str]):
        raise NotImplementedError

    async def delete_edge(self, source_node_id: str, target_node_id: str):
        raise NotImplementedError

    async def delete_edges_batch(self, edge_pairs: list[tuple[str, str]]):
        raise NotImplementedError

    async def clustering(self, algorithm: str, affected_node_ids: set[str] | None = None):
        raise NotImplementedError

    async def community_schema(self) -> dict[str, SingleCommunitySchema]:
        """Return the community representation with report and nodes"""
        raise NotImplementedError

    async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        raise NotImplementedError("Node embedding is not used in nano-graphrag.")


# =============================================================================
# GraphRAG Configuration
# =============================================================================


def _parse_bool(env_var: str, default: bool = False) -> bool:
    return _parse_bool_env(env_var, default)


def _parse_int(env_var: str, default: int, min_value: int | None = None) -> int:
    return _parse_int_env(env_var, default, min_value)


@dataclass
class _ConfigFields:
    """Single source of truth for config fields shared between GraphRAG and GraphRAGConfig.

    Do not instantiate directly — use GraphRAGConfig or GraphRAG instead.
    """

    # === Core ===
    working_dir: str = "./nano_graphrag"

    # === Shared API credentials (LLM + Embedding use same key by default) ===
    api_key: str | None = None
    api_base: str | None = None

    # === LLM (passed to LiteLLM) ===
    llm_model: str = DEFAULT_LLM_MODEL
    llm_cheap_model: str = DEFAULT_CHEAP_MODEL
    llm_api_base: str | None = None
    llm_api_key: str | None = None
    llm_max_async: int = 32
    llm_max_tokens: int = 32768
    llm_timeout: int = 120

    # === Embedding ===
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_api_base: str | None = None
    embedding_api_key: str | None = None
    embedding_dim: int = DEFAULT_EMBEDDING_DIM
    embedding_max_async: int = 16
    embedding_batch_size: int = 32

    # === Compute/Quality ===
    extraction_max_async: int = 16
    extraction_batch_size: int = 5
    doc_extraction_max_async: int = 4
    doc_flush_batch_size: int = 50
    entity_extraction_quality: str = "balanced"
    extraction_backend: str = "llm"
    graph_cluster_algorithm: str = "leiden"
    max_incremental_updates_before_full: int = 10
    alias_batch_size: int = 20
    enable_node_embedding: bool = False

    # === Features ===
    enable_local: bool = True
    enable_naive_rag: bool = False
    enable_llm_cache: bool = True
    enable_entity_linking: bool = False
    entity_linking_use_neighborhood_evidence: bool = True
    enable_community_reports: bool = True
    enable_temporal_extraction: bool = False
    entity_linking_similarity_threshold: float = 0.92
    entity_linking_max_candidates: int = 3
    entity_linking_iou_threshold: float = 0.3
    entity_linking_min_common_neighbors: int = 2
    alias_max_batches_in_flight: int = 5
    entity_count_min_ratio: float = 2.0
    entity_count_min_absolute: int = 3

    # === Logging ===
    log_level: str = "INFO"
    log_file: str | None = None


@dataclass
class GraphRAGConfig(_ConfigFields):
    """Backward-compatible configuration wrapper around GraphRAGSettings.

    Delegates loading/validation to Pydantic-backed GraphRAGSettings internally,
    but exposes the same flat dataclass interface as before.

    Supports three loading methods:
        - from_env(): Load from environment variables
        - from_yaml(path): Load from YAML (flat or nested format)
        - from_dict(data): Load from a dictionary (flat or nested)

    Example:
        config = GraphRAGConfig.from_yaml("config/settings.yaml")
        rag = GraphRAG.from_config(config)
    """

    @classmethod
    def _from_settings(cls, settings: GraphRAGSettings) -> GraphRAGConfig:
        flat = settings.to_flat_dict()
        if not flat.get("llm_api_key"):
            flat["llm_api_key"] = flat.get("api_key")
        if not flat.get("embedding_api_key"):
            flat["embedding_api_key"] = flat.get("api_key")
        if not flat.get("llm_api_base"):
            flat["llm_api_base"] = flat.get("api_base")
        if not flat.get("embedding_api_base"):
            flat["embedding_api_base"] = flat.get("api_base")
        return cls(**flat)

    @classmethod
    def from_env(cls) -> GraphRAGConfig:
        return cls._from_settings(GraphRAGSettings.from_env())

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> GraphRAGConfig:
        settings = GraphRAGSettings.from_dict(config)
        return cls._from_settings(settings)

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def __post_init__(self):
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        GraphRAGSettings.from_dict(data)

    def merge(self, overrides: dict[str, Any]) -> GraphRAGConfig:
        data = self.to_dict()
        merged = {**data, **{k: v for k, v in overrides.items() if v is not None}}
        return GraphRAGConfig.from_dict(merged)

    @classmethod
    def from_yaml(cls, path: str) -> GraphRAGConfig:
        settings = GraphRAGSettings.from_yaml(path)
        return cls._from_settings(settings)

    def to_yaml(self, path: str):
        data = self.to_dict()
        settings = GraphRAGSettings.from_dict(data)
        settings.to_yaml(path)
