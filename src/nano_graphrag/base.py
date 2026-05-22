from __future__ import annotations

import abc
from dataclasses import dataclass, field, fields
from typing import Any, Generic, Literal, TypedDict, TypeVar

import numpy as np

from ._config import (
    DEFAULT_CHEAP_MODEL,  # noqa: F401 - re-exported
    DEFAULT_EMBEDDING_DIM,  # noqa: F401 - re-exported
    DEFAULT_EMBEDDING_MODEL,  # noqa: F401 - re-exported
    DEFAULT_LLM_MODEL,  # noqa: F401 - re-exported
    FLAT_DEFAULTS,
    SUPPORTED_GRAPH_CLUSTERING,  # noqa: F401 - re-exported for graphrag_runtime.py
    GraphRAGSettings,
)
from ._utils import EmbeddingFunc


@dataclass
class QueryParam:
    """Parameters controlling query behavior.

    Attributes:
        mode: Query mode — ``"local"`` (entity-centric), ``"global"`` (community reports),
            ``"naive"`` (vector search), or ``"entity_grounded"`` (structured).
        only_need_context: If True, return retrieved context instead of LLM-generated answer.
        response_type: Format directive for the answer (e.g., ``"Concise Answer"``).
        level: Community hierarchy level for global mode.
        top_k: Number of top entities/chunks to retrieve.
    """

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
    # Phase 1: Structured Query Planning
    enable_query_planning: bool = False
    # Phase 2: Topological Structural Features
    structural_feature_weights: list[float] = field(default_factory=lambda: [0.5, 0.2, 0.1, 0.2])
    edge_gate_threshold: float = 0.0
    # Phase 3: Structurally-Gated Propagation
    propagation_hops: int = 1
    subgraph_prune_ratio: float = 0.0
    edge_gate_decay: float = 1.0

    @classmethod
    def from_config(cls, config, **overrides):
        params = {k: getattr(config, k) for k in _SAGE_QUERY_FIELDS if hasattr(config, k)}
        params.update(overrides)
        return cls(**params)


_SAGE_QUERY_FIELDS = (
    "enable_query_planning",
    "structural_feature_weights",
    "edge_gate_threshold",
    "propagation_hops",
    "subgraph_prune_ratio",
    "edge_gate_decay",
)


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
class StorageNameSpace(abc.ABC):
    namespace: str
    global_config: dict

    async def index_start_callback(self):  # noqa: B027
        """commit the storage operations after indexing"""
        pass

    async def index_done_callback(self):  # noqa: B027
        """commit the storage operations after indexing"""
        pass

    async def query_done_callback(self):  # noqa: B027
        """commit the storage operations after querying"""
        pass


@dataclass
class BaseVectorStorage(StorageNameSpace):
    embedding_func: EmbeddingFunc
    meta_fields: set = field(default_factory=set)

    @abc.abstractmethod
    async def query(self, query: str, top_k: int) -> list[dict]:
        raise NotImplementedError

    @abc.abstractmethod
    async def upsert(self, data: dict[str, dict]):
        raise NotImplementedError

    @abc.abstractmethod
    async def delete(self, ids: list[str]):
        raise NotImplementedError


@dataclass
class BaseKVStorage(Generic[T], StorageNameSpace):
    @abc.abstractmethod
    async def all_keys(self) -> list[str]:
        raise NotImplementedError

    @abc.abstractmethod
    async def get_by_id(self, id: str) -> T | None:
        raise NotImplementedError

    @abc.abstractmethod
    async def get_by_ids(self, ids: list[str], fields: set[str] | None = None) -> list[T | None]:
        raise NotImplementedError

    @abc.abstractmethod
    async def filter_keys(self, data: list[str]) -> set[str]:
        raise NotImplementedError

    @abc.abstractmethod
    async def upsert(self, data: dict[str, T]):
        raise NotImplementedError

    @abc.abstractmethod
    async def delete(self, ids: list[str]):
        raise NotImplementedError

    @abc.abstractmethod
    async def drop(self):
        raise NotImplementedError


@dataclass
class BaseGraphStorage(StorageNameSpace):
    @abc.abstractmethod
    async def has_node(self, node_id: str) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    async def node_degree(self, node_id: str) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    async def node_degrees_batch(self, node_ids: list[str]) -> list[int]:
        raise NotImplementedError

    @abc.abstractmethod
    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    async def edge_degrees_batch(self, edge_pairs: list[tuple[str, str]]) -> list[int]:
        raise NotImplementedError

    @abc.abstractmethod
    async def get_node(self, node_id: str) -> dict | None:
        raise NotImplementedError

    @abc.abstractmethod
    async def get_nodes_batch(self, node_ids: list[str]) -> list[dict | None]:
        raise NotImplementedError

    @abc.abstractmethod
    async def get_edge(self, source_node_id: str, target_node_id: str) -> dict | None:
        raise NotImplementedError

    @abc.abstractmethod
    async def get_edges_batch(self, edge_pairs: list[tuple[str, str]]) -> list[dict | None]:
        raise NotImplementedError

    @abc.abstractmethod
    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        raise NotImplementedError

    @abc.abstractmethod
    async def get_nodes_edges_batch(self, node_ids: list[str]) -> list[list[tuple[str, str]]]:
        raise NotImplementedError

    @abc.abstractmethod
    async def upsert_node(self, node_id: str, node_data: dict[str, Any]):
        raise NotImplementedError

    @abc.abstractmethod
    async def upsert_nodes_batch(self, nodes_data: list[tuple[str, dict[str, Any]]]):
        raise NotImplementedError

    @abc.abstractmethod
    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, Any]
    ):
        raise NotImplementedError

    @abc.abstractmethod
    async def upsert_edges_batch(self, edges_data: list[tuple[str, str, dict[str, Any]]]):
        raise NotImplementedError

    @abc.abstractmethod
    async def delete_node(self, node_id: str):
        raise NotImplementedError

    @abc.abstractmethod
    async def delete_nodes_batch(self, node_ids: list[str]):
        raise NotImplementedError

    @abc.abstractmethod
    async def delete_edge(self, source_node_id: str, target_node_id: str):
        raise NotImplementedError

    @abc.abstractmethod
    async def delete_edges_batch(self, edge_pairs: list[tuple[str, str]]):
        raise NotImplementedError

    @abc.abstractmethod
    async def clustering(self, algorithm: str, affected_node_ids: set[str] | None = None):
        raise NotImplementedError

    @abc.abstractmethod
    async def community_schema(self) -> dict[str, SingleCommunitySchema]:
        raise NotImplementedError

    async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        raise NotImplementedError("Node embedding is not used in nano-graphrag.")


# =============================================================================
# GraphRAG Configuration
# =============================================================================


@dataclass
class _ConfigFields:
    """Config fields shared between GraphRAG and GraphRAGConfig.

    Defaults are auto-synced from GraphRAGSettings via FLAT_DEFAULTS.
    Do not instantiate directly — use GraphRAGConfig or GraphRAG instead.
    """

    # === Core ===
    working_dir: str = FLAT_DEFAULTS["working_dir"]

    # === Shared API credentials (LLM + Embedding use same key by default) ===
    api_key: str | None = FLAT_DEFAULTS["api_key"]
    api_base: str | None = FLAT_DEFAULTS["api_base"]

    # === LLM (passed to LiteLLM) ===
    llm_model: str = FLAT_DEFAULTS["llm_model"]
    llm_cheap_model: str = FLAT_DEFAULTS["llm_cheap_model"]
    llm_api_base: str | None = FLAT_DEFAULTS["llm_api_base"]
    llm_api_key: str | None = FLAT_DEFAULTS["llm_api_key"]
    llm_max_async: int = FLAT_DEFAULTS["llm_max_async"]
    llm_max_tokens: int = FLAT_DEFAULTS["llm_max_tokens"]
    llm_timeout: int = FLAT_DEFAULTS["llm_timeout"]

    # === Embedding ===
    embedding_model: str = FLAT_DEFAULTS["embedding_model"]
    embedding_api_base: str | None = FLAT_DEFAULTS["embedding_api_base"]
    embedding_api_key: str | None = FLAT_DEFAULTS["embedding_api_key"]
    embedding_dim: int = FLAT_DEFAULTS["embedding_dim"]
    embedding_max_async: int = FLAT_DEFAULTS["embedding_max_async"]
    embedding_batch_size: int = FLAT_DEFAULTS["embedding_batch_size"]

    # === Compute/Quality ===
    extraction_max_async: int = FLAT_DEFAULTS["extraction_max_async"]
    extraction_batch_size: int = FLAT_DEFAULTS["extraction_batch_size"]
    doc_extraction_max_async: int = FLAT_DEFAULTS["doc_extraction_max_async"]
    doc_flush_batch_size: int = FLAT_DEFAULTS["doc_flush_batch_size"]
    entity_extraction_quality: str = FLAT_DEFAULTS["entity_extraction_quality"]
    extraction_backend: str = FLAT_DEFAULTS["extraction_backend"]
    graph_cluster_algorithm: str = FLAT_DEFAULTS["graph_cluster_algorithm"]
    max_incremental_updates_before_full: int = FLAT_DEFAULTS["max_incremental_updates_before_full"]
    alias_batch_size: int = FLAT_DEFAULTS["alias_batch_size"]
    enable_node_embedding: bool = FLAT_DEFAULTS["enable_node_embedding"]

    # === Features ===
    enable_local: bool = FLAT_DEFAULTS["enable_local"]
    enable_naive_rag: bool = FLAT_DEFAULTS["enable_naive_rag"]
    enable_llm_cache: bool = FLAT_DEFAULTS["enable_llm_cache"]
    enable_entity_linking: bool = FLAT_DEFAULTS["enable_entity_linking"]
    entity_linking_use_neighborhood_evidence: bool = FLAT_DEFAULTS[
        "entity_linking_use_neighborhood_evidence"
    ]
    enable_community_reports: bool = FLAT_DEFAULTS["enable_community_reports"]
    enable_temporal_extraction: bool = FLAT_DEFAULTS["enable_temporal_extraction"]
    entity_linking_similarity_threshold: float = FLAT_DEFAULTS[
        "entity_linking_similarity_threshold"
    ]
    entity_linking_max_candidates: int = FLAT_DEFAULTS["entity_linking_max_candidates"]
    entity_linking_iou_threshold: float = FLAT_DEFAULTS["entity_linking_iou_threshold"]
    entity_linking_min_common_neighbors: int = FLAT_DEFAULTS["entity_linking_min_common_neighbors"]
    alias_max_batches_in_flight: int = FLAT_DEFAULTS["alias_max_batches_in_flight"]
    entity_count_min_ratio: float = FLAT_DEFAULTS["entity_count_min_ratio"]
    entity_count_min_absolute: int = FLAT_DEFAULTS["entity_count_min_absolute"]

    # === Logging ===
    log_level: str = FLAT_DEFAULTS["log_level"]
    log_file: str | None = FLAT_DEFAULTS["log_file"]
    log_query_text: bool = FLAT_DEFAULTS["log_query_text"]

    # === Refinement Pipeline ===
    enable_refinement: bool = FLAT_DEFAULTS["enable_refinement"]
    refinement_merge_threshold: float = FLAT_DEFAULTS["refinement_merge_threshold"]
    refinement_enrich_min_chars: int = FLAT_DEFAULTS["refinement_enrich_min_chars"]
    refinement_infer_confidence: float = FLAT_DEFAULTS["refinement_infer_confidence"]
    refinement_batch_size: int = FLAT_DEFAULTS["refinement_batch_size"]
    refinement_infer_hub_cap: int = FLAT_DEFAULTS["refinement_infer_hub_cap"]
    relationship_confidence_threshold: float = FLAT_DEFAULTS["relationship_confidence_threshold"]

    # === Vault Export ===
    vault_path: str = FLAT_DEFAULTS["vault_path"]
    vault_export_communities: bool = FLAT_DEFAULTS["vault_export_communities"]

    # === SAGE-Inspired Improvements (Phases 1-4) ===
    enable_query_planning: bool = FLAT_DEFAULTS["enable_query_planning"]
    structural_feature_weights: list[float] = field(
        default_factory=lambda: list(FLAT_DEFAULTS["structural_feature_weights"])
    )
    edge_gate_threshold: float = FLAT_DEFAULTS["edge_gate_threshold"]
    propagation_hops: int = FLAT_DEFAULTS["propagation_hops"]
    subgraph_prune_ratio: float = FLAT_DEFAULTS["subgraph_prune_ratio"]
    edge_gate_decay: float = FLAT_DEFAULTS["edge_gate_decay"]
    enable_retrieval_feedback: bool = FLAT_DEFAULTS["enable_retrieval_feedback"]
    re_extraction_recall_threshold: float = FLAT_DEFAULTS["re_extraction_recall_threshold"]


@dataclass
class GraphRAGConfig(_ConfigFields):
    """Configuration wrapper around GraphRAGSettings.

    Supports three loading methods:
        - ``from_env()``: Load from environment variables
        - ``from_yaml(path)``: Load from YAML (flat or nested format)
        - ``from_dict(data)``: Load from a dictionary

    Example::

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
