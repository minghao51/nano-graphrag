import os
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Literal, Optional, TypedDict, TypeVar, Union

import numpy as np

from ._utils import EmbeddingFunc, logger


@dataclass
class QueryParam:
    mode: Literal["local", "global", "naive", "entity_grounded"] = "global"
    only_need_context: bool = False
    response_type: str = "Multiple Paragraphs"
    level: int = 2
    top_k: int = 20
    # naive search
    naive_max_token_for_text_unit = 12000
    # local search
    local_max_token_for_text_unit: int = 4000  # 12000 * 0.33
    local_max_token_for_local_context: int = 4800  # 12000 * 0.4
    local_max_token_for_community_report: int = 3200  # 12000 * 0.27
    local_community_single_one: bool = False
    # global search
    global_min_community_rating: float = 0
    global_max_consider_community: float = 512
    global_max_token_for_community_report: int = 16384
    global_special_community_map_llm_kwargs: dict = field(
        default_factory=lambda: {"response_format": {"type": "json_object"}}
    )
    # entity-grounded search
    entity_grounded_max_answer_length: int = 50  # tokens
    entity_grounded_require_entity_match: bool = True
    entity_grounded_fuzzy_threshold: float = 0.85


TextChunkSchema = TypedDict(
    "TextChunkSchema",
    {"tokens": int, "content": str, "full_doc_id": str, "chunk_order_index": int},
)

SingleCommunitySchema = TypedDict(
    "SingleCommunitySchema",
    {
        "level": int,
        "title": str,
        "edges": list[tuple[str, str]],
        "nodes": list[str],
        "chunk_ids": list[str],
        "occurrence": float,
        "sub_communities": list[str],
    },
)


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

    async def get_by_id(self, id: str) -> Union[T, None]:
        raise NotImplementedError

    async def get_by_ids(
        self, ids: list[str], fields: Union[set[str], None] = None
    ) -> list[Union[T, None]]:
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

    async def node_degrees_batch(self, node_ids: List[str]) -> List[str]:
        raise NotImplementedError

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        raise NotImplementedError

    async def edge_degrees_batch(self, edge_pairs: list[tuple[str, str]]) -> list[int]:
        raise NotImplementedError

    async def get_node(self, node_id: str) -> Union[dict, None]:
        raise NotImplementedError

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, Union[dict, None]]:
        raise NotImplementedError

    async def get_edge(self, source_node_id: str, target_node_id: str) -> Union[dict, None]:
        raise NotImplementedError

    async def get_edges_batch(self, edge_pairs: list[tuple[str, str]]) -> list[Union[dict, None]]:
        raise NotImplementedError

    async def get_node_edges(self, source_node_id: str) -> Union[list[tuple[str, str]], None]:
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

    async def clustering(self, algorithm: str, affected_node_ids: Optional[set[str]] = None):
        raise NotImplementedError

    async def community_schema(self) -> dict[str, SingleCommunitySchema]:
        """Return the community representation with report and nodes"""
        raise NotImplementedError

    async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        raise NotImplementedError("Node embedding is not used in nano-graphrag.")


# =============================================================================
# GraphRAG Configuration
# =============================================================================

DEFAULT_LLM_MODEL = "gpt-4o-mini"
DEFAULT_CHEAP_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "openrouter/qwen/qwen3-embedding-8b"
DEFAULT_EMBEDDING_DIM = 4096

# Alternative embedding models (set via EMBEDDING_MODEL env var):
# - openrouter/qwen/qwen3-embedding-8b (dim=4096) - Default, good quality, cost-effective
# - openrouter/openai/text-embedding-3-small (dim=1536) - OpenAI via OpenRouter
# - openrouter/baai/bge-m3 (dim=1024) - Multilingual support, long context
# - openrouter/sentence-transformers/all-mpnet-base-v2 (dim=768) - Lightweight
# - text-embedding-3-small (dim=1536) - Direct OpenAI
# Note: OpenRouter doesn't document specific concurrent call limits for embeddings

SUPPORTED_GRAPH_CLUSTERING = ("leiden",)


def _parse_bool(env_var: str, default: bool = False) -> bool:
    """Parse boolean from environment variable.

    Accepts: true, 1, yes, on (case-insensitive) as True
             false, 0, no, off (case-insensitive) as False
    """
    value = os.getenv(env_var, "").lower()
    if value in ("true", "1", "yes", "on"):
        return True
    if value in ("false", "0", "no", "off"):
        return False
    return default


def _parse_int(env_var: str, default: int, min_value: Optional[int] = None) -> int:
    """Parse integer from environment variable with validation.

    Args:
        env_var: Environment variable name
        default: Default value if not set
        min_value: Minimum valid value (optional)

    Returns:
        Parsed integer value or default if invalid
    """
    value = os.getenv(env_var, str(default))
    try:
        parsed = int(value)
        if min_value is not None and parsed < min_value:
            logger.warning(
                f"Invalid {env_var}={value}: Must be >= {min_value}, using default {default}"
            )
            return default
        return parsed
    except ValueError:
        logger.warning(f"Invalid {env_var}={value}: Not an integer, using default {default}")
        return default


@dataclass
class GraphRAGConfig:
    """Main configuration for GraphRAG.

    Can be loaded from environment variables, dict, or instantiated directly.

    Environment Variables:
        GRAPH_WORKING_DIR: Working directory for storage
        LLM_MODEL: LLM model name (e.g., "ollama/llama3.2", "gpt-4o-mini")
        LLM_CHEAP_MODEL: Cheap model for summarization
        LLM_API_BASE: Base URL for LLM API (e.g., "http://localhost:11434")
        LLM_API_KEY: API key for LLM
        LLM_MAX_ASYNC: Max concurrent LLM calls
        LLM_MAX_TOKENS: Max tokens for LLM response
        LLM_TIMEOUT: Timeout for LLM calls in seconds
        EMBEDDING_MODEL: Embedding model name
        EMBEDDING_API_BASE: Base URL for embedding API
        EMBEDDING_API_KEY: API key for embedding
        EMBEDDING_DIM: Embedding dimension
        EMBEDDING_MAX_ASYNC: Max concurrent embedding calls
        EMBEDDING_BATCH_SIZE: Batch size for embedding
        EXTRACTION_MAX_ASYNC: Max concurrent entity extraction
        CHUNK_BATCH_SIZE: Chunk batch size for processing
        ENTITY_EXTRACTION_QUALITY: "fast" | "balanced" | "thorough"
        GRAPH_CLUSTER_ALGORITHM: "leiden"
        ENABLE_NODE_EMBEDDING: Enable node embedding (default: False)
    """

    # === Core ===
    working_dir: str = "./nano_graphrag"

    # === LLM (passed to LiteLLM) ===
    llm_model: str = DEFAULT_LLM_MODEL
    llm_cheap_model: str = DEFAULT_CHEAP_MODEL
    llm_api_base: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_max_async: int = 32
    llm_max_tokens: int = 32768
    llm_timeout: int = 120

    # === Embedding ===
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_api_base: Optional[str] = None
    embedding_api_key: Optional[str] = None
    embedding_dim: int = DEFAULT_EMBEDDING_DIM
    embedding_max_async: int = 16
    embedding_batch_size: int = 32

    # === Compute/Quality ===
    extraction_max_async: int = 16
    entity_extraction_quality: str = "balanced"
    graph_cluster_algorithm: str = "leiden"
    enable_node_embedding: bool = False

    # === Features ===
    enable_local: bool = True
    enable_naive_rag: bool = False
    enable_llm_cache: bool = True

    # === Logging ===
    log_level: str = "INFO"
    log_file: Optional[str] = None

    # === Legacy/Deprecated (for backward compatibility) ===
    # These are mapped to the new config but will emit deprecation warnings

    @classmethod
    def from_env(cls) -> "GraphRAGConfig":
        """Load configuration from environment variables."""
        return cls(
            working_dir=os.getenv("GRAPH_WORKING_DIR", "./nano_graphrag"),
            llm_model=os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL),
            llm_cheap_model=os.getenv("LLM_CHEAP_MODEL", DEFAULT_CHEAP_MODEL),
            llm_api_base=os.getenv("LLM_API_BASE"),
            llm_api_key=os.getenv("LLM_API_KEY"),
            llm_max_async=_parse_int("LLM_MAX_ASYNC", 32, min_value=1),
            llm_max_tokens=_parse_int("LLM_MAX_TOKENS", 32768, min_value=1),
            llm_timeout=_parse_int("LLM_TIMEOUT", 120, min_value=1),
            embedding_model=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            embedding_api_base=os.getenv("EMBEDDING_API_BASE"),
            embedding_api_key=os.getenv("EMBEDDING_API_KEY"),
            embedding_dim=_parse_int("EMBEDDING_DIM", DEFAULT_EMBEDDING_DIM, min_value=1),
            embedding_max_async=_parse_int("EMBEDDING_MAX_ASYNC", 16, min_value=1),
            embedding_batch_size=_parse_int("EMBEDDING_BATCH_SIZE", 32, min_value=1),
            extraction_max_async=_parse_int("EXTRACTION_MAX_ASYNC", 16, min_value=1),
            entity_extraction_quality=os.getenv("ENTITY_EXTRACTION_QUALITY", "balanced"),
            graph_cluster_algorithm=os.getenv("GRAPH_CLUSTER_ALGORITHM", "leiden"),
            enable_node_embedding=_parse_bool("ENABLE_NODE_EMBEDDING", False),
            enable_local=_parse_bool("ENABLE_LOCAL", True),
            enable_naive_rag=_parse_bool("ENABLE_NAIVE_RAG", False),
            enable_llm_cache=_parse_bool("ENABLE_LLM_CACHE", True),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_file=os.getenv("LOG_FILE"),
        )

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "GraphRAGConfig":
        """Create config from dictionary, only setting non-None values."""
        # Filter out None values
        filtered = {k: v for k, v in config.items() if v is not None}
        return cls(**filtered)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "working_dir": self.working_dir,
            "llm_model": self.llm_model,
            "llm_cheap_model": self.llm_cheap_model,
            "llm_api_base": self.llm_api_base,
            "llm_api_key": self.llm_api_key,
            "llm_max_async": self.llm_max_async,
            "llm_max_tokens": self.llm_max_tokens,
            "llm_timeout": self.llm_timeout,
            "embedding_model": self.embedding_model,
            "embedding_api_base": self.embedding_api_base,
            "embedding_api_key": self.embedding_api_key,
            "embedding_dim": self.embedding_dim,
            "embedding_max_async": self.embedding_max_async,
            "embedding_batch_size": self.embedding_batch_size,
            "extraction_max_async": self.extraction_max_async,
            "entity_extraction_quality": self.entity_extraction_quality,
            "graph_cluster_algorithm": self.graph_cluster_algorithm,
            "enable_node_embedding": self.enable_node_embedding,
            "enable_local": self.enable_local,
            "enable_naive_rag": self.enable_naive_rag,
            "enable_llm_cache": self.enable_llm_cache,
            "log_level": self.log_level,
            "log_file": self.log_file,
        }

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate entity_extraction_quality
        valid_quality_modes = {"fast", "balanced", "thorough"}
        if self.entity_extraction_quality not in valid_quality_modes:
            raise ValueError(
                f"Invalid entity_extraction_quality={self.entity_extraction_quality!r}. "
                f"Must be one of: {sorted(valid_quality_modes)}"
            )

        if self.graph_cluster_algorithm not in SUPPORTED_GRAPH_CLUSTERING:
            raise ValueError(
                f"Invalid graph_cluster_algorithm={self.graph_cluster_algorithm!r}. "
                f"Must be one of: {sorted(SUPPORTED_GRAPH_CLUSTERING)}"
            )

        # Validate log_level

        valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level.upper() not in valid_log_levels:
            raise ValueError(
                f"Invalid log_level={self.log_level!r}. Must be one of: {sorted(valid_log_levels)}"
            )

    def merge(self, overrides: Dict[str, Any]) -> "GraphRAGConfig":
        """Merge with overrides (env < base < overrides)."""
        base_dict = self.to_dict()
        # Only override non-None values
        merged = {**base_dict, **{k: v for k, v in overrides.items() if v is not None}}
        return GraphRAGConfig.from_dict(merged)

    @classmethod
    def from_yaml(cls, path: str) -> "GraphRAGConfig":
        """Load config from YAML file.

        Example:
            config = GraphRAGConfig.from_yaml("config.yaml")
            rag = GraphRAG.from_config(config)
        """
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required to load YAML configs. Install with: uv add pyyaml"
            )

        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def to_yaml(self, path: str):
        """Save config to YAML file.

        Example:
            config = GraphRAGConfig.from_env()
            config.to_yaml("config.yaml")
        """
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required to save YAML configs. Install with: uv add pyyaml"
            )

        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
