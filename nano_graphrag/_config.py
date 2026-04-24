import os
from typing import Any, Dict, Literal, Optional, cast

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_LLM_MODEL = "openrouter/google/gemma-4-31b-it"
DEFAULT_CHEAP_MODEL = "openrouter/google/gemma-4-31b-it"
DEFAULT_EMBEDDING_MODEL = "openrouter/qwen/qwen3-embedding-8b"
DEFAULT_EMBEDDING_DIM = 4096
SUPPORTED_GRAPH_CLUSTERING = ("leiden", "louvain")


class LLMConfig(BaseModel):
    model: str = DEFAULT_LLM_MODEL
    cheap_model: str = DEFAULT_CHEAP_MODEL
    api_base: Optional[str] = None
    api_key: Optional[str] = Field(None, alias="LLM_API_KEY")
    max_async: int = Field(32, ge=1)
    max_tokens: int = Field(32768, ge=1)
    timeout: int = Field(120, ge=1)


class EmbeddingConfig(BaseModel):
    model: str = DEFAULT_EMBEDDING_MODEL
    api_base: Optional[str] = None
    api_key: Optional[str] = Field(None, alias="EMBEDDING_API_KEY")
    dim: int = Field(DEFAULT_EMBEDDING_DIM, ge=1)
    max_async: int = Field(16, ge=1)
    batch_size: int = Field(32, ge=1)


class ExtractionConfig(BaseModel):
    max_async: int = Field(16, ge=1)
    batch_size: int = Field(5, ge=1)
    doc_max_async: int = Field(4, ge=1)
    doc_flush_batch_size: int = Field(50, ge=1)
    quality: Literal["fast", "balanced", "thorough"] = "balanced"
    backend: Literal["llm", "gliner"] = "llm"


class ClusteringConfig(BaseModel):
    algorithm: Literal["leiden", "louvain"] = "leiden"
    max_incremental_updates_before_full: int = Field(10, ge=1)
    alias_batch_size: int = Field(20, ge=1)
    alias_max_batches_in_flight: int = Field(5, ge=1)


class EntityLinkingConfig(BaseModel):
    enabled: bool = False
    use_neighborhood_evidence: bool = True
    similarity_threshold: float = Field(0.92, ge=0.0, le=1.0)
    max_candidates: int = Field(3, ge=1)
    iou_threshold: float = Field(0.3, ge=0.0, le=1.0)
    min_common_neighbors: int = Field(2, ge=0)


class EntityFilterConfig(BaseModel):
    count_min_ratio: float = Field(2.0, ge=0.0)
    count_min_absolute: int = Field(3, ge=0)


class FeatureFlags(BaseModel):
    node_embedding: bool = False
    local_search: bool = True
    naive_rag: bool = False
    llm_cache: bool = True
    community_reports: bool = True


class LoggingConfig(BaseModel):
    level: str = "INFO"
    file: Optional[str] = None

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid:
            raise ValueError(f"Invalid log_level={v!r}. Must be one of: {sorted(valid)}")
        return v.upper()


class GraphRAGSettings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    working_dir: str = "./nano_graphrag"
    llm: LLMConfig = LLMConfig()  # type: ignore[call-arg]
    embedding: EmbeddingConfig = EmbeddingConfig()  # type: ignore[call-arg]
    extraction: ExtractionConfig = ExtractionConfig()  # type: ignore[call-arg]
    clustering: ClusteringConfig = ClusteringConfig()  # type: ignore[call-arg]
    entity_linking: EntityLinkingConfig = EntityLinkingConfig()  # type: ignore[call-arg]
    entity_filter: EntityFilterConfig = EntityFilterConfig()  # type: ignore[call-arg]
    features: FeatureFlags = FeatureFlags()  # type: ignore[call-arg]
    logging: LoggingConfig = LoggingConfig()  # type: ignore[call-arg]

    @classmethod
    def from_yaml(cls, path: str) -> "GraphRAGSettings":
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return cls._from_flat_or_nested(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphRAGSettings":
        return cls._from_flat_or_nested(data)

    @classmethod
    def from_env(cls) -> "GraphRAGSettings":
        return cls(
            working_dir=os.getenv("GRAPH_WORKING_DIR", "./nano_graphrag"),
            llm=LLMConfig(
                model=os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL),
                cheap_model=os.getenv("LLM_CHEAP_MODEL", DEFAULT_CHEAP_MODEL),
                api_base=os.getenv("LLM_API_BASE"),
                LLM_API_KEY=os.getenv("LLM_API_KEY"),
                max_async=_parse_int_env("LLM_MAX_ASYNC", 32, 1),
                max_tokens=_parse_int_env("LLM_MAX_TOKENS", 32768, 1),
                timeout=_parse_int_env("LLM_TIMEOUT", 120, 1),
            ),
            embedding=EmbeddingConfig(
                model=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
                api_base=os.getenv("EMBEDDING_API_BASE"),
                EMBEDDING_API_KEY=os.getenv("EMBEDDING_API_KEY"),
                dim=_parse_int_env("EMBEDDING_DIM", DEFAULT_EMBEDDING_DIM, 1),
                max_async=_parse_int_env("EMBEDDING_MAX_ASYNC", 16, 1),
                batch_size=_parse_int_env("EMBEDDING_BATCH_SIZE", 32, 1),
            ),
            extraction=ExtractionConfig(
                max_async=_parse_int_env("EXTRACTION_MAX_ASYNC", 16, 1),
                batch_size=_parse_int_env("EXTRACTION_BATCH_SIZE", 5, 1),
                doc_max_async=_parse_int_env("DOC_EXTRACTION_MAX_ASYNC", 4, 1),
                doc_flush_batch_size=_parse_int_env("DOC_FLUSH_BATCH_SIZE", 50, 1),
                quality=cast(
                    Literal["fast", "balanced", "thorough"],
                    os.getenv("ENTITY_EXTRACTION_QUALITY", "balanced"),
                ),
                backend=cast(Literal["llm", "gliner"], os.getenv("EXTRACTION_BACKEND", "llm")),
            ),
            clustering=ClusteringConfig(
                algorithm=cast(
                    Literal["leiden", "louvain"], os.getenv("GRAPH_CLUSTER_ALGORITHM", "leiden")
                ),
                max_incremental_updates_before_full=_parse_int_env(
                    "MAX_INCREMENTAL_UPDATES_BEFORE_FULL", 10, 1
                ),
                alias_batch_size=_parse_int_env("ALIAS_BATCH_SIZE", 20, 1),
                alias_max_batches_in_flight=_parse_int_env("ALIAS_MAX_BATCHES_IN_FLIGHT", 5, 1),
            ),
            entity_linking=EntityLinkingConfig(
                enabled=_parse_bool_env("ENABLE_ENTITY_LINKING", False),
                use_neighborhood_evidence=_parse_bool_env(
                    "ENTITY_LINKING_USE_NEIGHBORHOOD_EVIDENCE", True
                ),
                similarity_threshold=float(
                    os.getenv("ENTITY_LINKING_SIMILARITY_THRESHOLD", "0.92")
                ),
                max_candidates=_parse_int_env("ENTITY_LINKING_MAX_CANDIDATES", 3, 1),
                iou_threshold=float(os.getenv("ENTITY_LINKING_IOU_THRESHOLD", "0.3")),
                min_common_neighbors=_parse_int_env("ENTITY_LINKING_MIN_COMMON_NEIGHBORS", 2, 0),
            ),
            entity_filter=EntityFilterConfig(
                count_min_ratio=float(os.getenv("ENTITY_COUNT_MIN_RATIO", "2.0")),
                count_min_absolute=_parse_int_env("ENTITY_COUNT_MIN_ABSOLUTE", 3, 0),
            ),
            features=FeatureFlags(
                node_embedding=_parse_bool_env("ENABLE_NODE_EMBEDDING", False),
                local_search=_parse_bool_env("ENABLE_LOCAL", True),
                naive_rag=_parse_bool_env("ENABLE_NAIVE_RAG", False),
                llm_cache=_parse_bool_env("ENABLE_LLM_CACHE", True),
                community_reports=_parse_bool_env("ENABLE_COMMUNITY_REPORTS", True),
            ),
            logging=LoggingConfig(
                level=os.getenv("LOG_LEVEL", "INFO"),
                file=os.getenv("LOG_FILE"),
            ),
        )

    def merge(self, overrides: Dict[str, Any]) -> "GraphRAGSettings":
        base = self.to_flat_dict()
        merged = {**base, **{k: v for k, v in overrides.items() if v is not None}}
        return self._from_flat_or_nested(merged)

    def to_flat_dict(self) -> Dict[str, Any]:
        return _flatten_settings(self)

    def to_yaml(self, path: str):
        with open(path, "w") as f:
            yaml.dump(
                self.model_dump(mode="json"),
                f,
                default_flow_style=False,
                sort_keys=False,
            )

    @classmethod
    def _from_flat_or_nested(cls, data: Dict[str, Any]) -> "GraphRAGSettings":
        if _is_nested_format(data):
            return cls(**data)
        return cls(**_unflatten_data(data))


FLAT_FIELD_TO_ENV_VAR: Dict[str, str] = {
    "working_dir": "GRAPH_WORKING_DIR",
    "llm_model": "LLM_MODEL",
    "llm_cheap_model": "LLM_CHEAP_MODEL",
    "llm_api_base": "LLM_API_BASE",
    "llm_api_key": "LLM_API_KEY",
    "llm_max_async": "LLM_MAX_ASYNC",
    "llm_max_tokens": "LLM_MAX_TOKENS",
    "llm_timeout": "LLM_TIMEOUT",
    "embedding_model": "EMBEDDING_MODEL",
    "embedding_api_base": "EMBEDDING_API_BASE",
    "embedding_api_key": "EMBEDDING_API_KEY",
    "embedding_dim": "EMBEDDING_DIM",
    "embedding_max_async": "EMBEDDING_MAX_ASYNC",
    "embedding_batch_size": "EMBEDDING_BATCH_SIZE",
    "extraction_max_async": "EXTRACTION_MAX_ASYNC",
    "extraction_batch_size": "EXTRACTION_BATCH_SIZE",
    "doc_extraction_max_async": "DOC_EXTRACTION_MAX_ASYNC",
    "doc_flush_batch_size": "DOC_FLUSH_BATCH_SIZE",
    "entity_extraction_quality": "ENTITY_EXTRACTION_QUALITY",
    "extraction_backend": "EXTRACTION_BACKEND",
    "graph_cluster_algorithm": "GRAPH_CLUSTER_ALGORITHM",
    "max_incremental_updates_before_full": "MAX_INCREMENTAL_UPDATES_BEFORE_FULL",
    "alias_batch_size": "ALIAS_BATCH_SIZE",
    "alias_max_batches_in_flight": "ALIAS_MAX_BATCHES_IN_FLIGHT",
    "enable_node_embedding": "ENABLE_NODE_EMBEDDING",
    "enable_local": "ENABLE_LOCAL",
    "enable_naive_rag": "ENABLE_NAIVE_RAG",
    "enable_llm_cache": "ENABLE_LLM_CACHE",
    "enable_entity_linking": "ENABLE_ENTITY_LINKING",
    "entity_linking_use_neighborhood_evidence": "ENTITY_LINKING_USE_NEIGHBORHOOD_EVIDENCE",
    "enable_community_reports": "ENABLE_COMMUNITY_REPORTS",
    "entity_linking_similarity_threshold": "ENTITY_LINKING_SIMILARITY_THRESHOLD",
    "entity_linking_max_candidates": "ENTITY_LINKING_MAX_CANDIDATES",
    "entity_linking_iou_threshold": "ENTITY_LINKING_IOU_THRESHOLD",
    "entity_linking_min_common_neighbors": "ENTITY_LINKING_MIN_COMMON_NEIGHBORS",
    "entity_count_min_ratio": "ENTITY_COUNT_MIN_RATIO",
    "entity_count_min_absolute": "ENTITY_COUNT_MIN_ABSOLUTE",
    "log_level": "LOG_LEVEL",
    "log_file": "LOG_FILE",
}


def _is_nested_format(data: Dict[str, Any]) -> bool:
    nested_keys = {
        "llm",
        "embedding",
        "extraction",
        "clustering",
        "entity_linking",
        "entity_filter",
        "features",
        "logging",
    }
    return bool(set(data.keys()) & nested_keys)


def _unflatten_data(data: Dict[str, Any]) -> Dict[str, Any]:
    mapping = {
        "llm_model": ("llm", "model"),
        "llm_cheap_model": ("llm", "cheap_model"),
        "llm_api_base": ("llm", "api_base"),
        "llm_api_key": ("llm", "api_key"),
        "llm_max_async": ("llm", "max_async"),
        "llm_max_tokens": ("llm", "max_tokens"),
        "llm_timeout": ("llm", "timeout"),
        "embedding_model": ("embedding", "model"),
        "embedding_api_base": ("embedding", "api_base"),
        "embedding_api_key": ("embedding", "api_key"),
        "embedding_dim": ("embedding", "dim"),
        "embedding_max_async": ("embedding", "max_async"),
        "embedding_batch_size": ("embedding", "batch_size"),
        "extraction_max_async": ("extraction", "max_async"),
        "extraction_batch_size": ("extraction", "batch_size"),
        "doc_extraction_max_async": ("extraction", "doc_max_async"),
        "doc_flush_batch_size": ("extraction", "doc_flush_batch_size"),
        "entity_extraction_quality": ("extraction", "quality"),
        "extraction_backend": ("extraction", "backend"),
        "graph_cluster_algorithm": ("clustering", "algorithm"),
        "max_incremental_updates_before_full": (
            "clustering",
            "max_incremental_updates_before_full",
        ),
        "alias_batch_size": ("clustering", "alias_batch_size"),
        "alias_max_batches_in_flight": ("clustering", "alias_max_batches_in_flight"),
        "enable_entity_linking": ("entity_linking", "enabled"),
        "entity_linking_use_neighborhood_evidence": ("entity_linking", "use_neighborhood_evidence"),
        "entity_linking_similarity_threshold": ("entity_linking", "similarity_threshold"),
        "entity_linking_max_candidates": ("entity_linking", "max_candidates"),
        "entity_linking_iou_threshold": ("entity_linking", "iou_threshold"),
        "entity_linking_min_common_neighbors": ("entity_linking", "min_common_neighbors"),
        "entity_count_min_ratio": ("entity_filter", "count_min_ratio"),
        "entity_count_min_absolute": ("entity_filter", "count_min_absolute"),
        "enable_node_embedding": ("features", "node_embedding"),
        "enable_local": ("features", "local_search"),
        "enable_naive_rag": ("features", "naive_rag"),
        "enable_llm_cache": ("features", "llm_cache"),
        "enable_community_reports": ("features", "community_reports"),
        "log_level": ("logging", "level"),
        "log_file": ("logging", "file"),
    }
    result: Dict[str, Any] = {}
    used_keys: set = set()
    for flat_key, (section, nested_key) in mapping.items():
        if flat_key in data:
            result.setdefault(section, {})[nested_key] = data[flat_key]
            used_keys.add(flat_key)
    if "working_dir" in data:
        result["working_dir"] = data["working_dir"]
    for k, v in data.items():
        if k not in used_keys and k != "working_dir":
            result.setdefault("features", {})[k] = v
    return result


def _flatten_settings(settings: GraphRAGSettings) -> Dict[str, Any]:
    reverse_mapping = {
        ("llm", "model"): "llm_model",
        ("llm", "cheap_model"): "llm_cheap_model",
        ("llm", "api_base"): "llm_api_base",
        ("llm", "api_key"): "llm_api_key",
        ("llm", "max_async"): "llm_max_async",
        ("llm", "max_tokens"): "llm_max_tokens",
        ("llm", "timeout"): "llm_timeout",
        ("embedding", "model"): "embedding_model",
        ("embedding", "api_base"): "embedding_api_base",
        ("embedding", "api_key"): "embedding_api_key",
        ("embedding", "dim"): "embedding_dim",
        ("embedding", "max_async"): "embedding_max_async",
        ("embedding", "batch_size"): "embedding_batch_size",
        ("extraction", "max_async"): "extraction_max_async",
        ("extraction", "batch_size"): "extraction_batch_size",
        ("extraction", "doc_max_async"): "doc_extraction_max_async",
        ("extraction", "doc_flush_batch_size"): "doc_flush_batch_size",
        ("extraction", "quality"): "entity_extraction_quality",
        ("extraction", "backend"): "extraction_backend",
        ("clustering", "algorithm"): "graph_cluster_algorithm",
        (
            "clustering",
            "max_incremental_updates_before_full",
        ): "max_incremental_updates_before_full",
        ("clustering", "alias_batch_size"): "alias_batch_size",
        ("clustering", "alias_max_batches_in_flight"): "alias_max_batches_in_flight",
        ("entity_linking", "enabled"): "enable_entity_linking",
        ("entity_linking", "use_neighborhood_evidence"): "entity_linking_use_neighborhood_evidence",
        ("entity_linking", "similarity_threshold"): "entity_linking_similarity_threshold",
        ("entity_linking", "max_candidates"): "entity_linking_max_candidates",
        ("entity_linking", "iou_threshold"): "entity_linking_iou_threshold",
        ("entity_linking", "min_common_neighbors"): "entity_linking_min_common_neighbors",
        ("entity_filter", "count_min_ratio"): "entity_count_min_ratio",
        ("entity_filter", "count_min_absolute"): "entity_count_min_absolute",
        ("features", "node_embedding"): "enable_node_embedding",
        ("features", "local_search"): "enable_local",
        ("features", "naive_rag"): "enable_naive_rag",
        ("features", "llm_cache"): "enable_llm_cache",
        ("features", "community_reports"): "enable_community_reports",
        ("logging", "level"): "log_level",
        ("logging", "file"): "log_file",
    }
    result: Dict[str, Any] = {"working_dir": settings.working_dir}
    dump = settings.model_dump(mode="json")
    for section, key in reverse_mapping:
        section_data = dump.get(section, {})
        if key in section_data:
            result[reverse_mapping[(section, key)]] = section_data[key]
    return result


def _parse_bool_env(env_var: str, default: bool = False) -> bool:
    value = os.getenv(env_var, "").lower()
    if value in ("true", "1", "yes", "on"):
        return True
    if value in ("false", "0", "no", "off"):
        return False
    return default


def _parse_int_env(env_var: str, default: int, min_value: Optional[int] = None) -> int:
    value = os.getenv(env_var, str(default))
    try:
        parsed = int(value)
        if min_value is not None and parsed < min_value:
            return default
        return parsed
    except ValueError:
        return default
