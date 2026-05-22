from __future__ import annotations

import pytest

from nano_graphrag._exceptions import (
    AuthError,
    ConfigError,
    ExtractionError,
    GraphIntegrityError,
    GraphRAGError,
    LLMError,
    LLMExtractionError,
    ModeNotEnabledError,
    NoContextError,
    ParsingError,
    QueryError,
    RateLimitError,
    StorageConfigError,
    StorageError,
    VectorDBError,
)
from nano_graphrag._schemas import InsertResult, QueryResult, QuerySource, TokenUsage
from nano_graphrag.graphrag import _resolve_storage


class TestExceptionHierarchy:
    def test_base_error(self):
        e = GraphRAGError("test error")
        assert str(e) == "test error"
        assert e.message == "test error"
        assert e.details == {}

    def test_error_with_details(self):
        e = ConfigError("bad config", details={"field": "embedding_dim", "value": -1})
        assert "bad config" in str(e)
        assert e.details["field"] == "embedding_dim"

    def test_error_with_cause(self):
        original = ValueError("original")
        e = LLMError("wrapped", cause=original)
        assert e.__cause__ is original

    def test_inheritance_chain(self):
        assert issubclass(ConfigError, GraphRAGError)
        assert issubclass(StorageConfigError, ConfigError)
        assert issubclass(ExtractionError, GraphRAGError)
        assert issubclass(LLMExtractionError, ExtractionError)
        assert issubclass(ParsingError, ExtractionError)
        assert issubclass(QueryError, GraphRAGError)
        assert issubclass(ModeNotEnabledError, QueryError)
        assert issubclass(NoContextError, QueryError)
        assert issubclass(StorageError, GraphRAGError)
        assert issubclass(GraphIntegrityError, StorageError)
        assert issubclass(VectorDBError, StorageError)
        assert issubclass(LLMError, GraphRAGError)
        assert issubclass(RateLimitError, LLMError)
        assert issubclass(AuthError, LLMError)

    def test_all_catchable_as_base(self):
        errors = [
            ConfigError("c"),
            StorageConfigError("sc"),
            ExtractionError("e"),
            LLMExtractionError("le"),
            ParsingError("p"),
            QueryError("q"),
            ModeNotEnabledError("m"),
            NoContextError("n"),
            StorageError("s"),
            GraphIntegrityError("gi"),
            VectorDBError("v"),
            LLMError("l"),
            RateLimitError("r"),
            AuthError("a"),
        ]
        for err in errors:
            assert isinstance(err, GraphRAGError)


class TestQueryResult:
    def test_str_returns_answer(self):
        qr = QueryResult(answer="hello world", mode="local")
        assert str(qr) == "hello world"

    def test_bool_true(self):
        qr = QueryResult(answer="non-empty", mode="global")
        assert bool(qr) is True

    def test_bool_false(self):
        qr = QueryResult(answer="", mode="global")
        assert bool(qr) is False

    def test_with_sources(self):
        src = QuerySource(source_type="entity", id="e1", name="Entity1", relevance_score=0.95)
        qr = QueryResult(answer="test", mode="local", sources=[src])
        assert len(qr.sources) == 1
        assert qr.sources[0].source_type == "entity"
        assert qr.sources[0].name == "Entity1"

    def test_with_token_usage(self):
        tu = TokenUsage(
            prompt_tokens=100, completion_tokens=50, total_tokens=150, estimated_cost_usd=0.01
        )
        qr = QueryResult(answer="test", mode="local", tokens_used=tu)
        assert qr.tokens_used.total_tokens == 150
        assert qr.tokens_used.estimated_cost_usd == 0.01

    def test_defaults(self):
        qr = QueryResult(answer="test", mode="global")
        assert qr.sources == []
        assert qr.tokens_used is None
        assert qr.latency_ms == 0.0
        assert qr.metadata == {}


class TestInsertResult:
    def test_basic(self):
        ir = InsertResult(documents_processed=5)
        assert ir.documents_processed == 5
        assert ir.documents_skipped == 0
        assert ir.entities_created == 0
        assert ir.relationships_created == 0
        assert ir.communities_updated == 0
        assert ir.tokens_used is None
        assert ir.latency_ms == 0.0

    def test_with_all_fields(self):
        tu = TokenUsage(prompt_tokens=500, completion_tokens=200, total_tokens=700)
        ir = InsertResult(
            documents_processed=10,
            documents_skipped=3,
            entities_created=45,
            relationships_created=30,
            communities_updated=5,
            tokens_used=tu,
            latency_ms=1200.5,
        )
        assert ir.documents_skipped == 3
        assert ir.entities_created == 45
        assert ir.tokens_used.total_tokens == 700


class TestTokenUsage:
    def test_basic(self):
        tu = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        assert tu.estimated_cost_usd is None

    def test_with_cost(self):
        tu = TokenUsage(
            prompt_tokens=100, completion_tokens=50, total_tokens=150, estimated_cost_usd=0.005
        )
        assert tu.estimated_cost_usd == 0.005


class TestQuerySource:
    def test_entity_source(self):
        qs = QuerySource(source_type="entity", id="e1", name="Entity1")
        assert qs.source_type == "entity"
        assert qs.relevance_score is None
        assert qs.text_snippet is None

    def test_community_source(self):
        qs = QuerySource(source_type="community", id="c1", name="Community A", relevance_score=0.85)
        assert qs.source_type == "community"
        assert qs.relevance_score == 0.85


class TestStorageResolution:
    def test_resolve_known_backends(self):
        from nano_graphrag._storage import HNSWVectorStorage, JsonKVStorage, NetworkXStorage

        assert _resolve_storage("json") is JsonKVStorage
        assert _resolve_storage("sqlite") is JsonKVStorage
        assert _resolve_storage("hnsw") is HNSWVectorStorage
        assert _resolve_storage("networkx") is NetworkXStorage

    def test_resolve_case_insensitive(self):
        from nano_graphrag._storage import NetworkXStorage

        assert _resolve_storage("NetworkX") is NetworkXStorage
        assert _resolve_storage("NETWORKX") is NetworkXStorage

    def test_resolve_none_passthrough(self):
        assert _resolve_storage(None) is None

    def test_resolve_class_passthrough(self):
        from nano_graphrag._storage import NetworkXStorage

        assert _resolve_storage(NetworkXStorage) is NetworkXStorage

    def test_resolve_unknown_raises(self):
        with pytest.raises(StorageConfigError, match="Unknown storage backend"):
            _resolve_storage("nonexistent_backend")

    def test_resolve_whitespace_stripped(self):
        from nano_graphrag._storage import NetworkXStorage

        assert _resolve_storage("  networkx  ") is NetworkXStorage


class TestEagerValidation:
    def test_invalid_cluster_algorithm(self, clean_working_dir):
        from nano_graphrag import GraphRAG

        with pytest.raises(ConfigError, match="Unsupported graph_cluster_algorithm"):
            GraphRAG(working_dir=clean_working_dir, graph_cluster_algorithm="invalid")

    def test_invalid_extraction_quality(self, clean_working_dir):
        from nano_graphrag import GraphRAG

        with pytest.raises(ConfigError, match="entity_extraction_quality"):
            GraphRAG(working_dir=clean_working_dir, entity_extraction_quality="super")

    def test_invalid_extraction_batch_size(self, clean_working_dir):
        from nano_graphrag import GraphRAG

        with pytest.raises(ConfigError, match="extraction_batch_size"):
            GraphRAG(working_dir=clean_working_dir, extraction_batch_size=0)

    def test_invalid_embedding_dim(self, clean_working_dir):
        from nano_graphrag import GraphRAG

        with pytest.raises(ConfigError, match="embedding_dim"):
            GraphRAG(working_dir=clean_working_dir, embedding_dim=0)

    def test_valid_config_passes(self, clean_working_dir):
        from nano_graphrag import GraphRAG

        rag = GraphRAG(working_dir=clean_working_dir)
        assert rag is not None


class TestModePermissions:
    @pytest.mark.asyncio
    async def test_mode_not_enabled_error(self, clean_working_dir):
        from nano_graphrag import GraphRAG, QueryParam

        rag = GraphRAG(working_dir=clean_working_dir, enable_local=False, enable_naive_rag=False)
        with pytest.raises(ModeNotEnabledError, match="enable_local"):
            await rag.aquery("test", QueryParam(mode="local"))

    @pytest.mark.asyncio
    async def test_naive_mode_not_enabled(self, clean_working_dir):
        from nano_graphrag import GraphRAG, QueryParam

        rag = GraphRAG(working_dir=clean_working_dir, enable_naive_rag=False)
        with pytest.raises(ModeNotEnabledError, match="enable_naive_rag"):
            await rag.aquery("test", QueryParam(mode="naive"))


class TestAllExports:
    def test_all_exports_importable(self):
        import nano_graphrag

        for name in nano_graphrag.__all__:
            assert hasattr(nano_graphrag, name), (
                f"__all__ contains {name!r} but it is not importable"
            )
            getattr(nano_graphrag, name)

    def test_all_contains_key_types(self):
        import nano_graphrag

        assert "GraphRAG" in nano_graphrag.__all__
        assert "GraphRAGConfig" in nano_graphrag.__all__
        assert "QueryResult" in nano_graphrag.__all__
        assert "InsertResult" in nano_graphrag.__all__
        assert "TokenUsage" in nano_graphrag.__all__
        assert "QuerySource" in nano_graphrag.__all__
        assert "GraphRAGError" in nano_graphrag.__all__
        assert "ConfigError" in nano_graphrag.__all__
        assert "ModeNotEnabledError" in nano_graphrag.__all__
        assert "GraphIntegrityError" in nano_graphrag.__all__


class TestStringStorageSelection:
    def test_graphrag_with_string_storage(self, clean_working_dir):
        from nano_graphrag import GraphRAG

        rag = GraphRAG(
            working_dir=clean_working_dir,
            graph_storage_cls="networkx",
            key_string_value_json_storage_cls="json",
            vector_db_storage_cls="hnsw",
        )
        from nano_graphrag._storage import HNSWVectorStorage, JsonKVStorage, NetworkXStorage

        assert rag.graph_storage_cls is NetworkXStorage
        assert rag.key_string_value_json_storage_cls is JsonKVStorage
        assert rag.vector_db_storage_cls is HNSWVectorStorage

    def test_graphrag_with_invalid_string_storage(self, clean_working_dir):
        from nano_graphrag import GraphRAG

        with pytest.raises(StorageConfigError, match="Unknown storage backend"):
            GraphRAG(working_dir=clean_working_dir, graph_storage_cls="invalid_storage")
