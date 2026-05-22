from inspect import signature

import pytest

from nano_graphrag._config import GraphRAGSettings
from nano_graphrag.base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    GraphRAGConfig,
    QueryParam,
    ResponseType,
)
from nano_graphrag.graphrag import GraphRAG

pytestmark = [pytest.mark.unit, pytest.mark.contract]


class TestGraphRAGPublicAPISignatures:
    def test_ainsert_signature(self):
        sig = signature(GraphRAG.ainsert)
        params = list(sig.parameters.keys())
        assert params == ["self", "string_or_strings"]

    def test_aquery_signature(self):
        sig = signature(GraphRAG.aquery)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "query" in params
        assert "param" in params

    def test_arefine_signature(self):
        sig = signature(GraphRAG.arefine)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "phases" in params

    def test_aexport_vault_signature(self):
        sig = signature(GraphRAG.aexport_vault)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "path" in params
        assert "include_communities" in params

    def test_arebuild_graph_signature(self):
        sig = signature(GraphRAG.arebuild_graph)
        params = list(sig.parameters.keys())
        assert params == ["self"]


class TestConfigRoundTrip:
    def test_graphrag_config_to_dict_from_dict(self):
        original = GraphRAGConfig()
        data = original.to_dict()
        restored = GraphRAGConfig.from_dict(data)
        assert isinstance(restored, GraphRAGConfig)

    def test_graphrag_settings_from_dict_roundtrip(self):
        settings = GraphRAGSettings.from_dict({"llm_model": "test"})
        dumped = settings.model_dump()
        assert "llm" in dumped
        assert dumped["llm"]["model"] == "test"

    def test_query_param_from_config_fields(self):
        qp = QueryParam()
        for attr in ("mode", "only_need_context", "response_type", "level", "top_k"):
            assert hasattr(qp, attr)


class TestStorageABCContracts:
    def test_base_vector_storage_abstract_methods(self):
        for name in ("query", "upsert", "delete"):
            assert getattr(BaseVectorStorage, name).__isabstractmethod__

    def test_base_kv_storage_abstract_methods(self):
        for name in (
            "all_keys",
            "get_by_id",
            "get_by_ids",
            "filter_keys",
            "upsert",
            "delete",
            "drop",
        ):
            assert getattr(BaseKVStorage, name).__isabstractmethod__

    def test_base_graph_storage_abstract_methods(self):
        for name in (
            "has_node",
            "has_edge",
            "get_node",
            "get_edge",
            "upsert_node",
            "upsert_edge",
            "delete_node",
            "delete_edge",
            "clustering",
            "community_schema",
        ):
            assert getattr(BaseGraphStorage, name).__isabstractmethod__

    def test_storage_namespace_pattern(self):
        for cls in (BaseVectorStorage, BaseKVStorage, BaseGraphStorage):
            assert "namespace" in cls.__dataclass_fields__
            assert "global_config" in cls.__dataclass_fields__


class TestExportContracts:
    def test_graphrag_class_has_public_methods(self):
        for name in (
            "insert",
            "query",
            "ainsert",
            "aquery",
            "arefine",
            "aexport_vault",
            "arebuild_graph",
        ):
            assert callable(getattr(GraphRAG, name))

    def test_response_type_constants(self):
        assert isinstance(ResponseType.CONCISE, str) and ResponseType.CONCISE
        assert isinstance(ResponseType.SHORT, str) and ResponseType.SHORT
        assert isinstance(ResponseType.SINGLE_PARAGRAPH, str) and ResponseType.SINGLE_PARAGRAPH
        assert isinstance(ResponseType.BULLET_POINTS, str) and ResponseType.BULLET_POINTS
