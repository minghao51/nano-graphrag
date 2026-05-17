import asyncio
import json
import os
import shutil

import numpy as np
import pytest

from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag._ops.structural_features import (
    _compute_structural_features_sync,
    compute_composite_rank,
    ensure_structural_features,
)
from nano_graphrag._ops.retrieval_feedback import (
    RetrievalFeedback,
    compute_retrieval_feedback,
    log_retrieval_feedback,
)
from nano_graphrag._query_planner import (
    QueryAnalysis,
    SoftEntityMask,
    build_soft_entity_mask,
    compute_final_scores,
)
from nano_graphrag._utils import wrap_embedding_func_with_attrs

pytestmark = pytest.mark.unit

os.environ["OPENAI_API_KEY"] = "FAKE"

WORKING_DIR = "./tests/nano_graphrag_cache_SAGE"


@wrap_embedding_func_with_attrs(embedding_dim=384, max_token_size=8192)
async def local_embedding(texts: list[str]) -> np.ndarray:
    return np.random.rand(len(texts), 384)


FAKE_COMMUNITY_REPORT = json.dumps(
    {
        "title": "Test",
        "summary": "Test community.",
        "rating": 1,
        "rating_explanation": "Fixture",
        "findings": [{"summary": "Fixture", "explanation": "Fixture"}],
    }
)


async def sage_model(prompt, system_prompt=None, history_messages=None, **kwargs) -> str:
    if system_prompt is not None:
        return FAKE_COMMUNITY_REPORT
    if "Can the answer be fully deduced" in prompt:
        return "yes"
    if "Charles Dickens wrote A Christmas Carol" in prompt:
        return (
            '("entity"<|>CHARLES DICKENS<|>PERSON<|>Author of A Christmas Carol.)##'
            '("entity"<|>A CHRISTMAS CAROL<|>WORK<|>A novella by Charles Dickens.)##'
            '("relationship"<|>CHARLES DICKENS<|>A CHRISTMAS CAROL<|>Charles Dickens wrote A Christmas Carol.<|>1.0)<|COMPLETE|>'
        )
    if "Can the answer be fully deduced" in prompt:
        return "yes"
    if "Analyze the following question" in prompt:
        return json.dumps(
            {
                "entities": [{"name": "Charles Dickens", "aliases": ["Dickens"]}],
                "relation_clues": ["wrote"],
                "constraints": {},
                "answer_type": "work",
                "pseudo_queries": [],
            }
        )
    return FAKE_COMMUNITY_REPORT


def clean_working_dir():
    if os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR)
    os.mkdir(WORKING_DIR)


@pytest.fixture(autouse=True)
def _cleanup():
    clean_working_dir()
    yield
    if os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR)


def _build_rag(**kwargs):
    return GraphRAG(
        working_dir=WORKING_DIR,
        best_model_func=sage_model,
        cheap_model_func=sage_model,
        embedding_func=local_embedding,
        enable_naive_rag=True,
        **kwargs,
    )


class TestStructuralFeatures:
    def test_compute_features_empty_graph(self):
        import networkx as nx

        g = nx.MultiGraph()
        result = _compute_structural_features_sync(g)
        assert result == {}

    def test_compute_features_normalized(self):
        import networkx as nx

        g = nx.MultiGraph()
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        g.add_edge("c", "d")
        result = _compute_structural_features_sync(g)
        assert set(result.keys()) == {"a", "b", "c", "d"}
        for node_features in result.values():
            for v in node_features.values():
                assert 0.0 <= v <= 1.0

    def test_composite_rank_none(self):
        assert compute_composite_rank(None, [0.5, 0.2, 0.1, 0.2]) == 0.0

    def test_composite_rank_values(self):
        features = {
            "degree_centrality": 1.0,
            "pagerank": 0.8,
            "betweenness_centrality": 0.3,
            "clustering": 0.0,
        }
        rank = compute_composite_rank(features, [0.5, 0.2, 0.1, 0.2])
        expected = 0.5 * 1.0 + 0.2 * 0.8 + 0.1 * 0.3 + 0.2 * 0.0
        assert abs(rank - expected) < 1e-10

    def test_features_persisted_after_insert(self):
        rag = _build_rag()
        rag.insert_documents({"doc-1": "Charles Dickens wrote A Christmas Carol."})
        import asyncio

        loop = asyncio.new_event_loop()
        try:
            payload = loop.run_until_complete(
                rag.document_index.get_by_id("structural_features_payload")
            )
        finally:
            loop.close()
        assert isinstance(payload, dict)
        assert len(payload) > 0

    def test_ensure_features_loads_from_index(self):
        rag = _build_rag()
        rag.insert_documents({"doc-1": "Charles Dickens wrote A Christmas Carol."})
        graph = rag.chunk_entity_relation_graph._graph
        assert hasattr(graph, "_structural_features")
        del graph._structural_features
        import asyncio

        loop = asyncio.new_event_loop()
        try:
            features = loop.run_until_complete(
                ensure_structural_features(rag.chunk_entity_relation_graph, rag.document_index)
            )
        finally:
            loop.close()
        assert features is not None
        assert len(features) > 0


class TestEdgeGating:
    def test_edge_gate_threshold_filters_edges(self):
        rag = _build_rag()
        rag.insert_documents({"doc-1": "Charles Dickens wrote A Christmas Carol."})
        param = QueryParam(
            mode="local",
            only_need_context=True,
            edge_gate_threshold=999.0,
        )
        context = rag.query("Charles Dickens", param=param)
        assert context is not None

    def test_edge_gate_decay_formula(self):
        param = QueryParam(edge_gate_threshold=0.5, edge_gate_decay=0.5)
        assert param.edge_gate_threshold * (param.edge_gate_decay ** (1 - 1)) == 0.5
        assert param.edge_gate_threshold * (param.edge_gate_decay ** (2 - 1)) == 0.25
        assert param.edge_gate_threshold * (param.edge_gate_decay ** (3 - 1)) == 0.125


class TestRetrievalFeedback:
    @pytest.mark.asyncio
    async def test_feedback_deducible_from_context(self):
        config = {"cheap_model_func": sage_model}
        feedback = await compute_retrieval_feedback(
            query="Who wrote A Christmas Carol?",
            context_text="Charles Dickens wrote A Christmas Carol.",
            global_config=config,
        )
        assert feedback.deducible is True
        assert feedback.recall == 1.0

    @pytest.mark.asyncio
    async def test_feedback_not_deducible(self):
        async def say_no(prompt, **kw):
            return "no"

        config = {"cheap_model_func": say_no}
        feedback = await compute_retrieval_feedback(
            query="What is the capital of France?",
            context_text="The sky is blue.",
            global_config=config,
        )
        assert feedback.deducible is False
        assert feedback.recall == 0.0

    @pytest.mark.asyncio
    async def test_feedback_logs_to_document_index(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            from nano_graphrag._storage.kv_json import SQLiteKVStorage

            doc_index = SQLiteKVStorage(
                namespace="test_fb",
                global_config={"working_dir": tmpdir},
            )
            await doc_index.index_start_callback()
            config = {"cheap_model_func": sage_model}
            feedback = RetrievalFeedback(
                query="test",
                doc_ids_retrieved=["doc-1"],
                recall=0.8,
                deducible=True,
            )
            await log_retrieval_feedback(feedback, doc_index, config)
            stats = await doc_index.get_by_id("feedback_stats")
            assert isinstance(stats, dict)
            assert "doc-1" in stats
            assert stats["doc-1"]["total_queries"] == 1
            assert stats["doc-1"]["avg_recall"] == 0.8


class TestSoftEntityMask:
    def test_exact_match(self):
        analysis = QueryAnalysis(entities=[("Alice", [])])
        mask = build_soft_entity_mask(
            analysis,
            {"e1": "Alice"},
            {"alice": "e1"},
            {"e1": []},
        )
        assert mask.scores.get("e1") == 1.0

    def test_alias_match(self):
        analysis = QueryAnalysis(entities=[("A. Smith", ["Alice Smith"])])
        mask = build_soft_entity_mask(
            analysis,
            {"e1": "Alice Smith"},
            {"alice smith": "e1"},
            {"e1": ["Alice Smith"]},
        )
        assert mask.scores.get("e1") == 0.8

    def test_no_match(self):
        analysis = QueryAnalysis(entities=[("Unknown", [])])
        mask = build_soft_entity_mask(
            analysis,
            {"e1": "Alice"},
            {"alice": "e1"},
            {"e1": []},
        )
        assert "e1" not in mask.scores

    def test_compute_final_scores_combines(self):
        results = [
            {"id": "e1", "entity_name": "Alice", "similarity": 0.9},
            {"id": "e2", "entity_name": "Bob", "similarity": 0.8},
        ]
        mask = SoftEntityMask(scores={"e1": 1.0})
        scored = compute_final_scores(results, mask)
        e1_score = next(s for r, s in scored if r["id"] == "e1")
        e2_score = next(s for r, s in scored if r["id"] == "e2")
        assert e1_score > e2_score


class TestQueryParamFromConfig:
    def test_from_config_threads_sage_fields(self):
        rag = _build_rag(
            enable_query_planning=True,
            edge_gate_threshold=0.5,
            propagation_hops=3,
        )
        param = QueryParam.from_config(rag)
        assert param.enable_query_planning is True
        assert param.edge_gate_threshold == 0.5
        assert param.propagation_hops == 3

    def test_from_config_preserves_overrides(self):
        rag = _build_rag(enable_query_planning=True)
        param = QueryParam.from_config(rag, mode="local", enable_query_planning=False)
        assert param.mode == "local"
        assert param.enable_query_planning is False
