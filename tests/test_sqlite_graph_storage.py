import json
import os
import shutil

import numpy as np
import pytest

from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag._storage import SQLiteGraphStorage
from nano_graphrag._utils import generate_stable_entity_id, wrap_embedding_func_with_attrs

WORKING_DIR = "./tests/nano_graphrag_cache_sqlite_graph_storage"
FAKE_COMMUNITY_REPORT = json.dumps(
    {
        "title": "Test Community",
        "summary": "SQLite graph storage fixture report.",
        "rating": 1,
        "rating_explanation": "Fixture",
        "findings": [{"summary": "Fixture", "explanation": "Fixture"}],
    }
)


@wrap_embedding_func_with_attrs(embedding_dim=384, max_token_size=8192)
async def mock_embedding(texts: list[str]) -> np.ndarray:
    return np.random.rand(len(texts), 384)


async def sqlite_test_model(prompt, system_prompt=None, history_messages=None, **kwargs) -> str:
    if system_prompt is not None:
        return FAKE_COMMUNITY_REPORT
    if prompt == "continue_prompt" or "MANY entities were missed" in prompt:
        return ""
    if "Charles Dickens wrote A Christmas Carol" in prompt:
        return (
            '("entity"<|>CHARLES DICKENS<|>PERSON<|>Author of A Christmas Carol.)##'
            '("entity"<|>A CHRISTMAS CAROL<|>WORK<|>A novella by Charles Dickens.)##'
            '("relationship"<|>CHARLES DICKENS<|>A CHRISTMAS CAROL<|>Charles Dickens wrote A Christmas Carol.<|>1.0)<|COMPLETE|>'
        )
    return "Hello world"


@pytest.fixture(scope="function")
def setup_teardown():
    if os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR)
    os.mkdir(WORKING_DIR)

    yield

    shutil.rmtree(WORKING_DIR)


@pytest.fixture
def sqlite_storage(setup_teardown):
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        embedding_func=mock_embedding,
        graph_storage_cls=SQLiteGraphStorage,
    )
    return rag.chunk_entity_relation_graph


@pytest.mark.asyncio
async def test_upsert_and_get_node(sqlite_storage):
    await sqlite_storage.upsert_node("node1", {"attr1": "value1", "source_id": "chunk1"})

    result = await sqlite_storage.get_node("node1")
    assert result == {"attr1": "value1", "source_id": "chunk1"}
    assert await sqlite_storage.has_node("node1") is True


@pytest.mark.asyncio
async def test_upsert_and_get_edge_with_canonical_order(sqlite_storage):
    await sqlite_storage.upsert_node("node1", {"source_id": "chunk1"})
    await sqlite_storage.upsert_node("node2", {"source_id": "chunk2"})
    await sqlite_storage.upsert_edge("node2", "node1", {"weight": 1.0, "type": "connection"})

    assert await sqlite_storage.has_edge("node1", "node2") is True
    assert await sqlite_storage.has_edge("node2", "node1") is True
    assert await sqlite_storage.get_edge("node1", "node2") == {
        "weight": 1.0,
        "type": "connection",
    }
    assert await sqlite_storage.get_edge("node2", "node1") == {
        "weight": 1.0,
        "type": "connection",
    }


@pytest.mark.asyncio
async def test_batch_apis_and_degree_methods(sqlite_storage):
    await sqlite_storage.upsert_nodes_batch(
        [
            ("center", {"source_id": "chunk-center"}),
            ("left", {"source_id": "chunk-left"}),
            ("right", {"source_id": "chunk-right"}),
        ]
    )
    await sqlite_storage.upsert_edges_batch(
        [
            ("center", "left", {"weight": 1.0}),
            ("right", "center", {"weight": 2.0}),
        ]
    )

    assert await sqlite_storage.get_nodes_batch(["center", "left", "missing"]) == [
        {"source_id": "chunk-center"},
        {"source_id": "chunk-left"},
        None,
    ]
    assert await sqlite_storage.get_edges_batch(
        [("center", "left"), ("center", "right"), ("missing", "right")]
    ) == [{"weight": 1.0}, {"weight": 2.0}, None]
    assert await sqlite_storage.node_degree("center") == 2
    assert await sqlite_storage.node_degrees_batch(["center", "left", "missing"]) == [2, 1, 0]
    assert await sqlite_storage.edge_degree("center", "left") == 3
    assert await sqlite_storage.edge_degrees_batch([("center", "left"), ("left", "right")]) == [
        3,
        2,
    ]


@pytest.mark.asyncio
async def test_get_node_edges_returns_stable_canonical_tuples(sqlite_storage):
    await sqlite_storage.upsert_nodes_batch(
        [
            ("B", {"source_id": "chunk-b"}),
            ("A", {"source_id": "chunk-a"}),
            ("C", {"source_id": "chunk-c"}),
        ]
    )
    await sqlite_storage.upsert_edge("B", "A", {"weight": 1.0})
    await sqlite_storage.upsert_edge("C", "B", {"weight": 1.0})

    assert await sqlite_storage.get_node_edges("B") == [("A", "B"), ("B", "C")]
    assert await sqlite_storage.get_nodes_edges_batch(["A", "B", "missing"]) == [
        [("A", "B")],
        [("A", "B"), ("B", "C")],
        [],
    ]


@pytest.mark.asyncio
async def test_delete_and_nonexistent_behavior(sqlite_storage):
    await sqlite_storage.upsert_nodes_batch(
        [
            ("node1", {"source_id": "chunk1"}),
            ("node2", {"source_id": "chunk2"}),
        ]
    )
    await sqlite_storage.upsert_edge("node1", "node2", {"weight": 1.0})

    await sqlite_storage.delete_edge("node2", "node1")
    assert await sqlite_storage.get_edge("node1", "node2") is None

    await sqlite_storage.upsert_edge("node1", "node2", {"weight": 1.0})
    await sqlite_storage.delete_node("node1")
    assert await sqlite_storage.has_node("node1") is False
    assert await sqlite_storage.has_edge("node1", "node2") is False
    assert await sqlite_storage.get_node("missing") is None
    assert await sqlite_storage.get_edge("missing", "other") is None
    assert await sqlite_storage.get_node_edges("missing") is None
    assert await sqlite_storage.node_degree("missing") == 0
    assert await sqlite_storage.edge_degree("missing", "other") == 0


@pytest.mark.asyncio
async def test_persistence_across_reload(setup_teardown):
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        embedding_func=mock_embedding,
        graph_storage_cls=SQLiteGraphStorage,
    )
    storage = rag.chunk_entity_relation_graph
    await storage.upsert_node("node1", {"attr": "value", "source_id": "chunk1"})
    await storage.upsert_node("node2", {"attr": "value", "source_id": "chunk2"})
    await storage.upsert_edge("node1", "node2", {"weight": 1.0})
    await storage.index_done_callback()

    reloaded = SQLiteGraphStorage(namespace="chunk_entity_relation", global_config=rag.__dict__)
    assert await reloaded.has_node("node1")
    assert await reloaded.has_edge("node2", "node1")
    assert await reloaded.get_node("node1") == {"attr": "value", "source_id": "chunk1"}


@pytest.mark.asyncio
async def test_clustering_persists_clusters(setup_teardown):
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        embedding_func=mock_embedding,
        graph_storage_cls=SQLiteGraphStorage,
    )
    storage = rag.chunk_entity_relation_graph

    for i in range(6):
        await storage.upsert_node(f"NODE{i}", {"source_id": f"chunk{i}"})
    for i in range(5):
        await storage.upsert_edge(f"NODE{i}", f"NODE{i+1}", {"weight": 1.0})

    await storage.clustering("leiden")
    community_schema = await storage.community_schema()
    await storage.index_done_callback()

    reloaded = SQLiteGraphStorage(namespace="chunk_entity_relation", global_config=rag.__dict__)
    node = await reloaded.get_node("NODE0")

    assert len(community_schema) > 0
    assert "clusters" in node
    assert json.loads(node["clusters"])


@pytest.mark.asyncio
async def test_incremental_clustering_updates_frontier_only(sqlite_storage):
    sqlite_storage.global_config["addon_params"]["community_update_max_frontier_ratio"] = 0.9
    for node_id in ["A", "B", "C", "D", "E", "F"]:
        await sqlite_storage.upsert_node(node_id, {"source_id": f"chunk-{node_id}"})
    for source, target in [
        ("A", "B"),
        ("B", "C"),
        ("C", "A"),
        ("C", "D"),
        ("D", "E"),
        ("E", "F"),
        ("F", "D"),
    ]:
        await sqlite_storage.upsert_edge(source, target, {"weight": 1.0})

    await sqlite_storage.clustering("leiden")
    far_node_before = (await sqlite_storage.get_node("F"))["clusters"]

    await sqlite_storage.upsert_node("X", {"source_id": "chunk-X"})
    await sqlite_storage.upsert_edge("A", "X", {"weight": 1.0})
    await sqlite_storage.upsert_edge("B", "X", {"weight": 1.0})

    await sqlite_storage.clustering("leiden", affected_node_ids={"A", "B", "X"})

    far_node_after = (await sqlite_storage.get_node("F"))["clusters"]
    affected_node_after = json.loads((await sqlite_storage.get_node("A"))["clusters"])
    new_node_after = json.loads((await sqlite_storage.get_node("X"))["clusters"])

    assert sqlite_storage._last_clustering_was_incremental is True
    assert far_node_after == far_node_before
    assert affected_node_after[0]["cluster"].startswith("inc-")
    assert new_node_after[0]["cluster"].startswith("inc-")
    assert len(sqlite_storage._last_affected_community_ids) > 0


@pytest.mark.asyncio
async def test_community_schema_with_multiple_levels(sqlite_storage):
    await sqlite_storage.upsert_node(
        "node1",
        {"source_id": "chunk1", "clusters": json.dumps([{"level": 0, "cluster": "0"}])},
    )
    await sqlite_storage.upsert_node(
        "node2",
        {
            "source_id": "chunk2",
            "clusters": json.dumps([{"level": 0, "cluster": "0"}, {"level": 1, "cluster": "1"}]),
        },
    )
    await sqlite_storage.upsert_node(
        "node3",
        {
            "source_id": "chunk3",
            "clusters": json.dumps([{"level": 0, "cluster": "0"}, {"level": 1, "cluster": "2"}]),
        },
    )
    await sqlite_storage.upsert_edge("node1", "node2", {"weight": 1.0})
    await sqlite_storage.upsert_edge("node2", "node3", {"weight": 1.0})

    community_schema = await sqlite_storage.community_schema()

    assert set(community_schema.keys()) == {"0", "1", "2"}
    assert community_schema["0"]["level"] == 0
    assert set(community_schema["0"]["sub_communities"]) == {"1", "2"}
    assert sorted(community_schema["0"]["nodes"]) == ["node1", "node2", "node3"]


def test_graphrag_sqlite_storage_smoke_query(setup_teardown):
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        best_model_func=sqlite_test_model,
        cheap_model_func=sqlite_test_model,
        embedding_func=mock_embedding,
        graph_storage_cls=SQLiteGraphStorage,
    )
    rag.insert_documents({"doc-1": "Charles Dickens wrote A Christmas Carol."})

    dickens_id = generate_stable_entity_id("CHARLES DICKENS", "PERSON")
    context = rag.query("Dickens", param=QueryParam(mode="local", only_need_context=True))
    assert "CHARLES DICKENS" in context
    assert dickens_id not in context
