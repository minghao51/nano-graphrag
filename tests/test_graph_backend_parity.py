import os
import shutil

import numpy as np
import pytest

from nano_graphrag import GraphRAG
from nano_graphrag._storage import NetworkXStorage, SQLiteGraphStorage
from nano_graphrag._utils import wrap_embedding_func_with_attrs

pytestmark = pytest.mark.unit

WORKING_DIR_BASE = "./tests/nano_graphrag_cache_backend_parity"


@wrap_embedding_func_with_attrs(embedding_dim=64, max_token_size=8192)
async def mock_embedding(texts: list[str]) -> np.ndarray:
    return np.random.rand(len(texts), 64)


@pytest.fixture(params=[NetworkXStorage, SQLiteGraphStorage])
def graph_storage(request):
    storage_cls = request.param
    working_dir = f"{WORKING_DIR_BASE}_{storage_cls.__name__}"
    if os.path.exists(working_dir):
        shutil.rmtree(working_dir)
    os.mkdir(working_dir)

    rag = GraphRAG(
        working_dir=working_dir,
        embedding_func=mock_embedding,
        graph_storage_cls=storage_cls,
    )
    storage = rag.chunk_entity_relation_graph
    yield storage
    shutil.rmtree(working_dir)


async def test_edge_lookup_is_direction_agnostic(graph_storage):
    await graph_storage.upsert_nodes_batch(
        [
            ("A", {"source_id": "chunk-a"}),
            ("B", {"source_id": "chunk-b"}),
        ]
    )
    await graph_storage.upsert_edge("A", "B", {"weight": 1.0, "relationship_id": "rel_ab"})

    assert await graph_storage.has_edge("A", "B") is True
    assert await graph_storage.has_edge("B", "A") is True

    edge_ab = await graph_storage.get_edge("A", "B")
    edge_ba = await graph_storage.get_edge("B", "A")
    assert edge_ab is not None
    assert edge_ba is not None
    assert edge_ab["weight"] == edge_ba["weight"] == 1.0


async def test_node_edges_include_incident_neighbors(graph_storage):
    await graph_storage.upsert_nodes_batch(
        [
            ("A", {"source_id": "chunk-a"}),
            ("B", {"source_id": "chunk-b"}),
            ("C", {"source_id": "chunk-c"}),
        ]
    )
    await graph_storage.upsert_edge("A", "B", {"weight": 1.0, "relationship_id": "rel_ab"})
    await graph_storage.upsert_edge("C", "B", {"weight": 1.0, "relationship_id": "rel_cb"})

    edges = await graph_storage.get_node_edges("B")
    assert edges is not None
    assert len(edges) == 2
