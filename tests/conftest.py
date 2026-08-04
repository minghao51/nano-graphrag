import logging
import os
import shutil
from pathlib import Path
from unittest.mock import AsyncMock

import numpy as np
import pytest

from nano_graphrag._entity_registry import EntityRegistry

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

WORKING_DIR = Path("./test_cache")
os.environ.setdefault("OPENAI_API_KEY", "FAKE")


def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run tests marked as integration",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-integration"):
        return

    skip_integration = pytest.mark.skip(reason="need --run-integration option to run")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


@pytest.fixture
def working_dir():
    if WORKING_DIR.exists():
        shutil.rmtree(WORKING_DIR)
    WORKING_DIR.mkdir(parents=True, exist_ok=True)
    yield str(WORKING_DIR)
    if WORKING_DIR.exists():
        shutil.rmtree(WORKING_DIR)


@pytest.fixture
def clean_working_dir(working_dir):
    import uuid

    test_dir = Path(working_dir) / f"test_{uuid.uuid4().hex[:8]}"
    test_dir.mkdir(parents=True, exist_ok=True)
    yield str(test_dir)
    if test_dir.exists():
        shutil.rmtree(test_dir)


@pytest.fixture
def entity_registry():
    return EntityRegistry()


@pytest.fixture
def local_embedding():
    from nano_graphrag._utils import wrap_embedding_func_with_attrs

    @wrap_embedding_func_with_attrs(embedding_dim=384, max_token_size=8192)
    async def embedding_func(texts: list[str]) -> np.ndarray:
        return np.random.rand(len(texts), 384)

    return embedding_func


@pytest.fixture
def deterministic_embedding():
    from nano_graphrag._utils import wrap_embedding_func_with_attrs

    @wrap_embedding_func_with_attrs(embedding_dim=8, max_token_size=512)
    async def embedding_func(texts: list[str]) -> np.ndarray:
        rng = np.random.RandomState(42)
        return rng.rand(len(texts), 8)

    return embedding_func


@pytest.fixture
def content_hash_embedding():
    from nano_graphrag._utils import wrap_embedding_func_with_attrs

    @wrap_embedding_func_with_attrs(embedding_dim=8, max_token_size=512)
    async def embedding_func(texts: list[str]) -> np.ndarray:
        vectors = []
        for t in texts:
            seed = abs(hash(t)) % (2**31)
            rng = np.random.RandomState(seed)
            vectors.append(rng.rand(8))
        return np.array(vectors)

    return embedding_func


@pytest.fixture
def no_op_model():
    async def model(prompt, system_prompt=None, history_messages=None, **kwargs) -> str:
        return ""

    return model


@pytest.fixture
def fake_entity_grounded_model():
    async def model(prompt, system_prompt=None, history_messages=None, **kwargs) -> str:
        if system_prompt is not None:
            return '{"title":"Test Community","summary":"Entity-grounded fixture.","rating":1,"rating_explanation":"Fixture","findings":[{"summary":"Fixture","explanation":"Fixture"}]}'
        if prompt == "continue_prompt" or "MANY entities were missed" in prompt:
            return ""
        return (
            '("entity"<|>SAM BANKMAN-FRIED<|>PERSON<|>Founder of FTX.)'
            '("entity"<|>FTX<|>ORGANIZATION<|>Cryptocurrency exchange founded by Sam Bankman-Fried.)'
            '("relationship"<|>SAM BANKMAN-FRIED<|>FTX<|>Sam Bankman-Fried founded FTX.<|>1.0)<|COMPLETE|>'
        )

    return model


@pytest.fixture
def mock_networkx_storage(working_dir):
    from nano_graphrag._storage.gdb_networkx import NetworkXStorage

    return NetworkXStorage(namespace="test", global_config={"working_dir": working_dir})


@pytest.fixture
def mock_sqlite_kv_storage(working_dir):
    from nano_graphrag._storage.kv_json import SQLiteKVStorage

    return SQLiteKVStorage(namespace="test", global_config={"working_dir": working_dir})


@pytest.fixture
def mock_graph_storage():
    storage = AsyncMock()
    storage.has_node = AsyncMock(return_value=False)
    storage.has_edge = AsyncMock(return_value=False)
    storage.get_node = AsyncMock(return_value=None)
    storage.get_edge = AsyncMock(return_value=None)
    storage.get_nodes_batch = AsyncMock(return_value=[])
    storage.get_edges_batch = AsyncMock(return_value=[])
    storage.get_node_edges = AsyncMock(return_value=[])
    storage.get_nodes_edges_batch = AsyncMock(return_value=[])
    storage.node_degree = AsyncMock(return_value=0)
    storage.node_degrees_batch = AsyncMock(return_value=[])
    storage.edge_degree = AsyncMock(return_value=0)
    storage.edge_degrees_batch = AsyncMock(return_value=[])
    storage.upsert_node = AsyncMock()
    storage.upsert_nodes_batch = AsyncMock()
    storage.upsert_edge = AsyncMock()
    storage.upsert_edges_batch = AsyncMock()
    storage.delete_node = AsyncMock()
    storage.delete_nodes_batch = AsyncMock()
    storage.delete_edge = AsyncMock()
    storage.delete_edges_batch = AsyncMock()
    storage.clustering = AsyncMock()
    storage.community_schema = AsyncMock(return_value={})
    storage._graph = None
    return storage


@pytest.fixture
def mock_kv_storage():
    storage = AsyncMock()
    storage._store = {}

    async def _get_by_id(id):
        return storage._store.get(id)

    async def _get_by_ids(ids, fields=None):
        return [storage._store.get(i) for i in ids]

    async def _upsert(data):
        storage._store.update(data)

    async def _all_keys():
        return list(storage._store.keys())

    async def _filter_keys(data):
        return set(data) - set(storage._store.keys())

    async def _delete(ids):
        for i in ids:
            storage._store.pop(i, None)

    async def _drop():
        storage._store.clear()

    storage.get_by_id = AsyncMock(side_effect=_get_by_id)
    storage.get_by_ids = AsyncMock(side_effect=_get_by_ids)
    storage.upsert = AsyncMock(side_effect=_upsert)
    storage.all_keys = AsyncMock(side_effect=_all_keys)
    storage.filter_keys = AsyncMock(side_effect=_filter_keys)
    storage.delete = AsyncMock(side_effect=_delete)
    storage.drop = AsyncMock(side_effect=_drop)
    return storage


@pytest.fixture
def mock_entity_vdb():
    vdb = AsyncMock()
    vdb._store = {}

    async def _upsert(data):
        vdb._store.update(data)

    async def _query(query, top_k=5):
        return list(vdb._store.values())[:top_k]

    async def _delete(ids):
        for i in ids:
            vdb._store.pop(i, None)

    vdb.upsert = AsyncMock(side_effect=_upsert)
    vdb.query = AsyncMock(side_effect=_query)
    vdb.delete = AsyncMock(side_effect=_delete)
    return vdb


@pytest.fixture
def global_config(working_dir, no_op_model, deterministic_embedding):
    return {
        "working_dir": working_dir,
        "best_model_func": no_op_model,
        "cheap_model_func": no_op_model,
        "embedding_func": deterministic_embedding,
        "best_model_max_token_size": 32768,
        "cheap_model_max_token_size": 32768,
        "extraction_max_async": 4,
        "entity_summary_to_max_tokens": 500,
        "entity_registry": EntityRegistry(),
        "enable_refinement": True,
        "refinement_merge_threshold": 0.93,
        "refinement_enrich_min_chars": 80,
        "refinement_infer_confidence": 0.80,
        "refinement_infer_hub_cap": 3,
        "refinement_batch_size": 50,
    }
