import json
import os
import shutil

import pytest

from nano_graphrag import GraphRAG
from nano_graphrag._storage import JsonKVStorage
from nano_graphrag._utils import wrap_embedding_func_with_attrs

WORKING_DIR = "./tests/nano_graphrag_cache_kv_storage_test"


@pytest.fixture(scope="function")
def setup_teardown():
    if os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR)
    os.mkdir(WORKING_DIR)

    yield

    shutil.rmtree(WORKING_DIR)


@wrap_embedding_func_with_attrs(embedding_dim=8, max_token_size=1024)
async def mock_embedding(texts: list[str]):
    return [[0.0] * 8 for _ in texts]


@pytest.mark.asyncio
async def test_sqlite_kv_storage_migrates_legacy_json_data(setup_teardown):
    namespace = "migration"
    legacy_path = os.path.join(WORKING_DIR, f"kv_store_{namespace}.json")
    with open(legacy_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "doc-1": {"status": "ready", "count": 1},
                "doc-2": {"status": "queued", "count": 2},
            },
            f,
        )

    rag = GraphRAG(working_dir=WORKING_DIR, embedding_func=mock_embedding)
    storage = JsonKVStorage(namespace=namespace, global_config=rag.__dict__)

    assert await storage.get_by_id("doc-1") == {"status": "ready", "count": 1}
    assert await storage.get_by_ids(["doc-2", "missing"]) == [
        {"status": "queued", "count": 2},
        None,
    ]
    assert set(await storage.all_keys()) == {"doc-1", "doc-2"}
