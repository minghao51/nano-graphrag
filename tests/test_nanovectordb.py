import shutil
import tempfile

import numpy as np
import pytest

pytest.importorskip("nano_vectordb")
from nano_graphrag._storage.vdb_nanovectordb import NanoVectorDBStorage
from nano_graphrag._utils import wrap_embedding_func_with_attrs

pytestmark = pytest.mark.unit


@wrap_embedding_func_with_attrs(embedding_dim=8, max_token_size=512)
async def test_embedding(texts):
    rng = np.random.RandomState(42)
    result = rng.rand(len(texts), 8)
    for i, t in enumerate(texts):
        h = sum(ord(c) for c in t) % 100
        result[i] += h * 0.01
    return result


def _make_storage(work_dir, namespace="test", meta_fields=None, threshold=None):
    config = {"working_dir": work_dir}
    if threshold is not None:
        config["query_better_than_threshold"] = threshold
    return NanoVectorDBStorage(
        namespace=namespace,
        global_config=config,
        embedding_func=test_embedding,
        meta_fields=meta_fields or set(),
    )


async def test_nanovecdb_upsert_and_query():
    work_dir = tempfile.mkdtemp()
    try:
        storage = _make_storage(work_dir)
        await storage.upsert({"id1": {"content": "hello world", "name": "test1"}})
        results = await storage.query("hello", top_k=5)
        assert any(r["id"] == "id1" for r in results)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


async def test_nanovecdb_upsert_empty():
    work_dir = tempfile.mkdtemp()
    try:
        storage = _make_storage(work_dir)
        result = await storage.upsert({})
        assert result == []
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


async def test_nanovecdb_delete():
    work_dir = tempfile.mkdtemp()
    try:
        storage = _make_storage(work_dir)
        await storage.upsert({"id1": {"content": "hello world", "name": "test1"}})
        await storage.delete(["id1"])
        results = await storage.query("hello", top_k=5)
        assert not any(r["id"] == "id1" for r in results)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


async def test_nanovecdb_delete_empty_ids():
    work_dir = tempfile.mkdtemp()
    try:
        storage = _make_storage(work_dir)
        await storage.delete([])
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


async def test_nanovecdb_persistence():
    work_dir = tempfile.mkdtemp()
    try:
        storage = _make_storage(work_dir)
        await storage.upsert({"id1": {"content": "hello world", "name": "test1"}})
        await storage.index_done_callback()

        storage2 = _make_storage(work_dir)
        results = await storage2.query("hello", top_k=5)
        assert any(r["id"] == "id1" for r in results)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


async def test_nanovecdb_cosine_threshold():
    work_dir = tempfile.mkdtemp()
    try:
        storage = _make_storage(work_dir, threshold=0.99)
        await storage.upsert({"id1": {"content": "hello world", "name": "test1"}})
        results = await storage.query("completely unrelated xyz", top_k=5)
        assert len(results) == 0
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
