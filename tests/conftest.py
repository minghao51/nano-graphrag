import logging
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from nano_graphrag._entity_registry import EntityRegistry

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

WORKING_DIR = Path("./test_cache")
os.environ.setdefault("OPENAI_API_KEY", "FAKE")


@pytest.fixture
def working_dir():
    """Provide a clean working directory for tests."""
    if WORKING_DIR.exists():
        shutil.rmtree(WORKING_DIR)
    WORKING_DIR.mkdir(parents=True, exist_ok=True)
    yield str(WORKING_DIR)
    if WORKING_DIR.exists():
        shutil.rmtree(WORKING_DIR)


@pytest.fixture
def clean_working_dir(working_dir):
    """Clean existing working dir contents but keep the directory."""
    import uuid
    test_dir = Path(working_dir) / f"test_{uuid.uuid4().hex[:8]}"
    test_dir.mkdir(parents=True, exist_ok=True)
    yield str(test_dir)
    if test_dir.exists():
        shutil.rmtree(test_dir)


@pytest.fixture
def entity_registry():
    """Provide a fresh EntityRegistry instance."""
    return EntityRegistry()


@pytest.fixture
def local_embedding():
    """Provide a local embedding function for testing."""
    from nano_graphrag._utils import wrap_embedding_func_with_attrs

    @wrap_embedding_func_with_attrs(embedding_dim=384, max_token_size=8192)
    async def embedding_func(texts: list[str]) -> np.ndarray:
        return np.random.rand(len(texts), 384)

    return embedding_func


@pytest.fixture
def no_op_model():
    """Provide a no-op model function for testing."""

    async def model(prompt, system_prompt=None, history_messages=None, **kwargs) -> str:
        return ""

    return model


@pytest.fixture
def fake_entity_grounded_model():
    """Provide a fake model that returns entity-grounded responses."""

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
    """Provide a NetworkXStorage instance for testing."""
    from nano_graphrag._storage.gdb_networkx import NetworkXStorage

    return NetworkXStorage(namespace="test", global_config={"working_dir": working_dir})


@pytest.fixture
def mock_sqlite_kv_storage(working_dir):
    """Provide a SQLiteKVStorage instance for testing."""
    from nano_graphrag._storage.kv_json import SQLiteKVStorage

    return SQLiteKVStorage(namespace="test", global_config={"working_dir": working_dir})
