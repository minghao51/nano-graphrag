import pytest
import numpy as np

from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag._utils import wrap_embedding_func_with_attrs

pytestmark = pytest.mark.unit


async def fake_model(prompt, system_prompt=None, history_messages=None, **kwargs):
    return "ok"


@wrap_embedding_func_with_attrs(embedding_dim=64, max_token_size=8192)
async def mock_embedding(texts: list[str]) -> np.ndarray:
    return np.random.rand(len(texts), 64)


def _make_rag(log_query_text: bool):
    return GraphRAG(
        working_dir=f"./tests/nano_graphrag_cache_query_logging_{int(log_query_text)}",
        best_model_func=fake_model,
        cheap_model_func=fake_model,
        embedding_func=mock_embedding,
        log_query_text=log_query_text,
    )


def test_query_logging_redacts_text_by_default(monkeypatch):
    rag = _make_rag(log_query_text=False)
    calls = []

    def fake_info(event, **kwargs):
        calls.append((event, kwargs))

    monkeypatch.setattr("nano_graphrag.graphrag_query.logger.info", fake_info)
    rag.query("top secret question", param=QueryParam(mode="global"))

    query_start = [payload for event, payload in calls if event == "query_start"]
    assert query_start
    payload = query_start[0]
    assert "query" not in payload
    assert "query_hash" in payload
    assert payload["query_chars"] == len("top secret question")


def test_query_logging_can_include_query_text(monkeypatch):
    rag = _make_rag(log_query_text=True)
    calls = []

    def fake_info(event, **kwargs):
        calls.append((event, kwargs))

    monkeypatch.setattr("nano_graphrag.graphrag_query.logger.info", fake_info)
    rag.query("safe question", param=QueryParam(mode="global"))

    query_start = [payload for event, payload in calls if event == "query_start"]
    assert query_start
    payload = query_start[0]
    assert payload["query"] == "safe question"
