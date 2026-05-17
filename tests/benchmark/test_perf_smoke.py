import time

import numpy as np
import pytest

from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag._utils import wrap_embedding_func_with_attrs

pytestmark = [pytest.mark.benchmark, pytest.mark.slow]


@wrap_embedding_func_with_attrs(embedding_dim=128, max_token_size=8192)
async def mock_embedding(texts: list[str]) -> np.ndarray:
    return np.random.rand(len(texts), 128)


async def fake_model(prompt, system_prompt=None, history_messages=None, **kwargs):
    if system_prompt is not None:
        return (
            '{"title":"Perf","summary":"Perf","rating":1,'
            '"rating_explanation":"Perf","findings":[{"summary":"Perf","explanation":"Perf"}]}'
        )
    if prompt == "continue_prompt" or "MANY entities were missed" in prompt:
        return ""
    return (
        '("entity"<|>ENTITY A<|>ORG<|>Entity A.)##'
        '("entity"<|>ENTITY B<|>ORG<|>Entity B.)##'
        '("relationship"<|>ENTITY A<|>ENTITY B<|>Entity A relates to Entity B.<|>1.0)<|COMPLETE|>'
    )


def test_insert_query_perf_smoke():
    rag = GraphRAG(
        working_dir="./tests/nano_graphrag_cache_perf_smoke",
        best_model_func=fake_model,
        cheap_model_func=fake_model,
        embedding_func=mock_embedding,
    )

    docs = {f"doc-{i}": f"Document {i} about Entity A and Entity B." for i in range(20)}

    start_insert = time.perf_counter()
    rag.insert_documents(docs)
    insert_elapsed = time.perf_counter() - start_insert

    start_query = time.perf_counter()
    _ = rag.query("What connects Entity A and Entity B?", param=QueryParam(mode="local"))
    query_elapsed = time.perf_counter() - start_query

    assert insert_elapsed < 20.0
    assert query_elapsed < 10.0
