"""Benchmark coverage for larger document insert workloads."""

from __future__ import annotations

import asyncio
import shutil
import time
from pathlib import Path

import numpy as np
import pytest

from nano_graphrag import GraphRAG
from nano_graphrag._utils import wrap_embedding_func_with_attrs


@wrap_embedding_func_with_attrs(embedding_dim=384, max_token_size=8192)
async def local_embedding(texts: list[str]) -> np.ndarray:
    return np.random.rand(len(texts), 384)


async def synthetic_extraction_model(prompt, system_prompt=None, history_messages=None, **kwargs):
    if system_prompt is not None:
        return (
            '{"title":"Benchmark Community","summary":"Synthetic report.","rating":1,'
            '"rating_explanation":"Fixture","findings":[{"summary":"Fixture","explanation":"Fixture"}]}'
        )
    if prompt == "continue_prompt" or "MANY entities were missed" in prompt:
        return ""
    return (
        '("entity"<|>NODE A<|>PERSON<|>Synthetic person entity.)##'
        '("entity"<|>NODE B<|>ORG<|>Synthetic organization entity.)##'
        '("relationship"<|>NODE A<|>NODE B<|>Synthetic relation.<|>1.0)<|COMPLETE|>'
    )


@pytest.mark.benchmark
@pytest.mark.slow
def test_insert_documents_large_batch_completes_within_budget():
    working_dir = Path("./tests/nano_graphrag_cache_BENCH_INSERT")
    if working_dir.exists():
        shutil.rmtree(working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    rag = GraphRAG(
        working_dir=str(working_dir),
        best_model_func=synthetic_extraction_model,
        cheap_model_func=synthetic_extraction_model,
        embedding_func=local_embedding,
        enable_naive_rag=True,
        doc_extraction_max_async=4,
        extraction_max_async=16,
        doc_flush_batch_size=20,
    )

    docs = {
        "doc-{:04d}".format(i): "Synthetic content {}. NODE A works with NODE B.".format(i)
        for i in range(200)
    }

    started = time.perf_counter()
    rag.insert_documents(docs)
    elapsed = time.perf_counter() - started

    loop = asyncio.get_event_loop()
    persisted = loop.run_until_complete(rag.full_docs.all_keys())
    assert len(persisted) == 200
    # Coarse guardrail for regressions in CI runners; this should remain comfortably under budget.
    assert elapsed < 20.0
