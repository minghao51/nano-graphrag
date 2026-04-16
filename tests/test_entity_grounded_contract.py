import asyncio
import os

import numpy as np

from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag._utils import wrap_embedding_func_with_attrs

os.environ["OPENAI_API_KEY"] = "FAKE"

FAKE_COMMUNITY_REPORT = """{"title":"Test Community","summary":"Entity-grounded fixture.","rating":1,"rating_explanation":"Fixture","findings":[{"summary":"Fixture","explanation":"Fixture"}]}"""


@wrap_embedding_func_with_attrs(embedding_dim=384, max_token_size=8192)
async def local_embedding(texts: list[str]) -> np.ndarray:
    return np.array(
        [[float(index + dimension) for dimension in range(384)] for index, _ in enumerate(texts)],
        dtype=float,
    )


async def entity_grounded_fixture_model(
    prompt, system_prompt=None, history_messages=None, **kwargs
) -> str:
    if system_prompt is not None:
        return FAKE_COMMUNITY_REPORT
    if prompt == "continue_prompt" or "MANY entities were missed" in prompt:
        return ""
    return (
        '("entity"<|>SAM BANKMAN-FRIED<|>PERSON<|>Founder of FTX.)##'
        '("entity"<|>FTX<|>ORGANIZATION<|>Cryptocurrency exchange founded by Sam Bankman-Fried.)##'
        '("relationship"<|>SAM BANKMAN-FRIED<|>FTX<|>Sam Bankman-Fried founded FTX.<|>1.0)<|COMPLETE|>'
    )


def build_entity_grounded_rag(tmp_path):
    return GraphRAG(
        working_dir=str(tmp_path),
        best_model_func=entity_grounded_fixture_model,
        cheap_model_func=entity_grounded_fixture_model,
        embedding_func=local_embedding,
        enable_naive_rag=True,
    )


def test_graphrag_initializes_entity_registry(tmp_path):
    rag = build_entity_grounded_rag(tmp_path)

    assert hasattr(rag, "entity_registry")
    assert rag.entity_registry is not None
    assert len(rag.entity_registry) == 0


def test_query_param_accepts_entity_grounded_mode():
    param = QueryParam(mode="entity_grounded")

    assert param.mode == "entity_grounded"
    assert param.entity_grounded_max_answer_length == 50
    assert param.entity_grounded_require_entity_match is True
    assert param.entity_grounded_fuzzy_threshold == 0.85


def test_insert_documents_registers_entities_without_live_provider(tmp_path):
    rag = build_entity_grounded_rag(tmp_path)

    rag.insert_documents(
        {
            "doc-1": "Sam Bankman-Fried founded FTX.",
            "doc-2": "FTX was a cryptocurrency exchange.",
        }
    )

    canonical_id = rag.entity_registry.resolve_entity("SAM BANKMAN-FRIED", fuzzy_threshold=1.0)
    assert canonical_id is not None
    assert rag.entity_registry.get_canonical_name(canonical_id) == "SAM BANKMAN-FRIED"

    ftx_id = rag.entity_registry.resolve_entity("FTX", fuzzy_threshold=1.0)
    assert ftx_id is not None
    assert rag.entity_registry.get_canonical_name(ftx_id) == "FTX"

    manifest = asyncio.get_event_loop().run_until_complete(rag.document_index.get_by_id("doc-1"))
    assert manifest is not None
    assert canonical_id in manifest["entities"]
    assert ftx_id in manifest["entities"]
