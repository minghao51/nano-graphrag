import os
import json
import shutil
import asyncio
import numpy as np
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag._ops.extraction import extract_document_entity_relationships
from nano_graphrag._utils import (
    compute_mdhash_id,
    generate_stable_entity_id,
    wrap_embedding_func_with_attrs,
)

os.environ["OPENAI_API_KEY"] = "FAKE"

WORKING_DIR = "./tests/nano_graphrag_cache_RUNTIME"
FAKE_RESPONSE = "Hello world"
FAKE_JSON = json.dumps({"points": [{"description": "Hello world", "score": 1}]})
FAKE_COMMUNITY_REPORT = json.dumps(
    {
        "title": "Charles Dickens",
        "summary": "A small test community around Dickens and his novel.",
        "rating": 1,
        "rating_explanation": "Low impact test fixture.",
        "findings": [
            {"summary": "Dickens wrote the novel", "explanation": "Fixture data only."}
        ],
    }
)


def remove_if_exist(file):
    if os.path.exists(file):
        os.remove(file)


# We're using random embedding function for testing
@wrap_embedding_func_with_attrs(embedding_dim=384, max_token_size=8192)
async def local_embedding(texts: list[str]) -> np.ndarray:
    return np.random.rand(len(texts), 384)


async def fake_model(prompt, system_prompt=None, history_messages=None, **kwargs) -> str:
    return FAKE_RESPONSE


async def fake_json_model(prompt, system_prompt=None, history_messages=None, **kwargs) -> str:
    if system_prompt is not None:
        return FAKE_JSON
    return FAKE_COMMUNITY_REPORT


async def invalid_structured_model(prompt, system_prompt=None, history_messages=None, **kwargs) -> str:
    if system_prompt is not None:
        return "{not-json"
    if prompt == "continue_prompt" or "MANY entities were missed" in prompt:
        return ""
    return (
        '("entity"<|>CHARLES DICKENS<|>PERSON<|>Author of A Christmas Carol.)##'
        '("entity"<|>A CHRISTMAS CAROL<|>WORK<|>A novella by Charles Dickens.)##'
        '("relationship"<|>CHARLES DICKENS<|>A CHRISTMAS CAROL<|>Charles Dickens wrote A Christmas Carol.<|>1.0)<|COMPLETE|>'
    )


async def fake_entity_extraction(
    chunks,
    knowledge_graph_inst,
    entity_vdb,
    tokenizer_wrapper,
    global_config,
    using_amazon_bedrock=False,
):
    chunk_id = next(iter(chunks))
    dickens_id = generate_stable_entity_id("CHARLES DICKENS", "PERSON")
    carol_id = generate_stable_entity_id("A CHRISTMAS CAROL", "WORK")
    await knowledge_graph_inst.upsert_node(
        dickens_id,
        {
            "entity_name": "CHARLES DICKENS",
            "entity_type": "PERSON",
            "description": "Author of A Christmas Carol.",
            "source_id": chunk_id,
        },
    )
    await knowledge_graph_inst.upsert_node(
        carol_id,
        {
            "entity_name": "A CHRISTMAS CAROL",
            "entity_type": "WORK",
            "description": "A novella by Charles Dickens.",
            "source_id": chunk_id,
        },
    )
    await knowledge_graph_inst.upsert_edge(
        dickens_id,
        carol_id,
        {
            "description": "Charles Dickens wrote A Christmas Carol.",
            "weight": 1.0,
            "source_id": chunk_id,
        },
    )
    if entity_vdb is not None:
        await entity_vdb.upsert(
            {
                dickens_id: {
                    "content": "CHARLES DICKENS Author of A Christmas Carol.",
                    "entity_name": "CHARLES DICKENS",
                },
                carol_id: {
                    "content": "A CHRISTMAS CAROL Novella by Dickens.",
                    "entity_name": "A CHRISTMAS CAROL",
                },
            }
        )
    return knowledge_graph_inst


def clean_working_dir():
    if os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR)
    os.mkdir(WORKING_DIR)


def build_query_rag(best_model_func=fake_model, enable_naive_rag=False):
    clean_working_dir()
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        best_model_func=best_model_func,
        cheap_model_func=fake_model,
        embedding_func=local_embedding,
        enable_naive_rag=enable_naive_rag,
    )
    loop = asyncio.get_event_loop()
    chunk_id = "chunk-test-1"
    dickens_id = generate_stable_entity_id("CHARLES DICKENS", "PERSON")
    carol_id = generate_stable_entity_id("A CHRISTMAS CAROL", "WORK")
    loop.run_until_complete(
        rag.text_chunks.upsert(
            {
                chunk_id: {
                    "tokens": 8,
                    "content": "Charles Dickens wrote A Christmas Carol.",
                    "full_doc_id": "doc-test-1",
                    "chunk_order_index": 0,
                }
            }
        )
    )
    loop.run_until_complete(
        rag.chunk_entity_relation_graph.upsert_node(
            dickens_id,
            {
                "entity_name": "CHARLES DICKENS",
                "entity_type": "PERSON",
                "description": "Author of A Christmas Carol.",
                "source_id": chunk_id,
                "clusters": '[{"level": 0, "cluster": 1}]',
            },
        )
    )
    loop.run_until_complete(
        rag.chunk_entity_relation_graph.upsert_node(
            carol_id,
            {
                "entity_name": "A CHRISTMAS CAROL",
                "entity_type": "WORK",
                "description": "A novella by Charles Dickens.",
                "source_id": chunk_id,
                "clusters": '[{"level": 0, "cluster": 1}]',
            },
        )
    )
    loop.run_until_complete(
        rag.chunk_entity_relation_graph.upsert_edge(
            dickens_id,
            carol_id,
            {
                "description": "Charles Dickens wrote A Christmas Carol.",
                "weight": 1.0,
                "source_id": chunk_id,
            },
        )
    )
    loop.run_until_complete(
        rag.entities_vdb.upsert(
            {
                dickens_id: {
                    "content": "CHARLES DICKENS Author of A Christmas Carol.",
                    "entity_name": "CHARLES DICKENS",
                }
            }
        )
    )
    if enable_naive_rag:
        loop.run_until_complete(
            rag.chunks_vdb.upsert(
                {
                    chunk_id: {
                        "content": "Charles Dickens wrote A Christmas Carol.",
                    }
                }
            )
        )
    loop.run_until_complete(
        rag.community_reports.upsert(
            {
                "1": {
                    "report_string": "Charles Dickens wrote A Christmas Carol.",
                    "report_json": {"rating": 1},
                    "occurrence": 1.0,
                    "level": 0,
                    "title": "Charles Dickens",
                    "edges": [[dickens_id, carol_id]],
                    "nodes": [dickens_id, carol_id],
                    "chunk_ids": [chunk_id],
                    "sub_communities": [],
                }
            }
        )
    )
    return rag


def test_insert():
    clean_working_dir()
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        best_model_func=fake_json_model,
        cheap_model_func=fake_model,
        embedding_func=local_embedding,
        enable_naive_rag=True,
        entity_extraction_func=fake_entity_extraction,
    )
    with open("./tests/mock_data.txt", encoding="utf-8-sig") as f:
        rag.insert(f.read())


def test_local_query():
    rag = build_query_rag(best_model_func=fake_model)
    result = rag.query("Dickens", param=QueryParam(mode="local"))
    assert result == FAKE_RESPONSE


def test_global_query():
    rag = build_query_rag(best_model_func=fake_json_model)
    result = rag.query("Dickens")
    assert result == FAKE_JSON


def test_naive_query():
    rag = build_query_rag(best_model_func=fake_model, enable_naive_rag=True)
    result = rag.query("Dickens", param=QueryParam(mode="naive"))
    assert result == FAKE_RESPONSE


def test_subcommunity_insert():
    clean_working_dir()
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        best_model_func=fake_json_model,
        cheap_model_func=fake_model,
        embedding_func=local_embedding,
        enable_naive_rag=True,
        entity_extraction_func=fake_entity_extraction,
        addon_params={"force_to_use_sub_communities": True},
    )
    with open("./tests/mock_data.txt", encoding="utf-8-sig") as f:
        FAKE_TEXT = f.read()
    rag.insert(FAKE_TEXT)


def test_structured_extraction_falls_back_to_legacy_parsing():
    clean_working_dir()
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        best_model_func=invalid_structured_model,
        cheap_model_func=invalid_structured_model,
        embedding_func=local_embedding,
    )
    manifest = asyncio.get_event_loop().run_until_complete(
        extract_document_entity_relationships(
            {
                "chunk-1": {
                    "tokens": 8,
                    "content": "Charles Dickens wrote A Christmas Carol.",
                    "full_doc_id": "doc-test-1",
                    "chunk_order_index": 0,
                }
            },
            rag.tokenizer_wrapper,
            {
                **rag._runtime_config(),
                "_use_structured_extraction": True,
                "fallback_to_parsing": True,
            },
        )
    )

    assert len(manifest["entities"]) == 2
    assert len(manifest["relationships"]) == 1


def test_structured_extraction_can_disable_legacy_fallback():
    clean_working_dir()
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        best_model_func=invalid_structured_model,
        cheap_model_func=invalid_structured_model,
        embedding_func=local_embedding,
    )
    manifest = asyncio.get_event_loop().run_until_complete(
        extract_document_entity_relationships(
            {
                "chunk-1": {
                    "tokens": 8,
                    "content": "Charles Dickens wrote A Christmas Carol.",
                    "full_doc_id": "doc-test-1",
                    "chunk_order_index": 0,
                }
            },
            rag.tokenizer_wrapper,
            {
                **rag._runtime_config(),
                "_use_structured_extraction": True,
                "fallback_to_parsing": False,
            },
        )
    )

    assert manifest["entities"] == {}
    assert manifest["relationships"] == {}
