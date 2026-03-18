import asyncio
import os
import shutil

import numpy as np

from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag._utils import generate_stable_entity_id, wrap_embedding_func_with_attrs

os.environ["OPENAI_API_KEY"] = "FAKE"

WORKING_DIR = "./tests/nano_graphrag_cache_INCREMENTAL"
FAKE_COMMUNITY_REPORT = """{"title":"Test Community","summary":"Incremental test report.","rating":1,"rating_explanation":"Fixture","findings":[{"summary":"Fixture","explanation":"Fixture"}]}"""


@wrap_embedding_func_with_attrs(embedding_dim=384, max_token_size=8192)
async def local_embedding(texts: list[str]) -> np.ndarray:
    return np.random.rand(len(texts), 384)


async def incremental_model(prompt, system_prompt=None, history_messages=None, **kwargs) -> str:
    if system_prompt is not None:
        return FAKE_COMMUNITY_REPORT
    if prompt == "continue_prompt" or "MANY entities were missed" in prompt:
        return ""
    if "Charles Dickens wrote A Christmas Carol" in prompt:
        return (
            '("entity"<|>CHARLES DICKENS<|>PERSON<|>Author of A Christmas Carol.)##'
            '("entity"<|>A CHRISTMAS CAROL<|>WORK<|>A novella by Charles Dickens.)##'
            '("relationship"<|>CHARLES DICKENS<|>A CHRISTMAS CAROL<|>Charles Dickens wrote A Christmas Carol.<|>1.0)<|COMPLETE|>'
        )
    if "Charles Dickens wrote Oliver Twist" in prompt:
        return (
            '("entity"<|>CHARLES DICKENS<|>PERSON<|>Author of Oliver Twist.)##'
            '("entity"<|>OLIVER TWIST<|>WORK<|>A novel by Charles Dickens.)##'
            '("relationship"<|>CHARLES DICKENS<|>OLIVER TWIST<|>Charles Dickens wrote Oliver Twist.<|>1.0)<|COMPLETE|>'
        )
    if "Charles Dickens discussed A Christmas Carol" in prompt:
        return (
            '("entity"<|>CHARLES DICKENS<|>PERSON<|>Author discussed the novella.)##'
            '("entity"<|>A CHRISTMAS CAROL<|>WORK<|>A novella under discussion.)<|COMPLETE|>'
        )
    return FAKE_COMMUNITY_REPORT


def clean_working_dir():
    if os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR)
    os.mkdir(WORKING_DIR)


def build_incremental_rag():
    clean_working_dir()
    return GraphRAG(
        working_dir=WORKING_DIR,
        best_model_func=incremental_model,
        cheap_model_func=incremental_model,
        embedding_func=local_embedding,
        enable_naive_rag=True,
    )


def test_insert_documents_and_skip_unchanged():
    rag = build_incremental_rag()

    rag.insert_documents({"doc-1": "Charles Dickens wrote A Christmas Carol."})
    initial_doc = asyncio.get_event_loop().run_until_complete(rag.full_docs.get_by_id("doc-1"))
    initial_manifest = asyncio.get_event_loop().run_until_complete(
        rag.document_index.get_by_id("doc-1")
    )

    rag.insert_documents({"doc-1": "Charles Dickens wrote A Christmas Carol."})
    after_manifest = asyncio.get_event_loop().run_until_complete(rag.document_index.get_by_id("doc-1"))

    assert initial_doc["content"] == "Charles Dickens wrote A Christmas Carol."
    assert initial_manifest["content_hash"] == after_manifest["content_hash"]
    assert len(initial_manifest["chunk_ids"]) == len(after_manifest["chunk_ids"])


def test_insert_documents_replaces_changed_version():
    rag = build_incremental_rag()

    rag.insert_documents({"doc-1": "Charles Dickens wrote A Christmas Carol."})
    original_manifest = asyncio.get_event_loop().run_until_complete(
        rag.document_index.get_by_id("doc-1")
    )

    rag.insert_documents({"doc-1": "Charles Dickens wrote Oliver Twist."})
    updated_manifest = asyncio.get_event_loop().run_until_complete(
        rag.document_index.get_by_id("doc-1")
    )
    full_doc = asyncio.get_event_loop().run_until_complete(rag.full_docs.get_by_id("doc-1"))

    dickens_id = generate_stable_entity_id("CHARLES DICKENS", "PERSON")
    carol_id = generate_stable_entity_id("A CHRISTMAS CAROL", "WORK")
    oliver_id = generate_stable_entity_id("OLIVER TWIST", "WORK")

    assert original_manifest["content_hash"] != updated_manifest["content_hash"]
    assert full_doc["content"] == "Charles Dickens wrote Oliver Twist."
    assert not asyncio.get_event_loop().run_until_complete(
        rag.chunk_entity_relation_graph.has_node(carol_id)
    )
    assert asyncio.get_event_loop().run_until_complete(
        rag.chunk_entity_relation_graph.has_node(dickens_id)
    )
    assert asyncio.get_event_loop().run_until_complete(
        rag.chunk_entity_relation_graph.has_node(oliver_id)
    )


def test_insert_documents_removes_stale_relationship_and_community_edges():
    rag = build_incremental_rag()

    rag.insert_documents({"doc-1": "Charles Dickens wrote A Christmas Carol."})
    rag.insert_documents({"doc-1": "Charles Dickens discussed A Christmas Carol."})

    loop = asyncio.get_event_loop()
    dickens_id = generate_stable_entity_id("CHARLES DICKENS", "PERSON")
    carol_id = generate_stable_entity_id("A CHRISTMAS CAROL", "WORK")
    edge = loop.run_until_complete(rag.chunk_entity_relation_graph.get_edge(dickens_id, carol_id))
    reports = loop.run_until_complete(
        rag.community_reports.get_by_ids(loop.run_until_complete(rag.community_reports.all_keys()))
    )

    assert edge is None
    assert loop.run_until_complete(rag.chunk_entity_relation_graph.has_node(dickens_id))
    assert loop.run_until_complete(rag.chunk_entity_relation_graph.has_node(carol_id))
    assert all(
        [dickens_id, carol_id] not in (report or {}).get("edges", [])
        for report in reports
    )


def test_insert_documents_removes_shared_relationship_but_keeps_shared_nodes():
    rag = build_incremental_rag()

    rag.insert_documents(
        {
            "doc-1": "Charles Dickens wrote A Christmas Carol.",
            "doc-2": "Charles Dickens discussed A Christmas Carol.",
        }
    )
    rag.insert_documents({"doc-1": "Charles Dickens discussed A Christmas Carol."})

    loop = asyncio.get_event_loop()
    dickens_id = generate_stable_entity_id("CHARLES DICKENS", "PERSON")
    carol_id = generate_stable_entity_id("A CHRISTMAS CAROL", "WORK")

    assert loop.run_until_complete(rag.chunk_entity_relation_graph.get_edge(dickens_id, carol_id)) is None
    assert loop.run_until_complete(rag.chunk_entity_relation_graph.has_node(dickens_id))
    assert loop.run_until_complete(rag.chunk_entity_relation_graph.has_node(carol_id))


def test_insert_documents_persists_delta_state():
    rag = build_incremental_rag()
    rag.insert_documents({"doc-1": "Charles Dickens wrote A Christmas Carol."})

    reloaded = GraphRAG(
        working_dir=WORKING_DIR,
        best_model_func=incremental_model,
        cheap_model_func=incremental_model,
        embedding_func=local_embedding,
        enable_naive_rag=True,
    )
    reloaded.insert_documents({"doc-1": "Charles Dickens wrote A Christmas Carol."})

    manifest = asyncio.get_event_loop().run_until_complete(reloaded.document_index.get_by_id("doc-1"))
    assert len(manifest["entities"]) == 2


def test_local_query_context_uses_human_readable_names():
    rag = build_incremental_rag()
    rag.insert_documents({"doc-1": "Charles Dickens wrote A Christmas Carol."})

    context = rag.query("Dickens", param=QueryParam(mode="local", only_need_context=True))
    dickens_id = generate_stable_entity_id("CHARLES DICKENS", "PERSON")

    assert "CHARLES DICKENS" in context
    assert dickens_id not in context
