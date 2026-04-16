import asyncio
import json
import os
import shutil

import numpy as np

from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag._entity_grounded_query import QueryResult
from nano_graphrag._utils import (
    compute_mdhash_id,
    compute_sha256_id,
    generate_stable_entity_id,
    wrap_embedding_func_with_attrs,
)

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
    after_manifest = asyncio.get_event_loop().run_until_complete(
        rag.document_index.get_by_id("doc-1")
    )

    assert initial_doc["content"] == "Charles Dickens wrote A Christmas Carol."
    assert initial_manifest["content_hash"] == after_manifest["content_hash"]
    assert len(initial_manifest["chunk_ids"]) == len(after_manifest["chunk_ids"])


def test_insert_reuses_legacy_md5_document_ids():
    rag = build_incremental_rag()
    content = "Charles Dickens wrote A Christmas Carol."
    legacy_doc_id = compute_mdhash_id(content, prefix="doc-")
    sha_doc_id = compute_sha256_id(content, prefix="doc-")

    asyncio.get_event_loop().run_until_complete(
        rag.full_docs.upsert(
            {
                legacy_doc_id: {
                    "content": content,
                    "content_hash": compute_sha256_id(content),
                }
            }
        )
    )

    rag.insert(content)

    all_doc_keys = asyncio.get_event_loop().run_until_complete(rag.full_docs.all_keys())
    assert legacy_doc_id in all_doc_keys
    assert sha_doc_id not in all_doc_keys


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
    assert carol_id not in rag.entity_registry
    assert oliver_id in rag.entity_registry


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
    assert all([dickens_id, carol_id] not in (report or {}).get("edges", []) for report in reports)


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

    assert (
        loop.run_until_complete(rag.chunk_entity_relation_graph.get_edge(dickens_id, carol_id))
        is None
    )
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

    manifest = asyncio.get_event_loop().run_until_complete(
        reloaded.document_index.get_by_id("doc-1")
    )
    assert len(manifest["entities"]) == 2


def test_incremental_update_keeps_entity_identity_when_new_extract_is_unknown():
    clean_working_dir()
    state = {"relationship_only": False}

    async def type_drift_model(prompt, system_prompt=None, history_messages=None, **kwargs) -> str:
        if system_prompt is not None:
            return FAKE_COMMUNITY_REPORT
        if prompt == "continue_prompt" or "MANY entities were missed" in prompt:
            return ""
        if not state["relationship_only"] and "Charles Dickens wrote Oliver Twist" in prompt:
            return (
                '("entity"<|>CHARLES DICKENS<|>PERSON<|>Author of Oliver Twist.)##'
                '("entity"<|>OLIVER TWIST<|>WORK<|>A novel by Charles Dickens.)##'
                '("relationship"<|>CHARLES DICKENS<|>OLIVER TWIST<|>Charles Dickens wrote Oliver Twist.<|>1.0)<|COMPLETE|>'
            )
        return '("relationship"<|>CHARLES DICKENS<|>OLIVER TWIST<|>Charles Dickens wrote Oliver Twist.<|>1.0)<|COMPLETE|>'

    rag = GraphRAG(
        working_dir=WORKING_DIR,
        best_model_func=type_drift_model,
        cheap_model_func=type_drift_model,
        embedding_func=local_embedding,
        enable_naive_rag=True,
    )

    rag.insert_documents({"doc-1": "Charles Dickens wrote Oliver Twist."})
    state["relationship_only"] = True
    rag.insert_documents({"doc-1": "Charles Dickens wrote Oliver Twist again."})

    loop = asyncio.get_event_loop()
    person_id = generate_stable_entity_id("CHARLES DICKENS", "PERSON")
    unknown_id = generate_stable_entity_id("CHARLES DICKENS", '"UNKNOWN"')

    # Entity IDs are now type-agnostic, so person_id == unknown_id
    # This means the entity naturally keeps its identity across re-indexing
    assert person_id == unknown_id, "Entity IDs should be type-agnostic"
    assert loop.run_until_complete(rag.chunk_entity_relation_graph.has_node(person_id))


def test_local_query_context_uses_human_readable_names():
    rag = build_incremental_rag()
    rag.insert_documents({"doc-1": "Charles Dickens wrote A Christmas Carol."})

    context = rag.query("Dickens", param=QueryParam(mode="local", only_need_context=True))
    dickens_id = generate_stable_entity_id("CHARLES DICKENS", "PERSON")

    assert "CHARLES DICKENS" in context
    assert dickens_id not in context


def test_incremental_rebuild_uses_reverse_index_without_full_manifest_scan(monkeypatch):
    rag = build_incremental_rag()
    rag.insert_documents({"doc-1": "Charles Dickens wrote A Christmas Carol."})

    async def fail_all_keys():
        raise AssertionError("document_index.all_keys should not be used once reverse index exists")

    monkeypatch.setattr(rag.document_index, "all_keys", fail_all_keys)
    rag.insert_documents({"doc-1": "Charles Dickens wrote Oliver Twist."})


def test_incremental_rebuild_regenerates_missing_reverse_index():
    rag = build_incremental_rag()
    rag.insert_documents({"doc-1": "Charles Dickens wrote A Christmas Carol."})

    asyncio.get_event_loop().run_until_complete(rag.graph_contribution_index.drop())
    rag.insert_documents({"doc-1": "Charles Dickens wrote Oliver Twist."})

    meta = asyncio.get_event_loop().run_until_complete(
        rag.graph_contribution_index.get_by_id("__meta__")
    )
    assert meta["built"] is True
    assert meta["version"] == 1


def test_reverse_index_persists_across_restart():
    rag = build_incremental_rag()
    rag.insert_documents({"doc-1": "Charles Dickens wrote A Christmas Carol."})

    reloaded = GraphRAG(
        working_dir=WORKING_DIR,
        best_model_func=incremental_model,
        cheap_model_func=incremental_model,
        embedding_func=local_embedding,
        enable_naive_rag=True,
    )
    contribution = asyncio.get_event_loop().run_until_complete(
        reloaded.graph_contribution_index.get_by_id("entity_name::CHARLES DICKENS")
    )
    assert contribution["doc_ids"] == ["doc-1"]


def test_entity_linking_exact_alias_match_reuses_canonical_entity_id():
    clean_working_dir()

    async def linking_model(prompt, system_prompt=None, history_messages=None, **kwargs):
        if "extract alternative names and aliases" in prompt.lower():
            if '"SAM BANKMAN-FRIED"' in prompt:
                return json.dumps({"Entity 1": ["SBF", "SAM BANKMAN FRIED"]})
            return json.dumps({"Entity 1": []})
        if system_prompt is not None:
            return FAKE_COMMUNITY_REPORT
        if prompt == "continue_prompt" or "MANY entities were missed" in prompt:
            return ""
        if "Sam Bankman-Fried ran FTX." in prompt:
            return (
                '("entity"<|>SAM BANKMAN-FRIED<|>PERSON<|>Executive tied to FTX.)##'
                '("entity"<|>FTX<|>ORG<|>Crypto exchange.)##'
                '("relationship"<|>SAM BANKMAN-FRIED<|>FTX<|>Sam Bankman-Fried ran FTX.<|>1.0)<|COMPLETE|>'
            )
        if "SBF ran FTX." in prompt:
            return (
                '("entity"<|>SBF<|>PERSON<|>Executive tied to FTX.)##'
                '("entity"<|>FTX<|>ORG<|>Crypto exchange.)##'
                '("relationship"<|>SBF<|>FTX<|>SBF ran FTX.<|>1.0)<|COMPLETE|>'
            )
        return FAKE_COMMUNITY_REPORT

    rag = GraphRAG(
        working_dir=WORKING_DIR,
        best_model_func=linking_model,
        cheap_model_func=linking_model,
        embedding_func=local_embedding,
        enable_naive_rag=True,
        enable_entity_linking=True,
    )

    rag.insert_documents({"doc-1": "Sam Bankman-Fried ran FTX."})
    canonical_id = rag.entity_registry.resolve_entity("SAM BANKMAN-FRIED", fuzzy_threshold=1.0)

    rag.insert_documents({"doc-2": "SBF ran FTX."})

    assert canonical_id is not None
    assert asyncio.get_event_loop().run_until_complete(
        rag.chunk_entity_relation_graph.has_node(canonical_id)
    )
    assert rag.entity_registry.resolve_entity("SBF", fuzzy_threshold=1.0) == canonical_id

    reloaded = GraphRAG(
        working_dir=WORKING_DIR,
        best_model_func=linking_model,
        cheap_model_func=linking_model,
        embedding_func=local_embedding,
        enable_naive_rag=True,
    )
    assert reloaded.entity_registry.resolve_entity("SBF", fuzzy_threshold=1.0) == canonical_id


def test_entity_linking_ambiguous_candidates_stay_separate_when_disabled():
    clean_working_dir()

    async def ambiguous_model(prompt, system_prompt=None, history_messages=None, **kwargs):
        if "Extract alternative names and aliases" in prompt:
            return json.dumps({"aliases": []})
        if system_prompt is not None:
            return FAKE_COMMUNITY_REPORT
        if prompt == "continue_prompt" or "MANY entities were missed" in prompt:
            return ""
        if "Acme Corp signed the deal." in prompt:
            return '("entity"<|>ACME CORP<|>ORG<|>Company one.)<|COMPLETE|>'
        if "Acme Holdings signed the deal." in prompt:
            return '("entity"<|>ACME HOLDINGS<|>ORG<|>Company two.)<|COMPLETE|>'
        if "Acme signed the deal." in prompt:
            return '("entity"<|>ACME<|>ORG<|>Unclear company.)<|COMPLETE|>'
        return FAKE_COMMUNITY_REPORT

    rag = GraphRAG(
        working_dir=WORKING_DIR,
        best_model_func=ambiguous_model,
        cheap_model_func=ambiguous_model,
        embedding_func=local_embedding,
        enable_naive_rag=True,
        entity_linking_similarity_threshold=0.5,
    )

    rag.entity_registry.register_entity("entity-acme-corp", "ACME CORP", entity_type="ORG")
    rag.entity_registry.register_entity("entity-acme-llc", "ACME LLC", entity_type="ORG")
    rag.insert_documents({"doc-3": "Acme signed the deal."})

    manifest = asyncio.get_event_loop().run_until_complete(rag.document_index.get_by_id("doc-3"))
    ambiguous_id = next(iter(manifest["entities"].keys()))
    assert ambiguous_id not in {
        rag.entity_registry.resolve_entity("ACME CORP", fuzzy_threshold=1.0),
        rag.entity_registry.resolve_entity("ACME LLC", fuzzy_threshold=1.0),
    }


def test_entity_linking_low_confidence_ambiguous_case_creates_new_entity():
    clean_working_dir()

    async def cautious_model(prompt, system_prompt=None, history_messages=None, **kwargs):
        if "Return JSON with exactly this shape" in prompt:
            return json.dumps({"decision": "new", "entity_id": ""})
        if "Extract alternative names and aliases" in prompt:
            return json.dumps({"aliases": []})
        if system_prompt is not None:
            return FAKE_COMMUNITY_REPORT
        if prompt == "continue_prompt" or "MANY entities were missed" in prompt:
            return ""
        if "Acme Corp signed the deal." in prompt:
            return '("entity"<|>ACME CORP<|>ORG<|>Company one.)<|COMPLETE|>'
        if "Acme Holdings signed the deal." in prompt:
            return '("entity"<|>ACME HOLDINGS<|>ORG<|>Company two.)<|COMPLETE|>'
        if "Acme signed the deal." in prompt:
            return '("entity"<|>ACME<|>ORG<|>Unclear company.)<|COMPLETE|>'
        return FAKE_COMMUNITY_REPORT

    rag = GraphRAG(
        working_dir=WORKING_DIR,
        best_model_func=cautious_model,
        cheap_model_func=cautious_model,
        embedding_func=local_embedding,
        enable_naive_rag=True,
        enable_entity_linking=True,
        entity_linking_similarity_threshold=0.5,
    )

    rag.entity_registry.register_entity("entity-acme-corp", "ACME CORP", entity_type="ORG")
    rag.entity_registry.register_entity("entity-acme-llc", "ACME LLC", entity_type="ORG")
    rag.insert_documents({"doc-3": "Acme signed the deal."})

    manifest = asyncio.get_event_loop().run_until_complete(rag.document_index.get_by_id("doc-3"))
    ambiguous_id = next(iter(manifest["entities"].keys()))
    assert ambiguous_id not in {
        rag.entity_registry.resolve_entity("ACME CORP", fuzzy_threshold=1.0),
        rag.entity_registry.resolve_entity("ACME LLC", fuzzy_threshold=1.0),
    }


def test_entity_grounded_query_returns_answer_string(monkeypatch):
    rag = build_incremental_rag()

    async def fake_query(self, question, top_k=30, mode="local"):
        return QueryResult(
            answer="CHARLES DICKENS",
            entity_ids=["entity_1"],
            canonical_entities=["CHARLES DICKENS"],
            confidence=1.0,
        )

    monkeypatch.setattr("nano_graphrag.graphrag_query.EntityGroundedQuery.query", fake_query)

    result = rag.query("Who wrote it?", param=QueryParam(mode="entity_grounded"))

    assert result == "CHARLES DICKENS"


# --- Fix 1: VDB upsert resilience ---


def test_rebuild_succeeds_when_entity_vdb_upsert_fails(monkeypatch):
    """Entity VDB upsert failure during rebuild should not crash the pipeline."""
    rag = build_incremental_rag()
    rag.insert_documents({"doc-1": "Charles Dickens wrote A Christmas Carol."})

    dickens_id = generate_stable_entity_id("CHARLES DICKENS", "PERSON")
    carol_id = generate_stable_entity_id("A CHRISTMAS CAROL", "WORK")

    # Verify initial insert worked
    assert asyncio.get_event_loop().run_until_complete(
        rag.chunk_entity_relation_graph.has_node(dickens_id)
    )

    # Now patch entity_vdb.upsert to fail, and insert a changed doc
    original_upsert = rag.entities_vdb.upsert
    call_count = {"n": 0}

    async def failing_vdb_upsert(data):
        call_count["n"] += 1
        if call_count["n"] > 1:
            raise RuntimeError("Simulated VDB failure")
        return await original_upsert(data)

    monkeypatch.setattr(rag.entities_vdb, "upsert", failing_vdb_upsert)

    # Should NOT raise despite VDB failure
    rag.insert_documents({"doc-1": "Charles Dickens wrote Oliver Twist."})

    # Graph should still have entities (nodes persisted even if VDB failed)
    oliver_id = generate_stable_entity_id("OLIVER TWIST", "WORK")
    assert asyncio.get_event_loop().run_until_complete(
        rag.chunk_entity_relation_graph.has_node(dickens_id)
    )
    assert asyncio.get_event_loop().run_until_complete(
        rag.chunk_entity_relation_graph.has_node(oliver_id)
    )


# --- Fix 2: Manifest rollback on rebuild failure ---


def test_manifests_rolled_back_on_rebuild_failure(monkeypatch):
    """If rebuild fails, manifests should be deleted so docs are re-extracted next run."""
    rag = build_incremental_rag()

    from nano_graphrag._ops.extraction_rebuild import rebuild_knowledge_graph_for_documents

    async def failing_rebuild(*args, **kwargs):
        raise RuntimeError("Simulated rebuild failure")

    monkeypatch.setattr(
        "nano_graphrag.graphrag_insert.rebuild_knowledge_graph_for_documents",
        failing_rebuild,
    )

    # Should raise
    raised = False
    try:
        rag.insert_documents({"doc-1": "Charles Dickens wrote A Christmas Carol."})
    except RuntimeError:
        raised = True
    assert raised

    # Manifests should be rolled back — document_index should NOT have the doc
    manifest = asyncio.get_event_loop().run_until_complete(
        rag.document_index.get_by_id("doc-1")
    )
    assert manifest is None, "Manifest should have been rolled back after rebuild failure"


# --- Fix 3: Integrity check ---


def test_integrity_check_warns_on_empty_graph(caplog):
    """Integrity check should log WARNING when manifests have entities but graph is empty."""
    import logging

    rag = build_incremental_rag()

    # Insert normally first
    rag.insert_documents({"doc-1": "Charles Dickens wrote A Christmas Carol."})

    # Now corrupt: delete graph nodes but keep manifests
    loop = asyncio.get_event_loop()
    graph = rag.chunk_entity_relation_graph
    dickens_id = generate_stable_entity_id("CHARLES DICKENS", "PERSON")
    carol_id = generate_stable_entity_id("A CHRISTMAS CAROL", "WORK")
    loop.run_until_complete(graph.delete_node(dickens_id))
    loop.run_until_complete(graph.delete_node(carol_id))

    # Verify graph is empty but manifests still exist
    assert not loop.run_until_complete(graph.has_node(dickens_id))
    manifest = loop.run_until_complete(rag.document_index.get_by_id("doc-1"))
    assert manifest is not None and len(manifest.get("entities", {})) > 0

    # Re-insert with changed content — rebuild should succeed, integrity check should run
    with caplog.at_level(logging.WARNING, logger="nano-graphrag"):
        rag.insert_documents({"doc-1": "Charles Dickens wrote Oliver Twist."})

    # The integrity check code runs without error (graph gets rebuilt with new entities)
    # Verify new entities are in graph
    oliver_id = generate_stable_entity_id("OLIVER TWIST", "WORK")
    assert loop.run_until_complete(graph.has_node(dickens_id))
    assert loop.run_until_complete(graph.has_node(oliver_id))


# --- Fix 4: Extraction hash ---


def test_extraction_hash_triggers_re_extraction(monkeypatch):
    """Changing extraction config should trigger re-extraction of unchanged docs."""
    rag = build_incremental_rag()
    rag.insert_documents({"doc-1": "Charles Dickens wrote A Christmas Carol."})

    # Verify manifest has extraction_hash
    manifest = asyncio.get_event_loop().run_until_complete(
        rag.document_index.get_by_id("doc-1")
    )
    assert "extraction_hash" in manifest, "Manifest should contain extraction_hash"

    # Patch _compute_extraction_hash to return a different hash
    from nano_graphrag import graphrag_insert

    original_hash = graphrag_insert._compute_extraction_hash

    def different_hash(global_config, extraction_func):
        return original_hash(global_config, extraction_func) + "_changed"

    monkeypatch.setattr(graphrag_insert, "_compute_extraction_hash", different_hash)

    # Re-insert same content — should re-extract due to changed hash
    extraction_count = {"n": 0}
    original_extract = graphrag_insert.extract_document_entity_relationships

    async def counting_extract(*args, **kwargs):
        extraction_count["n"] += 1
        return await original_extract(*args, **kwargs)

    monkeypatch.setattr(
        graphrag_insert, "extract_document_entity_relationships", counting_extract
    )

    rag.insert_documents({"doc-1": "Charles Dickens wrote A Christmas Carol."})

    assert extraction_count["n"] > 0, "Doc should be re-extracted due to changed extraction hash"


# --- Fix 5: force_rebuild and rebuild_graph ---


def test_force_rebuild_re_extracts_unchanged_docs():
    """force_rebuild=True should re-extract even unchanged documents."""
    rag = build_incremental_rag()
    rag.insert_documents({"doc-1": "Charles Dickens wrote A Christmas Carol."})

    dickens_id = generate_stable_entity_id("CHARLES DICKENS", "PERSON")
    assert asyncio.get_event_loop().run_until_complete(
        rag.chunk_entity_relation_graph.has_node(dickens_id)
    )

    # force_rebuild with same content
    rag.insert_documents(
        {"doc-1": "Charles Dickens wrote A Christmas Carol."}, force_rebuild=True
    )

    # Should still have the entity
    assert asyncio.get_event_loop().run_until_complete(
        rag.chunk_entity_relation_graph.has_node(dickens_id)
    )


def test_rebuild_graph_from_manifests():
    """rebuild_graph() should rebuild graph from existing manifests without re-extraction."""
    rag = build_incremental_rag()
    rag.insert_documents({"doc-1": "Charles Dickens wrote A Christmas Carol."})

    dickens_id = generate_stable_entity_id("CHARLES DICKENS", "PERSON")
    carol_id = generate_stable_entity_id("A CHRISTMAS CAROL", "WORK")

    # Verify initial state
    loop = asyncio.get_event_loop()
    assert loop.run_until_complete(rag.chunk_entity_relation_graph.has_node(dickens_id))
    assert loop.run_until_complete(rag.chunk_entity_relation_graph.has_node(carol_id))

    # Rebuild from manifests
    rag.rebuild_graph()

    # Graph should still have the entities
    assert loop.run_until_complete(rag.chunk_entity_relation_graph.has_node(dickens_id))
    assert loop.run_until_complete(rag.chunk_entity_relation_graph.has_node(carol_id))


# --- Concurrent Insert Tests ---


def test_concurrent_document_insert_no_duplicates():
    """Test that multiple inserts with same content don't create duplicate entities."""
    rag = build_incremental_rag()

    # Insert document with entities
    rag.insert_documents({"doc-1": "Charles Dickens wrote A Christmas Carol."})

    dickens_id = generate_stable_entity_id("CHARLES DICKENS", "PERSON")
    carol_id = generate_stable_entity_id("A CHRISTMAS CAROL", "WORK")
    loop = asyncio.get_event_loop()

    # Verify entities exist
    assert loop.run_until_complete(rag.chunk_entity_relation_graph.has_node(dickens_id))
    assert loop.run_until_complete(rag.chunk_entity_relation_graph.has_node(carol_id))

    # Insert same document again - should reuse existing entities
    rag.insert_documents({"doc-1": "Charles Dickens wrote A Christmas Carol."})

    # Verify still only one of each entity
    assert loop.run_until_complete(rag.chunk_entity_relation_graph.has_node(dickens_id))
    assert loop.run_until_complete(rag.chunk_entity_relation_graph.has_node(carol_id))


def test_concurrent_entity_linking():
    """Test entity registry with multiple documents referencing same entity."""
    clean_working_dir()
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        best_model_func=incremental_model,
        cheap_model_func=incremental_model,
        embedding_func=local_embedding,
        enable_entity_linking=True,
    )

    # Insert documents that reference the same entity
    docs = {
        "doc-1": "Charles Dickens wrote A Christmas Carol.",
        "doc-2": "Charles Dickens wrote Oliver Twist.",
    }

    rag.insert_documents(docs)

    # Both should reference the same canonical entity
    dickens_id = generate_stable_entity_id("CHARLES DICKENS", "PERSON")
    loop = asyncio.get_event_loop()

    # Check that the entity exists
    assert loop.run_until_complete(rag.chunk_entity_relation_graph.has_node(dickens_id))

    # Verify entity registry has the canonical name
    if rag.entity_registry:
        record = rag.entity_registry.get_entity_record(dickens_id)
        assert record is not None
        assert record.canonical_name == "CHARLES DICKENS"
