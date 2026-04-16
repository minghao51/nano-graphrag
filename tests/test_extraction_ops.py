import json
import os
import shutil

import numpy as np
import pytest

from nano_graphrag._entity_registry import EntityRegistry
from nano_graphrag._ops.extraction_common import _normalize_entity_name
from nano_graphrag._ops.extraction_rebuild import (
    _append_doc_id,
    _entity_index_key,
    _entity_name_index_key,
    _manifest_index_keys,
    _relationship_index_key,
    rebuild_graph_contribution_index,
    rebuild_knowledge_graph_for_documents,
    update_graph_contribution_index_for_documents,
)
from nano_graphrag._ops.extraction_structured import _parse_single_result
from nano_graphrag._schemas import EntityExtractionOutput, ExtractedEntity, ExtractedRelationship
from nano_graphrag._storage.gdb_networkx import NetworkXStorage
from nano_graphrag._storage.kv_json import SQLiteKVStorage
from nano_graphrag._utils import (
    generate_stable_entity_id,
    wrap_embedding_func_with_attrs,
)

os.environ["OPENAI_API_KEY"] = "FAKE"

WORKING_DIR = "./tests/nano_graphrag_cache_EXTRACTION_OPS"


@wrap_embedding_func_with_attrs(embedding_dim=384, max_token_size=8192)
async def local_embedding(texts: list[str]) -> np.ndarray:
    return np.random.rand(len(texts), 384)


async def no_op_model(prompt, **kwargs) -> str:
    return ""


def _clean():
    if os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR)
    os.makedirs(WORKING_DIR, exist_ok=True)


def _make_global_config():
    return {
        "working_dir": WORKING_DIR,
        "best_model_func": no_op_model,
        "cheap_model_func": no_op_model,
        "entity_summary_to_max_tokens": 500,
        "cheap_model_max_token_size": 32768,
        "extraction_max_async": 4,
        "entity_registry": EntityRegistry(),
    }


def _make_manifest(entities=None, relationships=None):
    return {
        "entities": entities or {},
        "relationships": relationships or {},
    }


def _entity_entry(name, etype="PERSON", descriptions=None, chunk_ids=None):
    return {
        "entity_name": _normalize_entity_name(name),
        "entity_type": etype,
        "descriptions": descriptions or [f"Description of {name}"],
        "source_chunk_ids": chunk_ids or ["chunk-0"],
    }


def _rel_entry(src_id, tgt_id, desc="related", weight=1.0, chunk_id="chunk-0"):
    return {
        "src_entity_id": src_id,
        "tgt_entity_id": tgt_id,
        "description": desc,
        "weight": weight,
        "source_chunk_ids": [chunk_id],
        "relation_type": "related",
        "order": 1,
    }


# --- extraction_rebuild unit tests ---


class TestIndexKeyFunctions:
    def test_entity_index_key(self):
        key = _entity_index_key("entity_abc123")
        assert key == "entity::entity_abc123"

    def test_entity_name_index_key(self):
        key = _entity_name_index_key("John Smith")
        assert key == "entity_name::JOHN SMITH"

    def test_relationship_index_key(self):
        key = _relationship_index_key("rel_abc123")
        assert key == "relationship::rel_abc123"

    def test_manifest_index_keys_empty(self):
        keys = _manifest_index_keys({})
        assert keys == set()

    def test_manifest_index_keys_with_entities_and_relationships(self):
        eid = generate_stable_entity_id("FOO", "PERSON")
        manifest = _make_manifest(
            entities={eid: _entity_entry("FOO")},
            relationships={"rel-1": _rel_entry(eid, "tgt-1")},
        )
        keys = _manifest_index_keys(manifest)
        assert _entity_index_key(eid) in keys
        assert _entity_name_index_key("FOO") in keys
        assert _relationship_index_key("rel-1") in keys


class TestAppendDocId:
    def test_appends_new_doc_id(self):
        entries = {}
        _append_doc_id(entries, "key-1", "doc-1")
        assert entries["key-1"]["doc_ids"] == ["doc-1"]

    def test_does_not_duplicate(self):
        entries = {"key-1": {"doc_ids": ["doc-1"]}}
        _append_doc_id(entries, "key-1", "doc-1")
        assert entries["key-1"]["doc_ids"] == ["doc-1"]

    def test_appends_different_doc(self):
        entries = {"key-1": {"doc_ids": ["doc-1"]}}
        _append_doc_id(entries, "key-1", "doc-2")
        assert entries["key-1"]["doc_ids"] == ["doc-1", "doc-2"]


class TestRebuildGraphContributionIndex:
    @pytest.mark.asyncio
    async def test_builds_index_from_documents(self):
        _clean()
        config = _make_global_config()
        doc_index = SQLiteKVStorage(namespace="test_docs", global_config=config)
        contrib_index = SQLiteKVStorage(namespace="test_contrib", global_config=config)

        eid = generate_stable_entity_id("FOO", "PERSON")
        manifest = _make_manifest(
            entities={eid: _entity_entry("FOO")},
        )
        await doc_index.upsert({"doc-1": manifest})
        await rebuild_graph_contribution_index(contrib_index, doc_index)

        meta = await contrib_index.get_by_id("__meta__")
        assert meta is not None
        assert meta["built"] is True

        entity_key = _entity_index_key(eid)
        entry = await contrib_index.get_by_id(entity_key)
        assert entry is not None
        assert "doc-1" in entry["doc_ids"]

        doc_index.close()
        contrib_index.close()


class TestUpdateGraphContributionIndex:
    @pytest.mark.asyncio
    async def test_adds_new_document_contributions(self):
        _clean()
        config = _make_global_config()
        contrib_index = SQLiteKVStorage(namespace="test_contrib", global_config=config)

        eid = generate_stable_entity_id("BAR", "PERSON")
        old_manifest = _make_manifest()
        new_manifest = _make_manifest(entities={eid: _entity_entry("BAR")})

        await update_graph_contribution_index_for_documents(
            contrib_index, {"doc-1": old_manifest}, {"doc-1": new_manifest}
        )

        entry = await contrib_index.get_by_id(_entity_index_key(eid))
        assert entry is not None
        assert "doc-1" in entry["doc_ids"]

        contrib_index.close()


class TestRebuildKnowledgeGraph:
    @pytest.mark.asyncio
    async def test_rebuild_with_empty_manifests_returns_graph(self):
        _clean()
        config = _make_global_config()
        doc_index = SQLiteKVStorage(namespace="test_docs", global_config=config)
        contrib_index = SQLiteKVStorage(namespace="test_contrib", global_config=config)
        graph = NetworkXStorage(namespace="test_graph", global_config=config)

        result = await rebuild_knowledge_graph_for_documents(
            doc_index, contrib_index, graph, None, None, config, {}, {}
        )
        assert result is graph

        doc_index.close()
        contrib_index.close()

    @pytest.mark.asyncio
    async def test_rebuild_with_no_affected_ids_returns_early(self):
        _clean()
        config = _make_global_config()
        doc_index = SQLiteKVStorage(namespace="test_docs", global_config=config)
        contrib_index = SQLiteKVStorage(namespace="test_contrib", global_config=config)
        graph = NetworkXStorage(namespace="test_graph", global_config=config)

        old_manifest = _make_manifest(entities={}, relationships={})
        new_manifest = _make_manifest(entities={}, relationships={})
        result = await rebuild_knowledge_graph_for_documents(
            doc_index, contrib_index, graph, None, None, config,
            {"doc-1": old_manifest}, {"doc-1": new_manifest},
        )
        assert result is graph

        doc_index.close()
        contrib_index.close()


# --- extraction_structured unit tests ---


class TestParseSingleResult:
    def test_parses_entity_extraction_output(self):
        result = EntityExtractionOutput(
            entities=[
                ExtractedEntity(entity_name="Alice", entity_type="PERSON", description="A person"),
                ExtractedEntity(entity_name="Bob", entity_type="PERSON", description="Another person"),
            ],
            relationships=[
                ExtractedRelationship(
                    source="Alice", target="Bob",
                    description="knows", weight=1.0,
                ),
            ],
        )
        entities, relationships = _parse_single_result(result, "chunk-0")

        assert len(entities) == 2
        assert len(relationships) == 1

        alice_name = _normalize_entity_name("Alice")
        bob_name = _normalize_entity_name("Bob")
        alice_id = generate_stable_entity_id(alice_name, "PERSON")
        bob_id = generate_stable_entity_id(bob_name, "PERSON")
        assert alice_id in entities
        assert bob_id in entities
        assert entities[alice_id]["entity_name"] == alice_name

    def test_parses_empty_output(self):
        result = EntityExtractionOutput(entities=[], relationships=[])
        entities, relationships = _parse_single_result(result, "chunk-0")
        assert entities == {}
        assert relationships == {}

    def test_skips_entities_with_empty_names(self):
        result = EntityExtractionOutput(
            entities=[
                ExtractedEntity(entity_name="", entity_type="PERSON", description="empty name"),
                ExtractedEntity(entity_name="Valid", entity_type="PERSON", description="valid entity"),
            ],
            relationships=[],
        )
        entities, relationships = _parse_single_result(result, "chunk-0")
        assert len(entities) == 1
        valid_name = _normalize_entity_name("Valid")
        valid_id = generate_stable_entity_id(valid_name, "PERSON")
        assert valid_id in entities

    def test_creates_implicit_entities_for_relationship_endpoints(self):
        result = EntityExtractionOutput(
            entities=[],
            relationships=[
                ExtractedRelationship(
                    source="X", target="Y",
                    description="X knows Y", weight=1.0,
                ),
            ],
        )
        entities, relationships = _parse_single_result(result, "chunk-0")
        assert len(entities) == 2
        assert len(relationships) == 1

    def test_merges_duplicate_entities_within_chunk(self):
        result = EntityExtractionOutput(
            entities=[
                ExtractedEntity(entity_name="Alice", entity_type="PERSON", description="desc1"),
                ExtractedEntity(entity_name="Alice", entity_type="PERSON", description="desc2"),
            ],
            relationships=[],
        )
        entities, relationships = _parse_single_result(result, "chunk-0")
        alice_name = _normalize_entity_name("Alice")
        alice_id = generate_stable_entity_id(alice_name, "PERSON")
        assert len(entities) == 1
        assert len(entities[alice_id]["descriptions"]) == 2

    def test_parses_json_string_input(self):
        json_str = json.dumps({
            "entities": [
                {"entity_name": "Foo", "entity_type": "ORG", "description": "An org"},
            ],
            "relationships": [],
        })
        entities, relationships = _parse_single_result(json_str, "chunk-0")
        assert len(entities) == 1
