import json
import os
import shutil

import numpy as np
import pytest

from nano_graphrag._entity_registry import EntityRegistry
from nano_graphrag._ops.extraction import (
    _compute_neighborhood_iou,
    _disambiguate_entity_link,
    _get_graph_neighbor_names,
    _get_manifest_neighbor_names,
    _resolve_manifest_entity_link,
)
from nano_graphrag._ops.extraction_common import (
    _merge_results_into_manifest,
    _normalize_entity_name,
)
from nano_graphrag._ops.extraction_prompts import (
    _build_extraction_system_prompt,
    _build_temporal_instructions,
)
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

pytestmark = pytest.mark.unit

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


class TestGetManifestNeighborNames:
    def test_finds_neighbors_in_relationships(self):
        manifest = _make_manifest(
            entities={
                "e1": _entity_entry("Apple"),
                "e2": _entity_entry("iPhone"),
                "e3": _entity_entry("Tim Cook"),
            },
            relationships={
                "r1": _rel_entry("e1", "e2"),
                "r2": _rel_entry("e1", "e3"),
                "r3": _rel_entry("e2", "e3"),
            },
        )
        neighbors = _get_manifest_neighbor_names("e1", manifest)
        assert "iphone" in neighbors
        assert "tim cook" in neighbors
        assert len(neighbors) == 2

    def test_returns_empty_for_isolated_entity(self):
        manifest = _make_manifest(
            entities={
                "e1": _entity_entry("Apple"),
                "e2": _entity_entry("iPhone"),
            },
            relationships={},
        )
        neighbors = _get_manifest_neighbor_names("e1", manifest)
        assert neighbors == set()

    def test_ignores_unknown_entity_ids(self):
        manifest = _make_manifest(
            entities={"e1": _entity_entry("Apple")},
            relationships={"r1": _rel_entry("e1", "e_unknown")},
        )
        neighbors = _get_manifest_neighbor_names("e1", manifest)
        assert neighbors == set()


class TestComputeNeighborhoodIoU:
    def test_perfect_overlap(self):
        iou, common = _compute_neighborhood_iou({"a", "b"}, {"a", "b"})
        assert iou == 1.0
        assert common == {"a", "b"}

    def test_partial_overlap(self):
        iou, common = _compute_neighborhood_iou({"a", "b", "c"}, {"b", "c", "d"})
        assert abs(iou - 0.5) < 1e-9
        assert common == {"b", "c"}

    def test_no_overlap(self):
        iou, common = _compute_neighborhood_iou({"a", "b"}, {"c", "d"})
        assert iou == 0.0
        assert common == set()

    def test_empty_sets(self):
        iou, common = _compute_neighborhood_iou(set(), set())
        assert iou == 0.0
        assert common == set()

    def test_one_empty(self):
        iou, common = _compute_neighborhood_iou({"a"}, set())
        assert iou == 0.0
        assert common == set()


class TestGetGraphNeighborNames:
    async def test_returns_neighbor_names_from_registry(self):
        _clean()
        config = _make_global_config()
        graph = NetworkXStorage(namespace="test_graph", global_config=config)

        registry = EntityRegistry()
        registry.register_entity("e_apple", "Apple", entity_type="Organization")
        registry.register_entity("e_iphone", "iPhone", entity_type="Product")
        registry.register_entity("e_tim", "Tim Cook", entity_type="Person")

        await graph.upsert_node("e_apple", {"entity_name": "Apple", "entity_type": "Organization"})
        await graph.upsert_node("e_iphone", {"entity_name": "iPhone", "entity_type": "Product"})
        await graph.upsert_node("e_tim", {"entity_name": "Tim Cook", "entity_type": "Person"})
        await graph.upsert_edge("e_apple", "e_iphone", {"weight": 1.0})
        await graph.upsert_edge("e_apple", "e_tim", {"weight": 1.0})

        neighbors = await _get_graph_neighbor_names("e_apple", graph, registry)
        assert "iphone" in neighbors
        assert "tim cook" in neighbors
        assert "apple" not in neighbors

    async def test_returns_empty_for_isolated_node(self):
        _clean()
        config = _make_global_config()
        graph = NetworkXStorage(namespace="test_graph", global_config=config)
        registry = EntityRegistry()
        registry.register_entity("e1", "Foo")

        await graph.upsert_node("e1", {"entity_name": "Foo"})

        neighbors = await _get_graph_neighbor_names("e1", graph, registry)
        assert neighbors == set()


class TestResolveManifestEntityLinkWithNeighborhood:
    async def test_auto_links_single_candidate_with_high_iou(self):
        registry = EntityRegistry()
        registry.register_entity("e_apple_tech", "Apple", entity_type="Organization")
        registry.register_entity("e_iphone", "iPhone", entity_type="Product")
        registry.register_entity("e_tim", "Tim Cook", entity_type="Person")

        _clean()
        config = _make_global_config()
        config["entity_linking_iou_threshold"] = 0.3
        config["entity_linking_min_common_neighbors"] = 2
        config["entity_linking_similarity_threshold"] = 1.0

        graph = NetworkXStorage(namespace="test_graph", global_config=config)
        await graph.upsert_node("e_apple_tech", {"entity_name": "Apple", "entity_type": "Organization"})
        await graph.upsert_node("e_iphone", {"entity_name": "iPhone"})
        await graph.upsert_node("e_tim", {"entity_name": "Tim Cook"})
        await graph.upsert_edge("e_apple_tech", "e_iphone", {"weight": 1.0})
        await graph.upsert_edge("e_apple_tech", "e_tim", {"weight": 1.0})

        manifest_neighbors = {"iphone", "tim cook"}

        entity = {"entity_name": "Apple", "entity_type": "Organization"}

        result = await _resolve_manifest_entity_link(
            entity,
            registry,
            config,
            manifest_neighbors=manifest_neighbors,
            knowledge_graph_inst=graph,
        )
        assert result == "e_apple_tech"

    async def test_does_not_auto_link_with_no_neighbor_overlap(self):
        registry = EntityRegistry()
        registry.register_entity("e_apple_music", "Apple", entity_type="Organization")
        registry.register_entity("e_beatles", "Beatles", entity_type="Organization")

        _clean()
        config = _make_global_config()
        config["entity_linking_iou_threshold"] = 0.3
        config["entity_linking_min_common_neighbors"] = 2
        config["entity_linking_similarity_threshold"] = 0.6
        config["enable_entity_linking"] = False

        graph = NetworkXStorage(namespace="test_graph", global_config=config)
        await graph.upsert_node("e_apple_music", {"entity_name": "Apple"})
        await graph.upsert_node("e_beatles", {"entity_name": "Beatles"})
        await graph.upsert_edge("e_apple_music", "e_beatles", {"weight": 1.0})

        entity = {"entity_name": "Apple Computer", "entity_type": "Organization"}
        manifest_neighbors = {"iphone", "tim cook"}

        result = await _resolve_manifest_entity_link(
            entity,
            registry,
            config,
            manifest_neighbors=manifest_neighbors,
            knowledge_graph_inst=graph,
        )
        assert result is None

    async def test_returns_none_without_graph_storage(self):
        registry = EntityRegistry()
        config = _make_global_config()

        entity = {"entity_name": "Unknown", "entity_type": "Organization"}

        result = await _resolve_manifest_entity_link(
            entity, registry, config,
            manifest_neighbors={"foo"},
            knowledge_graph_inst=None,
        )
        assert result is None


class TestDisambiguateWithNeighborhoodEvidence:
    async def test_prompt_includes_neighborhood_evidence(self):
        registry = EntityRegistry()
        registry.register_entity("e_apple_tech", "Apple", entity_type="Organization")
        registry.register_entity("e_apple_music", "Apple Records", entity_type="Organization")
        registry.add_aliases("e_apple_music", ["Apple"])

        _clean()
        config = _make_global_config()

        graph = NetworkXStorage(namespace="test_graph", global_config=config)
        await graph.upsert_node("e_apple_tech", {"entity_name": "Apple"})
        await graph.upsert_node("e_iphone", {"entity_name": "iPhone"})
        await graph.upsert_edge("e_apple_tech", "e_iphone", {"weight": 1.0})

        captured_prompt = None

        async def capture_llm(prompt, **kwargs):
            nonlocal captured_prompt
            captured_prompt = prompt
            return '{"decision": "existing", "entity_id": "e_apple_tech"}'

        config["cheap_model_func"] = capture_llm
        config["enable_entity_linking"] = True
        config["entity_linking_similarity_threshold"] = 0.5

        entity = {"entity_name": "Apple", "entity_type": "Organization", "descriptions": ["Tech company"]}
        candidates = [("e_apple_tech", 0.95), ("e_apple_music", 0.90)]

        result = await _disambiguate_entity_link(
            entity, candidates, registry, config,
            manifest_neighbors={"iphone", "tim cook"},
            knowledge_graph_inst=graph,
        )

        assert result == "e_apple_tech"
        assert captured_prompt is not None
        assert "Structural Evidence" in captured_prompt
        assert "iphone" in captured_prompt.lower()
        assert "IoU" in captured_prompt


# --- extraction_common shared helper tests ---


class TestMergeResultsIntoManifest:
    def test_merges_single_result(self):
        eid = generate_stable_entity_id("Alice", "PERSON")
        results = [
            (
                {eid: {"entity_name": "Alice", "entity_type": "PERSON", "descriptions": ["desc"], "source_chunk_ids": ["c0"]}},
                {},
            )
        ]
        manifest = _merge_results_into_manifest(results, ["c0"])
        assert eid in manifest["entities"]
        assert manifest["entities"][eid]["entity_name"] == "Alice"

    def test_merges_duplicate_entity_across_chunks(self):
        eid = generate_stable_entity_id("Alice", "PERSON")
        results = [
            (
                {eid: {"entity_name": "Alice", "entity_type": "PERSON", "descriptions": ["desc1"], "source_chunk_ids": ["c0"], "aliases": []}},
                {},
            ),
            (
                {eid: {"entity_name": "Alice", "entity_type": "PERSON", "descriptions": ["desc2"], "source_chunk_ids": ["c1"], "aliases": ["Ali"]}},
                {},
            ),
        ]
        manifest = _merge_results_into_manifest(results, ["c0", "c1"])
        entity = manifest["entities"][eid]
        assert entity["descriptions"] == ["desc1", "desc2"]
        assert entity["source_chunk_ids"] == ["c0", "c1"]
        assert "Ali" in entity["aliases"]

    def test_merges_relationships_and_accumulates_weight(self):
        results = [
            (
                {},
                {"r1": {"src_entity_id": "e1", "tgt_entity_id": "e2", "relation_type": "related", "descriptions": ["d1"], "weight": 1.0, "source_chunk_ids": ["c0"]}},
            ),
            (
                {},
                {"r1": {"src_entity_id": "e1", "tgt_entity_id": "e2", "relation_type": "related", "descriptions": ["d2"], "weight": 2.0, "source_chunk_ids": ["c1"]}},
            ),
        ]
        manifest = _merge_results_into_manifest(results, ["c0", "c1"])
        assert manifest["relationships"]["r1"]["weight"] == 3.0
        assert manifest["relationships"]["r1"]["descriptions"] == ["d1", "d2"]

    def test_empty_results(self):
        manifest = _merge_results_into_manifest([], [])
        assert manifest["entities"] == {}
        assert manifest["relationships"] == {}

    def test_preserves_temporal_fields(self):
        results = [
            (
                {},
                {"r1": {"src_entity_id": "e1", "tgt_entity_id": "e2", "relation_type": "related", "descriptions": ["d1"], "weight": 1.0, "source_chunk_ids": ["c0"], "temporal_context": "since 2020", "valid_from": "2020", "valid_to": None}},
            ),
        ]
        manifest = _merge_results_into_manifest(results, ["c0"])
        rel = manifest["relationships"]["r1"]
        assert rel["temporal_context"] == "since 2020"
        assert rel["valid_from"] == "2020"

    def test_chunk_ids_preserved(self):
        manifest = _merge_results_into_manifest([], ["c0", "c1", "c2"])
        assert manifest["chunk_ids"] == ["c0", "c1", "c2"]


class TestBuildExtractionSystemPrompt:
    def test_fast_mode_single_is_short(self):
        config = {"entity_extraction_quality": "fast", "enable_temporal_extraction": False}
        prompt = _build_extraction_system_prompt(["PERSON", "ORG"], config)
        assert "Extract entities and relationships." in prompt
        assert "JSON" in prompt
        assert "assistant" not in prompt

    def test_fast_mode_batched_has_chunks(self):
        config = {"entity_extraction_quality": "fast", "enable_temporal_extraction": False}
        prompt = _build_extraction_system_prompt(["PERSON"], config, batched=True)
        assert "chunks" in prompt
        assert "chunk_id" in prompt

    def test_balanced_mode_has_assistant(self):
        config = {"entity_extraction_quality": "balanced", "enable_temporal_extraction": False}
        prompt = _build_extraction_system_prompt(["PERSON"], config)
        assert "entity extraction assistant" in prompt

    def test_balanced_batched_preserves_chunk_id(self):
        config = {"entity_extraction_quality": "balanced", "enable_temporal_extraction": False}
        prompt = _build_extraction_system_prompt(["PERSON"], config, batched=True)
        assert "chunk_id" in prompt

    def test_temporal_instructions_in_balanced(self):
        config = {"entity_extraction_quality": "balanced", "enable_temporal_extraction": True}
        prompt = _build_extraction_system_prompt(["PERSON"], config)
        assert "event_date" in prompt
        assert "temporal_context" in prompt

    def test_no_temporal_when_disabled(self):
        config = {"entity_extraction_quality": "balanced", "enable_temporal_extraction": False}
        prompt = _build_extraction_system_prompt(["PERSON"], config)
        assert "event_date" not in prompt


class TestBuildTemporalInstructions:
    def test_with_temporal_enabled(self):
        config = {"enable_temporal_extraction": True}
        instructions = _build_temporal_instructions(config)
        assert "event_date" in instructions
        assert "temporal_context" in instructions
        assert "valid_from" in instructions

    def test_without_temporal(self):
        config = {"enable_temporal_extraction": False}
        instructions = _build_temporal_instructions(config)
        assert "aliases" in instructions
        assert "event_date" not in instructions
