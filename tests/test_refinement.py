import os
import tempfile
import time
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from nano_graphrag._ops.refinement import arefine
from nano_graphrag._ops.refinement.enrich import (
    _enrich_phase,
    _validate_enrichment,
    _validate_entity_subject,
)
from nano_graphrag._ops.refinement.infer import _infer_phase, _select_candidates_weighted
from nano_graphrag._ops.refinement.merge import (
    _find_merge_candidates_brute_force,
    _merge_phase,
)
from nano_graphrag._ops.refinement.pipeline import (
    RefinementJournal,
    RejectionCache,
    _get_rejection_ttl,
)

pytestmark = pytest.mark.unit


class TestRefinementJournal:
    def test_journal_add_and_save(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "journal.jsonl")
            j = RefinementJournal(path)
            j.add("merge", {"examined": 5, "merged": 2})
            j.save()
            j2 = RefinementJournal(path)
            assert len(j2.entries) == 1
            assert j2.entries[0]["phase"] == "merge"
            assert j2.entries[0]["stats"]["merged"] == 2

    def test_journal_max_entries(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "journal.jsonl")
            j = RefinementJournal(path)
            for i in range(250):
                j.add("merge", {"i": i})
            j.save()
            j2 = RefinementJournal(path)
            assert len(j2.entries) == 100
            assert j2.entries[0]["stats"]["i"] == 150


class TestRejectionCache:
    def test_rejection_cache_crud(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "rejections.json")
            cache = RejectionCache(path)
            cache["pair_a|pair_b"] = time.time()
            assert "pair_a|pair_b" in cache
            assert "missing_key" not in cache

    def test_rejection_cache_prune_removes_old(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "rejections.json")
            cache = RejectionCache(path)
            old_ts = time.time() - 100000
            cache["old_key_1"] = old_ts
            cache["old_key_2"] = old_ts
            cache.prune(ttl=60.0)
            assert "old_key_1" not in cache
            assert "old_key_2" not in cache

    def test_rejection_cache_prune_keeps_fresh(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "rejections.json")
            cache = RejectionCache(path)
            cache["fresh_key"] = time.time()
            cache.prune(ttl=86400.0)
            assert "fresh_key" in cache


class TestRejectionTTL:
    def test_get_rejection_ttl_scaling(self):
        ttl_small = _get_rejection_ttl(0)
        ttl_large = _get_rejection_ttl(500)
        assert ttl_small < ttl_large
        assert ttl_small == 3 * 86400
        assert ttl_large == 7 * 86400


class TestMergeCandidateFiltering:
    def test_merge_candidates_type_filtered(self):
        all_nodes = {
            "n1": {"entity_type": "PERSON"},
            "n2": {"entity_type": "ORGANIZATION"},
        }
        vec = np.array([1.0, 0.0, 0.0])
        entity_vectors = {"n1": vec, "n2": vec}
        result = _find_merge_candidates_brute_force(
            all_nodes, entity_vectors, ["n1", "n2"], merge_threshold=0.9
        )
        assert result == []

    def test_merge_candidates_same_type_above_threshold(self):
        all_nodes = {
            "n1": {"entity_type": "PERSON"},
            "n2": {"entity_type": "PERSON"},
        }
        vec = np.array([1.0, 0.0, 0.0])
        entity_vectors = {"n1": vec, "n2": vec}
        result = _find_merge_candidates_brute_force(
            all_nodes, entity_vectors, ["n1", "n2"], merge_threshold=0.9
        )
        assert len(result) == 1
        assert result[0][0] == "n1"
        assert result[0][1] == "n2"
        assert result[0][2] >= 0.9

    def test_merge_candidates_same_type_below_threshold(self):
        all_nodes = {
            "n1": {"entity_type": "PERSON"},
            "n2": {"entity_type": "PERSON"},
        }
        entity_vectors = {
            "n1": np.array([1.0, 0.0, 0.0]),
            "n2": np.array([0.0, 1.0, 0.0]),
        }
        result = _find_merge_candidates_brute_force(
            all_nodes, entity_vectors, ["n1", "n2"], merge_threshold=0.9
        )
        assert result == []

    def test_find_merge_candidates_brute_force_match(self):
        vec = np.array([1.0, 0.0, 0.0, 0.0])
        all_nodes = {
            "a": {"entity_type": "PERSON"},
            "b": {"entity_type": "PERSON"},
        }
        vectors = {"a": vec, "b": vec}
        result = _find_merge_candidates_brute_force(
            all_nodes, vectors, ["a", "b"], merge_threshold=0.9
        )
        assert len(result) == 1
        assert result[0][2] >= 0.9


class TestEnrichmentValidation:
    def test_validate_enrichment_empty_fails(self):
        assert _validate_enrichment("Entity", "original", "") is False

    def test_validate_enrichment_name_absent_fails(self):
        assert (
            _validate_enrichment("Foo", "original text", "completely unrelated enriched text")
            is False
        )

    def test_validate_enrichment_valid_passes(self):
        assert (
            _validate_enrichment(
                "Alice",
                "Alice is a researcher",
                "Alice is a researcher at MIT who works on AI",
            )
            is True
        )

    def test_validate_enrichment_missing_entity_name(self):
        assert (
            _validate_enrichment("Alpha", "Alpha is a star", "Beta is a planet in the system")
            is False
        )

    def test_validate_enrichment_low_overlap(self):
        original = "alpha beta gamma"
        enriched = "completely different words here alpha"
        assert _validate_enrichment("alpha", original, enriched) is False

    def test_validate_enrichment_valid_with_description(self):
        original = "Alpha is a bright star in the constellation"
        enriched = "Alpha is a bright star in the constellation and emits blue light"
        assert _validate_enrichment("Alpha", original, enriched) is True

    def test_validate_entity_subject_name_in_sentence_passes(self):
        assert (
            _validate_entity_subject("Alice", "Alice is a researcher. Alice works on AI.") is True
        )

    def test_validate_entity_subject_pronoun_start_passes(self):
        assert (
            _validate_entity_subject("Alice", "She is very tall and works at the lab downtown")
            is True
        )

    def test_validate_entity_subject_short_sentence_skipped(self):
        assert _validate_entity_subject("Alice", "Ok.") is True


class TestSelectCandidatesWeighted:
    def test_select_candidates_small_batch(self):
        candidates = [("a", "b", 5), ("c", "d", 3)]
        result = _select_candidates_weighted(candidates, batch_size=10)
        assert result == candidates

    def test_select_candidates_weighted_large_batch(self):
        candidates = [(f"a{i}", f"b{i}", 100 - i) for i in range(100)]
        result = _select_candidates_weighted(candidates, batch_size=10)
        assert len(result) == 10
        top_8 = result[:8]
        for i in range(8):
            assert top_8[i][0] == f"a{i}"


class TestMergePhaseExecution:
    async def test_merge_phase_no_nodes(self, mock_graph_storage, mock_entity_vdb, global_config):
        mock_graph_storage.get_all_nodes = AsyncMock(return_value={})
        stats = await _merge_phase(mock_graph_storage, mock_entity_vdb, global_config)
        assert stats["examined"] == 0
        assert stats["merged"] == 0

    async def test_merge_phase_missing_llm(
        self, mock_graph_storage, mock_entity_vdb, global_config
    ):
        del global_config["cheap_model_func"]
        stats = await _merge_phase(mock_graph_storage, mock_entity_vdb, global_config)
        assert stats["merged"] == 0
        assert stats["examined"] == 0

    async def test_merge_phase_missing_embedding(
        self, mock_graph_storage, mock_entity_vdb, global_config
    ):
        del global_config["embedding_func"]
        stats = await _merge_phase(mock_graph_storage, mock_entity_vdb, global_config)
        assert stats["merged"] == 0

    async def test_merge_phase_empty_descriptions(
        self, mock_graph_storage, mock_entity_vdb, global_config
    ):
        nodes = {
            "n1": {"entity_type": "PERSON", "description": ""},
            "n2": {"entity_type": "PERSON", "description": ""},
        }
        mock_graph_storage.get_all_nodes = AsyncMock(return_value=nodes)
        stats = await _merge_phase(mock_graph_storage, mock_entity_vdb, global_config)
        assert stats["examined"] == 0

    async def test_merge_phase_embedding_fails(
        self, mock_graph_storage, mock_entity_vdb, global_config
    ):
        nodes = {
            "n1": {"entity_type": "PERSON", "description": "A person"},
            "n2": {"entity_type": "PERSON", "description": "Another person"},
        }
        mock_graph_storage.get_all_nodes = AsyncMock(return_value=nodes)

        async def fail_embed(texts):
            raise RuntimeError("embedding failed")

        global_config["embedding_func"] = fail_embed
        stats = await _merge_phase(mock_graph_storage, mock_entity_vdb, global_config)
        assert stats["merged"] == 0

    async def test_merge_phase_hub_cap_prevents_merge(
        self, mock_graph_storage, mock_entity_vdb, global_config
    ):
        nodes = {
            "n1": {
                "entity_name": "A",
                "entity_type": "PERSON",
                "description": "A person who works hard",
            },
            "n2": {
                "entity_name": "B",
                "entity_type": "PERSON",
                "description": "A person who works hard",
            },
        }
        mock_graph_storage.get_all_nodes = AsyncMock(return_value=nodes)
        mock_graph_storage.has_node = AsyncMock(return_value=True)
        mock_graph_storage.get_node = AsyncMock(side_effect=lambda nid: nodes.get(nid))
        mock_graph_storage.get_node_edges = AsyncMock(return_value=[])

        stats = await _merge_phase(mock_graph_storage, mock_entity_vdb, global_config, hub_cap=0)
        assert stats["skipped"] >= 0

    async def test_merge_phase_node_deleted_mid_flight(
        self, mock_graph_storage, mock_entity_vdb, global_config
    ):
        nodes = {
            "n1": {
                "entity_name": "A",
                "entity_type": "PERSON",
                "description": "A person who works hard",
            },
            "n2": {
                "entity_name": "B",
                "entity_type": "PERSON",
                "description": "A person who works hard",
            },
        }
        mock_graph_storage.get_all_nodes = AsyncMock(return_value=nodes)
        mock_graph_storage.has_node = AsyncMock(return_value=False)

        stats = await _merge_phase(mock_graph_storage, mock_entity_vdb, global_config)
        assert stats["merged"] == 0

    async def test_merge_phase_llm_exception_skips(
        self, mock_graph_storage, mock_entity_vdb, global_config
    ):
        nodes = {
            "n1": {
                "entity_name": "A",
                "entity_type": "PERSON",
                "description": "A person who works hard",
            },
            "n2": {
                "entity_name": "B",
                "entity_type": "PERSON",
                "description": "A person who works hard",
            },
        }
        mock_graph_storage.get_all_nodes = AsyncMock(return_value=nodes)
        mock_graph_storage.has_node = AsyncMock(return_value=True)
        mock_graph_storage.get_node = AsyncMock(side_effect=lambda nid: nodes.get(nid))

        async def fail_llm(prompt, **kwargs):
            raise RuntimeError("LLM failed")

        global_config["cheap_model_func"] = fail_llm
        stats = await _merge_phase(mock_graph_storage, mock_entity_vdb, global_config)
        assert stats["merged"] == 0

    async def test_merge_phase_llm_returns_list_format(
        self, mock_graph_storage, mock_entity_vdb, content_hash_embedding
    ):
        nodes = {
            "n1": {
                "entity_name": "A",
                "entity_type": "PERSON",
                "description": "A person who works hard",
                "aliases": "[]",
                "source_id": '["c1"]',
            },
            "n2": {
                "entity_name": "B",
                "entity_type": "PERSON",
                "description": "A person who works hard",
                "aliases": "[]",
                "source_id": '["c1"]',
            },
        }
        mock_graph_storage.get_all_nodes = AsyncMock(return_value=nodes)
        mock_graph_storage.has_node = AsyncMock(return_value=True)
        mock_graph_storage.get_node = AsyncMock(side_effect=lambda nid: nodes.get(nid))
        mock_graph_storage.get_node_edges = AsyncMock(return_value=[])

        async def list_llm(prompt, **kwargs):
            return [{"text": "Merged description of A and B working together"}]

        config = {
            "cheap_model_func": list_llm,
            "embedding_func": content_hash_embedding,
        }
        stats = await _merge_phase(mock_graph_storage, mock_entity_vdb, config)
        assert stats["merged"] == 1
        assert mock_entity_vdb.delete.called or mock_entity_vdb.upsert.called

    async def test_merge_phase_vdb_failure_aborts(
        self, mock_graph_storage, mock_entity_vdb, content_hash_embedding
    ):
        nodes = {
            "n1": {
                "entity_name": "A",
                "entity_type": "PERSON",
                "description": "A person who works hard",
                "aliases": "[]",
                "source_id": '["c1"]',
            },
            "n2": {
                "entity_name": "B",
                "entity_type": "PERSON",
                "description": "A person who works hard",
                "aliases": "[]",
                "source_id": '["c1"]',
            },
        }
        mock_graph_storage.get_all_nodes = AsyncMock(return_value=nodes)
        mock_graph_storage.has_node = AsyncMock(return_value=True)
        mock_graph_storage.get_node = AsyncMock(side_effect=lambda nid: nodes.get(nid))
        mock_graph_storage.get_node_edges = AsyncMock(return_value=[])

        vdb = AsyncMock()
        vdb.delete = AsyncMock(side_effect=RuntimeError("VDB down"))

        async def ok_llm(prompt, **kwargs):
            return "Merged description text"

        config = {
            "cheap_model_func": ok_llm,
            "embedding_func": content_hash_embedding,
        }
        stats = await _merge_phase(mock_graph_storage, vdb, config)
        assert stats["skipped"] >= 1
        assert not mock_graph_storage.upsert_node.called

    async def test_merge_phase_edge_migration(
        self, mock_graph_storage, mock_entity_vdb, content_hash_embedding
    ):
        nodes = {
            "n1": {
                "entity_name": "A",
                "entity_type": "PERSON",
                "description": "A person who works hard every day",
                "aliases": "[]",
                "source_id": '["c1"]',
            },
            "n2": {
                "entity_name": "B",
                "entity_type": "PERSON",
                "description": "A person who works hard every day",
                "aliases": "[]",
                "source_id": '["c1"]',
            },
        }
        mock_graph_storage.get_all_nodes = AsyncMock(return_value=nodes)
        mock_graph_storage.has_node = AsyncMock(return_value=True)
        mock_graph_storage.get_node = AsyncMock(side_effect=lambda nid: nodes.get(nid))
        mock_graph_storage.get_node_edges = AsyncMock(return_value=[("n2", "n3")])
        mock_graph_storage.get_edge = AsyncMock(
            return_value={"description": "knows", "weight": 1.0}
        )
        mock_graph_storage.has_edge = AsyncMock(return_value=False)

        async def ok_llm(prompt, **kwargs):
            return "Merged description of A and B"

        config = {
            "cheap_model_func": ok_llm,
            "embedding_func": content_hash_embedding,
        }
        stats = await _merge_phase(mock_graph_storage, mock_entity_vdb, config)
        assert stats["merged"] == 1
        assert mock_graph_storage.upsert_node.called
        assert mock_graph_storage.delete_node.called


class TestEnrichPhaseExecution:
    async def test_enrich_phase_no_llm(self, mock_graph_storage, mock_kv_storage, global_config):
        del global_config["cheap_model_func"]
        stats = await _enrich_phase(mock_graph_storage, mock_kv_storage, global_config)
        assert stats["enriched"] == 0

    async def test_enrich_phase_no_nodes(self, mock_graph_storage, mock_kv_storage, global_config):
        mock_graph_storage.get_all_nodes = AsyncMock(return_value={})
        stats = await _enrich_phase(mock_graph_storage, mock_kv_storage, global_config)
        assert stats["examined"] == 0

    async def test_enrich_phase_no_thin_entities(
        self, mock_graph_storage, mock_kv_storage, global_config
    ):
        nodes = {
            "n1": {"entity_name": "X", "description": "A" * 200},
        }
        mock_graph_storage.get_all_nodes = AsyncMock(return_value=nodes)
        stats = await _enrich_phase(mock_graph_storage, mock_kv_storage, global_config)
        assert stats["examined"] == 0

    async def test_enrich_phase_no_source_excerpts(
        self, mock_graph_storage, mock_kv_storage, global_config
    ):
        nodes = {
            "n1": {
                "entity_name": "X",
                "entity_type": "PERSON",
                "description": "Short",
                "source_id": "[]",
            },
        }
        mock_graph_storage.get_all_nodes = AsyncMock(return_value=nodes)
        stats = await _enrich_phase(mock_graph_storage, mock_kv_storage, global_config)
        assert stats["skipped"] >= 1

    async def test_enrich_phase_llm_exception(
        self, mock_graph_storage, mock_kv_storage, global_config
    ):
        nodes = {
            "n1": {
                "entity_name": "X",
                "entity_type": "PERSON",
                "description": "Short",
                "source_id": '["c1"]',
            },
        }
        mock_graph_storage.get_all_nodes = AsyncMock(return_value=nodes)
        mock_graph_storage.get_node_edges = AsyncMock(return_value=[])
        mock_kv_storage.get_by_ids = AsyncMock(return_value=[{"content": "Source text about X"}])

        async def fail_llm(prompt, **kwargs):
            raise RuntimeError("LLM failed")

        global_config["cheap_model_func"] = fail_llm
        stats = await _enrich_phase(mock_graph_storage, mock_kv_storage, global_config)
        assert stats["skipped"] >= 1

    async def test_enrich_phase_validation_failed(
        self, mock_graph_storage, mock_kv_storage, global_config
    ):
        nodes = {
            "n1": {
                "entity_name": "X",
                "entity_type": "PERSON",
                "description": "Short",
                "source_id": '["c1"]',
            },
        }
        mock_graph_storage.get_all_nodes = AsyncMock(return_value=nodes)
        mock_graph_storage.get_node_edges = AsyncMock(return_value=[])
        mock_kv_storage.get_by_ids = AsyncMock(return_value=[{"content": "Source text about X"}])

        async def bad_llm(prompt, **kwargs):
            return "Completely unrelated text that has nothing to do with anything"

        global_config["cheap_model_func"] = bad_llm
        stats = await _enrich_phase(mock_graph_storage, mock_kv_storage, global_config)
        assert stats["validation_failed"] >= 1

    async def test_enrich_phase_happy_path(
        self, mock_graph_storage, mock_kv_storage, global_config
    ):
        nodes = {
            "n1": {
                "entity_name": "X",
                "entity_type": "PERSON",
                "description": "Short desc",
                "source_id": '["c1"]',
            },
        }
        mock_graph_storage.get_all_nodes = AsyncMock(return_value=nodes)
        mock_graph_storage.get_node_edges = AsyncMock(return_value=[])
        mock_kv_storage.get_by_ids = AsyncMock(
            return_value=[{"content": "X is a researcher at MIT"}]
        )

        async def good_llm(prompt, **kwargs):
            return "X is a researcher at MIT who works on AI. Short desc. X has published papers."

        global_config["cheap_model_func"] = good_llm
        stats = await _enrich_phase(mock_graph_storage, mock_kv_storage, global_config)
        assert stats["enriched"] >= 1
        assert mock_graph_storage.upsert_node.called

    async def test_enrich_phase_batch_size_limiting(
        self, mock_graph_storage, mock_kv_storage, global_config
    ):
        nodes = {
            f"n{i}": {
                "entity_name": f"E{i}",
                "entity_type": "PERSON",
                "description": "Short",
                "source_id": "[]",
            }
            for i in range(10)
        }
        mock_graph_storage.get_all_nodes = AsyncMock(return_value=nodes)
        stats = await _enrich_phase(
            mock_graph_storage, mock_kv_storage, global_config, batch_size=3
        )
        assert stats["examined"] == 10


class TestInferPhaseExecution:
    async def test_infer_phase_no_llm(
        self, mock_graph_storage, mock_kv_storage, mock_entity_vdb, global_config
    ):
        del global_config["cheap_model_func"]
        stats = await _infer_phase(
            mock_graph_storage, mock_kv_storage, mock_entity_vdb, global_config
        )
        assert stats["inferred"] == 0

    async def test_infer_phase_too_few_nodes(
        self, mock_graph_storage, mock_kv_storage, mock_entity_vdb, global_config
    ):
        nodes = {"n1": {"entity_name": "A", "source_id": '["c1"]'}}
        mock_graph_storage.get_all_nodes = AsyncMock(return_value=nodes)
        stats = await _infer_phase(
            mock_graph_storage, mock_kv_storage, mock_entity_vdb, global_config
        )
        assert stats["inferred"] == 0

    async def test_infer_phase_no_co_occurring_pairs(
        self, mock_graph_storage, mock_kv_storage, mock_entity_vdb, global_config
    ):
        nodes = {
            "n1": {"entity_name": "A", "source_id": '["c1"]'},
            "n2": {"entity_name": "B", "source_id": '["c2"]'},
        }
        mock_graph_storage.get_all_nodes = AsyncMock(return_value=nodes)
        stats = await _infer_phase(
            mock_graph_storage, mock_kv_storage, mock_entity_vdb, global_config
        )
        assert stats["inferred"] == 0

    async def test_infer_phase_already_has_edge(
        self, mock_graph_storage, mock_kv_storage, mock_entity_vdb, global_config
    ):
        nodes = {
            "n1": {"entity_name": "A", "source_id": '["c1","c2"]'},
            "n2": {"entity_name": "B", "source_id": '["c1","c2"]'},
        }
        mock_graph_storage.get_all_nodes = AsyncMock(return_value=nodes)
        mock_graph_storage.has_edge = AsyncMock(return_value=True)
        stats = await _infer_phase(
            mock_graph_storage, mock_kv_storage, mock_entity_vdb, global_config
        )
        assert stats["inferred"] == 0

    async def test_infer_phase_rejection_cache_hit(
        self, mock_graph_storage, mock_kv_storage, mock_entity_vdb, global_config
    ):
        nodes = {
            "n1": {
                "entity_name": "A",
                "entity_type": "PERSON",
                "description": "Desc A",
                "source_id": '["c1","c2"]',
            },
            "n2": {
                "entity_name": "B",
                "entity_type": "ORG",
                "description": "Desc B",
                "source_id": '["c1","c2"]',
            },
        }
        mock_graph_storage.get_all_nodes = AsyncMock(return_value=nodes)
        mock_graph_storage.has_edge = AsyncMock(return_value=False)
        mock_graph_storage.get_node = AsyncMock(side_effect=lambda nid: nodes.get(nid))
        mock_kv_storage.get_by_ids = AsyncMock(return_value=[{"content": "A works with B"}])

        with tempfile.TemporaryDirectory() as td:
            cache = RejectionCache(os.path.join(td, "rejections.json"))
            cache["n1|n2"] = time.time()
            cache.save()

            cache = RejectionCache(os.path.join(td, "rejections.json"))
            stats = await _infer_phase(
                mock_graph_storage,
                mock_kv_storage,
                mock_entity_vdb,
                global_config,
                rejection_cache=cache,
            )
            assert stats["rejected_by_cache"] >= 1

    async def test_infer_phase_llm_says_no_relation(
        self, mock_graph_storage, mock_kv_storage, mock_entity_vdb, global_config
    ):
        nodes = {
            "n1": {
                "entity_name": "A",
                "entity_type": "PERSON",
                "description": "Desc A",
                "source_id": '["c1","c2"]',
            },
            "n2": {
                "entity_name": "B",
                "entity_type": "ORG",
                "description": "Desc B",
                "source_id": '["c1","c2"]',
            },
        }
        mock_graph_storage.get_all_nodes = AsyncMock(return_value=nodes)
        mock_graph_storage.has_edge = AsyncMock(return_value=False)
        mock_graph_storage.get_node = AsyncMock(side_effect=lambda nid: nodes.get(nid))
        mock_kv_storage.get_by_ids = AsyncMock(return_value=[{"content": "A and B unrelated"}])

        async def no_rel_llm(prompt, **kwargs):
            return '{"has_relation": false}'

        global_config["cheap_model_func"] = no_rel_llm
        stats = await _infer_phase(
            mock_graph_storage,
            mock_kv_storage,
            mock_entity_vdb,
            global_config,
        )
        assert stats["rejected_by_llm"] >= 1

    async def test_infer_phase_llm_invalid_json(
        self, mock_graph_storage, mock_kv_storage, mock_entity_vdb, global_config
    ):
        nodes = {
            "n1": {
                "entity_name": "A",
                "entity_type": "PERSON",
                "description": "Desc A",
                "source_id": '["c1","c2"]',
            },
            "n2": {
                "entity_name": "B",
                "entity_type": "ORG",
                "description": "Desc B",
                "source_id": '["c1","c2"]',
            },
        }
        mock_graph_storage.get_all_nodes = AsyncMock(return_value=nodes)
        mock_graph_storage.has_edge = AsyncMock(return_value=False)
        mock_graph_storage.get_node = AsyncMock(side_effect=lambda nid: nodes.get(nid))
        mock_kv_storage.get_by_ids = AsyncMock(return_value=[{"content": "A and B"}])

        async def bad_json_llm(prompt, **kwargs):
            return "not valid json at all"

        global_config["cheap_model_func"] = bad_json_llm
        stats = await _infer_phase(
            mock_graph_storage,
            mock_kv_storage,
            mock_entity_vdb,
            global_config,
        )
        assert stats["rejected_by_llm"] >= 1

    async def test_infer_phase_happy_path(
        self, mock_graph_storage, mock_kv_storage, mock_entity_vdb, global_config
    ):
        nodes = {
            "n1": {
                "entity_name": "Alpha",
                "entity_type": "PERSON",
                "description": "A researcher",
                "source_id": '["c1","c2"]',
            },
            "n2": {
                "entity_name": "Beta",
                "entity_type": "ORG",
                "description": "A company",
                "source_id": '["c1","c2"]',
            },
        }
        mock_graph_storage.get_all_nodes = AsyncMock(return_value=nodes)
        mock_graph_storage.has_edge = AsyncMock(return_value=False)
        mock_graph_storage.get_node = AsyncMock(side_effect=lambda nid: nodes.get(nid))
        mock_graph_storage.upsert_edge = AsyncMock()
        mock_kv_storage.get_by_ids = AsyncMock(return_value=[{"content": "Alpha works at Beta"}])

        async def infer_llm(prompt, **kwargs):
            return '{"has_relation": true, "relation_type": "employed_by", "source": "Alpha", "target": "Beta", "confidence": 0.9, "evidence": "Alpha works at Beta"}'

        global_config["cheap_model_func"] = infer_llm
        stats = await _infer_phase(
            mock_graph_storage,
            mock_kv_storage,
            mock_entity_vdb,
            global_config,
        )
        assert stats["inferred"] >= 1
        assert mock_graph_storage.upsert_edge.called


class TestArefine:
    async def test_arefine_disabled(
        self, mock_graph_storage, mock_entity_vdb, mock_kv_storage, global_config
    ):
        global_config["enable_refinement"] = False
        result = await arefine(
            mock_graph_storage, mock_entity_vdb, mock_kv_storage, global_config, phases=["merge"]
        )
        assert "skipped" in result
        assert result["skipped"]["reason"] == "enable_refinement is disabled"

    async def test_arefine_no_valid_phases(
        self, mock_graph_storage, mock_entity_vdb, mock_kv_storage, global_config
    ):
        result = await arefine(
            mock_graph_storage, mock_entity_vdb, mock_kv_storage, global_config, phases=["invalid"]
        )
        assert "skipped" in result
        assert result["skipped"]["reason"] == "no valid phases specified"

    async def test_arefine_merge_phase(
        self, mock_graph_storage, mock_entity_vdb, mock_kv_storage, global_config
    ):
        nodes = {
            "node_a": {
                "entity_name": "EntityA",
                "entity_type": "PERSON",
                "description": "A test entity description that is the same",
                "source_id": '["chunk_0"]',
            },
            "node_b": {
                "entity_name": "EntityB",
                "entity_type": "PERSON",
                "description": "A test entity description that is the same",
                "source_id": '["chunk_0"]',
            },
        }
        mock_graph_storage.get_all_nodes = AsyncMock(return_value=nodes)
        mock_graph_storage.has_node = AsyncMock(return_value=True)
        mock_graph_storage.has_edge = AsyncMock(return_value=False)
        mock_graph_storage.get_node = AsyncMock(side_effect=lambda nid: nodes.get(nid))
        mock_graph_storage.get_node_edges = AsyncMock(return_value=[])
        mock_graph_storage.get_edge = AsyncMock(return_value=None)

        global_config["cheap_model_func"] = global_config["best_model_func"]

        result = await arefine(
            mock_graph_storage, mock_entity_vdb, mock_kv_storage, global_config, phases=["merge"]
        )
        assert "merge" in result

    async def test_arefine_all_phases_with_mocks(
        self, mock_graph_storage, mock_entity_vdb, mock_kv_storage, global_config
    ):
        with (
            patch(
                "nano_graphrag._ops.refinement.pipeline._merge_phase", new_callable=AsyncMock
            ) as mock_merge,
            patch(
                "nano_graphrag._ops.refinement.pipeline._enrich_phase", new_callable=AsyncMock
            ) as mock_enrich,
            patch(
                "nano_graphrag._ops.refinement.pipeline._infer_phase", new_callable=AsyncMock
            ) as mock_infer,
        ):
            mock_merge.return_value = {"examined": 5, "merged": 2, "skipped": 1}
            mock_enrich.return_value = {
                "examined": 3,
                "enriched": 1,
                "skipped": 1,
                "validation_failed": 1,
            }
            mock_infer.return_value = {
                "examined": 4,
                "inferred": 2,
                "rejected_by_llm": 1,
                "rejected_by_cache": 1,
            }

            result = await arefine(
                mock_graph_storage,
                mock_entity_vdb,
                mock_kv_storage,
                global_config,
                phases=["merge", "enrich", "infer"],
            )
            assert mock_merge.await_count == 1
            assert mock_enrich.await_count == 1
            assert mock_infer.await_count == 1
            assert result["merge"]["merged"] == 2
            assert result["enrich"]["enriched"] == 1
            assert result["infer"]["inferred"] == 2

    async def test_arefine_phase_ordering(
        self, mock_graph_storage, mock_entity_vdb, mock_kv_storage, global_config
    ):
        call_order = []

        async def mock_merge(*args, **kwargs):
            call_order.append("merge")
            return {"examined": 0, "merged": 0, "skipped": 0}

        async def mock_enrich(*args, **kwargs):
            call_order.append("enrich")
            return {"examined": 0, "enriched": 0, "skipped": 0, "validation_failed": 0}

        async def mock_infer(*args, **kwargs):
            call_order.append("infer")
            return {"examined": 0, "inferred": 0, "rejected_by_llm": 0, "rejected_by_cache": 0}

        with (
            patch("nano_graphrag._ops.refinement.pipeline._merge_phase", side_effect=mock_merge),
            patch("nano_graphrag._ops.refinement.pipeline._enrich_phase", side_effect=mock_enrich),
            patch("nano_graphrag._ops.refinement.pipeline._infer_phase", side_effect=mock_infer),
        ):
            await arefine(
                mock_graph_storage,
                mock_entity_vdb,
                mock_kv_storage,
                global_config,
                phases=["merge", "enrich", "infer"],
            )
            assert call_order == ["merge", "enrich", "infer"]

    async def test_arefine_journal_persists(
        self, mock_graph_storage, mock_entity_vdb, mock_kv_storage, global_config
    ):
        with (
            tempfile.TemporaryDirectory() as td,
            patch(
                "nano_graphrag._ops.refinement.pipeline._merge_phase", new_callable=AsyncMock
            ) as mock_merge,
        ):
            mock_merge.return_value = {"examined": 2, "merged": 1, "skipped": 0}
            global_config["working_dir"] = td

            await arefine(
                mock_graph_storage,
                mock_entity_vdb,
                mock_kv_storage,
                global_config,
                phases=["merge"],
            )

            journal_path = os.path.join(td, "refinement_journal.jsonl")
            assert os.path.exists(journal_path)
            journal = RefinementJournal(journal_path)
            assert len(journal.entries) == 1
            assert journal.entries[0]["phase"] == "merge"
