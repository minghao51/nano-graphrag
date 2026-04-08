import os
import shutil

import pytest

from nano_graphrag import EntityExtractionOutput, ExtractedEntity, ExtractedRelationship, GraphRAG
from nano_graphrag._ops import (
    _handle_single_entity_extraction,
    _handle_single_relationship_extraction,
    _merge_edges_then_upsert,
    _merge_nodes_then_upsert,
    extract_document_entity_relationships,
    extract_entities,
    extract_entities_structured,
    rebuild_knowledge_graph_for_documents,
)
from nano_graphrag._ops.community import generate_community_report
from nano_graphrag._storage import NetworkXStorage
from nano_graphrag._utils import wrap_embedding_func_with_attrs


WORKING_DIR = "./tests/nano_graphrag_cache_refactor_seams"


@wrap_embedding_func_with_attrs(embedding_dim=8, max_token_size=128)
async def mock_embedding(texts: list[str]):
    import numpy as np

    return np.ones((len(texts), 8))


@pytest.fixture(autouse=True)
def setup_teardown():
    if os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR)
    os.mkdir(WORKING_DIR)
    yield
    shutil.rmtree(WORKING_DIR)


def test_ops_re_exports_survive_module_split():
    assert callable(_handle_single_entity_extraction)
    assert callable(_handle_single_relationship_extraction)
    assert callable(_merge_nodes_then_upsert)
    assert callable(_merge_edges_then_upsert)
    assert callable(extract_document_entity_relationships)
    assert callable(extract_entities)
    assert callable(extract_entities_structured)
    assert callable(rebuild_knowledge_graph_for_documents)


@pytest.mark.asyncio
async def test_structured_and_legacy_manifest_shapes_match():
    async def structured_model(prompt, system_prompt=None, history_messages=None, response_format=None, **kwargs):
        if response_format is not None:
            return EntityExtractionOutput(
                entities=[
                    ExtractedEntity(
                        entity_name="Charles Dickens",
                        entity_type="person",
                        description="Author of Oliver Twist.",
                    ),
                    ExtractedEntity(
                        entity_name="Oliver Twist",
                        entity_type="work",
                        description="A novel by Charles Dickens.",
                    ),
                ],
                relationships=[
                    ExtractedRelationship(
                        source="Charles Dickens",
                        target="Oliver Twist",
                        description="Charles Dickens wrote Oliver Twist.",
                        weight=1.0,
                    )
                ],
            )
        return ""

    async def legacy_model(prompt, system_prompt=None, history_messages=None, **kwargs):
        return (
            '("entity"<|>CHARLES DICKENS<|>PERSON<|>Author of Oliver Twist.)##'
            '("entity"<|>OLIVER TWIST<|>WORK<|>A novel by Charles Dickens.)##'
            '("relationship"<|>CHARLES DICKENS<|>OLIVER TWIST<|>Charles Dickens wrote Oliver Twist.<|>1.0)<|COMPLETE|>'
        )

    chunks = {
        "chunk-1": {
            "tokens": 8,
            "content": "Charles Dickens wrote Oliver Twist.",
            "full_doc_id": "doc-1",
            "chunk_order_index": 0,
        }
    }

    structured_rag = GraphRAG(
        working_dir=WORKING_DIR,
        best_model_func=structured_model,
        cheap_model_func=structured_model,
        embedding_func=mock_embedding,
    )
    legacy_rag = GraphRAG(
        working_dir=WORKING_DIR,
        best_model_func=legacy_model,
        cheap_model_func=legacy_model,
        embedding_func=mock_embedding,
    )

    structured_manifest = await extract_document_entity_relationships(
        chunks, structured_rag.tokenizer_wrapper, {**structured_rag._runtime_config(), "_use_structured_extraction": True}
    )
    legacy_manifest = await extract_document_entity_relationships(
        chunks, legacy_rag.tokenizer_wrapper, {**legacy_rag._runtime_config(), "_use_structured_extraction": False}
    )

    assert structured_manifest.keys() == legacy_manifest.keys()
    assert set(structured_manifest["entities"].keys()) == set(legacy_manifest["entities"].keys())
    assert set(structured_manifest["relationships"].keys()) == set(
        legacy_manifest["relationships"].keys()
    )

    structured_entity = next(iter(structured_manifest["entities"].values()))
    legacy_entity = next(iter(legacy_manifest["entities"].values()))
    assert structured_entity.keys() == legacy_entity.keys()

    structured_relationship = next(iter(structured_manifest["relationships"].values()))
    legacy_relationship = next(iter(legacy_manifest["relationships"].values()))
    assert structured_relationship.keys() == legacy_relationship.keys()


class _MemoryKV:
    def __init__(self, initial=None):
        self.data = dict(initial or {})

    async def all_keys(self):
        return list(self.data.keys())

    async def get_by_ids(self, ids, fields=None):
        return [self.data.get(i) for i in ids]

    async def upsert(self, data):
        self.data.update(data)

    async def delete(self, ids):
        for item in ids:
            self.data.pop(item, None)

    async def drop(self):
        self.data.clear()


class _GraphWithCommunities:
    async def community_schema(self):
        return {}


@pytest.mark.asyncio
async def test_generate_community_report_deletes_filtered_stale_reports():
    kv = _MemoryKV({"old-community": {"report_string": "stale"}})
    graph = _GraphWithCommunities()
    rag = GraphRAG(working_dir=WORKING_DIR, embedding_func=mock_embedding)

    await generate_community_report(
        kv,
        graph,
        rag.tokenizer_wrapper,
        rag._runtime_config(),
        only_community_ids={"old-community"},
    )

    assert kv.data == {}


@pytest.mark.asyncio
async def test_networkx_storage_export_still_uses_public_class():
    rag = GraphRAG(working_dir=WORKING_DIR, embedding_func=mock_embedding)
    storage = NetworkXStorage(namespace="test", global_config=rag.__dict__)
    await storage.upsert_node("NODE1", {"source_id": "chunk-1"})
    await storage.upsert_node("NODE2", {"source_id": "chunk-2"})
    await storage.upsert_edge("NODE1", "NODE2", {"weight": 1.0})
    await storage.clustering("leiden")

    assert len(await storage.community_schema()) > 0
