import asyncio
from unittest.mock import patch

import pytest

from nano_graphrag._entity_registry import EntityRegistry
from nano_graphrag._ops.extraction_writeback import (
    _process_entity_writeback,
    _process_relationship_writeback,
    _write_extraction_manifest,
)

pytestmark = pytest.mark.unit


async def _fake_summary(name, description, global_config, tokenizer_wrapper):
    return description


@pytest.fixture
def entity():
    return {
        "entity_name": "Alice",
        "entity_type": "PERSON",
        "descriptions": ["A researcher"],
        "source_chunk_ids": ["chunk1"],
        "aliases": [],
    }


@pytest.fixture
def relationship():
    return {
        "src_entity_id": "e1",
        "tgt_entity_id": "e2",
        "descriptions": ["related"],
        "weight": 1.0,
        "source_chunk_ids": ["chunk1"],
        "relation_type": "part_of",
        "confidence": 0.9,
    }


@pytest.mark.asyncio
async def test_process_entity_writeback_creates_node(entity, mock_graph_storage, global_config):
    with patch(
        "nano_graphrag._ops.extraction_writeback._handle_entity_relation_summary",
        new=_fake_summary,
    ):
        await _process_entity_writeback(
            "e1",
            entity,
            mock_graph_storage,
            None,
            {},
            asyncio.Semaphore(4),
            global_config,
            None,
        )
    assert mock_graph_storage.upsert_node.called
    call_args = mock_graph_storage.upsert_node.call_args
    assert call_args[0][0] == "e1"
    node_data = call_args[0][1]
    assert node_data["entity_name"] == "Alice"
    assert node_data["entity_type"] == "PERSON"


@pytest.mark.asyncio
async def test_process_entity_writeback_with_registry(entity, mock_graph_storage, global_config):
    registry = EntityRegistry()
    with patch(
        "nano_graphrag._ops.extraction_writeback._handle_entity_relation_summary",
        new=_fake_summary,
    ):
        await _process_entity_writeback(
            "e1",
            entity,
            mock_graph_storage,
            registry,
            {},
            asyncio.Semaphore(4),
            global_config,
            None,
        )
    assert registry.resolve_entity("Alice") == "e1"


@pytest.mark.asyncio
async def test_process_relationship_writeback_creates_edge(
    relationship, mock_graph_storage, global_config
):
    with patch(
        "nano_graphrag._ops.extraction_writeback._handle_entity_relation_summary",
        new=_fake_summary,
    ):
        await _process_relationship_writeback(
            "r1",
            relationship,
            mock_graph_storage,
            asyncio.Semaphore(4),
            global_config,
            None,
        )
    assert mock_graph_storage.upsert_edge.called
    call_args = mock_graph_storage.upsert_edge.call_args
    assert call_args[0][0] == "e1"
    assert call_args[0][1] == "e2"
    edge_data = call_args[0][2]
    assert edge_data["relation_type"] == "part_of"
    assert edge_data["confidence"] == 0.9


@pytest.mark.asyncio
async def test_process_relationship_writeback_temporal_fields(mock_graph_storage, global_config):
    relationship = {
        "src_entity_id": "e1",
        "tgt_entity_id": "e2",
        "descriptions": ["related"],
        "weight": 1.0,
        "source_chunk_ids": ["chunk1"],
        "relation_type": "part_of",
        "confidence": 0.9,
        "temporal_context": "in 2023",
        "valid_from": "2023-01-01",
        "valid_to": "2023-12-31",
    }
    with patch(
        "nano_graphrag._ops.extraction_writeback._handle_entity_relation_summary",
        new=_fake_summary,
    ):
        await _process_relationship_writeback(
            "r1",
            relationship,
            mock_graph_storage,
            asyncio.Semaphore(4),
            global_config,
            None,
        )
    edge_data = mock_graph_storage.upsert_edge.call_args[0][2]
    assert edge_data["temporal_context"] == "in 2023"
    assert edge_data["valid_from"] == "2023-01-01"
    assert edge_data["valid_to"] == "2023-12-31"


@pytest.mark.asyncio
async def test_write_manifest_empty_entities_returns_none(
    mock_graph_storage, mock_entity_vdb, global_config
):
    result = await _write_extraction_manifest(
        {"entities": {}, "relationships": {}},
        mock_graph_storage,
        mock_entity_vdb,
        None,
        global_config,
    )
    assert result is None


@pytest.mark.asyncio
async def test_write_manifest_upserts_entities(mock_graph_storage, mock_entity_vdb, global_config):
    manifest = {
        "entities": {
            "entity_id_1": {
                "entity_name": "Alice",
                "entity_type": "PERSON",
                "descriptions": ["A researcher at MIT"],
                "source_chunk_ids": ["chunk1"],
                "aliases": [],
            }
        },
        "relationships": {},
    }
    with patch(
        "nano_graphrag._ops.extraction_writeback._handle_entity_relation_summary",
        new=_fake_summary,
    ):
        result = await _write_extraction_manifest(
            manifest,
            mock_graph_storage,
            mock_entity_vdb,
            None,
            global_config,
        )
    assert result is mock_graph_storage
    assert mock_graph_storage.upsert_node.called
    assert mock_entity_vdb.upsert.called


@pytest.mark.asyncio
async def test_write_manifest_upserts_relationships(
    mock_graph_storage, mock_entity_vdb, global_config
):
    manifest = {
        "entities": {
            "entity_id_1": {
                "entity_name": "Alice",
                "entity_type": "PERSON",
                "descriptions": ["A researcher at MIT"],
                "source_chunk_ids": ["chunk1"],
                "aliases": [],
            },
            "entity_id_2": {
                "entity_name": "MIT",
                "entity_type": "ORGANIZATION",
                "descriptions": ["A university"],
                "source_chunk_ids": ["chunk1"],
                "aliases": [],
            },
        },
        "relationships": {
            "rel_id_1": {
                "src_entity_id": "entity_id_1",
                "tgt_entity_id": "entity_id_2",
                "descriptions": ["Alice works at MIT"],
                "weight": 1.0,
                "source_chunk_ids": ["chunk1"],
                "relation_type": "employed_by",
                "confidence": 0.9,
            }
        },
    }
    with patch(
        "nano_graphrag._ops.extraction_writeback._handle_entity_relation_summary",
        new=_fake_summary,
    ):
        result = await _write_extraction_manifest(
            manifest,
            mock_graph_storage,
            mock_entity_vdb,
            None,
            global_config,
        )
    assert result is mock_graph_storage
    assert mock_graph_storage.upsert_edge.called
    assert mock_graph_storage.upsert_node.called
