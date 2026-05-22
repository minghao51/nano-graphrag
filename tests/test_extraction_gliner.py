from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nano_graphrag._ops.extraction_gliner import (
    extract_document_entity_relationships_gliner,
    extract_entities_gliner,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def mock_model():
    schema_chain = MagicMock(
        return_value=MagicMock(
            entities=MagicMock(return_value=MagicMock(relations=MagicMock(return_value="schema")))
        )
    )
    model = MagicMock()
    model.create_schema = schema_chain
    return model


@pytest.fixture
def global_config():
    return {"extraction_max_async": 4}


@pytest.mark.asyncio
async def test_empty_chunks_returns_empty_manifest(mock_model, global_config):
    with patch(
        "nano_graphrag._ops.extraction_gliner._get_gliner_model",
        new_callable=AsyncMock,
        return_value=mock_model,
    ):
        result = await extract_document_entity_relationships_gliner({}, None, global_config)
    assert result["entities"] == {}
    assert result["relationships"] == {}


@pytest.mark.asyncio
async def test_single_chunk_extracts_entities(mock_model, global_config):
    mock_model.extract.return_value = {
        "entities": {"person": ["Alice"], "organization": ["MIT"]},
        "relation_extraction": {},
    }

    with patch(
        "nano_graphrag._ops.extraction_gliner._get_gliner_model",
        new_callable=AsyncMock,
        return_value=mock_model,
    ):
        result = await extract_document_entity_relationships_gliner(
            {"chunk1": {"content": "Alice works at MIT"}}, None, global_config
        )

    assert len(result["entities"]) == 2
    names = {e["entity_name"] for e in result["entities"].values()}
    assert "Alice" in names
    assert "MIT" in names
    types = {e["entity_type"] for e in result["entities"].values()}
    assert "PERSON" in types
    assert "ORGANIZATION" in types
    for e in result["entities"].values():
        assert "chunk1" in e["source_chunk_ids"]


@pytest.mark.asyncio
async def test_extracts_relationships(mock_model, global_config):
    mock_model.extract.return_value = {
        "entities": {"person": ["Alice"], "organization": ["MIT"]},
        "relation_extraction": {"works_for": [("Alice", "MIT")]},
    }

    with patch(
        "nano_graphrag._ops.extraction_gliner._get_gliner_model",
        new_callable=AsyncMock,
        return_value=mock_model,
    ):
        result = await extract_document_entity_relationships_gliner(
            {"chunk1": {"content": "Alice works at MIT"}}, None, global_config
        )

    assert len(result["relationships"]) == 1
    rel = next(iter(result["relationships"].values()))
    assert rel["src_entity_id"] in result["entities"]
    assert rel["tgt_entity_id"] in result["entities"]
    assert result["entities"][rel["src_entity_id"]]["entity_name"] == "Alice"
    assert result["entities"][rel["tgt_entity_id"]]["entity_name"] == "MIT"
    assert rel["relation_type"] == "works_for"


@pytest.mark.asyncio
async def test_multiple_chunks_merges_entities(mock_model, global_config):
    mock_model.extract.side_effect = [
        {"entities": {"person": ["Alice"]}, "relation_extraction": {}},
        {"entities": {"person": ["Alice"]}, "relation_extraction": {}},
    ]

    with patch(
        "nano_graphrag._ops.extraction_gliner._get_gliner_model",
        new_callable=AsyncMock,
        return_value=mock_model,
    ):
        result = await extract_document_entity_relationships_gliner(
            {
                "chunk1": {"content": "Alice is here"},
                "chunk2": {"content": "Alice is also here"},
            },
            None,
            global_config,
        )

    alice_entities = [e for e in result["entities"].values() if e["entity_name"] == "Alice"]
    assert len(alice_entities) == 1
    assert "chunk1" in alice_entities[0]["source_chunk_ids"]
    assert "chunk2" in alice_entities[0]["source_chunk_ids"]


@pytest.mark.asyncio
async def test_model_extract_exception_handled(mock_model, global_config):
    mock_model.extract.side_effect = Exception("extraction failed")

    with patch(
        "nano_graphrag._ops.extraction_gliner._get_gliner_model",
        new_callable=AsyncMock,
        return_value=mock_model,
    ):
        result = await extract_document_entity_relationships_gliner(
            {"chunk1": {"content": "Some text"}}, None, global_config
        )

    assert result["entities"] == {}
    assert result["relationships"] == {}


@pytest.mark.asyncio
async def test_extract_entities_gliner_empty_returns_none(global_config):
    empty_manifest = {
        "entities": {},
        "relationships": {},
        "chunk_ids": [],
    }

    with patch(
        "nano_graphrag._ops.extraction_gliner.extract_document_entity_relationships_gliner",
        new_callable=AsyncMock,
        return_value=empty_manifest,
    ):
        result = await extract_entities_gliner(
            {},
            MagicMock(),
            MagicMock(),
            MagicMock(),
            global_config,
        )

    assert result is None
