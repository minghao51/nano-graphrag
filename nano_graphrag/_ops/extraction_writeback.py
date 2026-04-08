from typing import Union

from .._utils import logger
from ..base import BaseGraphStorage, BaseVectorStorage, TextChunkSchema
from .extraction_common import _handle_entity_relation_summary, _join_unique
from .extraction_legacy import extract_document_entity_relationships_legacy
from .extraction_structured import extract_document_entity_relationships_structured


async def extract_entities_structured(
    chunks: dict[str, TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    tokenizer_wrapper,
    global_config: dict,
    using_amazon_bedrock: bool = False,
) -> Union[BaseGraphStorage, None]:
    manifest = await extract_document_entity_relationships_structured(
        chunks,
        tokenizer_wrapper,
        global_config,
    )
    if not manifest["entities"]:
        return None
    for entity_id, entity in manifest["entities"].items():
        description = _join_unique(entity["descriptions"])
        description = await _handle_entity_relation_summary(
            entity["entity_name"], description, global_config, tokenizer_wrapper
        )
        await knowledge_graph_inst.upsert_node(
            entity_id,
            {
                "entity_name": entity["entity_name"],
                "entity_type": entity["entity_type"],
                "description": description,
                "source_id": _join_unique(entity["source_chunk_ids"]),
            },
        )
    for relationship_id, relationship in manifest["relationships"].items():
        description = _join_unique(relationship["descriptions"])
        description = await _handle_entity_relation_summary(
            relationship_id, description, global_config, tokenizer_wrapper
        )
        await knowledge_graph_inst.upsert_edge(
            relationship["src_entity_id"],
            relationship["tgt_entity_id"],
            {
                "description": description,
                "weight": relationship["weight"],
                "source_id": _join_unique(relationship["source_chunk_ids"]),
                "order": 1,
                "relationship_id": relationship_id,
            },
        )
    if entity_vdb is not None:
        await entity_vdb.upsert(
            {
                entity_id: {
                    "content": entity["entity_name"] + _join_unique(entity["descriptions"]),
                    "entity_name": entity["entity_name"],
                }
                for entity_id, entity in manifest["entities"].items()
            }
        )
    return knowledge_graph_inst


async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    tokenizer_wrapper,
    global_config: dict,
    using_amazon_bedrock: bool = False,
) -> Union[BaseGraphStorage, None]:
    if global_config.get("_use_structured_extraction", False):
        return await extract_entities_structured(
            chunks,
            knowledge_graph_inst,
            entity_vdb,
            tokenizer_wrapper,
            global_config,
            using_amazon_bedrock,
        )
    manifest = await extract_document_entity_relationships_legacy(
        chunks,
        tokenizer_wrapper,
        global_config,
        using_amazon_bedrock,
    )
    if not manifest["entities"]:
        logger.warning("Didn't extract any entities, maybe your LLM is not working")
        return None
    for entity_id, entity in manifest["entities"].items():
        description = _join_unique(entity["descriptions"])
        description = await _handle_entity_relation_summary(
            entity["entity_name"], description, global_config, tokenizer_wrapper
        )
        await knowledge_graph_inst.upsert_node(
            entity_id,
            {
                "entity_name": entity["entity_name"],
                "entity_type": entity["entity_type"],
                "description": description,
                "source_id": _join_unique(entity["source_chunk_ids"]),
            },
        )
    for relationship_id, relationship in manifest["relationships"].items():
        description = _join_unique(relationship["descriptions"])
        description = await _handle_entity_relation_summary(
            relationship_id, description, global_config, tokenizer_wrapper
        )
        await knowledge_graph_inst.upsert_edge(
            relationship["src_entity_id"],
            relationship["tgt_entity_id"],
            {
                "description": description,
                "weight": relationship["weight"],
                "source_id": _join_unique(relationship["source_chunk_ids"]),
                "order": 1,
                "relationship_id": relationship_id,
            },
        )
    if entity_vdb is not None:
        await entity_vdb.upsert(
            {
                entity_id: {
                    "content": entity["entity_name"] + _join_unique(entity["descriptions"]),
                    "entity_name": entity["entity_name"],
                }
                for entity_id, entity in manifest["entities"].items()
            }
        )
    return knowledge_graph_inst
