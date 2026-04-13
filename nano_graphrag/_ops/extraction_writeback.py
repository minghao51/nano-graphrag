import asyncio
import json
from typing import Optional, Union

from .._utils import logger
from ..base import BaseGraphStorage, BaseVectorStorage, TextChunkSchema
from .extraction_common import _handle_entity_relation_summary, _join_unique


async def _extract_aliases_for_entity(
    entity_name: str,
    entity_type: str,
    source_chunk_ids: list[str],
    chunks: dict[str, TextChunkSchema],
    global_config: dict,
) -> list[str]:
    """Extract aliases for an entity from its source chunks.

    Args:
        entity_name: The canonical entity name
        entity_type: The entity type
        source_chunk_ids: IDs of chunks where this entity appears
        chunks: All text chunks
        global_config: Global configuration

    Returns:
        List of alias names for this entity
    """
    # Get the chunks where this entity appears
    relevant_chunks = [
        chunks[chunk_id]["content"] for chunk_id in source_chunk_ids if chunk_id in chunks
    ]

    if not relevant_chunks:
        return []

    # Use cheap model for alias extraction (faster, less critical task)
    use_llm_func = global_config.get("cheap_model_func")
    if use_llm_func is None:
        return []

    # Sample chunks if there are too many (limit to ~5 chunks)
    sample_chunks = relevant_chunks[:5]
    sample_text = "\n\n".join(sample_chunks)

    prompt = f"""Extract alternative names and aliases for the entity "{entity_name}" (type: {entity_type}) from the text below.

Rules:
1. Only extract names that refer to the SAME entity
2. Include abbreviations, nicknames, and alternative spellings
3. Exclude variations that are just case differences (e.g., "sam" vs "Sam")
4. Return as a JSON list of strings
5. If no aliases are found, return an empty list

Text:
{sample_text}

Return JSON format: {{"aliases": ["alias1", "alias2", ...]}}"""

    try:
        response = await use_llm_func(prompt)
        if isinstance(response, str):
            # Look for JSON object in response
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                data = json.loads(json_str)
                aliases = data.get("aliases", [])
            else:
                # Fallback: try parsing entire response
                data = json.loads(response)
                aliases = data.get("aliases", [])
        else:
            aliases = response.get("aliases", [])

        # Filter out empty strings and the canonical name itself
        filtered_aliases = [
            alias.strip()
            for alias in aliases
            if alias and alias.strip().lower() != entity_name.lower()
        ]

        return filtered_aliases

    except Exception as e:
        logger.debug(f"Failed to extract aliases for {entity_name}: {e}")
        return []


async def _write_extraction_manifest(
    manifest: dict,
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    tokenizer_wrapper,
    global_config: dict,
    chunks: Optional[dict[str, TextChunkSchema]] = None,
) -> Union[BaseGraphStorage, None]:
    if not manifest["entities"]:
        return None

    # Get entity_registry from global_config if available (for entity-grounded RAG)
    entity_registry = global_config.get("entity_registry")

    # Extract aliases for entities if chunks are provided
    aliases_to_extract = []
    if entity_registry is not None and chunks:
        for entity_id, entity in manifest["entities"].items():
            aliases_to_extract.append((entity_id, entity))

    # Batch extract aliases (parallel for efficiency)
    extracted_aliases: dict[str, list[str]] = {}
    if aliases_to_extract:
        chunks_by_id: dict[str, TextChunkSchema] = chunks  # type: ignore[assignment]
        extract_tasks = [
            _extract_aliases_for_entity(
                entity["entity_name"],
                entity["entity_type"],
                entity["source_chunk_ids"],
                chunks_by_id,
                global_config,
            )
            for entity_id, entity in aliases_to_extract
        ]
        alias_results = await asyncio.gather(*extract_tasks, return_exceptions=True)

        for (entity_id, entity), result in zip(aliases_to_extract, alias_results):
            if isinstance(result, Exception):
                logger.debug(f"Alias extraction failed for {entity['entity_name']}: {result}")
                extracted_aliases[entity_id] = []
            else:
                extracted_aliases[entity_id] = result  # type: ignore[assignment]

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

        # Register entity in EntityRegistry for entity-grounded RAG
        if entity_registry is not None:
            entity_registry.register_entity(
                entity_id=entity_id,
                canonical_name=entity["entity_name"],
                aliases=extracted_aliases.get(entity_id, []),
                entity_type=entity["entity_type"],
                metadata={"description": description},
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
                    "content": entity["entity_name"] + " - " + _join_unique(entity["descriptions"]),
                    "entity_name": entity["entity_name"],
                }
                for entity_id, entity in manifest["entities"].items()
            }
        )
    return knowledge_graph_inst


async def extract_entities_structured(
    chunks: dict[str, TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    tokenizer_wrapper,
    global_config: dict,
    using_amazon_bedrock: bool = False,
) -> Union[BaseGraphStorage, None]:
    from .extraction_structured import extract_document_entity_relationships_structured

    manifest = await extract_document_entity_relationships_structured(
        chunks,
        tokenizer_wrapper,
        global_config,
    )
    return await _write_extraction_manifest(
        manifest, knowledge_graph_inst, entity_vdb, tokenizer_wrapper, global_config, chunks
    )


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
    from .extraction_legacy import extract_document_entity_relationships_legacy

    manifest = await extract_document_entity_relationships_legacy(
        chunks,
        tokenizer_wrapper,
        global_config,
        using_amazon_bedrock,
    )
    if not manifest["entities"]:
        logger.warning("Didn't extract any entities, maybe your LLM is not working")
        return None
    return await _write_extraction_manifest(
        manifest, knowledge_graph_inst, entity_vdb, tokenizer_wrapper, global_config, chunks
    )
