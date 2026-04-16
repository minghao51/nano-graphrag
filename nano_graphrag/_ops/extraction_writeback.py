import asyncio
import json
from typing import Optional, Union

from .._utils import logger
from ..base import BaseGraphStorage, BaseVectorStorage, TextChunkSchema
from .extraction_common import _handle_entity_relation_summary, _join_unique


async def _process_entity_writeback(
    entity_id: str,
    entity: dict,
    knowledge_graph_inst: BaseGraphStorage,
    entity_registry,
    extracted_aliases: dict,
    summary_semaphore: asyncio.Semaphore,
    global_config: dict,
    tokenizer_wrapper,
):
    description = _join_unique(entity["descriptions"])
    async with summary_semaphore:
        description = await _handle_entity_relation_summary(
            entity["entity_name"], description, global_config, tokenizer_wrapper
        )
    await knowledge_graph_inst.upsert_node(
        entity_id,
        {
            "entity_name": entity["entity_name"],
            "entity_type": entity["entity_type"],
            "aliases": json.dumps(entity.get("aliases", [])),
            "description": description,
            "source_id": _join_unique(entity["source_chunk_ids"]),
        },
    )
    if entity_registry is not None:
        entity_registry.register_entity(
            entity_id=entity_id,
            canonical_name=entity["entity_name"],
            aliases=sorted(
                set(extracted_aliases.get(entity_id, [])).union(entity.get("aliases", []))
            ),
            entity_type=entity["entity_type"],
            metadata={"description": description},
        )


async def _process_relationship_writeback(
    relationship_id: str,
    relationship: dict,
    knowledge_graph_inst: BaseGraphStorage,
    summary_semaphore: asyncio.Semaphore,
    global_config: dict,
    tokenizer_wrapper,
):
    description = _join_unique(relationship["descriptions"])
    async with summary_semaphore:
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
    result = await _extract_aliases_for_batch(
        [(entity_name, entity_type, source_chunk_ids)], chunks, global_config
    )
    return result.get(entity_name, [])


async def _extract_aliases_for_batch(
    entities: list[tuple[str, str, list[str]]],
    chunks: dict[str, TextChunkSchema],
    global_config: dict,
) -> dict[str, list[str]]:
    """Extract aliases for a batch of entities in a single LLM call.

    Args:
        entities: List of (entity_name, entity_type, source_chunk_ids) tuples
        chunks: All text chunks
        global_config: Global configuration

    Returns:
        Dict mapping entity_name to list of aliases
    """
    if not entities:
        return {}

    use_llm_func = global_config.get("cheap_model_func")
    if use_llm_func is None:
        return {entity[0]: [] for entity in entities}

    entity_texts = []
    for entity_name, entity_type, source_chunk_ids in entities:
        relevant_chunks = [
            chunks[chunk_id]["content"] for chunk_id in source_chunk_ids if chunk_id in chunks
        ]
        if not relevant_chunks:
            entity_texts.append((entity_name, entity_type, ""))
        else:
            sample_chunks = relevant_chunks[:5]
            sample_text = "\n\n".join(sample_chunks)
            entity_texts.append((entity_name, entity_type, sample_text))

    prompt_parts = []
    for idx, (entity_name, entity_type, text) in enumerate(entity_texts, 1):
        prompt_parts.append(f"""Entity {idx}: "{entity_name}" (type: {entity_type})
Relevant text:
{text if text else "(no text available)"}""")

    entities_section = "\n\n".join(prompt_parts)

    prompt = f"""For each of the following entities, extract alternative names and aliases.

Rules:
1. Only extract names that refer to the SAME entity
2. Include abbreviations, nicknames, and alternative spellings
3. Exclude variations that are just case differences (e.g., "sam" vs "Sam")
4. If no aliases are found for an entity, use an empty list
5. Be conservative - only extract names that are clearly referring to the same entity

{entities_section}

Return JSON format with all entities:
{{
  "Entity 1": ["alias1", "alias2"],
  "Entity 2": [],
  ...
}}
"""

    try:
        response = await use_llm_func(prompt)
        if isinstance(response, str):
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                data = json.loads(json_str)
            else:
                data = json.loads(response)
        else:
            data = response if isinstance(response, dict) else {}

        result: dict[str, list[str]] = {}
        for idx, (entity_name, entity_type, _) in enumerate(entity_texts, 1):
            entity_key = f"Entity {idx}"
            aliases = data.get(entity_key, [])

            if isinstance(aliases, list):
                filtered_aliases = [
                    alias.strip()
                    for alias in aliases
                    if alias and alias.strip().lower() != entity_name.lower()
                ]
                result[entity_name] = filtered_aliases
            else:
                result[entity_name] = []

        return result

    except Exception as e:
        logger.debug(f"Failed to extract aliases for batch: {e}")
        return {entity[0]: [] for entity in entities}


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
        batch_size = global_config.get("alias_batch_size", 20)
        max_batches_in_flight = global_config.get("alias_max_batches_in_flight", 5)
        chunks_by_id: dict[str, TextChunkSchema] = chunks or {}

        batches = []
        for i in range(0, len(aliases_to_extract), batch_size):
            batch_entities = []
            for entity_id, entity in aliases_to_extract[i : i + batch_size]:
                batch_entities.append(
                    (
                        entity["entity_name"],
                        entity["entity_type"],
                        entity["source_chunk_ids"],
                    )
                )
            batches.append(batch_entities)

        # Limit concurrent batch processing for memory safety
        semaphore = asyncio.Semaphore(max_batches_in_flight)

        async def process_batch(batch):
            async with semaphore:
                batch_entity_names = [
                    e[0] for e in batch
                ]  # entity_names from (entity_name, entity_type, source_chunk_ids)
                result = await _extract_aliases_for_batch(batch, chunks_by_id, global_config)
                return batch_entity_names, result

        batch_results = await asyncio.gather(
            *[process_batch(batch) for batch in batches], return_exceptions=True
        )

        # Track batch failures and raise if too many fail
        failed_batches = 0
        total_batches = len(batch_results)

        for batch_result in batch_results:
            if isinstance(batch_result, Exception):
                failed_batches += 1
                logger.warning(f"Alias batch extraction failed: {batch_result}")
                continue
            entity_names, result_dict = batch_result
            for idx, entity_name in enumerate(entity_names, 1):
                entity_key = f"Entity {idx}"
                aliases = result_dict.get(entity_key, [])
                for eid, entity in aliases_to_extract:
                    if entity["entity_name"] == entity_name:
                        extracted_aliases[eid] = aliases

        # Raise error if too many batches failed
        if failed_batches > 0:
            failure_rate = failed_batches / total_batches
            logger.warning(
                f"Alias extraction completed with {failed_batches}/{total_batches} batch failures ({failure_rate:.1%})"
            )
            if failure_rate > 0.5:
                raise RuntimeError(
                    f"Alias extraction failed: {failure_rate:.1%} of batches failed ({failed_batches}/{total_batches})"
                )

    summary_semaphore = asyncio.Semaphore(global_config.get("extraction_max_async", 16))

    await asyncio.gather(
        *[
            _process_entity_writeback(
                eid,
                ent,
                knowledge_graph_inst,
                entity_registry,
                extracted_aliases,
                summary_semaphore,
                global_config,
                tokenizer_wrapper,
            )
            for eid, ent in manifest["entities"].items()
        ]
    )

    await asyncio.gather(
        *[
            _process_relationship_writeback(
                rid,
                rel,
                knowledge_graph_inst,
                summary_semaphore,
                global_config,
                tokenizer_wrapper,
            )
            for rid, rel in manifest["relationships"].items()
        ]
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
