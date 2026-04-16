import asyncio
import json
from typing import Any, Callable

from .._schemas import BatchedEntityExtractionOutput, EntityExtractionOutput
from .._utils import logger, pack_user_ass_to_openai_messages
from ..base import TextChunkSchema
from ..prompt import PROMPTS
from .extraction_common import (
    UNKNOWN_ENTITY_TYPE,
    _join_unique,
    _normalize_document_manifest,
    _normalize_entity_name,
    _normalize_entity_type,
    _parse_legacy_extraction_records,
    _upsert_document_entity,
    _upsert_document_relationship,
)


async def _process_chunk_with_legacy_prompt(
    chunk_key: str, content: str, global_config: dict
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    use_llm_func: Callable[..., Any] = global_config["best_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]
    entity_extract_prompt = PROMPTS["entity_extraction"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
    )
    continue_prompt = PROMPTS["entity_continue_extraction"]
    if_loop_prompt = PROMPTS["entity_if_loop_extraction"]
    hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)
    final_result = await use_llm_func(hint_prompt)
    history = pack_user_ass_to_openai_messages(hint_prompt, final_result, False)
    for now_glean_index in range(entity_extract_max_gleaning):
        glean_result = await use_llm_func(continue_prompt, history_messages=history)
        history += pack_user_ass_to_openai_messages(continue_prompt, glean_result, False)
        final_result += glean_result
        if now_glean_index == entity_extract_max_gleaning - 1:
            break
        if_loop_result: str = await use_llm_func(if_loop_prompt, history_messages=history)
        if if_loop_result.strip().strip('"').strip("'").lower() != "yes":
            break

    return await _parse_legacy_extraction_records(final_result, chunk_key, context_base, False)


def _parse_single_result(
    result, chunk_key: str
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Parse EntityExtractionOutput into entity/relationship dicts."""
    from .._schemas import EntityExtractionOutput

    entities = {}
    relationships = {}
    entity_name_to_id = {}

    if isinstance(result, str):
        result = EntityExtractionOutput(**json.loads(result))
    if not isinstance(result, EntityExtractionOutput):
        return entities, relationships

    for entity in result.entities:
        entity_name = _normalize_entity_name(entity.entity_name)
        if not entity_name:
            continue
        entity_type = _normalize_entity_type(entity.entity_type)
        entity_id = _upsert_document_entity(
            entities,
            entity_name,
            entity_type,
            entity.description,
            chunk_key,
        )
        entity_name_to_id[entity_name] = entity_id

    for relationship in result.relationships:
        src_name = _normalize_entity_name(relationship.source)
        tgt_name = _normalize_entity_name(relationship.target)
        if not src_name or not tgt_name:
            continue
        if src_name not in entity_name_to_id:
            entity_name_to_id[src_name] = _upsert_document_entity(
                entities,
                src_name,
                UNKNOWN_ENTITY_TYPE,
                relationship.description,
                chunk_key,
            )
        if tgt_name not in entity_name_to_id:
            entity_name_to_id[tgt_name] = _upsert_document_entity(
                entities,
                tgt_name,
                UNKNOWN_ENTITY_TYPE,
                relationship.description,
                chunk_key,
            )
        _upsert_document_relationship(
            relationships,
            entity_name_to_id[src_name],
            entity_name_to_id[tgt_name],
            relationship.description,
            relationship.weight,
            chunk_key,
        )

    return entities, relationships


async def _process_single_chunk(
    chunk_key: str,
    content: str,
    use_llm_func: Callable,
    entity_types: list[str],
    global_config: dict,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Extract entities from a single chunk via structured output."""
    fallback_to_parsing = global_config.get("fallback_to_parsing", True)
    try:
        result = await use_llm_func(
            content,
            system_prompt=f"""You are an entity extraction assistant. Extract entities and relationships from the text.
Entity types: {", ".join(entity_types)}.
Return a JSON with 'entities' (name, type, description) and 'relationships' (source, target, description, weight).""",
            response_format=EntityExtractionOutput,
        )
        return _parse_single_result(result, chunk_key)
    except Exception as e:
        logger.warning(f"Structured extraction failed for chunk {chunk_key}: {e}")
        if fallback_to_parsing:
            logger.info(f"Falling back to legacy parsing for chunk {chunk_key}")
            return await _process_chunk_with_legacy_prompt(chunk_key, content, global_config)
        return {}, {}


async def _process_batch_chunks(
    batch: list[tuple[str, TextChunkSchema]],
    use_llm_func: Callable,
    entity_types: list[str],
    global_config: dict,
) -> list[tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]]:
    """Extract entities from multiple chunks in a single LLM call."""
    fallback_to_parsing = global_config.get("fallback_to_parsing", True)

    numbered_text = []
    for i, (chunk_key, chunk_dp) in enumerate(batch):
        numbered_text.append(f"--- CHUNK {i} (id: {chunk_key}) ---\n{chunk_dp['content']}")
    combined_text = "\n\n".join(numbered_text)

    system_prompt = f"""You are an entity extraction assistant. Extract entities and relationships from each chunk below.
Entity types: {", ".join(entity_types)}.
Return a JSON with a 'chunks' array. Each element has: chunk_id (string matching the id in the header), entities (name, type, description), relationships (source, target, description, weight).
Preserve the chunk_id exactly as given."""

    try:
        result = await use_llm_func(
            combined_text,
            system_prompt=system_prompt,
            response_format=BatchedEntityExtractionOutput,
        )
        if isinstance(result, str):
            result = BatchedEntityExtractionOutput(**json.loads(result))

        # Build lookup from chunk_id
        results_by_id = {r.chunk_id: r for r in result.chunks}

        output = []
        for chunk_key, chunk_dp in batch:
            chunk_result = results_by_id.get(chunk_key)
            if chunk_result is None:
                logger.warning(f"Batch extraction missing chunk_id {chunk_key}, falling back")
                if fallback_to_parsing:
                    ents, rels = await _process_chunk_with_legacy_prompt(
                        chunk_key, chunk_dp["content"], global_config
                    )
                else:
                    ents, rels = {}, {}
            else:
                ents, rels = _parse_single_result(
                    EntityExtractionOutput(
                        entities=chunk_result.entities,
                        relationships=chunk_result.relationships,
                    ),
                    chunk_key,
                )
            output.append((ents, rels))
        return output
    except Exception as e:
        logger.warning(f"Batch extraction failed (batch of {len(batch)}): {e}")
        if fallback_to_parsing:
            # Fall back to individual extraction
            results = []
            for chunk_key, chunk_dp in batch:
                ents, rels = await _process_chunk_with_legacy_prompt(
                    chunk_key, chunk_dp["content"], global_config
                )
                results.append((ents, rels))
            return results
        return [({}, {})] * len(batch)


async def extract_document_entity_relationships_structured(
    chunks: dict[str, TextChunkSchema],
    tokenizer_wrapper,
    global_config: dict,
) -> dict:
    quality = global_config.get("entity_extraction_quality", "balanced")
    if quality == "fast":
        use_llm_func = global_config["cheap_model_func"]
    else:
        use_llm_func = global_config["best_model_func"]

    ordered_chunks = list(chunks.items())
    entity_types = PROMPTS["DEFAULT_ENTITY_TYPES"]
    already_processed = 0
    already_entities = 0
    already_relations = 0

    batch_size = global_config.get("extraction_batch_size", 1)
    max_concurrent = global_config.get("extraction_max_async", 16)
    semaphore = asyncio.Semaphore(max_concurrent)

    # Build batches
    batches = []
    for i in range(0, len(ordered_chunks), batch_size):
        batches.append(ordered_chunks[i : i + batch_size])

    async def _process_batch(batch):
        nonlocal already_processed, already_entities, already_relations
        batch_results = await _process_batch_chunks(
            batch,
            use_llm_func,
            entity_types,
            global_config,
        )
        for ents, rels in batch_results:
            already_processed += 1
            already_entities += len(ents)
            already_relations += len(rels)
        if already_processed % 10 == 0 or already_processed >= len(ordered_chunks):
            logger.info(
                f"Processed {already_processed}/{len(ordered_chunks)} chunks "
                f"({already_processed * 100 // len(ordered_chunks)}%), "
                f"{already_entities} entities, {already_relations} relations"
            )
        return batch_results

    async def _process_single_fallback(chunk_item):
        nonlocal already_processed, already_entities, already_relations
        chunk_key, chunk_dp = chunk_item
        ents, rels = await _process_single_chunk(
            chunk_key,
            chunk_dp["content"],
            use_llm_func,
            entity_types,
            global_config,
        )
        already_processed += 1
        already_entities += len(ents)
        already_relations += len(rels)
        if already_processed % 10 == 0 or already_processed == len(ordered_chunks):
            logger.info(
                f"Processed {already_processed}/{len(ordered_chunks)} chunks "
                f"({already_processed * 100 // len(ordered_chunks)}%), "
                f"{already_entities} entities, {already_relations} relations"
            )
        return ents, rels

    async def _process_with_semaphore(coroutine_fn, *args):
        async with semaphore:
            return await coroutine_fn(*args)

    if batch_size > 1 and len(batches) > 1:
        logger.info(
            f"[Batched Extraction] {len(ordered_chunks)} chunks in "
            f"{len(batches)} batches (batch_size={batch_size})"
        )
        batch_results = await asyncio.gather(
            *[_process_with_semaphore(_process_batch, b) for b in batches]
        )
        results = []
        for br in batch_results:
            results.extend(br)
    else:
        results = await asyncio.gather(
            *[_process_with_semaphore(_process_single_fallback, c) for c in ordered_chunks]
        )

    manifest_entities: dict[str, dict[str, Any]] = {}
    manifest_relationships: dict[str, dict[str, Any]] = {}
    manifest: dict[str, Any] = {
        "chunk_ids": list(chunks.keys()),
        "entities": manifest_entities,
        "relationships": manifest_relationships,
    }
    for entities, relationships in results:
        for entity_id, entity in entities.items():
            _upsert_document_entity(
                manifest_entities,
                entity["entity_name"],
                entity["entity_type"],
                _join_unique(entity["descriptions"]),
                entity["source_chunk_ids"][0],
            )
            manifest_entities[entity_id]["descriptions"].extend(entity["descriptions"][1:])
            manifest_entities[entity_id]["source_chunk_ids"].extend(entity["source_chunk_ids"][1:])
        for relationship_id, relationship in relationships.items():
            _upsert_document_relationship(
                manifest_relationships,
                relationship["src_entity_id"],
                relationship["tgt_entity_id"],
                _join_unique(relationship["descriptions"]),
                relationship["weight"],
                relationship["source_chunk_ids"][0],
                relationship.get("relation_type", "related"),
            )
            manifest_relationships[relationship_id]["descriptions"].extend(
                relationship["descriptions"][1:]
            )
            manifest_relationships[relationship_id]["source_chunk_ids"].extend(
                relationship["source_chunk_ids"][1:]
            )
    return _normalize_document_manifest(manifest)
