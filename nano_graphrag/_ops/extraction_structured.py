import asyncio
import json
import re
from typing import Any, Callable

from .._utils import logger, pack_user_ass_to_openai_messages, split_string_by_multi_markers
from ..base import TextChunkSchema
from ..prompt import PROMPTS
from .extraction_common import (
    UNKNOWN_ENTITY_TYPE,
    _handle_single_entity_extraction,
    _handle_single_relationship_extraction,
    _join_unique,
    _normalize_document_manifest,
    _normalize_entity_name,
    _normalize_entity_type,
    _upsert_document_entity,
    _upsert_document_relationship,
)


async def _process_chunk_with_legacy_prompt(
    chunk_key: str, content: str, global_config: dict
) -> tuple[dict[str, dict], dict[str, dict]]:
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

    records = split_string_by_multi_markers(
        final_result,
        [context_base["record_delimiter"], context_base["completion_delimiter"]],
    )
    entities: dict[str, dict[str, Any]] = {}
    relationships: dict[str, dict[str, Any]] = {}
    entity_name_to_id: dict[str, str] = {}
    for record in records:
        record_match = re.search(r"\((.*)\)", record)
        if record_match is None:
            continue
        record_attributes = split_string_by_multi_markers(
            record_match.group(1), [context_base["tuple_delimiter"]]
        )
        entity = await _handle_single_entity_extraction(record_attributes, chunk_key)
        if entity is not None:
            entity_id = _upsert_document_entity(
                entities,
                entity["entity_name"],
                entity["entity_type"],
                entity["description"],
                chunk_key,
            )
            entity_name_to_id[entity["entity_name"]] = entity_id
            continue

        relationship = await _handle_single_relationship_extraction(record_attributes, chunk_key)
        if relationship is None:
            continue
        src_name = relationship["src_name"]
        tgt_name = relationship["tgt_name"]
        if src_name not in entity_name_to_id:
            entity_name_to_id[src_name] = _upsert_document_entity(
                entities, src_name, UNKNOWN_ENTITY_TYPE, relationship["description"], chunk_key
            )
        if tgt_name not in entity_name_to_id:
            entity_name_to_id[tgt_name] = _upsert_document_entity(
                entities, tgt_name, UNKNOWN_ENTITY_TYPE, relationship["description"], chunk_key
            )
        _upsert_document_relationship(
            relationships,
            entity_name_to_id[src_name],
            entity_name_to_id[tgt_name],
            relationship["description"],
            relationship["weight"],
            chunk_key,
        )
    return entities, relationships


async def extract_document_entity_relationships_structured(
    chunks: dict[str, TextChunkSchema],
    tokenizer_wrapper,
    global_config: dict,
) -> dict:
    from .._schemas import EntityExtractionOutput

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
    fallback_to_parsing = global_config.get("fallback_to_parsing", True)

    max_concurrent = global_config.get("extraction_max_async", 16)
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        try:
            result = await use_llm_func(
                content,
                system_prompt=f"""You are an entity extraction assistant. Extract entities and relationships from the text.
Entity types: {", ".join(entity_types)}.
Return a JSON with 'entities' (name, type, description) and 'relationships' (source, target, description, weight).""",
                response_format=EntityExtractionOutput,
            )
            if isinstance(result, str):
                result = EntityExtractionOutput(**json.loads(result))
        except Exception as e:
            logger.warning(f"Structured extraction failed for chunk {chunk_key}: {e}")
            if fallback_to_parsing:
                logger.info(f"Falling back to legacy parsing for chunk {chunk_key}")
                entities, relationships = await _process_chunk_with_legacy_prompt(
                    chunk_key, content, global_config
                )
                already_processed += 1
                already_entities += len(entities)
                already_relations += len(relationships)
                if already_processed % 10 == 0 or already_processed == len(ordered_chunks):
                    logger.info(
                        f"Processed {already_processed}/{len(ordered_chunks)} chunks ({already_processed * 100 // len(ordered_chunks)}%), "
                        f"{already_entities} entities, {already_relations} relations"
                    )
                return entities, relationships
            result = None

        entities = {}
        relationships = {}
        entity_name_to_id = {}

        if isinstance(result, EntityExtractionOutput):
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

        already_processed += 1
        already_entities += len(entities)
        already_relations += len(relationships)
        if already_processed % 10 == 0 or already_processed == len(ordered_chunks):
            logger.info(
                f"Processed {already_processed}/{len(ordered_chunks)} chunks ({already_processed * 100 // len(ordered_chunks)}%), "
                f"{already_entities} entities, {already_relations} relations"
            )
        return entities, relationships

    async def _process_single_content_with_semaphore(chunk_item):
        async with semaphore:
            return await _process_single_content(chunk_item)

    results = await asyncio.gather(
        *[_process_single_content_with_semaphore(c) for c in ordered_chunks]
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
