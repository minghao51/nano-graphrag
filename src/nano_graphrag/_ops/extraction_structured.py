from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

from .._schemas import BatchedEntityExtractionOutput, EntityExtractionOutput
from .._utils import logger
from ..base import TextChunkSchema
from ..prompt import PROMPTS
from .extraction_common import (
    UNKNOWN_ENTITY_TYPE,
    _ExtractionProgress,
    _merge_results_into_manifest,
    _normalize_entity_name,
    _normalize_entity_type,
    _parse_legacy_extraction_records,
    _run_gleaning_loop,
    _upsert_document_entity,
    _upsert_document_relationship,
)
from .extraction_prompts import _build_extraction_system_prompt


async def _process_chunk_with_legacy_prompt(
    chunk_key: str, content: str, global_config: dict
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    quality = global_config.get("entity_extraction_quality", "balanced")
    if quality == "fast":
        use_llm_func: Callable[..., Any] = global_config["cheap_model_func"]
    else:
        use_llm_func = global_config["best_model_func"]
    max_gleaning = 0 if quality == "fast" else global_config["entity_extract_max_gleaning"]
    entity_extract_prompt = PROMPTS["entity_extraction"]
    context_base = {
        "tuple_delimiter": PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        "record_delimiter": PROMPTS["DEFAULT_RECORD_DELIMITER"],
        "completion_delimiter": PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        "entity_types": ",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
    }
    final_result = await _run_gleaning_loop(
        use_llm_func, content, entity_extract_prompt, context_base, max_gleaning
    )
    return await _parse_legacy_extraction_records(final_result, chunk_key, context_base, False)


def _parse_single_result(
    result, chunk_key: str
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Parse EntityExtractionOutput into entity/relationship dicts."""
    from .._schemas import EntityExtractionOutput

    entities: dict[str, dict[str, Any]] = {}
    relationships: dict[str, dict[str, Any]] = {}
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
        aliases = [a for a in getattr(entity, "aliases", []) if a]
        event_date = getattr(entity, "event_date", None)
        entity_id = _upsert_document_entity(
            entities,
            entity_name,
            entity_type,
            entity.description,
            chunk_key,
            aliases=aliases,
            event_date=event_date,
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
            relation_type=relationship.relation_type,
            confidence=relationship.confidence,
            temporal_context=getattr(relationship, "temporal_context", None),
            valid_from=getattr(relationship, "valid_from", None),
            valid_to=getattr(relationship, "valid_to", None),
        )

    return entities, relationships


async def _process_single_chunk(
    chunk_key: str,
    content: str,
    use_llm_func: Callable,
    entity_types: list[str],
    global_config: dict,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    fallback_to_parsing = global_config.get("fallback_to_parsing", True)
    system_prompt = _build_extraction_system_prompt(entity_types, global_config)
    try:
        result = await use_llm_func(
            content,
            system_prompt=system_prompt,
            response_format=EntityExtractionOutput,
        )
        return _parse_single_result(result, chunk_key)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        logger.warning("structured_extraction_failed", chunk_key=chunk_key, error=str(e))
        if fallback_to_parsing:
            logger.info("fallback_to_legacy_parsing", chunk_key=chunk_key)
            return await _process_chunk_with_legacy_prompt(chunk_key, content, global_config)
        return {}, {}


async def _process_batch_chunks(
    batch: list[tuple[str, TextChunkSchema]],
    use_llm_func: Callable,
    entity_types: list[str],
    global_config: dict,
) -> list[tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]]:
    fallback_to_parsing = global_config.get("fallback_to_parsing", True)

    numbered_text = []
    for i, (chunk_key, chunk_dp) in enumerate(batch):
        numbered_text.append(f"--- CHUNK {i} (id: {chunk_key}) ---\n{chunk_dp['content']}")
    combined_text = "\n\n".join(numbered_text)

    system_prompt = _build_extraction_system_prompt(entity_types, global_config, batched=True)

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
                logger.warning("batch_extraction_missing_chunk", chunk_key=chunk_key)
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
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        logger.warning("batch_extraction_failed", batch_size=len(batch), error=str(e))
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
    progress = _ExtractionProgress(len(ordered_chunks))

    batch_size = global_config.get("extraction_batch_size", 1)
    if quality == "fast" and batch_size < 8:
        batch_size = 8
    max_concurrent = global_config.get("extraction_max_async", 16)
    semaphore = asyncio.Semaphore(max_concurrent)

    batches = []
    for i in range(0, len(ordered_chunks), batch_size):
        batches.append(ordered_chunks[i : i + batch_size])

    async def _process_batch(batch):
        batch_results = await _process_batch_chunks(
            batch,
            use_llm_func,
            entity_types,
            global_config,
        )
        for ents, rels in batch_results:
            progress.update(len(ents), len(rels))
        return batch_results

    async def _process_single_fallback(chunk_item):
        chunk_key, chunk_dp = chunk_item
        ents, rels = await _process_single_chunk(
            chunk_key,
            chunk_dp["content"],
            use_llm_func,
            entity_types,
            global_config,
        )
        progress.update(len(ents), len(rels))
        return ents, rels

    async def _process_with_semaphore(coroutine_fn, *args):
        async with semaphore:
            return await coroutine_fn(*args)

    if batch_size > 1 and len(batches) > 1:
        logger.info(
            "batched_extraction_config",
            total_chunks=len(ordered_chunks),
            batch_count=len(batches),
            batch_size=batch_size,
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

    return _merge_results_into_manifest(results, list(chunks.keys()))
