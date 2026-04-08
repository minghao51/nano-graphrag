import asyncio
from typing import Any, Callable

from .._utils import logger, pack_user_ass_to_openai_messages
from ..base import TextChunkSchema
from ..prompt import PROMPTS
from .extraction_common import (
    _normalize_document_manifest,
    _parse_legacy_extraction_records,
    generate_stable_entity_id,
)


async def extract_document_entity_relationships_legacy(
    chunks: dict[str, TextChunkSchema],
    tokenizer_wrapper,
    global_config: dict,
    using_amazon_bedrock: bool = False,
) -> dict:
    use_llm_func: Callable[..., Any] = global_config["best_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]
    ordered_chunks = list(chunks.items())

    entity_extract_prompt = PROMPTS["entity_extraction"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
    )
    continue_prompt = PROMPTS["entity_continue_extraction"]
    if_loop_prompt = PROMPTS["entity_if_loop_extraction"]

    already_processed = 0
    already_entities = 0
    already_relations = 0

    max_concurrent = global_config.get("extraction_max_async", 16)
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)
        final_result = await use_llm_func(hint_prompt)
        if isinstance(final_result, list):
            final_result = final_result[0]["text"]

        history = pack_user_ass_to_openai_messages(hint_prompt, final_result, using_amazon_bedrock)
        for now_glean_index in range(entity_extract_max_gleaning):
            glean_result = await use_llm_func(continue_prompt, history_messages=history)
            history += pack_user_ass_to_openai_messages(
                continue_prompt, glean_result, using_amazon_bedrock
            )
            final_result += glean_result
            if now_glean_index == entity_extract_max_gleaning - 1:
                break
            if_loop_result: str = await use_llm_func(if_loop_prompt, history_messages=history)
            if if_loop_result.strip().strip('"').strip("'").lower() != "yes":
                break

        entities, relationships = await _parse_legacy_extraction_records(
            final_result, chunk_key, context_base, using_amazon_bedrock
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
        for entity in entities.values():
            entity_id = generate_stable_entity_id(entity["entity_name"], entity["entity_type"])
            target = manifest_entities.setdefault(
                entity_id,
                {
                    "entity_name": entity["entity_name"],
                    "entity_type": entity["entity_type"],
                    "descriptions": [],
                    "source_chunk_ids": [],
                },
            )
            target["descriptions"].extend(entity["descriptions"])
            target["source_chunk_ids"].extend(entity["source_chunk_ids"])
        for relationship_id, relationship in relationships.items():
            target = manifest_relationships.setdefault(
                relationship_id,
                {
                    "src_entity_id": relationship["src_entity_id"],
                    "tgt_entity_id": relationship["tgt_entity_id"],
                    "relation_type": relationship.get("relation_type", "related"),
                    "descriptions": [],
                    "weight": 0.0,
                    "source_chunk_ids": [],
                },
            )
            target["descriptions"].extend(relationship["descriptions"])
            target["weight"] += relationship["weight"]
            target["source_chunk_ids"].extend(relationship["source_chunk_ids"])
    return _normalize_document_manifest(manifest)
