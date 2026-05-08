from __future__ import annotations

import asyncio
from typing import Any, Callable

from ..base import TextChunkSchema
from ..prompt import PROMPTS
from .extraction_common import (
    _ExtractionProgress,
    _merge_results_into_manifest,
    _parse_legacy_extraction_records,
    _run_gleaning_loop,
)


async def extract_document_entity_relationships_legacy(
    chunks: dict[str, TextChunkSchema],
    tokenizer_wrapper,
    global_config: dict,
    using_amazon_bedrock: bool = False,
) -> dict:
    quality = global_config.get("entity_extraction_quality", "balanced")
    if quality == "fast":
        use_llm_func: Callable[..., Any] = global_config["cheap_model_func"]
    else:
        use_llm_func = global_config["best_model_func"]
    entity_extract_max_gleaning = (
        0 if quality == "fast" else global_config["entity_extract_max_gleaning"]
    )
    ordered_chunks = list(chunks.items())

    if global_config.get("enable_temporal_extraction", False):
        entity_extract_prompt = PROMPTS["entity_extraction_temporal"]
    else:
        entity_extract_prompt = PROMPTS["entity_extraction"]
    context_base = {
        "tuple_delimiter": PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        "record_delimiter": PROMPTS["DEFAULT_RECORD_DELIMITER"],
        "completion_delimiter": PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        "entity_types": ",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
    }

    progress = _ExtractionProgress(len(ordered_chunks))

    max_concurrent = global_config.get("extraction_max_async", 16)
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        chunk_key = chunk_key_dp[0]
        content = chunk_key_dp[1]["content"]
        final_result = await _run_gleaning_loop(
            use_llm_func,
            content,
            entity_extract_prompt,
            context_base,
            entity_extract_max_gleaning,
            using_amazon_bedrock,
        )
        entities, relationships = await _parse_legacy_extraction_records(
            final_result, chunk_key, context_base, using_amazon_bedrock
        )
        progress.update(len(entities), len(relationships))
        return entities, relationships

    async def _process_single_content_with_semaphore(chunk_item):
        async with semaphore:
            return await _process_single_content(chunk_item)

    results = await asyncio.gather(
        *[_process_single_content_with_semaphore(c) for c in ordered_chunks]
    )
    return _merge_results_into_manifest(results, list(chunks.keys()))
