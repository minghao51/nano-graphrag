import asyncio
import json
import re
from collections import Counter, defaultdict
from typing import Any, Callable, Optional, Union

from .._utils import (
    TokenizerWrapper,
    clean_str,
    generate_stable_entity_id,
    generate_stable_relationship_id,
    is_float_regex,
    logger,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
)
from ..base import BaseGraphStorage, BaseKVStorage, BaseVectorStorage, TextChunkSchema
from ..prompt import GRAPH_FIELD_SEP, PROMPTS


def _join_unique(values: list[str]) -> str:
    return GRAPH_FIELD_SEP.join(sorted(set(v for v in values if v)))


def _normalize_entity_name(value: str) -> str:
    return clean_str(value.upper())


def _normalize_entity_type(value: str) -> str:
    return clean_str(value.upper()) or '"UNKNOWN"'


async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
    tokenizer_wrapper: TokenizerWrapper,
) -> str:
    use_llm_func: Callable[..., Any] = global_config["cheap_model_func"]
    llm_max_tokens = global_config["cheap_model_max_token_size"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]

    tokens = tokenizer_wrapper.encode(description)
    if len(tokens) < summary_max_tokens:
        return description
    prompt_template = PROMPTS["summarize_entity_descriptions"]

    use_description = tokenizer_wrapper.decode(tokens[:llm_max_tokens])
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
    )
    use_prompt = prompt_template.format(**context_base)
    logger.debug(f"Trigger summary: {entity_or_relation_name}")
    summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)
    return summary


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None
    entity_name = _normalize_entity_name(record_attributes[1])
    if not entity_name.strip():
        return None
    entity_type = _normalize_entity_type(record_attributes[2])
    entity_description = clean_str(record_attributes[3])
    entity_source_id = chunk_key
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=entity_source_id,
    )


async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
        return None
    source = _normalize_entity_name(record_attributes[1])
    target = _normalize_entity_name(record_attributes[2])
    edge_description = clean_str(record_attributes[3])
    edge_source_id = chunk_key
    weight = float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    return dict(
        src_name=source,
        tgt_name=target,
        weight=weight,
        description=edge_description,
        source_id=edge_source_id,
    )


async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
    tokenizer_wrapper,
):
    already_entitiy_types = []
    already_source_ids = []
    already_description = []

    already_node = await knowledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        already_entitiy_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])

    entity_type = sorted(
        Counter([dp["entity_type"] for dp in nodes_data] + already_entitiy_types).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    description = await _handle_entity_relation_summary(
        entity_name, description, global_config, tokenizer_wrapper
    )
    node_data = dict(
        entity_type=entity_type,
        description=description,
        source_id=source_id,
    )
    await knowledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    return node_data


async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
    tokenizer_wrapper,
):
    already_weights = []
    already_source_ids = []
    already_description = []
    already_order = []
    if await knowledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knowledge_graph_inst.get_edge(src_id, tgt_id)
        if already_edge is not None:
            already_weights.append(already_edge["weight"])
            already_source_ids.extend(
                split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
            )
            already_description.append(already_edge["description"])
            already_order.append(already_edge.get("order", 1))

    order = min([dp.get("order", 1) for dp in edges_data] + already_order)
    weight = sum([dp["weight"] for dp in edges_data] + already_weights)
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in edges_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in edges_data] + already_source_ids)
    )
    for need_insert_id in [src_id, tgt_id]:
        if not (await knowledge_graph_inst.has_node(need_insert_id)):
            await knowledge_graph_inst.upsert_node(
                need_insert_id,
                node_data={
                    "source_id": source_id,
                    "description": description,
                    "entity_type": '"UNKNOWN"',
                },
            )
    description = await _handle_entity_relation_summary(
        f"{src_id}->{tgt_id}", description, global_config, tokenizer_wrapper
    )
    await knowledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(weight=weight, description=description, source_id=source_id, order=order),
    )


def _upsert_document_entity(
    entities: dict[str, dict],
    entity_name: str,
    entity_type: str,
    description: str,
    chunk_key: str,
) -> str:
    entity_id = generate_stable_entity_id(entity_name, entity_type)
    entity_entry = entities.setdefault(
        entity_id,
        {
            "entity_name": entity_name,
            "entity_type": entity_type,
            "descriptions": [],
            "source_chunk_ids": [],
        },
    )
    entity_entry["entity_name"] = entity_name
    entity_entry["entity_type"] = entity_type
    entity_entry["descriptions"].append(description)
    entity_entry["source_chunk_ids"].append(chunk_key)
    return entity_id


def _upsert_document_relationship(
    relationships: dict[str, dict],
    src_entity_id: str,
    tgt_entity_id: str,
    description: str,
    weight: float,
    chunk_key: str,
    relation_type: str = "related",
) -> str:
    relationship_id = generate_stable_relationship_id(src_entity_id, tgt_entity_id, relation_type)
    relationship_entry = relationships.setdefault(
        relationship_id,
        {
            "src_entity_id": src_entity_id,
            "tgt_entity_id": tgt_entity_id,
            "relation_type": relation_type,
            "descriptions": [],
            "weight": 0.0,
            "source_chunk_ids": [],
        },
    )
    relationship_entry["descriptions"].append(description)
    relationship_entry["weight"] += weight
    relationship_entry["source_chunk_ids"].append(chunk_key)
    return relationship_id


def _normalize_document_manifest(manifest: dict) -> dict:
    normalized_entities = {}
    for entity_id, entity in manifest.get("entities", {}).items():
        normalized_entities[entity_id] = {
            "entity_name": entity["entity_name"],
            "entity_type": entity["entity_type"],
            "descriptions": sorted(set(entity.get("descriptions", []))),
            "source_chunk_ids": sorted(set(entity.get("source_chunk_ids", []))),
        }

    normalized_relationships = {}
    for relationship_id, relationship in manifest.get("relationships", {}).items():
        normalized_relationships[relationship_id] = {
            "src_entity_id": relationship["src_entity_id"],
            "tgt_entity_id": relationship["tgt_entity_id"],
            "relation_type": relationship.get("relation_type", "related"),
            "descriptions": sorted(set(relationship.get("descriptions", []))),
            "weight": relationship.get("weight", 0.0),
            "source_chunk_ids": sorted(set(relationship.get("source_chunk_ids", []))),
        }

    return {
        "content_hash": manifest.get("content_hash"),
        "chunk_ids": sorted(set(manifest.get("chunk_ids", []))),
        "entities": normalized_entities,
        "relationships": normalized_relationships,
    }


def _combine_entity_contributions(contributions: list[dict]) -> Optional[dict]:
    if not contributions:
        return None
    entity_name = contributions[-1]["entity_name"]
    entity_type = Counter([c["entity_type"] for c in contributions]).most_common(1)[0][0]
    descriptions = []
    source_chunk_ids = []
    for contribution in contributions:
        descriptions.extend(contribution.get("descriptions", []))
        source_chunk_ids.extend(contribution.get("source_chunk_ids", []))
    return {
        "entity_name": entity_name,
        "entity_type": entity_type,
        "description": _join_unique(descriptions),
        "source_id": _join_unique(source_chunk_ids),
    }


def _select_canonical_entity_id(
    entity_ids: list[str], contributions: list[dict], preferred_entity_id: Optional[str] = None
) -> str:
    if preferred_entity_id is not None:
        return preferred_entity_id
    type_counts = Counter(contribution["entity_type"] for contribution in contributions)
    preferred_types = sorted(
        type_counts.items(),
        key=lambda item: (item[0] != '"UNKNOWN"', item[1], item[0]),
        reverse=True,
    )
    for entity_type, _ in preferred_types:
        candidate_id = generate_stable_entity_id(contributions[-1]["entity_name"], entity_type)
        if candidate_id in entity_ids:
            return candidate_id
    return sorted(entity_ids)[0]


def _combine_relationship_contributions(contributions: list[dict]) -> Optional[dict]:
    if not contributions:
        return None
    first = contributions[0]
    descriptions = []
    source_chunk_ids = []
    total_weight = 0.0
    for contribution in contributions:
        descriptions.extend(contribution.get("descriptions", []))
        source_chunk_ids.extend(contribution.get("source_chunk_ids", []))
        total_weight += float(contribution.get("weight", 0.0))
    return {
        "src_entity_id": first["src_entity_id"],
        "tgt_entity_id": first["tgt_entity_id"],
        "description": _join_unique(descriptions),
        "source_id": _join_unique(source_chunk_ids),
        "weight": total_weight,
        "order": 1,
        "relation_type": first.get("relation_type", "related"),
    }


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

    # Concurrency limit for entity extraction
    max_concurrent = global_config.get("extraction_max_async", 16)
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _process_chunk_with_legacy_prompt(
        chunk_key: str, content: str
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
        continue_prompt = PROMPTS["entiti_continue_extraction"]
        if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]
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

            relationship = await _handle_single_relationship_extraction(
                record_attributes, chunk_key
            )
            if relationship is None:
                continue
            src_name = relationship["src_name"]
            tgt_name = relationship["tgt_name"]
            if src_name not in entity_name_to_id:
                entity_name_to_id[src_name] = _upsert_document_entity(
                    entities, src_name, '"UNKNOWN"', relationship["description"], chunk_key
                )
            if tgt_name not in entity_name_to_id:
                entity_name_to_id[tgt_name] = _upsert_document_entity(
                    entities, tgt_name, '"UNKNOWN"', relationship["description"], chunk_key
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
                    chunk_key, content
                )
                already_processed += 1
                already_entities += len(entities)
                already_relations += len(relationships)
                now_ticks = PROMPTS["process_tickers"][
                    already_processed % len(PROMPTS["process_tickers"])
                ]
                print(
                    f"{now_ticks} Processed {already_processed}({already_processed * 100 // len(ordered_chunks)}%) chunks, {already_entities} entities, {already_relations} relations\r",
                    end="",
                    flush=True,
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
                        '"UNKNOWN"',
                        relationship.description,
                        chunk_key,
                    )
                if tgt_name not in entity_name_to_id:
                    entity_name_to_id[tgt_name] = _upsert_document_entity(
                        entities,
                        tgt_name,
                        '"UNKNOWN"',
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
        now_ticks = PROMPTS["process_tickers"][already_processed % len(PROMPTS["process_tickers"])]
        print(
            f"{now_ticks} Processed {already_processed}({already_processed * 100 // len(ordered_chunks)}%) chunks, {already_entities} entities, {already_relations} relations\r",
            end="",
            flush=True,
        )
        return entities, relationships

    async def _process_single_content_with_semaphore(chunk_item):
        async with semaphore:
            return await _process_single_content(chunk_item)

    results = await asyncio.gather(
        *[_process_single_content_with_semaphore(c) for c in ordered_chunks]
    )
    print()
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
    continue_prompt = PROMPTS["entiti_continue_extraction"]
    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]

    already_processed = 0
    already_entities = 0
    already_relations = 0

    # Concurrency limit for entity extraction
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

            relationship = await _handle_single_relationship_extraction(
                record_attributes, chunk_key
            )
            if relationship is None:
                continue
            src_name = relationship["src_name"]
            tgt_name = relationship["tgt_name"]
            if src_name not in entity_name_to_id:
                entity_name_to_id[src_name] = _upsert_document_entity(
                    entities, src_name, '"UNKNOWN"', relationship["description"], chunk_key
                )
            if tgt_name not in entity_name_to_id:
                entity_name_to_id[tgt_name] = _upsert_document_entity(
                    entities, tgt_name, '"UNKNOWN"', relationship["description"], chunk_key
                )
            _upsert_document_relationship(
                relationships,
                entity_name_to_id[src_name],
                entity_name_to_id[tgt_name],
                relationship["description"],
                relationship["weight"],
                chunk_key,
            )

        already_processed += 1
        already_entities += len(entities)
        already_relations += len(relationships)
        now_ticks = PROMPTS["process_tickers"][already_processed % len(PROMPTS["process_tickers"])]
        print(
            f"{now_ticks} Processed {already_processed}({already_processed * 100 // len(ordered_chunks)}%) chunks,  {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return entities, relationships

    async def _process_single_content_with_semaphore(chunk_item):
        async with semaphore:
            return await _process_single_content(chunk_item)

    results = await asyncio.gather(
        *[_process_single_content_with_semaphore(c) for c in ordered_chunks]
    )
    print()
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


async def extract_document_entity_relationships(
    chunks: dict[str, TextChunkSchema],
    tokenizer_wrapper,
    global_config: dict,
    using_amazon_bedrock: bool = False,
) -> dict:
    manifest = (
        await extract_document_entity_relationships_structured(
            chunks,
            tokenizer_wrapper,
            global_config,
        )
        if global_config.get("_use_structured_extraction", False)
        else await extract_document_entity_relationships_legacy(
            chunks,
            tokenizer_wrapper,
            global_config,
            using_amazon_bedrock,
        )
    )
    if not manifest["entities"]:
        logger.warning("Didn't extract any entities, maybe your LLM is not working")
    return manifest


async def rebuild_knowledge_graph_for_documents(
    document_index: BaseKVStorage[dict],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: Optional[BaseVectorStorage],
    tokenizer_wrapper,
    global_config: dict,
    old_document_manifests: dict[str, dict],
    new_document_manifests: dict[str, dict],
):
    affected_entity_ids: set[str] = set()
    affected_relationship_ids: set[str] = set()
    affected_entity_names: set[str] = set()
    preferred_entity_ids_by_name: dict[str, str] = {}
    removed_relationship_lookup: dict[str, tuple[str, str]] = {}
    all_changed_doc_ids = set(old_document_manifests.keys()).union(new_document_manifests.keys())
    for doc_id in all_changed_doc_ids:
        old_manifest = old_document_manifests.get(doc_id) or {}
        new_manifest = new_document_manifests.get(doc_id) or {}
        old_entities = old_manifest.get("entities", {})
        new_entities = new_manifest.get("entities", {})
        old_relationships = old_manifest.get("relationships", {})
        new_relationships = new_manifest.get("relationships", {})
        affected_entity_ids.update(old_entities.keys())
        affected_entity_ids.update(new_entities.keys())
        for entity_id, entity in old_entities.items():
            entity_name = entity.get("entity_name")
            if not entity_name:
                continue
            existing_id = preferred_entity_ids_by_name.get(entity_name)
            if existing_id is None or entity.get("entity_type") != '"UNKNOWN"':
                preferred_entity_ids_by_name[entity_name] = entity_id
        affected_entity_names.update(
            entity["entity_name"] for entity in old_entities.values() if entity.get("entity_name")
        )
        affected_entity_names.update(
            entity["entity_name"] for entity in new_entities.values() if entity.get("entity_name")
        )
        affected_relationship_ids.update(old_relationships.keys())
        affected_relationship_ids.update(new_relationships.keys())
        for relationship_id, relationship in old_relationships.items():
            if relationship_id not in new_relationships:
                removed_relationship_lookup[relationship_id] = (
                    relationship["src_entity_id"],
                    relationship["tgt_entity_id"],
                )

    if not affected_entity_ids and not affected_relationship_ids:
        return knowledge_graph_inst

    document_keys = await document_index.all_keys()
    all_documents = await document_index.get_by_ids(document_keys)
    entity_contributions: dict[str, list[dict]] = defaultdict(list)
    relationship_contributions: list[dict] = []

    for document in all_documents:
        if document is None:
            continue
        document_entities = document.get("entities", {})
        document_entity_names = {
            entity_id: entity.get("entity_name")
            for entity_id, entity in document_entities.items()
            if entity.get("entity_name")
        }
        for entity_id, entity in document.get("entities", {}).items():
            if (
                entity_id in affected_entity_ids
                or entity.get("entity_name") in affected_entity_names
            ):
                entity_contributions[entity_id].append(entity)
        for relationship_id, relationship in document.get("relationships", {}).items():
            src_name = document_entity_names.get(relationship["src_entity_id"])
            tgt_name = document_entity_names.get(relationship["tgt_entity_id"])
            if (
                relationship_id in affected_relationship_ids
                or src_name in affected_entity_names
                or tgt_name in affected_entity_names
            ):
                relationship_contributions.append(relationship)

    entity_name_groups: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for entity_id, contributions in entity_contributions.items():
        for contribution in contributions:
            entity_name = contribution.get("entity_name")
            if entity_name:
                entity_name_groups[entity_name].append((entity_id, contribution))

    entity_id_remap: dict[str, str] = {}
    combined_entities = {}
    for entity_name, grouped_contributions in entity_name_groups.items():
        grouped_entity_ids = [entity_id for entity_id, _ in grouped_contributions]
        grouped_entities = [contribution for _, contribution in grouped_contributions]
        canonical_entity_id = _select_canonical_entity_id(
            grouped_entity_ids,
            grouped_entities,
            preferred_entity_id=preferred_entity_ids_by_name.get(entity_name),
        )
        combined_entities[canonical_entity_id] = _combine_entity_contributions(grouped_entities)
        for entity_id in grouped_entity_ids:
            entity_id_remap[entity_id] = canonical_entity_id

    canonical_relationship_contributions: dict[str, list[dict]] = defaultdict(list)
    for relationship in relationship_contributions:
        src_entity_id = entity_id_remap.get(
            relationship["src_entity_id"], relationship["src_entity_id"]
        )
        tgt_entity_id = entity_id_remap.get(
            relationship["tgt_entity_id"], relationship["tgt_entity_id"]
        )
        canonical_relationship_id = generate_stable_relationship_id(
            src_entity_id,
            tgt_entity_id,
            relationship.get("relation_type", "related"),
        )
        canonical_relationship_contributions[canonical_relationship_id].append(
            {
                **relationship,
                "src_entity_id": src_entity_id,
                "tgt_entity_id": tgt_entity_id,
            }
        )

    combined_relationships = {
        relationship_id: _combine_relationship_contributions(contributions)
        for relationship_id, contributions in canonical_relationship_contributions.items()
    }

    entity_ids_to_refresh = set(affected_entity_ids).union(entity_id_remap.values())
    for entity_id in entity_ids_to_refresh:
        combined = combined_entities.get(entity_id_remap.get(entity_id, entity_id))
        if entity_id_remap.get(entity_id, entity_id) != entity_id:
            await knowledge_graph_inst.delete_node(entity_id)
            if entity_vdb is not None:
                await entity_vdb.delete([entity_id])
            continue
        if combined is None:
            await knowledge_graph_inst.delete_node(entity_id)
            if entity_vdb is not None:
                await entity_vdb.delete([entity_id])
            continue
        combined["description"] = await _handle_entity_relation_summary(
            combined["entity_name"],
            combined["description"],
            global_config,
            tokenizer_wrapper,
        )
        await knowledge_graph_inst.upsert_node(entity_id, combined)
        if entity_vdb is not None:
            await entity_vdb.upsert(
                {
                    entity_id: {
                        "content": combined["entity_name"] + combined["description"],
                        "entity_name": combined["entity_name"],
                    }
                }
            )

    relationship_ids_to_refresh = set(affected_relationship_ids).union(
        combined_relationships.keys()
    )
    for relationship_id in relationship_ids_to_refresh:
        removed_edge = removed_relationship_lookup.get(relationship_id)
        if removed_edge is not None:
            removed_edge = (
                entity_id_remap.get(removed_edge[0], removed_edge[0]),
                entity_id_remap.get(removed_edge[1], removed_edge[1]),
            )
        combined = combined_relationships.get(relationship_id)
        if combined is None and removed_edge is not None:
            canonical_relationship_id = generate_stable_relationship_id(
                removed_edge[0], removed_edge[1]
            )
            combined = combined_relationships.get(canonical_relationship_id)
        if combined is None:
            if removed_edge is not None:
                await knowledge_graph_inst.delete_edge(removed_edge[0], removed_edge[1])
            continue
        for endpoint in [combined["src_entity_id"], combined["tgt_entity_id"]]:
            if not await knowledge_graph_inst.has_node(endpoint):
                endpoint_combined = combined_entities.get(endpoint)
                if endpoint_combined is not None:
                    await knowledge_graph_inst.upsert_node(endpoint, endpoint_combined)
        combined["description"] = await _handle_entity_relation_summary(
            f"{combined['src_entity_id']}->{combined['tgt_entity_id']}",
            combined["description"],
            global_config,
            tokenizer_wrapper,
        )
        await knowledge_graph_inst.upsert_edge(
            combined["src_entity_id"],
            combined["tgt_entity_id"],
            {
                "description": combined["description"],
                "weight": combined["weight"],
                "source_id": combined["source_id"],
                "order": combined["order"],
                "relationship_id": relationship_id,
                "relation_type": combined["relation_type"],
            },
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
