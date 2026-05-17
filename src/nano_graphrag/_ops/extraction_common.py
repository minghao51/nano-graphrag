from __future__ import annotations

import re
import time
from collections import Counter
from collections.abc import Callable
from typing import Any

from .._utils import (
    TokenizerWrapper,
    clean_str,
    deserialize_source_ids,
    generate_stable_entity_id,
    generate_stable_relationship_id,
    is_float_regex,
    logger,
    pack_user_ass_to_openai_messages,
    serialize_source_ids,
    split_string_by_multi_markers,
)
from ..base import BaseGraphStorage
from ..prompt import GRAPH_FIELD_SEP, PROMPTS

UNKNOWN_ENTITY_TYPE = '"UNKNOWN"'


async def _parse_legacy_extraction_records(
    final_result: str,
    chunk_key: str,
    context_base: dict,
    using_amazon_bedrock: bool = False,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
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
                aliases=entity.get("aliases"),
                event_date=entity.get("event_date"),
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
            temporal_context=relationship.get("temporal_context"),
            valid_from=relationship.get("valid_from"),
            valid_to=relationship.get("valid_to"),
        )
    return entities, relationships


def _join_unique(values: list[str]) -> str:
    return GRAPH_FIELD_SEP.join(sorted({v for v in values if v}))


def _normalize_entity_name(value: str) -> str:
    return clean_str(value.upper())


def _normalize_entity_type(value: str) -> str:
    return clean_str(value.upper()) or UNKNOWN_ENTITY_TYPE


async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
    tokenizer_wrapper: TokenizerWrapper,
) -> str:
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]

    tokens = tokenizer_wrapper.encode(description)
    if len(tokens) < summary_max_tokens:
        return description

    quality = global_config.get("entity_extraction_quality", "balanced")
    if quality == "fast":
        return tokenizer_wrapper.decode(tokens[:summary_max_tokens])

    use_llm_func: Callable[..., Any] = global_config["cheap_model_func"]
    llm_max_tokens = global_config["cheap_model_max_token_size"]

    prompt_template = PROMPTS["summarize_entity_descriptions"]

    use_description = tokenizer_wrapper.decode(tokens[:llm_max_tokens])
    context_base = {
        "entity_name": entity_or_relation_name,
        "description_list": use_description.split(GRAPH_FIELD_SEP),
    }
    use_prompt = prompt_template.format(**context_base)
    logger.debug("trigger_summary", entity_name=entity_or_relation_name)
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
    entity_aliases_raw = clean_str(record_attributes[4]) if len(record_attributes) >= 5 else ""
    entity_aliases = [
        a.strip()
        for a in entity_aliases_raw.split(",")
        if a.strip() and a.strip().lower() != entity_name.lower()
    ]
    event_date = clean_str(record_attributes[5]) if len(record_attributes) >= 6 else None
    if event_date and not event_date.strip():
        event_date = None
    entity_source_id = chunk_key
    return {
        "entity_name": entity_name,
        "entity_type": entity_type,
        "description": entity_description,
        "aliases": entity_aliases,
        "source_id": entity_source_id,
        "event_date": event_date,
    }


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
    temporal_context = clean_str(record_attributes[5]) if len(record_attributes) >= 7 else None
    valid_from = clean_str(record_attributes[6]) if len(record_attributes) >= 8 else None
    valid_to = clean_str(record_attributes[7]) if len(record_attributes) >= 9 else None
    if temporal_context and not temporal_context.strip():
        temporal_context = None
    if valid_from and not valid_from.strip():
        valid_from = None
    if valid_to and not valid_to.strip():
        valid_to = None
    return {
        "src_name": source,
        "tgt_name": target,
        "weight": weight,
        "description": edge_description,
        "source_id": edge_source_id,
        "temporal_context": temporal_context,
        "valid_from": valid_from,
        "valid_to": valid_to,
    }


async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
    tokenizer_wrapper,
):
    already_entity_types = []
    already_source_ids = []
    already_description = []

    already_node = await knowledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        already_entity_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])

    type_counter = Counter(dp["entity_type"] for dp in nodes_data)
    type_counter.update(already_entity_types)
    if not type_counter:
        entity_type = UNKNOWN_ENTITY_TYPE
    else:
        entity_type = type_counter.most_common(1)[0][0]
    description = GRAPH_FIELD_SEP.join(
        sorted({dp["description"] for dp in nodes_data}.union(already_description))
    )
    all_source_ids = set()
    for dp in nodes_data:
        all_source_ids.update(deserialize_source_ids(dp["source_id"]))
    all_source_ids.update(already_source_ids)
    source_id = serialize_source_ids(list(all_source_ids))
    description = await _handle_entity_relation_summary(
        entity_name, description, global_config, tokenizer_wrapper
    )
    node_data = {
        "entity_type": entity_type,
        "description": description,
        "source_id": source_id,
    }
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
    already_temporal_context = None
    already_valid_from = None
    already_valid_to = None
    already_relation_type = "related_to"
    already_confidence = 0.8
    if await knowledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knowledge_graph_inst.get_edge(src_id, tgt_id)
        if already_edge is not None:
            already_weights.append(already_edge["weight"])
            already_source_ids.extend(
                split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
            )
            already_description.append(already_edge["description"])
            already_order.append(already_edge.get("order", 1))
            already_temporal_context = already_edge.get("temporal_context")
            already_valid_from = already_edge.get("valid_from")
            already_valid_to = already_edge.get("valid_to")
            already_relation_type = already_edge.get("relation_type", "related_to")
            already_confidence = already_edge.get("confidence", 0.8)

    order = min([dp.get("order", 1) for dp in edges_data] + already_order)
    weight = sum(dp["weight"] for dp in edges_data) + sum(already_weights)
    description = GRAPH_FIELD_SEP.join(
        sorted({dp["description"] for dp in edges_data}.union(already_description))
    )
    all_source_ids = set()
    for dp in edges_data:
        all_source_ids.update(deserialize_source_ids(dp["source_id"]))
    all_source_ids.update(already_source_ids)
    source_id = serialize_source_ids(list(all_source_ids))
    for need_insert_id in [src_id, tgt_id]:
        if not (await knowledge_graph_inst.has_node(need_insert_id)):
            await knowledge_graph_inst.upsert_node(
                need_insert_id,
                node_data={
                    "source_id": source_id,
                    "description": description,
                    "entity_type": UNKNOWN_ENTITY_TYPE,
                },
            )
    description = await _handle_entity_relation_summary(
        f"{src_id}->{tgt_id}", description, global_config, tokenizer_wrapper
    )
    temporal_context = None
    valid_from = None
    valid_to = None
    for dp in edges_data:
        if not temporal_context and dp.get("temporal_context"):
            temporal_context = dp["temporal_context"]
        if not valid_from and dp.get("valid_from"):
            valid_from = dp["valid_from"]
        if not valid_to and dp.get("valid_to"):
            valid_to = dp["valid_to"]
    if not temporal_context:
        temporal_context = already_temporal_context
    if not valid_from:
        valid_from = already_valid_from
    if not valid_to:
        valid_to = already_valid_to
    edge_data = {
        "weight": weight,
        "description": description,
        "source_id": source_id,
        "order": order,
    }
    if temporal_context:
        edge_data["temporal_context"] = temporal_context
    if valid_from:
        edge_data["valid_from"] = valid_from
    if valid_to:
        edge_data["valid_to"] = valid_to
    relation_types = [
        dp.get("relation_type", "related_to")
        for dp in edges_data
        if dp.get("relation_type", "related_to") != "related_to"
    ]
    if already_relation_type != "related_to":
        relation_types.append(already_relation_type)
    if relation_types:
        edge_data["relation_type"] = Counter(relation_types).most_common(1)[0][0]
    else:
        edge_data["relation_type"] = "related_to"
    confidences = [dp.get("confidence", 0.8) for dp in edges_data] + [already_confidence]
    if confidences:
        edge_data["confidence"] = max(confidences)
    await knowledge_graph_inst.upsert_edge(src_id, tgt_id, edge_data=edge_data)


def _upsert_document_entity(
    entities: dict[str, dict],
    entity_name: str,
    entity_type: str,
    description: str,
    chunk_key: str,
    aliases: list[str] | None = None,
    event_date: str | None = None,
) -> str:
    entity_id = generate_stable_entity_id(entity_name, entity_type)
    entity_entry = entities.setdefault(
        entity_id,
        {
            "entity_name": entity_name,
            "entity_type": entity_type,
            "descriptions": [],
            "source_chunk_ids": [],
            "aliases": [],
            "event_date": event_date,
        },
    )
    entity_entry["descriptions"].append(description)
    entity_entry["source_chunk_ids"].append(chunk_key)
    if aliases:
        existing = set(entity_entry.get("aliases", []))
        entity_entry["aliases"] = sorted(existing.union(a for a in aliases if a))
    if event_date and not entity_entry.get("event_date"):
        entity_entry["event_date"] = event_date
    return entity_id


def _upsert_document_relationship(
    relationships: dict[str, dict],
    src_entity_id: str,
    tgt_entity_id: str,
    description: str,
    weight: float,
    chunk_key: str,
    relation_type: str = "related_to",
    confidence: float = 0.8,
    temporal_context: str | None = None,
    valid_from: str | None = None,
    valid_to: str | None = None,
) -> str:
    from .._schemas import normalize_relation_type

    relation_type = normalize_relation_type(relation_type)
    relationship_id = generate_stable_relationship_id(
        src_entity_id, tgt_entity_id, relation_type, temporal_context=temporal_context
    )
    relationship_entry = relationships.setdefault(
        relationship_id,
        {
            "src_entity_id": src_entity_id,
            "tgt_entity_id": tgt_entity_id,
            "relation_type": relation_type,
            "descriptions": [],
            "weight": 0.0,
            "source_chunk_ids": [],
            "confidence": confidence,
            "temporal_context": temporal_context,
            "valid_from": valid_from,
            "valid_to": valid_to,
        },
    )
    relationship_entry["descriptions"].append(description)
    relationship_entry["weight"] += weight
    relationship_entry["source_chunk_ids"].append(chunk_key)
    if confidence > relationship_entry.get("confidence", 0.0):
        relationship_entry["confidence"] = confidence
    if temporal_context and not relationship_entry.get("temporal_context"):
        relationship_entry["temporal_context"] = temporal_context
    if valid_from and not relationship_entry.get("valid_from"):
        relationship_entry["valid_from"] = valid_from
    if valid_to and not relationship_entry.get("valid_to"):
        relationship_entry["valid_to"] = valid_to
    return relationship_id


def _normalize_document_manifest(manifest: dict) -> dict:
    normalized_entities = {}
    for entity_id, entity in manifest.get("entities", {}).items():
        normalized_entities[entity_id] = {
            "entity_name": entity["entity_name"],
            "entity_type": entity["entity_type"],
            "aliases": sorted(set(entity.get("aliases", []))),
            "descriptions": sorted(set(entity.get("descriptions", []))),
            "source_chunk_ids": sorted(set(entity.get("source_chunk_ids", []))),
            "event_date": entity.get("event_date"),
        }

    normalized_relationships = {}
    for relationship_id, relationship in manifest.get("relationships", {}).items():
        normalized_relationships[relationship_id] = {
            "src_entity_id": relationship["src_entity_id"],
            "tgt_entity_id": relationship["tgt_entity_id"],
            "relation_type": relationship.get("relation_type", "related_to"),
            "descriptions": sorted(set(relationship.get("descriptions", []))),
            "weight": relationship.get("weight", 0.0),
            "source_chunk_ids": sorted(set(relationship.get("source_chunk_ids", []))),
            "confidence": relationship.get("confidence", 0.8),
            "temporal_context": relationship.get("temporal_context"),
            "valid_from": relationship.get("valid_from"),
            "valid_to": relationship.get("valid_to"),
        }

    return {
        "content_hash": manifest.get("content_hash"),
        "chunk_ids": sorted(set(manifest.get("chunk_ids", []))),
        "entities": normalized_entities,
        "relationships": normalized_relationships,
    }


def _combine_entity_contributions(contributions: list[dict]) -> dict | None:
    if not contributions:
        return None
    entity_name_counts = Counter(c["entity_name"] for c in contributions)
    entity_name = entity_name_counts.most_common(1)[0][0]
    entity_type = Counter([c["entity_type"] for c in contributions]).most_common(1)[0][0]
    descriptions = []
    source_chunk_ids = []
    aliases = []
    event_date = None
    for contribution in contributions:
        aliases.extend(contribution.get("aliases", []))
        descriptions.extend(contribution.get("descriptions", []))
        source_chunk_ids.extend(contribution.get("source_chunk_ids", []))
        if not event_date and contribution.get("event_date"):
            event_date = contribution["event_date"]
    return {
        "entity_name": entity_name,
        "entity_type": entity_type,
        "aliases": sorted({a for a in aliases if a and a != entity_name}),
        "description": _join_unique(descriptions),
        "source_id": _join_unique(source_chunk_ids),
        "event_date": event_date,
    }


def _select_canonical_entity_id(
    entity_ids: list[str], contributions: list[dict], preferred_entity_id: str | None = None
) -> str:
    if preferred_entity_id is not None:
        return preferred_entity_id
    type_counts = Counter(contribution["entity_type"] for contribution in contributions)
    preferred_types = sorted(
        type_counts.items(),
        key=lambda item: (item[0] != UNKNOWN_ENTITY_TYPE, item[1], item[0]),
        reverse=True,
    )
    for entity_type, _ in preferred_types:
        candidate_id = generate_stable_entity_id(contributions[-1]["entity_name"], entity_type)
        if candidate_id in entity_ids:
            return candidate_id
    return sorted(entity_ids)[0]


def _combine_relationship_contributions(contributions: list[dict]) -> dict | None:
    if not contributions:
        return None
    first = contributions[0]
    descriptions = []
    source_chunk_ids = []
    total_weight = 0.0
    temporal_context = None
    valid_from = None
    valid_to = None
    for contribution in contributions:
        descriptions.extend(contribution.get("descriptions", []))
        source_chunk_ids.extend(contribution.get("source_chunk_ids", []))
        total_weight += float(contribution.get("weight", 0.0))
        if not temporal_context and contribution.get("temporal_context"):
            temporal_context = contribution["temporal_context"]
        if not valid_from and contribution.get("valid_from"):
            valid_from = contribution["valid_from"]
        if not valid_to and contribution.get("valid_to"):
            valid_to = contribution["valid_to"]
    return {
        "src_entity_id": first["src_entity_id"],
        "tgt_entity_id": first["tgt_entity_id"],
        "description": _join_unique(descriptions),
        "source_id": _join_unique(source_chunk_ids),
        "weight": total_weight,
        "order": 1,
        "relation_type": first.get("relation_type", "related_to"),
        "confidence": first.get("confidence", 0.8),
        "temporal_context": temporal_context,
        "valid_from": valid_from,
        "valid_to": valid_to,
    }


def _merge_results_into_manifest(
    results: list[tuple[dict, dict]],
    chunk_keys: list[str],
) -> dict:
    manifest_entities: dict[str, dict[str, Any]] = {}
    manifest_relationships: dict[str, dict[str, Any]] = {}
    for entities, relationships in results:
        for entity_id, entity in entities.items():
            target = manifest_entities.setdefault(
                entity_id,
                {
                    "entity_name": entity["entity_name"],
                    "entity_type": entity["entity_type"],
                    "descriptions": [],
                    "source_chunk_ids": [],
                    "aliases": [],
                    "event_date": entity.get("event_date"),
                },
            )
            target["descriptions"].extend(entity.get("descriptions", []))
            target["source_chunk_ids"].extend(entity.get("source_chunk_ids", []))
            existing_aliases = set(target.get("aliases", []))
            new_aliases = existing_aliases.union(a for a in entity.get("aliases", []) if a)
            target["aliases"] = sorted(new_aliases)
            if entity.get("event_date") and not target.get("event_date"):
                target["event_date"] = entity["event_date"]
        for relationship_id, relationship in relationships.items():
            target = manifest_relationships.setdefault(
                relationship_id,
                {
                    "src_entity_id": relationship["src_entity_id"],
                    "tgt_entity_id": relationship["tgt_entity_id"],
                    "relation_type": relationship.get("relation_type", "related_to"),
                    "descriptions": [],
                    "weight": 0.0,
                    "source_chunk_ids": [],
                    "confidence": relationship.get("confidence", 0.8),
                    "temporal_context": relationship.get("temporal_context"),
                    "valid_from": relationship.get("valid_from"),
                    "valid_to": relationship.get("valid_to"),
                },
            )
            target["descriptions"].extend(relationship.get("descriptions", []))
            target["weight"] += relationship.get("weight", 0.0)
            target["source_chunk_ids"].extend(relationship.get("source_chunk_ids", []))
            if relationship.get("temporal_context") and not target.get("temporal_context"):
                target["temporal_context"] = relationship["temporal_context"]
            if relationship.get("valid_from") and not target.get("valid_from"):
                target["valid_from"] = relationship["valid_from"]
            if relationship.get("valid_to") and not target.get("valid_to"):
                target["valid_to"] = relationship["valid_to"]
    manifest = {
        "chunk_ids": chunk_keys,
        "entities": manifest_entities,
        "relationships": manifest_relationships,
    }
    return _normalize_document_manifest(manifest)


async def _run_gleaning_loop(
    use_llm_func: Callable,
    content: str,
    entity_extract_prompt: str,
    context_base: dict,
    max_gleaning: int,
    using_amazon_bedrock: bool = False,
) -> str:
    continue_prompt = PROMPTS["entity_continue_extraction"]
    if_loop_prompt = PROMPTS["entity_if_loop_extraction"]
    hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)
    final_result = await use_llm_func(hint_prompt)
    if isinstance(final_result, list):
        final_result = final_result[0]["text"]
    history = pack_user_ass_to_openai_messages(hint_prompt, final_result, using_amazon_bedrock)
    for now_glean_index in range(max_gleaning):
        glean_result = await use_llm_func(continue_prompt, history_messages=history)
        history += pack_user_ass_to_openai_messages(
            continue_prompt, glean_result, using_amazon_bedrock
        )
        final_result += glean_result
        if now_glean_index == max_gleaning - 1:
            break
        if_loop_result: str = await use_llm_func(if_loop_prompt, history_messages=history)
        if if_loop_result.strip().strip('"').strip("'").lower() != "yes":
            break
    return final_result


class _ExtractionProgress:
    def __init__(self, total: int):
        self.total = total
        self.processed = 0
        self.entities = 0
        self.relations = 0
        self._start: float | None = None

    def update(self, num_entities: int, num_relations: int):
        if self._start is None:
            self._start = time.time()
        self.processed += 1
        self.entities += num_entities
        self.relations += num_relations
        if self.processed % 10 == 0 or self.processed >= self.total:
            elapsed = time.time() - self._start if self._start else 0
            logger.info(
                "extraction_chunk_progress",
                processed=self.processed,
                total=self.total,
                pct=self.processed * 100 // self.total,
                entities=self.entities,
                relations=self.relations,
                elapsed_s=round(elapsed, 1),
            )
