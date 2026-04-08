import re
from collections import Counter
from typing import Any, Callable, Optional

from .._utils import (
    TokenizerWrapper,
    clean_str,
    deserialize_source_ids,
    generate_stable_entity_id,
    generate_stable_relationship_id,
    is_float_regex,
    logger,
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



def _join_unique(values: list[str]) -> str:
    return GRAPH_FIELD_SEP.join(sorted(set(v for v in values if v)))


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
        key=lambda item: (item[0] != UNKNOWN_ENTITY_TYPE, item[1], item[0]),
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
