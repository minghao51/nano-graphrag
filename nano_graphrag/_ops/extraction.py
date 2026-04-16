import json
from typing import Optional

from .._utils import generate_stable_relationship_id
from .extraction_common import (
    _combine_entity_contributions,
    _combine_relationship_contributions,
    _handle_single_entity_extraction,
    _handle_single_relationship_extraction,
    _merge_edges_then_upsert,
    _merge_nodes_then_upsert,
    _normalize_document_manifest,
)
from .extraction_legacy import extract_document_entity_relationships_legacy
from .extraction_rebuild import rebuild_knowledge_graph_for_documents
from .extraction_structured import extract_document_entity_relationships_structured
from .extraction_writeback import extract_entities, extract_entities_structured


async def extract_document_entity_relationships(
    chunks,
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
        from .._utils import logger

        logger.warning("Didn't extract any entities, maybe your LLM is not working")
        return manifest

    # Check for suspiciously low entity counts
    entity_count = len(manifest["entities"])
    chunk_count = len(manifest.get("chunk_ids", []))
    entity_count_min_ratio = global_config.get("entity_count_min_ratio", 2.0)
    entity_count_min_absolute = global_config.get("entity_count_min_absolute", 3)
    expected_min_entities = max(
        entity_count_min_absolute, int(chunk_count * entity_count_min_ratio)
    )

    if entity_count < expected_min_entities:
        from .._utils import logger

        logger.warning(
            f"Low entity count detected: {entity_count} entities from {chunk_count} chunks "
            f"(expected at least {expected_min_entities}). "
            f"This may indicate poor extraction quality. Consider: "
            f"1) Using entity_extraction_quality='balanced' or 'thorough' "
            f"2) Using a higher quality LLM model "
            f"3) Checking if your documents contain sufficient named entities"
        )
    manifest = await _enrich_manifest_aliases(manifest, chunks, global_config)
    return await _apply_entity_linking(manifest, global_config)


async def _enrich_manifest_aliases(manifest: dict, chunks, global_config: dict) -> dict:
    from .extraction_writeback import _extract_aliases_for_entity

    enriched_entities = {}
    for entity_id, entity in manifest["entities"].items():
        aliases = await _extract_aliases_for_entity(
            entity["entity_name"],
            entity["entity_type"],
            entity.get("source_chunk_ids", []),
            chunks,
            global_config,
        )
        enriched_entities[entity_id] = {**entity, "aliases": sorted(set(aliases))}
    return {**manifest, "entities": enriched_entities}


async def _apply_entity_linking(manifest: dict, global_config: dict) -> dict:
    entity_registry = global_config.get("entity_registry")
    if entity_registry is None or not manifest.get("entities"):
        return manifest

    entity_id_remap: dict[str, str] = {}
    grouped_entities: dict[str, list[dict]] = {}
    for entity_id, entity in manifest["entities"].items():
        linked_entity_id = await _resolve_manifest_entity_link(
            entity, entity_registry, global_config
        )
        target_entity_id = linked_entity_id or entity_id
        entity_id_remap[entity_id] = target_entity_id
        canonical_name = entity["entity_name"]
        aliases = set(entity.get("aliases", []))
        if linked_entity_id:
            record = entity_registry.get_entity_record(linked_entity_id)
            if record is not None:
                canonical_name = record.canonical_name
                aliases.update(record.aliases)
                aliases.add(entity["entity_name"])
        grouped_entities.setdefault(target_entity_id, []).append(
            {
                **entity,
                "entity_name": canonical_name,
                "aliases": sorted(a for a in aliases if a and a != canonical_name),
            }
        )

    normalized_entities = {}
    for entity_id, contributions in grouped_entities.items():
        combined = _combine_entity_contributions(contributions)
        if combined is None:
            continue
        normalized_entities[entity_id] = {
            "entity_name": combined["entity_name"],
            "entity_type": combined["entity_type"],
            "aliases": combined.get("aliases", []),
            "descriptions": sorted(
                {
                    description
                    for contribution in contributions
                    for description in contribution.get("descriptions", [])
                }
            ),
            "source_chunk_ids": sorted(
                {
                    chunk_id
                    for contribution in contributions
                    for chunk_id in contribution.get("source_chunk_ids", [])
                }
            ),
        }

    grouped_relationships: dict[str, list[dict]] = {}
    for relationship in manifest["relationships"].values():
        src_entity_id = entity_id_remap.get(
            relationship["src_entity_id"], relationship["src_entity_id"]
        )
        tgt_entity_id = entity_id_remap.get(
            relationship["tgt_entity_id"], relationship["tgt_entity_id"]
        )
        relationship_id = generate_stable_relationship_id(
            src_entity_id, tgt_entity_id, relationship.get("relation_type", "related")
        )
        grouped_relationships.setdefault(relationship_id, []).append(
            {
                **relationship,
                "src_entity_id": src_entity_id,
                "tgt_entity_id": tgt_entity_id,
            }
        )

    normalized_relationships = {}
    for relationship_id, contributions in grouped_relationships.items():
        combined = _combine_relationship_contributions(contributions)
        if combined is None:
            continue
        normalized_relationships[relationship_id] = {
            "src_entity_id": combined["src_entity_id"],
            "tgt_entity_id": combined["tgt_entity_id"],
            "relation_type": combined.get("relation_type", "related"),
            "descriptions": sorted(
                {
                    description
                    for contribution in contributions
                    for description in contribution.get("descriptions", [])
                }
            ),
            "weight": combined["weight"],
            "source_chunk_ids": sorted(
                {
                    chunk_id
                    for contribution in contributions
                    for chunk_id in contribution.get("source_chunk_ids", [])
                }
            ),
        }

    return _normalize_document_manifest(
        {
            **manifest,
            "entities": normalized_entities,
            "relationships": normalized_relationships,
        }
    )


async def _resolve_manifest_entity_link(
    entity: dict, entity_registry, global_config: dict
) -> Optional[str]:
    exact_match = entity_registry.resolve_entity(entity["entity_name"], fuzzy_threshold=1.0)
    if exact_match is not None:
        return exact_match

    threshold = global_config.get("entity_linking_similarity_threshold", 0.92)
    candidates = entity_registry.find_candidates(
        entity["entity_name"],
        entity_type=entity.get("entity_type"),
        fuzzy_threshold=threshold,
        limit=global_config.get("entity_linking_max_candidates", 3),
    )
    if len(candidates) == 1:
        return candidates[0][0]
    if not candidates or not global_config.get("enable_entity_linking", False):
        return None
    return await _disambiguate_entity_link(entity, candidates, entity_registry, global_config)


async def _disambiguate_entity_link(
    entity: dict, candidates, entity_registry, global_config: dict
) -> Optional[str]:
    llm_func = global_config.get("cheap_model_func")
    if llm_func is None:
        return None
    candidate_lines = []
    valid_ids = set()
    for entity_id, score in candidates:
        record = entity_registry.get_entity_record(entity_id)
        if record is None:
            continue
        valid_ids.add(entity_id)
        candidate_lines.append(
            f"- id: {entity_id}\n  canonical_name: {record.canonical_name}\n  entity_type: {record.entity_type}\n  score: {score:.3f}\n  aliases: {sorted(record.aliases)}"
        )
    if not candidate_lines:
        return None
    prompt = f"""Decide whether the extracted entity matches one of the existing entities.

Extracted entity:
- name: {entity["entity_name"]}
- type: {entity["entity_type"]}
- descriptions: {entity.get("descriptions", [])}

Candidates:
{chr(10).join(candidate_lines)}

Return JSON with exactly this shape:
{{"decision": "existing" | "new", "entity_id": "<candidate-id-or-empty>"}}

Choose "new" unless one candidate is clearly the same real-world entity."""
    try:
        response = await llm_func(prompt)
        if not isinstance(response, str):
            return None
        start_idx = response.find("{")
        end_idx = response.rfind("}") + 1
        payload = json.loads(response[start_idx:end_idx] if start_idx >= 0 else response)
        if payload.get("decision") == "existing" and payload.get("entity_id") in valid_ids:
            return payload["entity_id"]
    except Exception:
        return None
    return None


__all__ = [
    "_handle_single_entity_extraction",
    "_handle_single_relationship_extraction",
    "_merge_edges_then_upsert",
    "_merge_nodes_then_upsert",
    "extract_document_entity_relationships",
    "extract_entities",
    "extract_entities_structured",
    "rebuild_knowledge_graph_for_documents",
]
