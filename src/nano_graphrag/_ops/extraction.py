from __future__ import annotations

import json

from .._utils import generate_stable_relationship_id, logger
from ..base import BaseGraphStorage
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


def _get_manifest_neighbor_names(entity_id: str, manifest: dict) -> set[str]:
    names: set[str] = set()
    for rel in manifest["relationships"].values():
        src, tgt = rel["src_entity_id"], rel["tgt_entity_id"]
        if src == entity_id and tgt in manifest["entities"]:
            names.add(manifest["entities"][tgt]["entity_name"].lower())
        elif tgt == entity_id and src in manifest["entities"]:
            names.add(manifest["entities"][src]["entity_name"].lower())
    return names


def _compute_neighborhood_iou(
    manifest_neighbors: set[str],
    graph_neighbors: set[str],
) -> tuple[float, set[str]]:
    if not manifest_neighbors and not graph_neighbors:
        return 0.0, set()
    common = manifest_neighbors & graph_neighbors
    union = manifest_neighbors | graph_neighbors
    iou = len(common) / len(union) if union else 0.0
    return iou, common


async def _get_graph_neighbor_names(
    candidate_entity_id: str,
    knowledge_graph_inst: BaseGraphStorage,
    entity_registry,
) -> set[str]:
    edges = await knowledge_graph_inst.get_node_edges(candidate_entity_id)
    if not edges:
        return set()
    neighbor_ids: set[str] = set()
    for src, tgt in edges:
        neighbor_ids.add(src)
        neighbor_ids.add(tgt)
    neighbor_ids.discard(candidate_entity_id)
    if not neighbor_ids:
        return set()
    names: set[str] = set()
    unresolved: list[str] = []
    for nid in neighbor_ids:
        record = entity_registry.get_entity_record(nid)
        if record:
            names.add(record.canonical_name.lower())
        else:
            unresolved.append(nid)
    if unresolved:
        nodes = await knowledge_graph_inst.get_nodes_batch(unresolved)
        for node in nodes:
            if node is not None:
                names.add(node.get("entity_name", "").lower())
    return names


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
        logger.warning("extraction_no_entities_found")
        return manifest

    entity_count = len(manifest["entities"])
    chunk_count = len(manifest.get("chunk_ids", []))
    entity_count_min_ratio = global_config.get("entity_count_min_ratio", 2.0)
    entity_count_min_absolute = global_config.get("entity_count_min_absolute", 3)
    expected_min_entities = max(
        entity_count_min_absolute, int(chunk_count * entity_count_min_ratio)
    )

    if entity_count < expected_min_entities:
        logger.warning(
            "low_entity_count",
            entity_count=entity_count,
            chunk_count=chunk_count,
            expected_min=expected_min_entities,
        )
    manifest = await _enrich_manifest_aliases(manifest, chunks, global_config)
    return await _apply_entity_linking(manifest, global_config)


async def _enrich_manifest_aliases(manifest: dict, chunks, global_config: dict) -> dict:
    quality = global_config.get("entity_extraction_quality", "balanced")
    if quality == "fast":
        return manifest

    from .extraction_writeback import _extract_aliases_for_batch

    entities_to_enrich = [
        (entity["entity_name"], entity["entity_type"], entity.get("source_chunk_ids", []))
        for entity in manifest["entities"].values()
        if not entity.get("aliases")
    ]
    if not entities_to_enrich:
        return manifest

    enriched = await _extract_aliases_for_batch(entities_to_enrich, chunks, global_config)
    enriched_entities = {}
    for entity_id, entity in manifest["entities"].items():
        aliases = set(entity.get("aliases", []))
        batch_aliases = enriched.get(entity["entity_name"], [])
        aliases.update(batch_aliases)
        enriched_entities[entity_id] = {**entity, "aliases": sorted(set(aliases))}
    return {**manifest, "entities": enriched_entities}


async def _apply_entity_linking(manifest: dict, global_config: dict) -> dict:
    entity_registry = global_config.get("entity_registry")
    if entity_registry is None or not manifest.get("entities"):
        return manifest

    knowledge_graph_inst: BaseGraphStorage | None = global_config.get("knowledge_graph_inst")

    entity_id_remap: dict[str, str] = {}
    grouped_entities: dict[str, list[dict]] = {}
    for entity_id, entity in manifest["entities"].items():
        manifest_neighbors = _get_manifest_neighbor_names(entity_id, manifest)
        linked_entity_id = await _resolve_manifest_entity_link(
            entity,
            entity_registry,
            global_config,
            manifest_neighbors=manifest_neighbors,
            knowledge_graph_inst=knowledge_graph_inst,
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


def _heuristic_resolve_candidates(
    entity: dict,
    candidates: list[tuple[str, float]],
    entity_registry,
) -> str | None:
    entity_name_lower = entity["entity_name"].lower()
    entity_type = entity.get("entity_type", "")
    entity_aliases = {a.lower() for a in entity.get("aliases", [])}

    for candidate_id, _ in candidates:
        record = entity_registry.get_entity_record(candidate_id)
        if record is None:
            continue

        candidate_name_lower = record.canonical_name.lower()
        candidate_aliases = {a.lower() for a in record.aliases}
        candidate_type = record.entity_type
        type_match = entity_type and entity_type == candidate_type

        if entity_aliases & candidate_aliases:
            return candidate_id

        if candidate_name_lower == entity_name_lower and (type_match or not entity_type):
            return candidate_id

        if entity_name_lower in candidate_aliases or candidate_name_lower in entity_aliases:
            return candidate_id

    return None


async def _resolve_manifest_entity_link(
    entity: dict,
    entity_registry,
    global_config: dict,
    *,
    manifest_neighbors: set[str] | None = None,
    knowledge_graph_inst: BaseGraphStorage | None = None,
) -> str | None:
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

    if not candidates:
        return None

    iou_threshold = global_config.get("entity_linking_iou_threshold", 0.3)
    min_common = global_config.get("entity_linking_min_common_neighbors", 2)
    use_neighborhood_evidence = global_config.get("entity_linking_use_neighborhood_evidence", True)
    linking_enabled = global_config.get("enable_entity_linking", False)

    neighborhood_checked = False
    if (
        use_neighborhood_evidence
        and knowledge_graph_inst is not None
        and manifest_neighbors
        and len(candidates) == 1
    ):
        neighborhood_checked = True
        try:
            graph_neighbors = await _get_graph_neighbor_names(
                candidates[0][0], knowledge_graph_inst, entity_registry
            )
            iou, common = _compute_neighborhood_iou(manifest_neighbors, graph_neighbors)
            if iou >= iou_threshold and len(common) >= min_common:
                return candidates[0][0]
        except Exception:
            # Fall back to simple matching if neighborhood check fails
            pass

    if len(candidates) == 1:
        if neighborhood_checked:
            return None
        return candidates[0][0]

    if not linking_enabled:
        return None

    heuristic_result = _heuristic_resolve_candidates(entity, candidates, entity_registry)
    if heuristic_result is not None:
        return heuristic_result

    quality = global_config.get("entity_extraction_quality", "balanced")
    if quality == "fast":
        return None

    return await _disambiguate_entity_link(
        entity,
        candidates,
        entity_registry,
        global_config,
        manifest_neighbors=manifest_neighbors,
        knowledge_graph_inst=knowledge_graph_inst,
    )


async def _disambiguate_entity_link(
    entity: dict,
    candidates,
    entity_registry,
    global_config: dict,
    *,
    manifest_neighbors: set[str] | None = None,
    knowledge_graph_inst: BaseGraphStorage | None = None,
) -> str | None:
    llm_func = global_config.get("cheap_model_func")
    if llm_func is None:
        return None
    candidate_lines = []
    valid_ids = set()
    neighborhood_lines: list[str] = []

    for entity_id, score in candidates:
        record = entity_registry.get_entity_record(entity_id)
        if record is None:
            continue
        valid_ids.add(entity_id)
        candidate_lines.append(
            f"- id: {entity_id}\n"
            f"  canonical_name: {record.canonical_name}\n"
            f"  entity_type: {record.entity_type}\n"
            f"  score: {score:.3f}\n"
            f"  aliases: {sorted(record.aliases)}"
        )

        if manifest_neighbors and knowledge_graph_inst:
            try:
                graph_neighbors = await _get_graph_neighbor_names(
                    entity_id, knowledge_graph_inst, entity_registry
                )
                iou, common = _compute_neighborhood_iou(manifest_neighbors, graph_neighbors)
                if common:
                    common_display = ", ".join(sorted(common)[:5])
                    neighborhood_lines.append(
                        f"- Candidate '{record.canonical_name}' (ID: {entity_id}) "
                        f"shares {len(common)} common neighbor(s) ({common_display}) "
                        f"with the extracted entity. [IoU: {iou:.2f}]"
                    )
                else:
                    neighborhood_lines.append(
                        f"- Candidate '{record.canonical_name}' (ID: {entity_id}) "
                        f"shares NO common neighbors with the extracted entity. [IoU: 0.00]"
                    )
            except Exception:
                # If neighborhood lookup fails, skip it and continue without this evidence
                pass

    if not candidate_lines:
        return None

    related_section = ""
    if manifest_neighbors:
        related_section = f"- related to (in document): {sorted(manifest_neighbors)}\n"

    neighborhood_section = ""
    if neighborhood_lines:
        neighborhood_section = (
            "\nStructural Evidence (neighbor overlap with existing graph):\n"
            + "\n".join(neighborhood_lines)
            + "\n\nHigher IoU = stronger evidence that this is the same entity.\n"
        )

    prompt = (
        "Decide whether the extracted entity matches one of the existing entities.\n"
        "\n"
        "Extracted entity:\n"
        f"- name: {entity['entity_name']}\n"
        f"- type: {entity['entity_type']}\n"
        f"- descriptions: {entity.get('descriptions', [])}\n"
        f"{related_section}"
        "\n"
        "Candidates:\n"
        f"{chr(10).join(candidate_lines)}\n"
        f"{neighborhood_section}"
        "\n"
        "Return JSON with exactly this shape:\n"
        '{{"decision": "existing" | "new", "entity_id": "<candidate-id-or-empty>"}}\n'
        "\n"
        'Choose "new" unless one candidate is clearly the same real-world entity. '
        "Neighborhood overlap is strong evidence."
    )

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
    "_compute_neighborhood_iou",
    "_get_graph_neighbor_names",
    "_get_manifest_neighbor_names",
    "_handle_single_entity_extraction",
    "_handle_single_relationship_extraction",
    "_merge_edges_then_upsert",
    "_merge_nodes_then_upsert",
    "extract_document_entity_relationships",
    "extract_entities",
    "extract_entities_structured",
    "rebuild_knowledge_graph_for_documents",
]
