from collections import defaultdict
from typing import Optional

from .._utils import generate_stable_relationship_id
from ..base import BaseGraphStorage, BaseKVStorage, BaseVectorStorage
from .extraction_common import (
    UNKNOWN_ENTITY_TYPE,
    _combine_entity_contributions,
    _combine_relationship_contributions,
    _handle_entity_relation_summary,
    _select_canonical_entity_id,
)


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
            if existing_id is None or entity.get("entity_type") != UNKNOWN_ENTITY_TYPE:
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
