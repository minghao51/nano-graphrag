import json
from collections import defaultdict
from typing import Optional

from .._utils import generate_stable_relationship_id, logger
from ..base import BaseGraphStorage, BaseKVStorage, BaseVectorStorage
from .extraction_common import (
    UNKNOWN_ENTITY_TYPE,
    _combine_entity_contributions,
    _combine_relationship_contributions,
    _handle_entity_relation_summary,
    _normalize_entity_name,
    _select_canonical_entity_id,
)

GRAPH_INDEX_META_KEY = "__meta__"
GRAPH_INDEX_VERSION = 1


def _entity_index_key(entity_id: str) -> str:
    return f"entity::{entity_id}"


def _entity_name_index_key(entity_name: str) -> str:
    return f"entity_name::{_normalize_entity_name(entity_name)}"


def _relationship_index_key(relationship_id: str) -> str:
    return f"relationship::{relationship_id}"


async def ensure_graph_contribution_index(
    contribution_index: BaseKVStorage[dict], document_index: BaseKVStorage[dict]
) -> None:
    meta = await contribution_index.get_by_id(GRAPH_INDEX_META_KEY)
    if meta is not None and meta.get("version") == GRAPH_INDEX_VERSION and meta.get("built"):
        return
    await rebuild_graph_contribution_index(contribution_index, document_index)


async def rebuild_graph_contribution_index(
    contribution_index: BaseKVStorage[dict],
    document_index: BaseKVStorage[dict],
    batch_size: int = 100,
) -> None:
    await contribution_index.drop()
    doc_ids = await document_index.all_keys()
    index_entries: dict[str, dict] = {}

    for i in range(0, len(doc_ids), batch_size):
        batch_ids = doc_ids[i : i + batch_size]
        manifests = await document_index.get_by_ids(batch_ids)

        for doc_id, manifest in zip(batch_ids, manifests):
            if manifest is None:
                continue
            for entity_id, entity in manifest.get("entities", {}).items():
                _append_doc_id(index_entries, _entity_index_key(entity_id), doc_id)
                entity_name = entity.get("entity_name")
                if entity_name:
                    _append_doc_id(index_entries, _entity_name_index_key(entity_name), doc_id)
            for relationship_id in manifest.get("relationships", {}).keys():
                _append_doc_id(index_entries, _relationship_index_key(relationship_id), doc_id)

        if len(index_entries) >= batch_size * 10:
            await contribution_index.upsert(index_entries)
            index_entries.clear()

    index_entries[GRAPH_INDEX_META_KEY] = {"version": GRAPH_INDEX_VERSION, "built": True}
    await contribution_index.upsert(index_entries)


async def _propagate_entity_remap_to_all_documents(
    document_index: BaseKVStorage[dict],
    contribution_index: BaseKVStorage[dict],
    entity_id_remap: dict[str, str],
) -> None:
    """Propagate entity ID remapping to ALL documents' relationships.

    When entities merge (e.g., entity_2 merged into entity_1 as canonical),
    relationships in unaffected documents that reference entity_2 need to be
    updated to reference entity_1. This ensures no orphaned edges.

    Also updates the contribution index to reflect the new entity IDs.

    Uses the contribution index to find affected documents instead of scanning
    all documents with all_keys().
    """
    from .._utils import logger

    old_entity_ids = set(entity_id_remap.keys())
    new_entity_ids = set(entity_id_remap.values())

    contrib_keys_to_check = [_entity_index_key(eid) for eid in old_entity_ids]
    if not contrib_keys_to_check:
        return

    contrib_entries = await contribution_index.get_by_ids(contrib_keys_to_check)

    affected_doc_ids: set[str] = set()
    for contrib_key, entry in zip(contrib_keys_to_check, contrib_entries):
        if entry is not None:
            affected_doc_ids.update(entry.get("doc_ids", []))

    if not affected_doc_ids:
        return

    affected_manifests = await document_index.get_by_ids(sorted(affected_doc_ids))

    docs_to_update: dict[str, dict] = {}
    contrib_updates: dict[str, dict] = {}

    for doc_id, manifest in zip(sorted(affected_doc_ids), affected_manifests):
        if manifest is None:
            continue

        relationships = manifest.get("relationships", {})
        if not relationships:
            continue

        needs_update = False
        updated_relationships = {}

        for rel_id, rel in relationships.items():
            src_id = rel.get("src_entity_id", "")
            tgt_id = rel.get("tgt_entity_id", "")

            new_src = entity_id_remap.get(src_id)
            new_tgt = entity_id_remap.get(tgt_id)

            if new_src is not None or new_tgt is not None:
                new_rel = dict(rel)
                if new_src is not None:
                    new_rel["src_entity_id"] = new_src
                if new_tgt is not None:
                    new_rel["tgt_entity_id"] = new_tgt
                updated_relationships[rel_id] = new_rel
                needs_update = True

        if needs_update:
            updated_manifest = dict(manifest)
            updated_manifest["relationships"] = dict(relationships)
            updated_manifest["relationships"].update(updated_relationships)
            docs_to_update[doc_id] = updated_manifest

            logger.debug(
                f"Propagated entity remap to document {doc_id}: "
                f"{len(updated_relationships)} relationships updated"
            )

    if docs_to_update:
        await document_index.upsert(docs_to_update)

        for doc_id in docs_to_update:
            for old_entity_id in old_entity_ids:
                contrib_key = _entity_index_key(old_entity_id)
                entry = await contribution_index.get_by_id(contrib_key)
                if entry and doc_id in entry.get("doc_ids", []):
                    doc_ids_list = list(entry["doc_ids"])
                    doc_ids_list.remove(doc_id)
                    if doc_ids_list:
                        contrib_updates[contrib_key] = {"doc_ids": sorted(doc_ids_list)}
                    else:
                        contrib_updates[contrib_key] = None

            for new_entity_id in new_entity_ids:
                contrib_key = _entity_index_key(new_entity_id)
                entry = await contribution_index.get_by_id(contrib_key)
                existing_doc_ids = set(entry["doc_ids"]) if entry else set()
                existing_doc_ids.add(doc_id)
                contrib_updates[contrib_key] = {"doc_ids": sorted(existing_doc_ids)}

        for contrib_key, value in contrib_updates.items():
            if value is None:
                await contribution_index.delete([contrib_key])
            elif value.get("doc_ids"):
                await contribution_index.upsert({contrib_key: value})

        logger.info(
            f"Entity remap propagated to {len(docs_to_update)} documents, "
            f"{len(contrib_updates)} contribution index entries updated"
        )


async def update_graph_contribution_index_for_documents(
    contribution_index: BaseKVStorage[dict],
    old_document_manifests: dict[str, dict],
    new_document_manifests: dict[str, dict],
) -> None:
    all_keys: set[str] = set()
    for manifest in list(old_document_manifests.values()) + list(new_document_manifests.values()):
        if manifest is None:
            continue
        all_keys.update(_manifest_index_keys(manifest))

    existing_entries = await contribution_index.get_by_ids(sorted(all_keys)) if all_keys else []
    existing_lookup = {
        key: value or {"doc_ids": []} for key, value in zip(sorted(all_keys), existing_entries)
    }

    upserts: dict[str, dict] = {}
    deletes: list[str] = []
    changed_doc_ids = set(old_document_manifests.keys()).union(new_document_manifests.keys())
    for doc_id in changed_doc_ids:
        old_manifest = old_document_manifests.get(doc_id) or {}
        new_manifest = new_document_manifests.get(doc_id) or {}
        removed_keys = _manifest_index_keys(old_manifest) - _manifest_index_keys(new_manifest)
        added_keys = _manifest_index_keys(new_manifest)

        for key in removed_keys.union(added_keys):
            record = existing_lookup.setdefault(key, {"doc_ids": []})
            doc_ids = set(record.get("doc_ids", []))
            if key in removed_keys:
                doc_ids.discard(doc_id)
            if key in added_keys:
                doc_ids.add(doc_id)
            if doc_ids:
                upserts[key] = {"doc_ids": sorted(doc_ids)}
            else:
                deletes.append(key)

    upserts[GRAPH_INDEX_META_KEY] = {"version": GRAPH_INDEX_VERSION, "built": True}
    if deletes:
        await contribution_index.delete(sorted(set(deletes)))
    if upserts:
        await contribution_index.upsert(upserts)


def _append_doc_id(index_entries: dict[str, dict], key: str, doc_id: str) -> None:
    entry = index_entries.setdefault(key, {"doc_ids": []})
    if doc_id not in entry["doc_ids"]:
        entry["doc_ids"].append(doc_id)


def _manifest_index_keys(manifest: dict) -> set[str]:
    keys = set()
    for entity_id, entity in manifest.get("entities", {}).items():
        keys.add(_entity_index_key(entity_id))
        entity_name = entity.get("entity_name")
        if entity_name:
            keys.add(_entity_name_index_key(entity_name))
    for relationship_id in manifest.get("relationships", {}).keys():
        keys.add(_relationship_index_key(relationship_id))
    return keys


async def rebuild_knowledge_graph_for_documents(
    document_index: BaseKVStorage[dict],
    contribution_index: BaseKVStorage[dict],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: Optional[BaseVectorStorage],
    tokenizer_wrapper,
    global_config: dict,
    old_document_manifests: dict[str, dict],
    new_document_manifests: dict[str, dict],
) -> BaseGraphStorage:
    entity_registry = global_config.get("entity_registry")
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

    await ensure_graph_contribution_index(contribution_index, document_index)
    contribution_keys = sorted(
        {_entity_index_key(entity_id) for entity_id in affected_entity_ids}.union(
            {_entity_name_index_key(entity_name) for entity_name in affected_entity_names}
        ).union(
            {
                _relationship_index_key(relationship_id)
                for relationship_id in affected_relationship_ids
            }
        )
    )
    indexed_contributions = await contribution_index.get_by_ids(contribution_keys)
    document_keys = sorted(
        {
            doc_id
            for contribution in indexed_contributions
            if contribution is not None
            for doc_id in contribution.get("doc_ids", [])
        }
    )
    if not document_keys and contribution_keys:
        await rebuild_graph_contribution_index(contribution_index, document_index)
        indexed_contributions = await contribution_index.get_by_ids(contribution_keys)
        document_keys = sorted(
            {
                doc_id
                for contribution in indexed_contributions
                if contribution is not None
                for doc_id in contribution.get("doc_ids", [])
            }
        )
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

    if entity_id_remap:
        await _propagate_entity_remap_to_all_documents(
            document_index, contribution_index, entity_id_remap
        )

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
    entity_vdb_batch: dict[str, dict] = {}
    for entity_id in entity_ids_to_refresh:
        combined = combined_entities.get(entity_id_remap.get(entity_id, entity_id))
        if entity_id_remap.get(entity_id, entity_id) != entity_id:
            await knowledge_graph_inst.delete_node(entity_id)
            if entity_vdb is not None:
                try:
                    await entity_vdb.delete([entity_id])
                except Exception as e:
                    logger.warning(f"Failed to delete entity {entity_id} from entity_vdb: {e}")
            if entity_registry is not None:
                entity_registry.remove_entity(entity_id)
            continue
        if combined is None:
            await knowledge_graph_inst.delete_node(entity_id)
            if entity_vdb is not None:
                try:
                    await entity_vdb.delete([entity_id])
                except Exception as e:
                    logger.warning(f"Failed to delete entity {entity_id} from entity_vdb: {e}")
            if entity_registry is not None:
                entity_registry.remove_entity(entity_id)
            continue
        combined["description"] = await _handle_entity_relation_summary(
            combined["entity_name"],
            combined["description"],
            global_config,
            tokenizer_wrapper,
        )
        node_payload = {**combined, "aliases": json.dumps(combined.get("aliases", []))}
        await knowledge_graph_inst.upsert_node(entity_id, node_payload)
        if entity_vdb is not None:
            entity_vdb_batch[entity_id] = {
                "content": combined["entity_name"] + " - " + combined["description"],
                "entity_name": combined["entity_name"],
            }
        if entity_registry is not None:
            existing_record = entity_registry.get_entity_record(entity_id)
            entity_registry.register_entity(
                entity_id=entity_id,
                canonical_name=combined["entity_name"],
                aliases=sorted(
                    set(combined.get("aliases", [])).union(
                        existing_record.aliases if existing_record is not None else set()
                    )
                ),
                entity_type=combined.get("entity_type", "unknown"),
                metadata={"description": combined["description"]},
            )

    if entity_vdb is not None and entity_vdb_batch:
        try:
            await entity_vdb.upsert(entity_vdb_batch)
        except Exception as e:
            logger.warning(
                f"Failed to batch upsert {len(entity_vdb_batch)} entities "
                f"to entity_vdb: {e}. Entities in graph but not vector-searchable."
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
