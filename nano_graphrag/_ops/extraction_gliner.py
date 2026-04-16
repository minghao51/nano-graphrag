import asyncio
from typing import Any

from .._utils import generate_stable_entity_id, generate_stable_relationship_id, logger
from .extraction_common import _normalize_document_manifest

GLiNER_MODEL = None
GLiNER_MODEL_NAME = "fastino/gliner2-base-v1"

ENTITY_TYPES = [
    "person",
    "organization",
    "location",
    "event",
    "product",
    "technology",
    "date",
    "money",
    "film",
    "actor",
    "director",
    "award",
]

RELATION_TYPES = [
    "directed",
    "acted_in",
    "spouse_of",
    "parent_of",
    "child_of",
    "sibling_of",
    "born_in",
    "died_in",
    "located_in",
    "works_for",
    "founded_by",
    "member_of",
    "happened_in",
    "happened_on",
    "is_a",
    "also_known_as",
    "created_by",
    "released",
    "married_to",
    "mother_of",
    "father_of",
    "son_of",
    "daughter_of",
    "brother_of",
    "sister_of",
]


async def _get_gliner_model():
    global GLiNER_MODEL
    if GLiNER_MODEL is None:
        try:
            from gliner2 import GLiNER2
        except ImportError:
            raise ImportError("GLiNER2 is not installed. Install with: uv pip install gliner2")
        logger.info(f"Loading GLiNER2 model: {GLiNER_MODEL_NAME}")
        GLiNER_MODEL = GLiNER2.from_pretrained(GLiNER_MODEL_NAME)
        logger.info("GLiNER2 model loaded successfully")
    return GLiNER_MODEL


async def extract_document_entity_relationships_gliner(
    chunks: dict[str, Any],
    tokenizer_wrapper: Any,
    global_config: dict,
) -> dict:
    model = await _get_gliner_model()

    schema = (
        model.create_schema()
        .entities(ENTITY_TYPES)
        .relations({rel: f"{rel.replace('_', ' ')} relationship" for rel in RELATION_TYPES})
    )

    manifest_entities: dict[str, dict[str, Any]] = {}
    manifest_relationships: dict[str, dict[str, Any]] = {}
    chunk_ids = list(chunks.keys())

    batch_size = global_config.get("extraction_max_async", 16)
    semaphore = asyncio.Semaphore(batch_size)

    async def _process_chunk(chunk_key: str, chunk_data: dict[str, Any]):
        async with semaphore:
            content = chunk_data.get("content", "")
            if not content.strip():
                return [], []

            try:
                results = model.extract(content, schema)
            except Exception as e:
                logger.warning(f"GLiNER2 extraction failed for chunk: {e}")
                return [], []

            entities_by_type = results.get("entities", {})
            relations_by_type = results.get("relation_extraction", {})

            chunk_entities = {}
            entity_name_to_id = {}

            for entity_type, entity_list in entities_by_type.items():
                for entity_name in entity_list:
                    entity_name = entity_name.strip()
                    if not entity_name:
                        continue

                    entity_id = generate_stable_entity_id(entity_name, entity_type.upper())
                    entity_record = {
                        "entity_name": entity_name,
                        "entity_type": entity_type.upper(),
                        "descriptions": [],
                        "source_chunk_ids": [chunk_key],
                    }
                    if entity_id not in chunk_entities:
                        chunk_entities[entity_id] = entity_record
                        entity_name_to_id[entity_name.lower()] = entity_id

                    if entity_id not in manifest_entities:
                        manifest_entities[entity_id] = entity_record
                    else:
                        manifest_entities[entity_id]["source_chunk_ids"].append(chunk_key)

            chunk_relationships = {}
            for relation_type, relation_tuples in relations_by_type.items():
                for head_name, tail_name in relation_tuples:
                    head_name = head_name.strip()
                    tail_name = tail_name.strip()
                    if not head_name or not tail_name:
                        continue

                    head_lower = head_name.lower()
                    tail_lower = tail_name.lower()

                    eid1 = entity_name_to_id.get(head_lower)
                    eid2 = entity_name_to_id.get(tail_lower)

                    if eid1 and eid2:
                        rel_id = generate_stable_relationship_id(eid1, eid2, relation_type)
                        ent1 = chunk_entities[eid1]
                        ent2 = chunk_entities[eid2]
                        chunk_relationships[rel_id] = {
                            "src_entity_id": eid1,
                            "tgt_entity_id": eid2,
                            "relation_type": relation_type,
                            "descriptions": [
                                f"{ent1['entity_name']} {relation_type.replace('_', ' ')} {ent2['entity_name']}"
                            ],
                            "weight": 1.0,
                            "source_chunk_ids": [chunk_key],
                        }

            logger.info(
                f"Processed chunk {chunk_key[:8]}...: {len(chunk_entities)} entities, {len(chunk_relationships)} relations"
            )
            return list(chunk_entities.values()), list(chunk_relationships.values())

    tasks = [_process_chunk(k, v) for k, v in chunks.items()]
    results = await asyncio.gather(*tasks)

    for entities, relationships in results:
        for entity in entities:
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
            target["descriptions"].extend(entity.get("descriptions", []))
            target["source_chunk_ids"].extend(entity.get("source_chunk_ids", []))

        for relationship in relationships:
            rel_id = generate_stable_relationship_id(
                relationship["src_entity_id"],
                relationship["tgt_entity_id"],
                relationship.get("relation_type", "related"),
            )
            target = manifest_relationships.setdefault(
                rel_id,
                {
                    "src_entity_id": relationship["src_entity_id"],
                    "tgt_entity_id": relationship["tgt_entity_id"],
                    "relation_type": relationship.get("relation_type", "related"),
                    "descriptions": [],
                    "weight": 0.0,
                    "source_chunk_ids": [],
                },
            )
            target["descriptions"].extend(relationship.get("descriptions", []))
            target["weight"] += relationship.get("weight", 1.0)
            target["source_chunk_ids"].extend(relationship.get("source_chunk_ids", []))

    manifest = {
        "chunk_ids": chunk_ids,
        "entities": manifest_entities,
        "relationships": manifest_relationships,
    }
    return _normalize_document_manifest(manifest)


async def extract_entities_gliner(
    chunks: dict[str, Any],
    knowledge_graph_inst: Any,
    entity_vdb: Any,
    tokenizer_wrapper: Any,
    global_config: dict,
) -> Any:
    from .._ops.extraction_writeback import _write_extraction_manifest

    manifest = await extract_document_entity_relationships_gliner(
        chunks,
        tokenizer_wrapper,
        global_config,
    )
    if not manifest["entities"]:
        logger.warning("GLiNER2 didn't extract any entities")
        return None
    return await _write_extraction_manifest(
        manifest, knowledge_graph_inst, entity_vdb, tokenizer_wrapper, global_config, chunks
    )
