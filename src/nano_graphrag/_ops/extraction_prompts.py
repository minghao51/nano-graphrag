from __future__ import annotations

from .._schemas import RELATION_VOCABULARY_SORTED


def _build_temporal_instructions(global_config: dict) -> str:
    temporal = global_config.get("enable_temporal_extraction", True)
    if temporal:
        return (
            " For aliases: include alternative names, abbreviations, or nicknames."
            " Use an empty list if none.\n"
            "For event_date: only for EVENT type entities, when the event occurred"
            " (ISO-8601 or free-text). Null for non-events.\n"
            "For temporal_context: free-text about when the relationship was true"
            ' (e.g. "since 2020", "from 2018 to 2022"). Null if unknown.\n'
            "For valid_from/valid_to: ISO-8601 dates if known, null if unknown."
        )
    return (
        " For aliases: include alternative names, abbreviations, or nicknames."
        " Use an empty list if none."
    )


def _build_extraction_system_prompt(
    entity_types: list[str],
    global_config: dict,
    batched: bool = False,
) -> str:
    quality = global_config.get("entity_extraction_quality", "balanced")
    temporal_instructions = _build_temporal_instructions(global_config)
    types_str = ", ".join(entity_types)
    relation_types_str = ", ".join(RELATION_VOCABULARY_SORTED)

    if quality == "fast":
        if batched:
            return (
                f"Extract entities and relationships from each chunk.\n"
                f"Entity types: {types_str}.\n"
                f"Relation types: {relation_types_str}.\n"
                f"For each relationship, assign a relation_type from the list above and a confidence score (0.0-1.0).\n"
                f'Return JSON: {{"chunks": [{{"chunk_id": str, "entities": ['
                f'{{"entity_name": str, "entity_type": str, "description": str}}], '
                f'"relationships": [{{"source": str, "target": str, "description": str, '
                f'"relation_type": str, "confidence": float, "weight": float}}]}}]}}'
            )
        return (
            f"Extract entities and relationships.\n"
            f"Entity types: {types_str}.\n"
            f"Relation types: {relation_types_str}.\n"
            f"For each relationship, assign a relation_type from the list above and a confidence score (0.0-1.0).\n"
            f'Return JSON: {{"entities": [{{"entity_name": str, "entity_type": str, "description": str}}], '
            f'"relationships": [{{"source": str, "target": str, "description": str, '
            f'"relation_type": str, "confidence": float, "weight": float}}]}}'
        )

    if batched:
        return (
            f"You are an entity extraction assistant. Extract entities and relationships from each chunk below.\n"
            f"Entity types: {types_str}.\n"
            f"Relation types: {relation_types_str}.\n"
            f"For each relationship, choose the most specific relation_type from the list above "
            f"(avoid 'related_to' if a more specific type fits). Also assign a confidence score "
            f"from 0.0 to 1.0 indicating how certain you are about the relationship.\n"
            f"Return a JSON with a 'chunks' array. Each element has: chunk_id (string matching the id in the header), "
            f"entities (name, type, description, aliases), relationships (source, target, description, "
            f"relation_type, confidence, weight)."
            f"{temporal_instructions}\n"
            f"Preserve the chunk_id exactly as given."
        )
    return (
        f"You are an entity extraction assistant. Extract entities and relationships from the text.\n"
        f"Entity types: {types_str}.\n"
        f"Relation types: {relation_types_str}.\n"
        f"For each relationship, choose the most specific relation_type from the list above "
        f"(avoid 'related_to' if a more specific type fits). Also assign a confidence score "
        f"from 0.0 to 1.0 indicating how certain you are about the relationship.\n"
        f"Return a JSON with 'entities' (name, type, description, aliases) and "
        f"'relationships' (source, target, description, relation_type, confidence, weight)."
        f"{temporal_instructions}"
    )
