"""Compatibility wrapper around the split operation modules."""

from ._ops import (
    _handle_single_entity_extraction,
    _handle_single_relationship_extraction,
    _merge_edges_then_upsert,
    _merge_nodes_then_upsert,
    chunking_by_seperators,
    chunking_by_token_size,
    extract_document_entity_relationships,
    extract_entities,
    extract_entities_structured,
    generate_community_report,
    get_chunks,
    global_query,
    local_query,
    naive_query,
    rebuild_knowledge_graph_for_documents,
)

__all__ = [
    "_handle_single_entity_extraction",
    "_handle_single_relationship_extraction",
    "_merge_edges_then_upsert",
    "_merge_nodes_then_upsert",
    "extract_document_entity_relationships",
    "chunking_by_seperators",
    "chunking_by_token_size",
    "extract_entities",
    "extract_entities_structured",
    "generate_community_report",
    "get_chunks",
    "global_query",
    "local_query",
    "naive_query",
    "rebuild_knowledge_graph_for_documents",
]
