from .extraction_common import (
    _handle_single_entity_extraction,
    _handle_single_relationship_extraction,
    _merge_edges_then_upsert,
    _merge_nodes_then_upsert,
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
