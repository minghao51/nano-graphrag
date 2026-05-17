from __future__ import annotations

import asyncio
import re
from typing import Any

from ..._utils import _safe_json_loads, get_all_nodes_safe, logger

DREAM_ENRICH_PROMPT = """You are a knowledge-graph curator. An entity has a thin description that needs to be
enriched using evidence from source documents.

Entity type: {entity_type}
Name: "{entity_name}"
Current description: {current_description}

Known relationships:
{relationships}

Relevant source excerpts:
{source_excerpts}

Write an improved description that:
1. Keeps ALL information from the current description (do not remove anything)
2. Adds ONLY facts that are explicitly about "{entity_name}" — not about any other entity
3. Is factual, third-person, knowledge-base style
4. Is concise but more informative than the current description

STRICT RULES (violations will corrupt the knowledge base):
- MOST IMPORTANT: Every fact you include MUST have "{entity_name}" as its subject in the
  source text. If a sentence is about another entity, do NOT include it.
- Do NOT copy facts about other entities onto this entity.
- ONLY include facts explicitly stated in the excerpts or current description
- Do NOT invent details not in the text
- If unsure whether a detail applies to "{entity_name}" specifically, leave it out

Return ONLY the enriched description text. No preamble, no explanation."""

_MIN_DESCRIPTION_CHARS = 80
_MIN_WORD_OVERLAP_RATIO = 0.4


async def _enrich_phase(
    knowledge_graph_inst,
    text_chunks_kv,
    global_config: dict,
    min_chars: int = _MIN_DESCRIPTION_CHARS,
    batch_size: int = 50,
) -> dict[str, Any]:
    stats = {"examined": 0, "enriched": 0, "skipped": 0, "validation_failed": 0}

    use_llm_func = global_config.get("cheap_model_func")
    if use_llm_func is None:
        logger.warning("refinement_enrich_skipped", reason="missing_llm_func")
        return stats

    all_nodes = await get_all_nodes_safe(knowledge_graph_inst)
    if not all_nodes:
        return stats

    thin_entities = []
    for node_id, node_data in all_nodes.items():
        desc = node_data.get("description", "")
        if len(desc) < min_chars:
            thin_entities.append((node_id, node_data))

    if not thin_entities:
        return stats

    stats["examined"] = len(thin_entities)
    logger.info("refinement_enrich_start", thin_count=len(thin_entities))

    semaphore = asyncio.Semaphore(global_config.get("extraction_max_async", 16))

    async def _fetch_chunk_excerpts(source_ids: list[str]) -> list[str]:
        if not source_ids or text_chunks_kv is None:
            return []
        sids = source_ids[:5]
        raw = await text_chunks_kv.get_by_ids(sids)
        excerpts = []
        for _sid, chunk in zip(sids, raw, strict=False):
            if chunk is not None:
                excerpts.append(chunk.get("content", ""))
        return excerpts

    async def _enrich_one(node_id: str, node_data: dict) -> bool:
        async with semaphore:
            entity_name = node_data.get("entity_name", node_id)
            entity_type = node_data.get("entity_type", "UNKNOWN")
            current_desc = node_data.get("description", "")

            source_ids = _safe_json_loads(node_data.get("source_id", "[]"), [])

            excerpts = await _fetch_chunk_excerpts(source_ids)
            if not excerpts:
                stats["skipped"] += 1
                return False

            edges = await knowledge_graph_inst.get_node_edges(node_id)
            relationships = []
            if edges:
                for src, tgt in edges[:10]:
                    edge = await knowledge_graph_inst.get_edge(src, tgt)
                    if edge:
                        relationships.append(f"- {src} -> {tgt}: {edge.get('description', '')}")

            prompt = DREAM_ENRICH_PROMPT.format(
                entity_type=entity_type,
                entity_name=entity_name,
                current_description=current_desc,
                relationships="\n".join(relationships[:10]),
                source_excerpts="\n\n".join(excerpts[:3]),
            )

            try:
                enriched = await use_llm_func(prompt)
                if isinstance(enriched, list):
                    enriched = enriched[0].get("text", str(enriched))
            except Exception as e:
                logger.debug("refinement_enrich_llm_failed", error=str(e))
                stats["skipped"] += 1
                return False

            if not _validate_enrichment(entity_name, current_desc, enriched):
                stats["validation_failed"] += 1
                return False

            merged_data = dict(node_data)
            merged_data["description"] = enriched
            await knowledge_graph_inst.upsert_node(node_id, merged_data)
            stats["enriched"] += 1
            return True

    batch = thin_entities[:batch_size]
    await asyncio.gather(*[_enrich_one(nid, nd) for nid, nd in batch])

    logger.info("refinement_enrich_done", **stats)
    return stats


def _validate_enrichment(entity_name: str, original: str, enriched: str) -> bool:
    if not enriched or not enriched.strip():
        return False
    enriched_lower = enriched.lower()
    if entity_name.lower() not in enriched_lower:
        return False
    original_words = set(original.lower().split())
    enriched_words = set(enriched.lower().split())
    if original_words:
        overlap = len(original_words & enriched_words) / len(original_words)
        if overlap < _MIN_WORD_OVERLAP_RATIO:
            return False
    if not _validate_entity_subject(entity_name, enriched):
        return False
    return True


def _validate_entity_subject(entity_name: str, text: str) -> bool:
    name_lower = entity_name.lower()
    name_parts = set(name_lower.split())
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    pronouns = {"it", "he", "she", "they", "this", "these", "the", "its", "their"}

    for sentence in sentences:
        s = sentence.strip()
        if not s or len(s) < 10:
            continue
        s_lower = s.lower()
        if name_lower in s_lower:
            continue
        first_words = s_lower.split()[:4]
        if any(w in pronouns for w in first_words):
            continue
        proper_nouns = [w for w in first_words if w[0].isupper() and w not in pronouns]
        if proper_nouns and not (name_parts & set(first_words)):
            logger.debug(
                "enrich_identity_skip",
                entity=entity_name,
                suspicious=s[:80],
            )
            return False

    return True
