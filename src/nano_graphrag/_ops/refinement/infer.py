from __future__ import annotations

import asyncio
import json
import time
from collections import Counter
from typing import Any

from ..._schemas import normalize_relation_type
from ..._utils import logger

DREAM_INFER_PROMPT = """You are a knowledge-graph curator with high standards. Two entities co-occur in
{co_occurrence_count} source document(s) but have no edge in the graph. Determine whether
a SPECIFIC, FACTUAL relationship exists.

Entity A — type: {type_a}, name: "{name_a}"
Description: {description_a}

Entity B — type: {type_b}, name: "{name_b}"
Description: {description_b}

Source excerpt where both appear:
{source_excerpt}

RULES — read carefully:
1. Provide brief evidence from the excerpt or entity descriptions that supports the
   relationship. If there is no supporting evidence, return has_relation: false.
2. The relationship must be SPECIFIC. Generic labels like "related_to", "associated_with",
   or "connected_to" are NEVER acceptable.
3. Choose the correct DIRECTION. "source" is the entity that performs the action or holds
   the role. Example: "Einstein authored Theory of Relativity" →
   source="Einstein", target="Theory of Relativity", relation_type="authored_by".
4. Confidence: 1.0 = explicitly stated, 0.8-0.9 = clearly implied. Below 0.80 = do not return.
5. Co-occurrence alone is NOT evidence. Two entities in the same document does NOT mean
   they are related. You MUST find a specific statement or clear implication linking them.

Allowed relation types: part_of, contains, member_of, located_in, headquartered_in,
operates_in, originates_from, created_by, authored_by, founded_by, managed_by, led_by,
developed_by, influences, cites, builds_on, extends, contradicts, supports, references,
precedes, causes, enables, prevents, uses, depends_on, produces, consumes, implements,
provides, employed_by, collaborates_with, works_on, invests_in, competes_with, instance_of,
has_characteristic, classified_as

If confident, return:
{{"has_relation": true, "relation_type": "<type>", "source": "<name_of_source_entity>", "target": "<name_of_target_entity>", "confidence": <0.80-1.0>, "evidence": "<brief evidence>"}}

If no clear relationship, return:
{{"has_relation": false}}

Return ONLY the JSON object. No other text."""


async def _infer_phase(
    knowledge_graph_inst,
    text_chunks_kv,
    entity_vdb,
    global_config: dict,
    min_confidence: float = 0.80,
    hub_cap: int = 3,
    batch_size: int = 50,
    rejection_cache: dict | None = None,
) -> dict[str, Any]:
    stats = {"examined": 0, "inferred": 0, "rejected_by_llm": 0, "rejected_by_cache": 0}

    use_llm_func = global_config.get("cheap_model_func")
    if use_llm_func is None:
        logger.warning("refinement_infer_skipped", reason="missing_llm_func")
        return stats

    all_nodes = await _get_all_nodes_safe(knowledge_graph_inst)
    if len(all_nodes) < 2:
        return stats

    node_to_chunks: dict[str, set[str]] = {}
    for node_id, node_data in all_nodes.items():
        source_ids_raw = node_data.get("source_id", "[]")
        try:
            sids = json.loads(source_ids_raw) if isinstance(source_ids_raw, str) else []
        except (json.JSONDecodeError, TypeError):
            sids = []
        node_to_chunks[node_id] = set(sids)

    co_occurrence: dict[tuple[str, str], int] = {}
    chunk_to_nodes: dict[str, list[str]] = {}
    for node_id, chunk_set in node_to_chunks.items():
        for chunk_id in chunk_set:
            chunk_to_nodes.setdefault(chunk_id, []).append(node_id)
    for _chunk_id, nodes in chunk_to_nodes.items():
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                pair = tuple(sorted([nodes[i], nodes[j]]))
                co_occurrence[pair] = co_occurrence.get(pair, 0) + 1

    chunks_data = {}
    if text_chunks_kv is not None:
        all_chunk_keys = await text_chunks_kv.all_keys()
        chunks_raw = await text_chunks_kv.get_by_ids(all_chunk_keys)
        for k, v in zip(all_chunk_keys, chunks_raw):
            if v is not None:
                chunks_data[k] = v.get("content", "")

    candidate_pairs = []
    for (n_a, n_b), count in co_occurrence.items():
        if count < 2:
            continue
        if await knowledge_graph_inst.has_edge(n_a, n_b):
            continue
        candidate_pairs.append((n_a, n_b, count))

    candidate_pairs.sort(key=lambda x: x[2], reverse=True)
    candidate_pairs = candidate_pairs[:batch_size]

    semaphore = asyncio.Semaphore(global_config.get("extraction_max_async", 16))
    hub_counter: dict[str, int] = Counter()

    async def _evaluate_pair(n_a: str, n_b: str, co_count: int) -> None:
        async with semaphore:
            if hub_counter.get(n_a, 0) >= hub_cap or hub_counter.get(n_b, 0) >= hub_cap:
                return
            stats["examined"] += 1

            cache_key = f"{n_a}|{n_b}"
            if rejection_cache and cache_key in rejection_cache:
                stats["rejected_by_cache"] += 1
                return

            nd_a = await knowledge_graph_inst.get_node(n_a)
            nd_b = await knowledge_graph_inst.get_node(n_b)
            if nd_a is None or nd_b is None:
                return

            shared_chunks = node_to_chunks.get(n_a, set()) & node_to_chunks.get(n_b, set())
            excerpt = ""
            for cid in list(shared_chunks)[:1]:
                if cid in chunks_data:
                    excerpt = chunks_data[cid][:2000]
                    break

            prompt = DREAM_INFER_PROMPT.format(
                co_occurrence_count=co_count,
                type_a=nd_a.get("entity_type", "UNKNOWN"),
                name_a=nd_a.get("entity_name", n_a),
                description_a=nd_a.get("description", "")[:500],
                type_b=nd_b.get("entity_type", "UNKNOWN"),
                name_b=nd_b.get("entity_name", n_b),
                description_b=nd_b.get("description", "")[:500],
                source_excerpt=excerpt,
            )

            try:
                response = await use_llm_func(prompt)
                if isinstance(response, list):
                    response = response[0].get("text", str(response))
            except Exception as e:
                logger.debug("refinement_infer_llm_failed", error=str(e))
                return

            try:
                result = json.loads(response.strip())
            except (json.JSONDecodeError, TypeError):
                stats["rejected_by_llm"] += 1
                return

            if not result.get("has_relation", False):
                if rejection_cache is not None:
                    rejection_cache[cache_key] = time.time()
                stats["rejected_by_llm"] += 1
                return

            rel_type = normalize_relation_type(result.get("relation_type", "related_to"))
            confidence = float(result.get("confidence", 0.0))
            if confidence < min_confidence:
                if rejection_cache is not None:
                    rejection_cache[cache_key] = time.time()
                stats["rejected_by_llm"] += 1
                return

            src_name = result.get("source", "").upper()
            tgt_name = result.get("target", "").upper()
            name_a_upper = nd_a.get("entity_name", n_a).upper()
            name_b_upper = nd_b.get("entity_name", n_b).upper()

            if src_name == name_a_upper and tgt_name == name_b_upper:
                src_id, tgt_id = n_a, n_b
            elif src_name == name_b_upper and tgt_name == name_a_upper:
                src_id, tgt_id = n_b, n_a
            else:
                src_id, tgt_id = n_a, n_b

            from ..._utils import generate_stable_relationship_id

            rel_id = generate_stable_relationship_id(src_id, tgt_id, rel_type)
            edge_data = {
                "description": result.get("evidence", ""),
                "weight": 1.0,
                "source_id": "[]",
                "order": 1,
                "relationship_id": rel_id,
                "relation_type": rel_type,
                "confidence": confidence,
            }
            await knowledge_graph_inst.upsert_edge(src_id, tgt_id, edge_data)

            hub_counter[n_a] += 1
            hub_counter[n_b] += 1
            stats["inferred"] += 1
            logger.info(
                "refinement_inferred",
                source=src_id,
                target=tgt_id,
                relation_type=rel_type,
                confidence=round(confidence, 2),
            )

    await asyncio.gather(*[_evaluate_pair(a, b, c) for a, b, c in candidate_pairs])
    logger.info("refinement_infer_done", **stats)
    return stats


async def _get_all_nodes_safe(knowledge_graph_inst) -> dict[str, dict]:
    if hasattr(knowledge_graph_inst, "get_all_nodes"):
        return await knowledge_graph_inst.get_all_nodes()
    if hasattr(knowledge_graph_inst, "_graph"):
        graph = knowledge_graph_inst._graph
        result = {}
        for node_id in graph.nodes():
            result[node_id] = dict(graph.nodes[node_id])
        return result
    return {}
