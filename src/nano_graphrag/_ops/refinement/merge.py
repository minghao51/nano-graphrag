from __future__ import annotations

from typing import Any

from ..._utils import logger

DREAM_MERGE_PROMPT = """You are a knowledge-graph curator. Two entities of type "{entity_type}" have been detected
as near-duplicates and need to be merged into one.

Entity A — name: "{name_a}"
Description: {description_a}

Entity B — name: "{name_b}"
Description: {description_b}

Write a single, merged description that preserves ALL information from BOTH descriptions.
Do not lose any facts. Be concise but complete. Write in factual, third-person,
knowledge-base style.

Return ONLY the merged description text. No preamble, no explanation."""


async def _merge_phase(
    knowledge_graph_inst,
    entity_vdb,
    global_config: dict,
    merge_threshold: float = 0.93,
    hub_cap: int = 3,
) -> dict[str, Any]:
    stats = {"examined": 0, "merged": 0, "skipped": 0}

    use_llm_func = global_config.get("cheap_model_func")
    embedding_func = global_config.get("embedding_func")

    if use_llm_func is None or embedding_func is None:
        logger.warning("refinement_merge_skipped", reason="missing_llm_or_embedding_func")
        return stats

    all_nodes = await _get_all_nodes_safe(knowledge_graph_inst)
    if len(all_nodes) < 2:
        return stats

    entity_descriptions = {}
    entity_vectors = {}
    for node_id, node_data in all_nodes.items():
        desc = node_data.get("description", "")
        if desc:
            entity_descriptions[node_id] = desc

    if not entity_descriptions:
        return stats

    text_list = list(entity_descriptions.values())
    node_ids = list(entity_descriptions.keys())
    try:
        vectors = await embedding_func(text_list)
    except Exception as e:
        logger.error("refinement_merge_embedding_failed", error=str(e))
        return stats

    for i, node_id in enumerate(node_ids):
        entity_vectors[node_id] = vectors[i]

    import numpy as np

    merge_candidates = []
    node_ids_list = list(entity_vectors.keys())
    for i in range(len(node_ids_list)):
        for j in range(i + 1, len(node_ids_list)):
            nid_a = node_ids_list[i]
            nid_b = node_ids_list[j]
            nd_a = all_nodes[nid_a]
            nd_b = all_nodes[nid_b]
            if nd_a.get("entity_type") != nd_b.get("entity_type"):
                continue
            va = entity_vectors[nid_a]
            vb = entity_vectors[nid_b]
            sim = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-8))
            if sim >= merge_threshold:
                merge_candidates.append((nid_a, nid_b, sim))

    merge_candidates.sort(key=lambda x: x[2], reverse=True)

    hub_counter: dict[str, int] = {}
    for nid_a, nid_b, sim in merge_candidates:
        stats["examined"] += 1
        if hub_counter.get(nid_a, 0) >= hub_cap or hub_counter.get(nid_b, 0) >= hub_cap:
            stats["skipped"] += 1
            continue
        if not await knowledge_graph_inst.has_node(nid_a) or not knowledge_graph_inst.has_node(
            nid_b
        ):
            continue

        nd_a = await knowledge_graph_inst.get_node(nid_a)
        nd_b = await knowledge_graph_inst.get_node(nid_b)
        if nd_a is None or nd_b is None:
            continue

        prompt = DREAM_MERGE_PROMPT.format(
            entity_type=nd_a.get("entity_type", "UNKNOWN"),
            name_a=nd_a.get("entity_name", nid_a),
            description_a=nd_a.get("description", ""),
            name_b=nd_b.get("entity_name", nid_b),
            description_b=nd_b.get("description", ""),
        )
        try:
            merged_desc = await use_llm_func(prompt)
            if isinstance(merged_desc, list):
                merged_desc = merged_desc[0].get("text", str(merged_desc))
        except Exception as e:
            logger.debug("refinement_merge_llm_failed", error=str(e))
            continue

        keeper = (
            nid_a if len(nd_a.get("description", "")) >= len(nd_b.get("description", "")) else nid_b
        )
        absorbed = nid_b if keeper == nid_a else nid_a
        keeper_data = nd_a if keeper == nid_a else nd_b

        merged_data = {
            "entity_name": keeper_data.get("entity_name"),
            "entity_type": keeper_data.get("entity_type"),
            "description": merged_desc,
            "source_id": keeper_data.get("source_id", ""),
        }
        aliases_a = nd_a.get("aliases", "[]")
        aliases_b = nd_b.get("aliases", "[]")
        import json

        try:
            a_aliases = json.loads(aliases_a) if isinstance(aliases_a, str) else aliases_a
        except (json.JSONDecodeError, TypeError):
            a_aliases = []
        try:
            b_aliases = json.loads(aliases_b) if isinstance(aliases_b, str) else aliases_b
        except (json.JSONDecodeError, TypeError):
            b_aliases = []
        absorbed_name = nd_b.get("entity_name") if keeper == nid_a else nd_a.get("entity_name")
        merged_aliases = sorted(set(a_aliases + b_aliases + [absorbed_name]))
        merged_data["aliases"] = json.dumps(merged_aliases)

        await knowledge_graph_inst.upsert_node(keeper, merged_data)

        edges = await knowledge_graph_inst.get_node_edges(absorbed)
        if edges:
            for src, tgt in edges:
                edge_data = await knowledge_graph_inst.get_edge(src, tgt)
                if edge_data is None:
                    continue
                new_src = keeper if src == absorbed else src
                new_tgt = keeper if tgt == absorbed else tgt
                if new_src == new_tgt:
                    continue
                if not await knowledge_graph_inst.has_edge(new_src, new_tgt):
                    await knowledge_graph_inst.upsert_edge(new_src, new_tgt, edge_data)
            await knowledge_graph_inst.delete_node(absorbed)

        if entity_vdb is not None:
            try:
                await entity_vdb.delete([absorbed])
                content = merged_data["entity_name"] + " - " + merged_desc
                await entity_vdb.upsert(
                    {keeper: {"content": content, "entity_name": merged_data["entity_name"]}}
                )
            except Exception as e:
                logger.debug("refinement_merge_vdb_update_failed", error=str(e))

        hub_counter[nid_a] = hub_counter.get(nid_a, 0) + 1
        hub_counter[nid_b] = hub_counter.get(nid_b, 0) + 1
        stats["merged"] += 1
        logger.info(
            "refinement_merged",
            keeper=keeper,
            absorbed=absorbed,
            similarity=round(sim, 3),
        )

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
    logger.warning("refinement_cannot_list_nodes")
    return {}
