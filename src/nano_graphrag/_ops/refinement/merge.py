from __future__ import annotations

import json
from typing import Any

import numpy as np

from ..._utils import _safe_json_loads, get_all_nodes_safe, logger

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

_ANN_K = 50


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

    all_nodes = await get_all_nodes_safe(knowledge_graph_inst)
    if len(all_nodes) < 2:
        return stats

    entity_descriptions = {}
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
    except (RuntimeError, ValueError) as e:
        logger.error("refinement_merge_embedding_failed", error=str(e))
        return stats

    entity_vectors = {}
    for i, node_id in enumerate(node_ids):
        entity_vectors[node_id] = vectors[i]

    vector_dim = vectors.shape[1] if len(vectors.shape) > 1 else len(vectors)
    merge_candidates = _find_merge_candidates_ann(
        all_nodes, entity_vectors, node_ids, vector_dim, merge_threshold
    )

    merge_candidates.sort(key=lambda x: x[2], reverse=True)

    hub_counter: dict[str, int] = {}
    for nid_a, nid_b, sim in merge_candidates:
        stats["examined"] += 1
        if hub_counter.get(nid_a, 0) >= hub_cap or hub_counter.get(nid_b, 0) >= hub_cap:
            stats["skipped"] += 1
            continue
        if not await knowledge_graph_inst.has_node(
            nid_a
        ) or not await knowledge_graph_inst.has_node(nid_b):
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
        a_aliases = _safe_json_loads(nd_a.get("aliases", "[]"), [])
        b_aliases = _safe_json_loads(nd_b.get("aliases", "[]"), [])
        absorbed_name = nd_b.get("entity_name") if keeper == nid_a else nd_a.get("entity_name")
        merged_aliases = sorted(set(a_aliases + b_aliases + [absorbed_name]))
        merged_data["aliases"] = json.dumps(merged_aliases)

        # VDB-first: update vector DB before graph mutation so failures don't corrupt graph
        vdb_ok = True
        if entity_vdb is not None:
            try:
                await entity_vdb.delete([absorbed])
                content = merged_data["entity_name"] + " - " + merged_desc
                await entity_vdb.upsert(
                    {keeper: {"content": content, "entity_name": merged_data["entity_name"]}}
                )
            except Exception as e:
                logger.debug("refinement_merge_vdb_update_failed", error=str(e))
                vdb_ok = False

        if not vdb_ok:
            stats["skipped"] += 1
            continue

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


def _find_merge_candidates_ann(
    all_nodes: dict,
    entity_vectors: dict,
    node_ids: list[str],
    vector_dim: int,
    merge_threshold: float,
) -> list[tuple[str, str, float]]:
    n = len(node_ids)
    if n < 500:
        return _find_merge_candidates_brute_force(
            all_nodes, entity_vectors, node_ids, merge_threshold
        )

    try:
        import hnswlib
    except ImportError:
        return _find_merge_candidates_brute_force(
            all_nodes, entity_vectors, node_ids, merge_threshold
        )

    index = hnswlib.Index(space="cosine", dim=vector_dim)
    index.init_index(max_elements=n, ef_construction=200, M=16)
    index.set_ef(_ANN_K * 2)

    node_id_to_idx = {}
    vectors_array = np.zeros((n, vector_dim), dtype=np.float32)
    for i, nid in enumerate(node_ids):
        vec = np.asarray(entity_vectors[nid], dtype=np.float32)
        vectors_array[i] = vec
        node_id_to_idx[nid] = i
    index.add_items(vectors_array)

    idx_to_node_id = {v: k for k, v in node_id_to_idx.items()}

    merge_candidates = []
    seen_pairs: set[tuple[str, str]] = set()

    for nid in node_ids:
        idx = node_id_to_idx[nid]
        labels, distances = index.knn_query(vectors_array[idx : idx + 1], k=min(_ANN_K, n))
        for label, dist in zip(labels[0], distances[0], strict=False):
            other_nid = idx_to_node_id.get(int(label))
            if other_nid is None or other_nid == nid:
                continue
            pair = (nid, other_nid) if nid < other_nid else (other_nid, nid)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            nd_a = all_nodes[pair[0]]
            nd_b = all_nodes[pair[1]]
            if nd_a.get("entity_type") != nd_b.get("entity_type"):
                continue

            sim = 1.0 - float(dist)
            if sim >= merge_threshold:
                merge_candidates.append((pair[0], pair[1], sim))

    return merge_candidates


def _find_merge_candidates_brute_force(
    all_nodes: dict,
    entity_vectors: dict,
    node_ids: list[str],
    merge_threshold: float,
) -> list[tuple[str, str, float]]:
    merge_candidates = []
    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            nid_a = node_ids[i]
            nid_b = node_ids[j]
            nd_a = all_nodes[nid_a]
            nd_b = all_nodes[nid_b]
            if nd_a.get("entity_type") != nd_b.get("entity_type"):
                continue
            va = entity_vectors[nid_a]
            vb = entity_vectors[nid_b]
            sim = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-8))
            if sim >= merge_threshold:
                merge_candidates.append((nid_a, nid_b, sim))
    return merge_candidates
