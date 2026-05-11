from __future__ import annotations

import asyncio
import json
from collections import Counter
from typing import Any

from .._utils import (
    list_of_list_to_csv,
    logger,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
)
from ..base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    CommunitySchema,
    QueryParam,
    TextChunkSchema,
)
from ..prompt import GRAPH_FIELD_SEP, PROMPTS


def _normalize_date_str(s):
    if not s:
        return ""
    s = s.strip()
    if len(s) == 4 and s.isdigit():
        return f"{s}-01-01"
    if len(s) == 7 and s[4] == "-":
        return f"{s}-01"
    return s


def _edge_matches_time_range(edge_data: dict, query_param: QueryParam) -> bool:
    if query_param.time_range is None:
        return True
    valid_from = _normalize_date_str(edge_data.get("valid_from"))
    valid_to = _normalize_date_str(edge_data.get("valid_to"))
    if not valid_from and not valid_to:
        return True
    query_start = _normalize_date_str(query_param.time_range[0])
    query_end = _normalize_date_str(query_param.time_range[1])
    if query_param.temporal_mode == "at_point":
        if valid_from and query_start < valid_from:
            return False
        if valid_to and query_start > valid_to:
            return False
        return True
    edge_start = valid_from or ""
    edge_end = valid_to or "9999"
    return not (edge_end < query_start or edge_start > query_end)


async def _find_most_related_community_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    community_reports: BaseKVStorage[CommunitySchema],
    tokenizer_wrapper,
):
    related_communities = []
    for node_d in node_datas:
        if "clusters" not in node_d:
            continue
        related_communities.extend(json.loads(node_d["clusters"]))
    related_community_dup_keys = [
        str(dp["cluster"]) for dp in related_communities if dp["level"] <= query_param.level
    ]
    related_community_keys_counts = dict(Counter(related_community_dup_keys))
    _related_community_datas = await asyncio.gather(
        *[community_reports.get_by_id(k) for k in related_community_keys_counts.keys()]
    )
    related_community_datas = {
        k: v
        for k, v in zip(related_community_keys_counts.keys(), _related_community_datas)
        if v is not None
    }
    related_community_keys = sorted(
        [k for k in related_community_keys_counts.keys() if k in related_community_datas],
        key=lambda k: (
            related_community_keys_counts[k],
            related_community_datas[k]["report_json"].get("rating", -1),
        ),
        reverse=True,
    )
    sorted_community_datas = [related_community_datas[k] for k in related_community_keys]

    use_community_reports = truncate_list_by_token_size(
        sorted_community_datas,
        key=lambda x: x["report_string"],
        max_token_size=query_param.local_max_token_for_community_report,
        tokenizer_wrapper=tokenizer_wrapper,
    )
    if query_param.local_community_single_one:
        use_community_reports = use_community_reports[:1]
    return use_community_reports


async def _find_most_related_text_unit_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    tokenizer_wrapper,
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP]) for dp in node_datas
    ]
    edges = await knowledge_graph_inst.get_nodes_edges_batch([dp["id"] for dp in node_datas])
    all_one_hop_nodes: set[str] = set()
    for this_edges in edges:
        if not this_edges:
            continue
        all_one_hop_nodes.update([e[1] for e in this_edges])
    all_one_hop_node_list: list[str] = list(all_one_hop_nodes)
    all_one_hop_nodes_data = await knowledge_graph_inst.get_nodes_batch(all_one_hop_node_list)
    all_one_hop_text_units_lookup = {
        k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
        for k, v in zip(all_one_hop_node_list, all_one_hop_nodes_data)
        if v is not None
    }

    all_chunk_ids = set()
    for this_text_units in text_units:
        all_chunk_ids.update(this_text_units)

    all_chunk_data = await text_chunks_db.get_by_ids(list(all_chunk_ids))
    chunk_data_lookup = {
        cid: data for cid, data in zip(all_chunk_ids, all_chunk_data) if data is not None
    }

    all_text_units_lookup = {}
    for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
        for c_id in this_text_units:
            if c_id in all_text_units_lookup:
                continue
            if c_id not in chunk_data_lookup:
                continue
            relation_counts = 0
            for e in this_edges:
                if (
                    e[1] in all_one_hop_text_units_lookup
                    and c_id in all_one_hop_text_units_lookup[e[1]]
                ):
                    relation_counts += 1
            all_text_units_lookup[c_id] = {
                "data": chunk_data_lookup[c_id],
                "order": index,
                "relation_counts": relation_counts,
            }
    if any(v is None for v in all_text_units_lookup.values()):
        logger.warning("text_chunks_missing")
    all_text_units = [{"id": k, **v} for k, v in all_text_units_lookup.items() if v is not None]
    all_text_units = sorted(all_text_units, key=lambda x: (x["order"], -int(x["relation_counts"])))  # type: ignore[call-overload]
    all_text_units = truncate_list_by_token_size(
        all_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.local_max_token_for_text_unit,
        tokenizer_wrapper=tokenizer_wrapper,
    )
    return [t["data"] for t in all_text_units]


async def _find_most_related_edges_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
    tokenizer_wrapper,
    global_config: dict | None = None,
):
    all_related_edges = await knowledge_graph_inst.get_nodes_edges_batch(
        [dp["id"] for dp in node_datas]
    )

    all_edges: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for this_edges in all_related_edges:
        for e in this_edges:
            sorted_edge: tuple[str, str] = tuple(sorted(e))  # type: ignore[assignment]
            if sorted_edge not in seen:
                seen.add(sorted_edge)
                all_edges.append(sorted_edge)

    all_edges_pack = await knowledge_graph_inst.get_edges_batch(all_edges)
    all_edges_degree = await knowledge_graph_inst.edge_degrees_batch(all_edges)
    related_node_ids = sorted({node_id for edge in all_edges for node_id in edge})
    related_nodes = await knowledge_graph_inst.get_nodes_batch(related_node_ids)
    node_name_lookup = {
        node_id: node_data.get("entity_name", node_id)
        for node_id, node_data in zip(related_node_ids, related_nodes)
        if node_data is not None
    }
    all_edges_data = [
        {
            "src_tgt": k,
            "src_entity_name": node_name_lookup.get(k[0], k[0]),
            "tgt_entity_name": node_name_lookup.get(k[1], k[1]),
            "rank": d,
            **v,
        }
        for k, v, d in zip(all_edges, all_edges_pack, all_edges_degree)
        if v is not None
    ]
    if query_param.time_range is not None:
        all_edges_data = [e for e in all_edges_data if _edge_matches_time_range(e, query_param)]
    confidence_threshold = (global_config or {}).get("relationship_confidence_threshold", 0.0)
    if confidence_threshold > 0:
        all_edges_data = [
            e for e in all_edges_data if e.get("confidence", 0.8) >= confidence_threshold
        ]
    all_edges_data = sorted(all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True)
    return truncate_list_by_token_size(
        all_edges_data,
        key=lambda x: x["description"],
        max_token_size=query_param.local_max_token_for_local_context,
        tokenizer_wrapper=tokenizer_wrapper,
    )


async def _build_local_query_context(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    tokenizer_wrapper,
    global_config: dict | None = None,
):
    results = await entities_vdb.query(query, top_k=query_param.top_k)
    if not len(results):
        return None
    node_datas_raw = await knowledge_graph_inst.get_nodes_batch([r["id"] for r in results])
    if not all(n is not None for n in node_datas_raw):
        logger.warning("some_nodes_missing")
    node_degrees = await knowledge_graph_inst.node_degrees_batch([r["id"] for r in results])
    node_datas: list[dict[Any, Any]] = [
        {**n, "id": k["id"], "entity_name": n.get("entity_name", k["entity_name"]), "rank": d}
        for k, n, d in zip(results, node_datas_raw, node_degrees)
        if n is not None
    ]
    use_communities = await _find_most_related_community_from_entities(
        node_datas, query_param, community_reports, tokenizer_wrapper
    )
    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas, query_param, text_chunks_db, knowledge_graph_inst, tokenizer_wrapper
    )
    use_relations = await _find_most_related_edges_from_entities(
        node_datas,
        query_param,
        knowledge_graph_inst,
        tokenizer_wrapper,
        global_config=global_config,
    )
    logger.info(
        "local_query_context",
        entities=len(node_datas),
        communities=len(use_communities),
        relations=len(use_relations),
        text_units=len(use_text_units),
    )
    entites_section_list: list[list[Any]] = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)

    relations_section_list: list[list[Any]] = [
        ["id", "source", "target", "description", "relation_type", "weight", "rank", "temporal"]
    ]
    for i, e in enumerate(use_relations):
        temporal = e.get("temporal_context") or ""
        valid_from = e.get("valid_from") or ""
        valid_to = e.get("valid_to") or ""
        if valid_from or valid_to:
            temporal = f"{valid_from or '?'} to {valid_to or 'now'}"
        relations_section_list.append(
            [
                i,
                e.get("src_entity_name", e["src_tgt"][0]),
                e.get("tgt_entity_name", e["src_tgt"][1]),
                e["description"],
                e.get("relation_type", "related_to"),
                e["weight"],
                e["rank"],
                temporal,
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)

    communities_section_list: list[list[Any]] = [["id", "content"]]
    for i, c in enumerate(use_communities):
        communities_section_list.append([i, c["report_string"]])
    communities_context = list_of_list_to_csv(communities_section_list)

    text_units_section_list: list[list[Any]] = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    return f"""
-----Reports-----
```csv
{communities_context}
```
-----Entities-----
```csv
{entities_context}
```
-----Relationships-----
```csv
{relations_context}
```
-----Sources-----
```csv
{text_units_context}
```
"""


async def local_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    tokenizer_wrapper,
    global_config: dict,
) -> str:
    context = await _build_local_query_context(
        query,
        knowledge_graph_inst,
        entities_vdb,
        community_reports,
        text_chunks_db,
        query_param,
        tokenizer_wrapper,
        global_config=global_config,
    )
    if query_param.only_need_context:
        return context
    if context is None:
        return PROMPTS["fail_response"]
    sys_prompt = PROMPTS["local_rag_response"].format(
        context_data=context, response_type=query_param.response_type
    )
    return await global_config["best_model_func"](query, system_prompt=sys_prompt)


async def local_query_stream(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    tokenizer_wrapper,
    global_config: dict,
):
    context = await _build_local_query_context(
        query,
        knowledge_graph_inst,
        entities_vdb,
        community_reports,
        text_chunks_db,
        query_param,
        tokenizer_wrapper,
        global_config=global_config,
    )
    if query_param.only_need_context:
        if context is not None:
            yield context
        return
    if context is None:
        yield PROMPTS["fail_response"]
        return
    sys_prompt = PROMPTS["local_rag_response"].format(
        context_data=context, response_type=query_param.response_type
    )
    stream_func = global_config["best_model_stream_func"]
    async for chunk in stream_func(query, system_prompt=sys_prompt):
        yield chunk


async def _map_global_communities(
    query: str,
    communities_data: list[CommunitySchema],
    query_param: QueryParam,
    global_config: dict,
    tokenizer_wrapper,
):
    use_string_json_convert_func = global_config["convert_response_to_json_func"]
    use_model_func = global_config["best_model_func"]
    community_groups = []
    while len(communities_data):
        this_group = truncate_list_by_token_size(
            communities_data,
            key=lambda x: x["report_string"],
            max_token_size=query_param.global_max_token_for_community_report,
            tokenizer_wrapper=tokenizer_wrapper,
        )
        community_groups.append(this_group)
        communities_data = communities_data[len(this_group) :]

    async def _process(community_truncated_datas: list[CommunitySchema]) -> dict:
        communities_section_list: list[list[Any]] = [["id", "content", "rating", "importance"]]
        for i, c in enumerate(community_truncated_datas):
            communities_section_list.append(
                [
                    i,
                    c["report_string"],
                    c["report_json"].get("rating", 0),
                    c["occurrence"],
                ]
            )
        community_context = list_of_list_to_csv(communities_section_list)
        sys_prompt_temp = PROMPTS["global_map_rag_points"]
        sys_prompt = sys_prompt_temp.format(context_data=community_context)
        response = await use_model_func(
            query,
            system_prompt=sys_prompt,
            **query_param.global_special_community_map_llm_kwargs,
        )
        data = use_string_json_convert_func(response)
        return data.get("points", [])

    logger.info("global_search_groups", groups=len(community_groups))
    return await asyncio.gather(*[_process(c) for c in community_groups])


async def _build_global_query_context(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    query_param: QueryParam,
    tokenizer_wrapper,
    global_config: dict,
):
    community_schema = await knowledge_graph_inst.community_schema()
    community_schema = {
        k: v for k, v in community_schema.items() if v["level"] <= query_param.level
    }
    if not len(community_schema):
        return None
    sorted_community_schemas = sorted(
        community_schema.items(),
        key=lambda x: x[1]["occurrence"],
        reverse=True,
    )
    sorted_community_schemas = sorted_community_schemas[: query_param.global_max_consider_community]
    community_datas_raw = await community_reports.get_by_ids(
        [k[0] for k in sorted_community_schemas]
    )
    community_datas: list[CommunitySchema] = [c for c in community_datas_raw if c is not None]
    community_datas = [
        c
        for c in community_datas
        if c["report_json"].get("rating", 0) >= query_param.global_min_community_rating
    ]
    community_datas = sorted(
        community_datas,
        key=lambda x: (x["occurrence"], x["report_json"].get("rating", 0)),
        reverse=True,
    )
    logger.info("global_retrieved_communities", count=len(community_datas))

    map_communities_points = await _map_global_communities(
        query, community_datas, query_param, global_config, tokenizer_wrapper
    )
    final_support_points = []
    for i, mc in enumerate(map_communities_points):
        for point in mc:
            if "description" not in point:
                continue
            final_support_points.append(
                {
                    "analyst": i,
                    "answer": point["description"],
                    "score": point.get("score", 1),
                }
            )
    final_support_points = [p for p in final_support_points if p["score"] > 0]
    if not len(final_support_points):
        return None
    final_support_points = sorted(final_support_points, key=lambda x: x["score"], reverse=True)
    final_support_points = truncate_list_by_token_size(
        final_support_points,
        key=lambda x: x["answer"],
        max_token_size=query_param.global_max_token_for_community_report,
        tokenizer_wrapper=tokenizer_wrapper,
    )
    return "\n".join(
        f"""----Analyst {dp["analyst"]}----
Importance Score: {dp["score"]}
{dp["answer"]}
"""
        for dp in final_support_points
    )


async def global_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    tokenizer_wrapper,
    global_config: dict,
) -> str:
    points_context = await _build_global_query_context(
        query,
        knowledge_graph_inst,
        community_reports,
        query_param,
        tokenizer_wrapper,
        global_config,
    )
    if points_context is None:
        return PROMPTS["fail_response"]
    if query_param.only_need_context:
        return points_context
    sys_prompt = PROMPTS["global_reduce_rag_response"].format(
        report_data=points_context, response_type=query_param.response_type
    )
    return await global_config["best_model_func"](query, system_prompt=sys_prompt)


async def global_query_stream(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    tokenizer_wrapper,
    global_config: dict,
):
    points_context = await _build_global_query_context(
        query,
        knowledge_graph_inst,
        community_reports,
        query_param,
        tokenizer_wrapper,
        global_config,
    )
    if points_context is None:
        yield PROMPTS["fail_response"]
        return
    if query_param.only_need_context:
        yield points_context
        return
    prompt = PROMPTS["global_reduce_rag_response"].format(
        report_data=points_context, response_type=query_param.response_type
    )
    stream_func = global_config["best_model_stream_func"]
    async for chunk in stream_func(query, system_prompt=prompt):
        yield chunk


async def _build_naive_query_context(
    query,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    tokenizer_wrapper,
):
    results = await chunks_vdb.query(query, top_k=query_param.top_k)
    if not len(results):
        return None
    chunks_ids = [r["id"] for r in results]
    chunks = await text_chunks_db.get_by_ids(chunks_ids)
    maybe_trun_chunks = truncate_list_by_token_size(
        chunks,
        key=lambda x: x["content"],
        max_token_size=query_param.naive_max_token_for_text_unit,
        tokenizer_wrapper=tokenizer_wrapper,
    )
    logger.info("truncate_chunks", before=len(chunks), after=len(maybe_trun_chunks))
    return "--New Chunk--\n".join([c["content"] for c in maybe_trun_chunks])


async def naive_query(
    query,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    tokenizer_wrapper,
    global_config: dict,
) -> str:
    section = await _build_naive_query_context(
        query,
        chunks_vdb,
        text_chunks_db,
        query_param,
        tokenizer_wrapper,
    )
    if section is None:
        return PROMPTS["fail_response"]
    if query_param.only_need_context:
        return section
    sys_prompt = PROMPTS["naive_rag_response"].format(
        content_data=section, response_type=query_param.response_type
    )
    return await global_config["best_model_func"](query, system_prompt=sys_prompt)


async def naive_query_stream(
    query,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    tokenizer_wrapper,
    global_config: dict,
):
    section = await _build_naive_query_context(
        query,
        chunks_vdb,
        text_chunks_db,
        query_param,
        tokenizer_wrapper,
    )
    if section is None:
        yield PROMPTS["fail_response"]
        return
    if query_param.only_need_context:
        yield section
        return
    sys_prompt = PROMPTS["naive_rag_response"].format(
        content_data=section, response_type=query_param.response_type
    )
    stream_func = global_config["best_model_stream_func"]
    async for chunk in stream_func(query, system_prompt=sys_prompt):
        yield chunk
