import asyncio
import json
from collections import Counter

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
        related_community_keys_counts.keys(),
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
    all_one_hop_nodes = set()
    for this_edges in edges:
        if not this_edges:
            continue
        all_one_hop_nodes.update([e[1] for e in this_edges])
    all_one_hop_nodes = list(all_one_hop_nodes)
    all_one_hop_nodes_data = await knowledge_graph_inst.get_nodes_batch(all_one_hop_nodes)
    all_one_hop_text_units_lookup = {
        k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
        for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
        if v is not None
    }
    all_text_units_lookup = {}
    for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
        for c_id in this_text_units:
            if c_id in all_text_units_lookup:
                continue
            relation_counts = 0
            for e in this_edges:
                if (
                    e[1] in all_one_hop_text_units_lookup
                    and c_id in all_one_hop_text_units_lookup[e[1]]
                ):
                    relation_counts += 1
            all_text_units_lookup[c_id] = {
                "data": await text_chunks_db.get_by_id(c_id),
                "order": index,
                "relation_counts": relation_counts,
            }
    if any([v is None for v in all_text_units_lookup.values()]):
        logger.warning("Text chunks are missing, maybe the storage is damaged")
    all_text_units = [{"id": k, **v} for k, v in all_text_units_lookup.items() if v is not None]
    all_text_units = sorted(all_text_units, key=lambda x: (x["order"], -x["relation_counts"]))
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
):
    all_related_edges = await knowledge_graph_inst.get_nodes_edges_batch(
        [dp["id"] for dp in node_datas]
    )

    all_edges = []
    seen = set()

    for this_edges in all_related_edges:
        for e in this_edges:
            sorted_edge = tuple(sorted(e))
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
):
    results = await entities_vdb.query(query, top_k=query_param.top_k)
    if not len(results):
        return None
    node_datas = await knowledge_graph_inst.get_nodes_batch([r["id"] for r in results])
    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")
    node_degrees = await knowledge_graph_inst.node_degrees_batch([r["id"] for r in results])
    node_datas = [
        {**n, "id": k["id"], "entity_name": n.get("entity_name", k["entity_name"]), "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]
    use_communities = await _find_most_related_community_from_entities(
        node_datas, query_param, community_reports, tokenizer_wrapper
    )
    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas, query_param, text_chunks_db, knowledge_graph_inst, tokenizer_wrapper
    )
    use_relations = await _find_most_related_edges_from_entities(
        node_datas, query_param, knowledge_graph_inst, tokenizer_wrapper
    )
    logger.info(
        f"Using {len(node_datas)} entites, {len(use_communities)} communities, {len(use_relations)} relations, {len(use_text_units)} text units"
    )
    entites_section_list = [["id", "entity", "type", "description", "rank"]]
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

    relations_section_list = [["id", "source", "target", "description", "weight", "rank"]]
    for i, e in enumerate(use_relations):
        relations_section_list.append(
            [
                i,
                e.get("src_entity_name", e["src_tgt"][0]),
                e.get("tgt_entity_name", e["src_tgt"][1]),
                e["description"],
                e["weight"],
                e["rank"],
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)

    communities_section_list = [["id", "content"]]
    for i, c in enumerate(use_communities):
        communities_section_list.append([i, c["report_string"]])
    communities_context = list_of_list_to_csv(communities_section_list)

    text_units_section_list = [["id", "content"]]
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
    use_model_func = global_config["best_model_func"]
    context = await _build_local_query_context(
        query,
        knowledge_graph_inst,
        entities_vdb,
        community_reports,
        text_chunks_db,
        query_param,
        tokenizer_wrapper,
    )
    if query_param.only_need_context:
        return context
    if context is None:
        return PROMPTS["fail_response"]
    sys_prompt_temp = PROMPTS["local_rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context, response_type=query_param.response_type
    )
    return await use_model_func(
        query,
        system_prompt=sys_prompt,
    )


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
        communities_section_list = [["id", "content", "rating", "importance"]]
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

    logger.info(f"Grouping to {len(community_groups)} groups for global search")
    return await asyncio.gather(*[_process(c) for c in community_groups])


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
    community_schema = await knowledge_graph_inst.community_schema()
    community_schema = {
        k: v for k, v in community_schema.items() if v["level"] <= query_param.level
    }
    if not len(community_schema):
        return PROMPTS["fail_response"]
    use_model_func = global_config["best_model_func"]

    sorted_community_schemas = sorted(
        community_schema.items(),
        key=lambda x: x[1]["occurrence"],
        reverse=True,
    )
    sorted_community_schemas = sorted_community_schemas[: query_param.global_max_consider_community]
    community_datas = await community_reports.get_by_ids([k[0] for k in sorted_community_schemas])
    community_datas = [c for c in community_datas if c is not None]
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
    logger.info(f"Revtrieved {len(community_datas)} communities")

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
        return PROMPTS["fail_response"]
    final_support_points = sorted(final_support_points, key=lambda x: x["score"], reverse=True)
    final_support_points = truncate_list_by_token_size(
        final_support_points,
        key=lambda x: x["answer"],
        max_token_size=query_param.global_max_token_for_community_report,
        tokenizer_wrapper=tokenizer_wrapper,
    )
    points_context = []
    for dp in final_support_points:
        points_context.append(
            f"""----Analyst {dp["analyst"]}----
Importance Score: {dp["score"]}
{dp["answer"]}
"""
        )
    points_context = "\n".join(points_context)
    if query_param.only_need_context:
        return points_context
    sys_prompt_temp = PROMPTS["global_reduce_rag_response"]
    return await use_model_func(
        query,
        sys_prompt_temp.format(report_data=points_context, response_type=query_param.response_type),
    )


async def naive_query(
    query,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    tokenizer_wrapper,
    global_config: dict,
):
    use_model_func = global_config["best_model_func"]
    results = await chunks_vdb.query(query, top_k=query_param.top_k)
    if not len(results):
        return PROMPTS["fail_response"]
    chunks_ids = [r["id"] for r in results]
    chunks = await text_chunks_db.get_by_ids(chunks_ids)

    maybe_trun_chunks = truncate_list_by_token_size(
        chunks,
        key=lambda x: x["content"],
        max_token_size=query_param.naive_max_token_for_text_unit,
        tokenizer_wrapper=tokenizer_wrapper,
    )
    logger.info(f"Truncate {len(chunks)} to {len(maybe_trun_chunks)} chunks")
    section = "--New Chunk--\n".join([c["content"] for c in maybe_trun_chunks])
    if query_param.only_need_context:
        return section
    sys_prompt_temp = PROMPTS["naive_rag_response"]
    sys_prompt = sys_prompt_temp.format(
        content_data=section, response_type=query_param.response_type
    )
    return await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
