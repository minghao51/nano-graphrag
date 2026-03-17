import asyncio
from typing import Any, Callable

from .._utils import TokenizerWrapper, list_of_list_to_csv, logger, truncate_list_by_token_size
from ..base import BaseGraphStorage, BaseKVStorage, CommunitySchema, SingleCommunitySchema
from ..prompt import PROMPTS


def _pack_single_community_by_sub_communities(
    community: SingleCommunitySchema,
    max_token_size: int,
    already_reports: dict[str, CommunitySchema],
    tokenizer_wrapper: TokenizerWrapper,
) -> tuple[str, int, set, set]:
    all_sub_communities = [
        already_reports[k] for k in community["sub_communities"] if k in already_reports
    ]
    all_sub_communities = sorted(all_sub_communities, key=lambda x: x["occurrence"], reverse=True)

    may_trun_all_sub_communities = truncate_list_by_token_size(
        all_sub_communities,
        key=lambda x: x["report_string"],
        max_token_size=max_token_size,
        tokenizer_wrapper=tokenizer_wrapper,
    )
    sub_fields = ["id", "report", "rating", "importance"]
    sub_communities_describe = list_of_list_to_csv(
        [sub_fields]
        + [
            [
                i,
                c["report_string"],
                c["report_json"].get("rating", -1),
                c["occurrence"],
            ]
            for i, c in enumerate(may_trun_all_sub_communities)
        ]
    )
    already_nodes = []
    already_edges = []
    for c in may_trun_all_sub_communities:
        already_nodes.extend(c["nodes"])
        already_edges.extend([tuple(e) for e in c["edges"]])

    return (
        sub_communities_describe,
        len(tokenizer_wrapper.encode(sub_communities_describe)),
        set(already_nodes),
        set(already_edges),
    )


async def _pack_single_community_describe(
    knwoledge_graph_inst: BaseGraphStorage,
    community: SingleCommunitySchema,
    tokenizer_wrapper: TokenizerWrapper,
    max_token_size: int = 12000,
    already_reports: dict[str, CommunitySchema] = {},
    global_config: dict = {},
) -> str:
    nodes_in_order = sorted(community["nodes"])
    edges_in_order = sorted(community["edges"], key=lambda x: x[0] + x[1])

    nodes_data = await asyncio.gather(*[knwoledge_graph_inst.get_node(n) for n in nodes_in_order])
    edges_data = await asyncio.gather(
        *[knwoledge_graph_inst.get_edge(src, tgt) for src, tgt in edges_in_order]
    )

    final_template = """-----Reports-----
```csv
{reports}
```
-----Entities-----
```csv
{entities}
```
-----Relationships-----
```csv
{relationships}
```"""
    base_template_tokens = len(
        tokenizer_wrapper.encode(final_template.format(reports="", entities="", relationships=""))
    )
    remaining_budget = max_token_size - base_template_tokens

    report_describe = ""
    contain_nodes = set()
    contain_edges = set()

    truncated = len(nodes_in_order) > 100 or len(edges_in_order) > 100

    need_to_use_sub_communities = truncated and community["sub_communities"] and already_reports
    force_to_use_sub_communities = global_config["addon_params"].get(
        "force_to_use_sub_communities", False
    )

    if need_to_use_sub_communities or force_to_use_sub_communities:
        logger.debug(f"Community {community['title']} using sub-communities")
        result = _pack_single_community_by_sub_communities(
            community, remaining_budget, already_reports, tokenizer_wrapper
        )
        report_describe, report_size, contain_nodes, contain_edges = result
        remaining_budget = max(0, remaining_budget - report_size)

    def format_row(row: list) -> str:
        return ",".join('"{}"'.format(str(item).replace('"', '""')) for item in row)

    node_fields = ["id", "entity", "type", "description", "degree"]
    edge_fields = ["id", "source", "target", "description", "rank"]

    node_degrees = await knwoledge_graph_inst.node_degrees_batch(nodes_in_order)
    edge_degrees = await knwoledge_graph_inst.edge_degrees_batch(edges_in_order)

    nodes_list_data = [
        [
            i,
            name,
            data.get("entity_type", "UNKNOWN"),
            data.get("description", "UNKNOWN"),
            node_degrees[i],
        ]
        for i, (name, data) in enumerate(zip(nodes_in_order, nodes_data))
        if name not in contain_nodes
    ]

    edges_list_data = [
        [i, edge[0], edge[1], data.get("description", "UNKNOWN"), edge_degrees[i]]
        for i, (edge, data) in enumerate(zip(edges_in_order, edges_data))
        if (edge[0], edge[1]) not in contain_edges
    ]

    nodes_list_data.sort(key=lambda x: x[-1], reverse=True)
    edges_list_data.sort(key=lambda x: x[-1], reverse=True)

    header_tokens = len(
        tokenizer_wrapper.encode(
            list_of_list_to_csv([node_fields]) + "\n" + list_of_list_to_csv([edge_fields])
        )
    )

    data_budget = max(0, remaining_budget - header_tokens)
    total_items = len(nodes_list_data) + len(edges_list_data)
    node_ratio = len(nodes_list_data) / max(1, total_items)
    edge_ratio = 1 - node_ratio

    nodes_final = truncate_list_by_token_size(
        nodes_list_data,
        key=format_row,
        max_token_size=int(data_budget * node_ratio),
        tokenizer_wrapper=tokenizer_wrapper,
    )
    edges_final = truncate_list_by_token_size(
        edges_list_data,
        key=format_row,
        max_token_size=int(data_budget * edge_ratio),
        tokenizer_wrapper=tokenizer_wrapper,
    )

    nodes_describe = list_of_list_to_csv([node_fields] + nodes_final)
    edges_describe = list_of_list_to_csv([edge_fields] + edges_final)

    return final_template.format(
        reports=report_describe,
        entities=nodes_describe,
        relationships=edges_describe,
    )


def _community_report_json_to_str(parsed_output: dict) -> str:
    title = parsed_output.get("title", "Report")
    summary = parsed_output.get("summary", "")
    findings = parsed_output.get("findings", [])

    def finding_summary(finding: dict):
        if isinstance(finding, str):
            return finding
        return finding.get("summary")

    def finding_explanation(finding: dict):
        if isinstance(finding, str):
            return ""
        return finding.get("explanation")

    report_sections = "\n\n".join(
        f"## {finding_summary(f)}\n\n{finding_explanation(f)}" for f in findings
    )
    return f"# {title}\n\n{summary}\n\n{report_sections}"


async def generate_community_report(
    community_report_kv: BaseKVStorage[CommunitySchema],
    knwoledge_graph_inst: BaseGraphStorage,
    tokenizer_wrapper: TokenizerWrapper,
    global_config: dict,
):
    llm_extra_kwargs = global_config["special_community_report_llm_kwargs"]
    use_llm_func: Callable[..., Any] = global_config["best_model_func"]
    use_string_json_convert_func: Callable[..., Any] = global_config[
        "convert_response_to_json_func"
    ]

    communities_schema = await knwoledge_graph_inst.community_schema()
    community_keys, community_values = (
        list(communities_schema.keys()),
        list(communities_schema.values()),
    )
    already_processed = 0

    prompt_template = PROMPTS["community_report"]
    prompt_overhead = len(tokenizer_wrapper.encode(prompt_template.format(input_text="")))

    async def _form_single_community_report(
        community: SingleCommunitySchema, already_reports: dict[str, CommunitySchema]
    ):
        nonlocal already_processed
        describe = await _pack_single_community_describe(
            knwoledge_graph_inst,
            community,
            tokenizer_wrapper=tokenizer_wrapper,
            max_token_size=global_config["best_model_max_token_size"] - prompt_overhead - 200,
            already_reports=already_reports,
            global_config=global_config,
        )
        prompt = prompt_template.format(input_text=describe)

        response = await use_llm_func(prompt, **llm_extra_kwargs)
        data = use_string_json_convert_func(response)
        already_processed += 1
        now_ticks = PROMPTS["process_tickers"][already_processed % len(PROMPTS["process_tickers"])]
        print(f"{now_ticks} Processed {already_processed} communities\r", end="", flush=True)
        return data

    levels = sorted(set([c["level"] for c in community_values]), reverse=True)
    logger.info(f"Generating by levels: {levels}")
    community_datas = {}
    for level in levels:
        this_level_community_keys, this_level_community_values = zip(
            *[(k, v) for k, v in zip(community_keys, community_values) if v["level"] == level]
        )
        this_level_communities_reports = await asyncio.gather(
            *[
                _form_single_community_report(c, community_datas)
                for c in this_level_community_values
            ]
        )
        community_datas.update(
            {
                k: {
                    "report_string": _community_report_json_to_str(r),
                    "report_json": r,
                    **v,
                }
                for k, r, v in zip(
                    this_level_community_keys,
                    this_level_communities_reports,
                    this_level_community_values,
                )
            }
        )
    print()
    await community_report_kv.upsert(community_datas)
