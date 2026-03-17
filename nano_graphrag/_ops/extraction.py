import asyncio
import json
import re
from collections import Counter, defaultdict
from typing import Any, Callable, Union

from .._utils import (
    TokenizerWrapper,
    clean_str,
    compute_mdhash_id,
    is_float_regex,
    logger,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
)
from ..base import BaseGraphStorage, BaseVectorStorage, TextChunkSchema
from ..prompt import GRAPH_FIELD_SEP, PROMPTS


async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
    tokenizer_wrapper: TokenizerWrapper,
) -> str:
    use_llm_func: Callable[..., Any] = global_config["cheap_model_func"]
    llm_max_tokens = global_config["cheap_model_max_token_size"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]

    tokens = tokenizer_wrapper.encode(description)
    if len(tokens) < summary_max_tokens:
        return description
    prompt_template = PROMPTS["summarize_entity_descriptions"]

    use_description = tokenizer_wrapper.decode(tokens[:llm_max_tokens])
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
    )
    use_prompt = prompt_template.format(**context_base)
    logger.debug(f"Trigger summary: {entity_or_relation_name}")
    summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)
    return summary


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    entity_source_id = chunk_key
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=entity_source_id,
    )


async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
        return None
    source = clean_str(record_attributes[1].upper())
    target = clean_str(record_attributes[2].upper())
    edge_description = clean_str(record_attributes[3])
    edge_source_id = chunk_key
    weight = float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    return dict(
        src_id=source,
        tgt_id=target,
        weight=weight,
        description=edge_description,
        source_id=edge_source_id,
    )


async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knwoledge_graph_inst: BaseGraphStorage,
    global_config: dict,
    tokenizer_wrapper,
):
    already_entitiy_types = []
    already_source_ids = []
    already_description = []

    already_node = await knwoledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        already_entitiy_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])

    entity_type = sorted(
        Counter([dp["entity_type"] for dp in nodes_data] + already_entitiy_types).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    description = await _handle_entity_relation_summary(
        entity_name, description, global_config, tokenizer_wrapper
    )
    node_data = dict(
        entity_type=entity_type,
        description=description,
        source_id=source_id,
    )
    await knwoledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    return node_data


async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knwoledge_graph_inst: BaseGraphStorage,
    global_config: dict,
    tokenizer_wrapper,
):
    already_weights = []
    already_source_ids = []
    already_description = []
    already_order = []
    if await knwoledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knwoledge_graph_inst.get_edge(src_id, tgt_id)
        already_weights.append(already_edge["weight"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_edge["description"])
        already_order.append(already_edge.get("order", 1))

    order = min([dp.get("order", 1) for dp in edges_data] + already_order)
    weight = sum([dp["weight"] for dp in edges_data] + already_weights)
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in edges_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in edges_data] + already_source_ids)
    )
    for need_insert_id in [src_id, tgt_id]:
        if not (await knwoledge_graph_inst.has_node(need_insert_id)):
            await knwoledge_graph_inst.upsert_node(
                need_insert_id,
                node_data={
                    "source_id": source_id,
                    "description": description,
                    "entity_type": '"UNKNOWN"',
                },
            )
    description = await _handle_entity_relation_summary(
        (src_id, tgt_id), description, global_config, tokenizer_wrapper
    )
    await knwoledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(weight=weight, description=description, source_id=source_id, order=order),
    )


async def extract_entities_structured(
    chunks: dict[str, TextChunkSchema],
    knwoledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    tokenizer_wrapper,
    global_config: dict,
    using_amazon_bedrock: bool = False,
) -> Union[BaseGraphStorage, None]:
    """Extract entities and relationships using structured LLM output.

    This function uses Pydantic models for structured output, which provides
    type-safe entity extraction with a single LLM pass per chunk.

    Quality Modes (via entity_extraction_quality in global_config):
        - "fast": Use cheap model (faster, lower cost, may miss some entities)
        - "balanced": Use best model (default, good quality)
        - "thorough": Use best model (currently same as balanced)

    Note: This function uses single-pass extraction. For multi-pass extraction
    with gleaning (iterative refinement), use the legacy extract_entities()
    function instead by setting use_litellm=False in GraphRAG config.

    Args:
        chunks: Dictionary of chunk_id to chunk data
        knwoledge_graph_inst: Graph storage instance
        entity_vdb: Vector storage instance for entities
        tokenizer_wrapper: Tokenizer for text processing
        global_config: Configuration dictionary containing:
            - entity_extraction_quality: "fast" | "balanced" | "thorough"
            - cheap_model_func: LLM function for fast mode
            - best_model_func: LLM function for balanced/thorough modes
            - structured_output: Enable structured output
            - fallback_to_parsing: Enable parsing fallback
        using_amazon_bedrock: Whether using Amazon Bedrock

    Returns:
        Updated graph storage instance, or None if extraction failed
    """
    from .._schemas import EntityExtractionOutput

    quality = global_config.get("entity_extraction_quality", "balanced")

    if quality == "fast":
        use_llm_func = global_config["cheap_model_func"]
    elif quality == "thorough":
        use_llm_func = global_config["best_model_func"]
    else:
        use_llm_func = global_config["best_model_func"]

    use_structured_output = global_config.get("structured_output", True)
    fallback_to_parsing = global_config.get("fallback_to_parsing", True)

    ordered_chunks = list(chunks.items())
    entity_types = PROMPTS["DEFAULT_ENTITY_TYPES"]

    already_processed = 0
    already_entities = 0
    already_relations = 0

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]

        try:
            if use_structured_output:
                result = await use_llm_func(
                    content,
                    system_prompt=f"""You are an entity extraction assistant. Extract entities and relationships from the text.
Entity types: {", ".join(entity_types)}.
Return a JSON with 'entities' (name, type, description) and 'relationships' (source, target, description, weight).""",
                    response_format=EntityExtractionOutput,
                )
                if isinstance(result, str):
                    try:
                        parsed_data = json.loads(result)
                        result = EntityExtractionOutput(**parsed_data)
                    except Exception as e:
                        logger.warning(f"Failed to parse structured output from string: {e}")
                        result = None

                if isinstance(result, EntityExtractionOutput):
                    maybe_nodes = {
                        clean_str(e.entity_name.upper()): [
                            {
                                "entity_name": clean_str(e.entity_name.upper()),
                                "entity_type": clean_str(e.entity_type.upper()),
                                "description": e.description,
                                "source_id": chunk_key,
                            }
                        ]
                        for e in result.entities
                    }
                    maybe_edges = {
                        tuple(sorted([clean_str(r.source.upper()), clean_str(r.target.upper())])): [
                            {
                                "src_id": clean_str(r.source.upper()),
                                "tgt_id": clean_str(r.target.upper()),
                                "description": r.description,
                                "weight": r.weight,
                                "source_id": chunk_key,
                            }
                        ]
                        for r in result.relationships
                    }
                else:
                    maybe_nodes, maybe_edges = {}, {}
            else:
                maybe_nodes, maybe_edges = {}, {}

            if not maybe_nodes or not use_structured_output:
                entity_extract_prompt = PROMPTS["entity_extraction"]
                context_base = dict(
                    tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
                    record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
                    completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
                    entity_types=",".join(entity_types),
                )
                hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)
                final_result = await use_llm_func(hint_prompt)

                records = split_string_by_multi_markers(
                    final_result,
                    [context_base["record_delimiter"], context_base["completion_delimiter"]],
                )
                maybe_nodes = defaultdict(list)
                maybe_edges = defaultdict(list)
                for record in records:
                    record_match = re.search(r"\((.*)\)", record)
                    if record_match is None:
                        continue
                    record_attrs = split_string_by_multi_markers(
                        record_match.group(1), [context_base["tuple_delimiter"]]
                    )
                    if len(record_attrs) >= 4 and record_attrs[0] == '"entity"':
                        entity_data = {
                            "entity_name": clean_str(record_attrs[1].upper()),
                            "entity_type": clean_str(record_attrs[2].upper()),
                            "description": clean_str(record_attrs[3]),
                            "source_id": chunk_key,
                        }
                        if entity_data["entity_name"].strip():
                            maybe_nodes[entity_data["entity_name"]].append(entity_data)
                    elif len(record_attrs) >= 5 and record_attrs[0] == '"relationship"':
                        rel_data = {
                            "src_id": clean_str(record_attrs[1].upper()),
                            "tgt_id": clean_str(record_attrs[2].upper()),
                            "description": clean_str(record_attrs[3]),
                            "weight": float(record_attrs[-1])
                            if is_float_regex(record_attrs[-1])
                            else 1.0,
                            "source_id": chunk_key,
                        }
                        maybe_edges[tuple(sorted([rel_data["src_id"], rel_data["tgt_id"]]))].append(
                            rel_data
                        )

        except Exception as e:
            logger.warning(f"Structured extraction failed for chunk {chunk_key}: {e}")
            if not fallback_to_parsing:
                raise
            maybe_nodes, maybe_edges = {}, {}

        already_processed += 1
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][already_processed % len(PROMPTS["process_tickers"])]
        print(
            f"{now_ticks} Processed {already_processed}({already_processed * 100 // len(ordered_chunks)}%) chunks, {already_entities} entities, {already_relations} relations\r",
            end="",
            flush=True,
        )
        return dict(maybe_nodes), dict(maybe_edges)

    results = await asyncio.gather(*[_process_single_content(c) for c in ordered_chunks])
    print()
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            maybe_edges[tuple(sorted(k))].extend(v)

    if not len(maybe_nodes):
        logger.warning("Didn't extract any entities")
        return None

    all_entities_data = await asyncio.gather(
        *[
            _merge_nodes_then_upsert(k, v, knwoledge_graph_inst, global_config, tokenizer_wrapper)
            for k, v in maybe_nodes.items()
        ]
    )
    await asyncio.gather(
        *[
            _merge_edges_then_upsert(
                k[0], k[1], v, knwoledge_graph_inst, global_config, tokenizer_wrapper
            )
            for k, v in maybe_edges.items()
        ]
    )

    if entity_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + dp["description"],
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)

    return knwoledge_graph_inst


async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    knwoledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    tokenizer_wrapper,
    global_config: dict,
    using_amazon_bedrock: bool = False,
) -> Union[BaseGraphStorage, None]:
    if global_config.get("_use_structured_extraction", False):
        return await extract_entities_structured(
            chunks,
            knwoledge_graph_inst,
            entity_vdb,
            tokenizer_wrapper,
            global_config,
            using_amazon_bedrock,
        )

    use_llm_func: Callable[..., Any] = global_config["best_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]

    ordered_chunks = list(chunks.items())

    entity_extract_prompt = PROMPTS["entity_extraction"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
    )
    continue_prompt = PROMPTS["entiti_continue_extraction"]
    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]

    already_processed = 0
    already_entities = 0
    already_relations = 0

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)
        final_result = await use_llm_func(hint_prompt)
        if isinstance(final_result, list):
            final_result = final_result[0]["text"]

        history = pack_user_ass_to_openai_messages(hint_prompt, final_result, using_amazon_bedrock)
        for now_glean_index in range(entity_extract_max_gleaning):
            glean_result = await use_llm_func(continue_prompt, history_messages=history)

            history += pack_user_ass_to_openai_messages(
                continue_prompt, glean_result, using_amazon_bedrock
            )
            final_result += glean_result
            if now_glean_index == entity_extract_max_gleaning - 1:
                break

            if_loop_result: str = await use_llm_func(if_loop_prompt, history_messages=history)
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break

        records = split_string_by_multi_markers(
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )

        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(
                record, [context_base["tuple_delimiter"]]
            )
            if_entities = await _handle_single_entity_extraction(record_attributes, chunk_key)
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue

            if_relation = await _handle_single_relationship_extraction(record_attributes, chunk_key)
            if if_relation is not None:
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(if_relation)
        already_processed += 1
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][already_processed % len(PROMPTS["process_tickers"])]
        print(
            f"{now_ticks} Processed {already_processed}({already_processed * 100 // len(ordered_chunks)}%) chunks,  {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return dict(maybe_nodes), dict(maybe_edges)

    results = await asyncio.gather(*[_process_single_content(c) for c in ordered_chunks])
    print()
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            maybe_edges[tuple(sorted(k))].extend(v)
    all_entities_data = await asyncio.gather(
        *[
            _merge_nodes_then_upsert(k, v, knwoledge_graph_inst, global_config, tokenizer_wrapper)
            for k, v in maybe_nodes.items()
        ]
    )
    await asyncio.gather(
        *[
            _merge_edges_then_upsert(
                k[0], k[1], v, knwoledge_graph_inst, global_config, tokenizer_wrapper
            )
            for k, v in maybe_edges.items()
        ]
    )
    if not len(all_entities_data):
        logger.warning("Didn't extract any entities, maybe your LLM is not working")
        return None
    if entity_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + dp["description"],
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)
    return knwoledge_graph_inst
