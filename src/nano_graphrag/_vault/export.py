from __future__ import annotations

import os
from typing import Any

from .._utils import logger

_SPARSE_ENTITY_MIN_CHARS = 20


async def aexport_vault(
    knowledge_graph_inst,
    community_reports_kv,
    global_config: dict,
    path: str | None = None,
    include_communities: bool = True,
) -> dict[str, Any]:
    vault_path = path or global_config.get("vault_path", "./vault")
    stats = {"entities": 0, "communities": 0, "sparse_rolled_up": 0}

    all_nodes = await _get_all_nodes_safe(knowledge_graph_inst)
    if not all_nodes:
        logger.warning("vault_export_no_nodes")
        return stats

    entities_dir = os.path.join(vault_path, "entities")
    os.makedirs(entities_dir, exist_ok=True)

    entities_by_type: dict[str, list[tuple[str, dict]]] = {}
    for node_id, node_data in all_nodes.items():
        etype = node_data.get("entity_type", "UNKNOWN")
        entities_by_type.setdefault(etype, []).append((node_id, node_data))

    sparse_entities: list[tuple[str, dict]] = []

    for etype, entities in entities_by_type.items():
        type_dir = os.path.join(entities_dir, etype)
        os.makedirs(type_dir, exist_ok=True)

        type_index_lines = []
        for node_id, node_data in entities:
            entity_name = node_data.get("entity_name", node_id)
            description = node_data.get("description", "")

            if len(description) < _SPARSE_ENTITY_MIN_CHARS:
                sparse_entities.append((node_id, node_data))
                continue

            await _write_entity_markdown(type_dir, node_id, node_data, knowledge_graph_inst)
            type_index_lines.append(f"- [[{entity_name}]]")
            stats["entities"] += 1

        if type_index_lines:
            _write_index(type_dir, type_index_lines)

    if sparse_entities:
        stats["sparse_rolled_up"] = len(sparse_entities)
        _write_sparse_rollup(entities_dir, sparse_entities)

    master_lines = []
    for etype in sorted(entities_by_type.keys()):
        count = len(entities_by_type[etype])
        master_lines.append(f"- [[{etype}/_index|{etype}]] ({count})")
    _write_index(entities_dir, master_lines)

    if include_communities and community_reports_kv is not None:
        await _export_communities(vault_path, community_reports_kv, stats)

    _write_graph_stats(vault_path, all_nodes, stats)
    logger.info("vault_export_done", path=vault_path, **stats)
    return stats


async def _write_entity_markdown(
    type_dir: str,
    node_id: str,
    node_data: dict,
    knowledge_graph_inst,
) -> None:
    import json
    from datetime import datetime

    entity_name = node_data.get("entity_name", node_id)
    entity_type = node_data.get("entity_type", "UNKNOWN")
    description = node_data.get("description", "")
    aliases_raw = node_data.get("aliases", "[]")
    try:
        aliases = json.loads(aliases_raw) if isinstance(aliases_raw, str) else aliases_raw
    except (json.JSONDecodeError, TypeError):
        aliases = []
    source_ids_raw = node_data.get("source_id", "[]")
    try:
        source_ids = json.loads(source_ids_raw) if isinstance(source_ids_raw, str) else []
    except (json.JSONDecodeError, TypeError):
        source_ids = []

    edges = await knowledge_graph_inst.get_node_edges(node_id)
    connections = []
    if edges:
        for src, tgt in edges:
            edge = await knowledge_graph_inst.get_edge(src, tgt)
            if edge is None:
                continue
            rel_type = edge.get("relation_type", "related_to")
            confidence = edge.get("confidence", 0.8)
            other_id = tgt if src == node_id else src
            other_node = await knowledge_graph_inst.get_node(other_id)
            other_name = other_node.get("entity_name", other_id) if other_node else other_id
            connections.append(f"- {rel_type} → [[{other_name}]] (confidence: {confidence:.1f})")

    safe_name = _safe_filename(entity_name)
    filepath = os.path.join(type_dir, f"{safe_name}.md")

    frontmatter = _yaml_frontmatter(
        {
            "id": node_id,
            "type": entity_type,
            "name": entity_name,
            "aliases": aliases,
            "source_chunks": source_ids[:10],
            "confidence": node_data.get("confidence", 1.0),
            "created": datetime.now().strftime("%Y-%m-%d"),
            "updated": datetime.now().strftime("%Y-%m-%d"),
        }
    )

    md = f"""{frontmatter}
# {entity_name}

{description}

## Connections
{chr(10).join(connections) if connections else "No connections found."}
"""
    with open(filepath, "w") as f:
        f.write(md)


async def _export_communities(
    vault_path: str,
    community_reports_kv,
    stats: dict,
) -> None:
    communities_dir = os.path.join(vault_path, "communities")
    os.makedirs(communities_dir, exist_ok=True)

    all_keys = await community_reports_kv.all_keys()
    if not all_keys:
        return

    reports = await community_reports_kv.get_by_ids(all_keys)
    index_lines = []
    for key, report in zip(all_keys, reports, strict=False):
        if report is None:
            continue
        title = report.get("report_json", {}).get("title", key)
        content = report.get("report_string", "")
        safe_title = _safe_filename(title)
        filepath = os.path.join(communities_dir, f"{safe_title}.md")

        frontmatter = _yaml_frontmatter(
            {
                "id": key,
                "title": title,
                "rating": report.get("report_json", {}).get("rating", 0),
            }
        )
        with open(filepath, "w") as f:
            f.write(f"{frontmatter}\n# {title}\n\n{content}\n")
        index_lines.append(f"- [[{safe_title}|{title}]]")
        stats["communities"] += 1

    if index_lines:
        _write_index(communities_dir, index_lines)


def _write_graph_stats(vault_path: str, all_nodes: dict, stats: dict) -> None:
    filepath = os.path.join(vault_path, "graph-stats.md")
    types_count: dict[str, int] = {}
    for node_data in all_nodes.values():
        etype = node_data.get("entity_type", "UNKNOWN")
        types_count[etype] = types_count.get(etype, 0) + 1

    lines = ["# Graph Statistics\n"]
    lines.append(f"- Total entities: {len(all_nodes)}")
    lines.append(f"- Exported entities: {stats['entities']}")
    lines.append(f"- Sparse (rolled up): {stats['sparse_rolled_up']}")
    lines.append(f"- Communities: {stats['communities']}\n")
    lines.append("## Entity Types\n")
    for etype, count in sorted(types_count.items()):
        lines.append(f"- {etype}: {count}")

    with open(filepath, "w") as f:
        f.write("\n".join(lines))


def _write_index(directory: str, lines: list[str]) -> None:
    filepath = os.path.join(directory, "_index.md")
    with open(filepath, "w") as f:
        f.write("\n".join(lines) + "\n")


def _write_sparse_rollup(entities_dir: str, sparse: list[tuple[str, dict]]) -> None:
    lines = ["# Sparse Entities\n\n"]
    lines.append("Entities with descriptions shorter than 20 characters are listed here.\n")
    for node_id, node_data in sparse:
        name = node_data.get("entity_name", node_id)
        etype = node_data.get("entity_type", "UNKNOWN")
        desc = node_data.get("description", "")
        lines.append(f"- **{name}** ({etype}): {desc or '(no description)'}")
    filepath = os.path.join(entities_dir, "sparse_entities.md")
    with open(filepath, "w") as f:
        f.write("\n".join(lines))


def _yaml_frontmatter(data: dict) -> str:
    lines = ["---"]
    for k, v in data.items():
        if isinstance(v, list):
            lines.append(f"{k}:")
            for item in v:
                lines.append(f"  - {item}")
        elif isinstance(v, float):
            lines.append(f"{k}: {v:.2f}")
        else:
            lines.append(f"{k}: {v}")
    lines.append("---")
    return "\n".join(lines)


def _safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in " ._-" else "_" for c in name).strip(". ")


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
