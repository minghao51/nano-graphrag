from __future__ import annotations

import html
import os
from dataclasses import dataclass, field
from typing import Literal

from ._utils import logger


@dataclass
class GraphStatus:
    entity_count: int = 0
    entity_types: dict[str, int] = field(default_factory=dict)
    relationship_count: int = 0
    community_count: int = 0
    community_levels: dict[int, int] = field(default_factory=dict)
    storage_size_bytes: int = 0
    last_insert_timestamp: str | None = None
    document_count: int = 0
    chunk_count: int = 0
    health: Literal["healthy", "warning", "error"] = "healthy"
    warnings: list[str] = field(default_factory=list)


async def compute_status(working_dir: str, rag=None) -> GraphStatus:
    """Compute comprehensive graph status without loading the full graph.

    Args:
        working_dir: Path to the working directory.
        rag: Optional GraphRAG instance. If provided, uses existing storages.

    Returns:
        GraphStatus with health assessment.
    """
    status = GraphStatus()
    warnings = []

    if not os.path.exists(working_dir):
        status.health = "error"
        status.warnings = ["Working directory does not exist"]
        return status

    total_size = 0
    for root, _dirs, files in os.walk(working_dir):
        for f in files:
            total_size += os.path.getsize(os.path.join(root, f))
    status.storage_size_bytes = total_size

    if rag is not None:
        try:
            all_docs = await rag.full_docs.all_keys()
            status.document_count = len(all_docs)
        except Exception:
            logger.debug("storage_read_failed", storage="full_docs", exc_info=True)
            warnings.append("Could not read document index")

        try:
            all_chunks = await rag.text_chunks.all_keys()
            status.chunk_count = len(all_chunks)
        except Exception:
            logger.debug("storage_read_failed", storage="text_chunks", exc_info=True)
            warnings.append("Could not read text chunks")

        try:
            all_reports = await rag.community_reports.all_keys()
            status.community_count = len(all_reports)
        except Exception:
            logger.debug("storage_read_failed", storage="community_reports", exc_info=True)
            warnings.append("Could not read community reports")

        graph = rag.chunk_entity_relation_graph
        if graph is not None and hasattr(graph, "_graph") and graph._graph is not None:
            nx_graph = graph._graph
            status.entity_count = nx_graph.number_of_nodes()
            status.relationship_count = nx_graph.number_of_edges()

            entity_types: dict[str, int] = {}
            for node_id in nx_graph.nodes():
                node_data = nx_graph.nodes[node_id]
                etype = node_data.get("entity_type", "UNKNOWN")
                entity_types[etype] = entity_types.get(etype, 0) + 1
            status.entity_types = entity_types

            if hasattr(nx_graph, "graph") and "clustering" in nx_graph.graph:
                levels: dict[int, int] = {}
                for node_id in nx_graph.nodes():
                    node_data = nx_graph.nodes[node_id]
                    for key in node_data:
                        if key.startswith("community_"):
                            try:
                                level = int(key.split("_")[1])
                            except (IndexError, ValueError):
                                continue
                            communities = node_data[key]
                            if isinstance(communities, list):
                                levels[level] = len(communities)
                status.community_levels = levels

    if status.entity_count == 0 and status.document_count > 0:
        warnings.append("Empty graph — documents exist but no entities extracted")
        status.health = "warning"
    elif status.entity_count == 0:
        warnings.append("Empty graph — no data inserted")
        status.health = "warning"

    if status.community_count == 0 and status.entity_count > 0:
        warnings.append("No community reports — run insert to generate")

    status.warnings = warnings
    return status


async def visualize_graph(
    graph_storage,
    output: str = "graph.html",
    max_nodes: int = 200,
    **kwargs,
) -> str:
    """Generate interactive HTML visualization of the knowledge graph.

    Uses pyvis for visualization. Falls back to a simple HTML table if pyvis
    is not available.

    Args:
        graph_storage: The graph storage backend instance.
        output: Output HTML file path.
        max_nodes: Maximum number of nodes to render (for performance).
        kwargs: Passed to pyvis Network (physics, node_size, etc.).

    Returns:
        Path to the generated HTML file.
    """
    nx_graph = getattr(graph_storage, "_graph", None)
    if nx_graph is None:
        from ._exceptions import StorageError

        raise StorageError("Graph storage has no _graph attribute (NetworkX backend required)")

    max_nodes = max(1, max_nodes)

    node_count = nx_graph.number_of_nodes()
    if node_count == 0:
        with open(output, "w") as f:
            f.write(
                "<html><body><h2>Empty Graph</h2><p>No entities to visualize.</p></body></html>"
            )
        return output

    try:
        from pyvis.network import Network

        net = Network(
            height="800px",
            width="100%",
            directed=True,
            notebook=False,
        )
        net.heading = f"Knowledge Graph ({min(node_count, max_nodes)} of {node_count} nodes)"

        physics = kwargs.pop("physics", {"barnesHut": {"gravitationalConstant": -3000}})
        net.set_physics(physics)

        nodes_added = 0
        for node_id in list(nx_graph.nodes())[:max_nodes]:
            node_data = nx_graph.nodes[node_id]
            label = node_data.get("entity_name", str(node_id)[:20])
            etype = node_data.get("entity_type", "UNKNOWN")
            net.add_node(node_id, label=label, title=f"{label} ({etype})", group=etype)
            nodes_added += 1

        visible_node_ids = set(list(nx_graph.nodes())[:max_nodes])

        for src, tgt in list(nx_graph.edges())[: max_nodes * 3]:
            if src in visible_node_ids and tgt in visible_node_ids:
                edge_data = nx_graph.edges[src, tgt]
                weight = edge_data.get("weight", 1.0)
                net.add_edge(src, tgt, value=weight, title=edge_data.get("description", ""))

        net.save_graph(output)
        logger.info("graph_visualized", output=output, nodes=nodes_added)
        return output

    except ImportError:
        logger.warning("pyvis_not_available", fallback="html_table")
        return _generate_simple_html(nx_graph, output, max_nodes)


def _generate_simple_html(nx_graph, output: str, max_nodes: int) -> str:
    max_nodes = max(1, max_nodes)
    nodes = list(nx_graph.nodes(data=True))[:max_nodes]
    edges = list(nx_graph.edges(data=True))[: max_nodes * 2]

    rows = []
    for node_id, data in nodes:
        name = html.escape(data.get("entity_name", str(node_id)))
        etype = html.escape(data.get("entity_type", "UNKNOWN"))
        desc = html.escape(data.get("description", "")[:100])
        rows.append(f"<tr><td>{name}</td><td>{etype}</td><td>{desc}</td></tr>")

    edge_rows = []
    for src, tgt, data in edges:
        src_name = html.escape(nx_graph.nodes[src].get("entity_name", str(src)))
        tgt_name = html.escape(nx_graph.nodes[tgt].get("entity_name", str(tgt)))
        desc = html.escape(data.get("description", "")[:80])
        edge_rows.append(f"<tr><td>{src_name}</td><td>{tgt_name}</td><td>{desc}</td></tr>")

    html_content = f"""<!DOCTYPE html>
<html><head><title>Knowledge Graph</title>
<style>
body {{ font-family: system-ui; margin: 20px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
th {{ background: #f4f4f4; }}
</style></head>
<body>
<h2>Entities ({len(nodes)} shown)</h2>
<table><tr><th>Name</th><th>Type</th><th>Description</th></tr>
{"".join(rows)}
</table>
<h2>Relationships ({len(edge_rows)} shown)</h2>
<table><tr><th>Source</th><th>Target</th><th>Description</th></tr>
{"".join(edge_rows)}
</table>
</body></html>"""

    with open(output, "w") as f:
        f.write(html_content)
    return output


def _repr_html_(status: GraphStatus) -> str:
    """Generate HTML representation for Jupyter notebooks."""
    entity_rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>"
        for k, v in sorted(status.entity_types.items())
    )
    warning_html = ""
    if status.warnings:
        items = "".join(f"<li>{html.escape(w)}</li>" for w in status.warnings)
        warning_html = (
            f'<div style="color:orange;"><strong>Warnings:</strong><ul>{items}</ul></div>'
        )

    return f"""
    <div style="font-family:system-ui;border:1px solid #ddd;border-radius:8px;padding:16px;max-width:600px;">
        <h3 style="margin:0 0 8px 0;">nano-graphrag</h3>
        <p style="color:#666;margin:0 0 12px 0;">
            {status.entity_count} entities &middot;
            {status.relationship_count} relationships &middot;
            {status.community_count} communities &middot;
            {status.document_count} documents &middot;
            {status.chunk_count} chunks
        </p>
        {"<table><tr><th>Type</th><th>Count</th></tr>" + entity_rows + "</table>" if entity_rows else ""}
        {warning_html}
    </div>
    """


GraphStatus._repr_html_ = _repr_html_  # type: ignore[attr-defined]
