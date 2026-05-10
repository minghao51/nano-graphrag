"""
Quick Start — Get running in 5 minutes
=======================================
Objective: Insert a document into a knowledge graph, query it, and explore
the resulting graph structure with rich visualization and analysis.

Run:  dotenvx run -- uv run notebooks/01_quick_start.py
"""

import json
import logging
import time
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.WARNING)

WORKING_DIR = Path("./_cache/01_quick_start")
WORKING_DIR.mkdir(parents=True, exist_ok=True)

# %% Imports
from nano_graphrag import GraphRAG, QueryParam

# %% Load sample text — A Christmas Carol (first 400 lines)
with open("../tests/fixtures/mock_data.txt", encoding="utf-8-sig") as f:
    lines = f.readlines()

text = "".join(lines[:400])
print(f"Loaded {len(text):,} characters ({len(text.splitlines())} lines)")

# %% Initialize GraphRAG with default settings
# API credentials come from environment variables (dotenvx).
rag = GraphRAG(
    working_dir=str(WORKING_DIR),
    enable_llm_cache=True,
)
print(f"LLM model:      {rag.llm_model}")
print(f"Embedding model: {rag.embedding_model}")

# %% Insert — chunk, extract entities, build graph, detect communities
print("Indexing...")
start = time.time()
await rag.ainsert(text)
elapsed = time.time() - start
print(f"Done in {elapsed:.1f}s")

# %% Global query — aggregates community reports for broad questions
result = await rag.aquery(
    "What are the main themes in A Christmas Carol?",
    param=QueryParam(mode="global"),
)
print(result)

# %% Local query — entity-level retrieval for specific questions
result = await rag.aquery(
    "What is the relationship between Scrooge and Bob Cratchit?",
    param=QueryParam(mode="local"),
)
print(result)

# ============================================================
# GRAPH ANALYSIS & VISUALIZATION
# ============================================================

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})

graphml_path = WORKING_DIR / "graph_chunk_entity_relation.graphml"
G = nx.read_graphml(str(graphml_path))

# %% Helper: extract node/edge metadata
node_names = {n: G.nodes[n].get("entity_name", n) for n in G.nodes()}
node_types = {n: G.nodes[n].get("entity_type", "UNKNOWN") for n in G.nodes()}
edge_weights = [G.edges[u, v].get("weight", 1.0) for u, v in G.edges()]
edge_descriptions = [G.edges[u, v].get("description", "") for u, v in G.edges()]

# Build community mapping from node cluster data
community_map = {}
for nid, data in G.nodes(data=True):
    clusters = json.loads(data.get("clusters", "[]"))
    for c in clusters:
        cname = c["cluster"]
        if cname not in community_map:
            community_map[cname] = []
        community_map[cname].append(nid)

l0_communities = {k: v for k, v in community_map.items() if k.startswith("l0_")}

# %% --- Figure 1: Graph Overview Stats (full-width) ---
fig, axes = plt.subplots(2, 2, figsize=(20, 14))
fig.suptitle("Knowledge Graph Overview — A Christmas Carol", fontsize=18, fontweight="bold", y=0.98)

# (a) Degree distribution
degrees = [d for _, d in G.degree()]
axes[0, 0].hist(degrees, bins=range(1, max(degrees) + 2), edgecolor="white", color="#4C78A8", alpha=0.85)
axes[0, 0].set_xlabel("Degree (number of connections)")
axes[0, 0].set_ylabel("Number of entities")
axes[0, 0].set_title("(a) Degree Distribution")
axes[0, 0].set_yscale("log")
axes[0, 0].axvline(x=np.median(degrees), color="red", linestyle="--", label=f"median={np.median(degrees):.0f}")
axes[0, 0].legend()

# (b) Entity type breakdown
type_counts = Counter(node_types.values())
sorted_types = type_counts.most_common()
type_names = [t for t, _ in sorted_types]
type_vals = [c for _, c in sorted_types]
colors_types = ["#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B", "#EECA3B"]
bars = axes[0, 1].barh(range(len(type_names)), type_vals, color=colors_types[:len(type_names)], edgecolor="white")
axes[0, 1].set_yticks(range(len(type_names)))
axes[0, 1].set_yticklabels(type_names)
axes[0, 1].set_xlabel("Count")
axes[0, 1].set_title("(b) Entities by Type")
axes[0, 1].invert_yaxis()
for bar, val in zip(bars, type_vals):
    axes[0, 1].text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2, str(val), va="center", fontweight="bold")

# (c) Top-20 entities by degree
top20 = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:20]
top20_names = [node_names[nid][:28] for nid, _ in top20]
top20_degs = [d for _, d in top20]
top20_colors = [colors_types[type_names.index(node_types[nid]) % len(colors_types)] for nid, _ in top20]

axes[1, 0].barh(range(len(top20_names)), top20_degs, color=top20_colors, edgecolor="white")
axes[1, 0].set_yticks(range(len(top20_names)))
axes[1, 0].set_yticklabels(top20_names, fontsize=9)
axes[1, 0].set_xlabel("Degree")
axes[1, 0].set_title("(c) Top 20 Entities by Degree")
axes[1, 0].invert_yaxis()

# (d) Edge weight distribution
axes[1, 1].hist(edge_weights, bins=20, edgecolor="white", color="#72B7B2", alpha=0.85)
axes[1, 1].set_xlabel("Edge Weight")
axes[1, 1].set_ylabel("Count")
axes[1, 1].set_title("(d) Relationship Weight Distribution")

plt.tight_layout(rect=[0, 0, 1, 0.95])
path = WORKING_DIR / "fig1_graph_overview.png"
plt.savefig(path, bbox_inches="tight")
plt.close()
print(f"Saved {path}")

# Print summary stats
print("\nGraph Statistics:")
print(f"  Nodes (entities):     {G.number_of_nodes()}")
print(f"  Edges (relations):    {G.number_of_edges()}")
print(f"  Density:              {nx.density(G):.4f}")
print(f"  Avg clustering:       {nx.average_clustering(G):.4f}")
print(f"  Connected components: {nx.number_connected_components(G)}")
print(f"  Avg degree:           {np.mean(degrees):.1f}")
print(f"  Median degree:        {np.median(degrees):.0f}")
print(f"  L0 communities:       {len(l0_communities)}")

# %% --- Figure 2: Full Network Visualization (community-colored) ---
fig, ax = plt.subplots(1, 1, figsize=(22, 22))

pos = nx.spring_layout(G, k=2.5 / (G.number_of_nodes() ** 0.5), seed=42, iterations=100)

# Assign each node to its L0 community
node_to_l0 = {}
for cname, members in l0_communities.items():
    for nid in members:
        node_to_l0[nid] = cname

# Assign colors by community
community_colors = {}
cmap_communities = plt.cm.tab10
l0_names = sorted(l0_communities.keys())
for i, cname in enumerate(l0_names):
    community_colors[cname] = cmap_communities(i / max(len(l0_names) - 1, 1))

node_colors = []
for n in G.nodes():
    cname = node_to_l0.get(n)
    node_colors.append(community_colors.get(cname, "#CCCCCC"))

node_sizes = [120 + 60 * G.degree(n) for n in G.nodes()]
edge_widths = [0.3 + 0.7 * float(G.edges[u, v].get("weight", 1.0)) for u, v in G.edges()]

nx.draw_networkx_edges(G, pos, alpha=0.12, width=edge_widths, edge_color="#888888", ax=ax)
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.85, edgecolors="white", linewidths=0.5, ax=ax)

# Label top-15 entities by degree
top15_nodes = {nid for nid, _ in sorted(G.degree(), key=lambda x: x[1], reverse=True)[:15]}
labels = {nid: node_names[nid][:22] for nid in top15_nodes}
nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight="bold", ax=ax)

handles = []
for cname in l0_names:
    member_names = [node_names[n][:18] for n in l0_communities[cname][:4]]
    label = f"{cname}: {', '.join(member_names)}"
    handles.append(mpatches.Patch(color=community_colors[cname], label=label))
ax.legend(handles=handles, title="Level-0 Communities", loc="upper left", fontsize=9, title_fontsize=10, framealpha=0.9)

ax.set_title("Knowledge Graph — Community Structure", fontsize=18, fontweight="bold", pad=20)
ax.axis("off")

path = WORKING_DIR / "fig2_network_communities.png"
plt.savefig(path, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved {path}")

# %% --- Figure 3: Entity Type Network (separate subgraphs by type) ---
fig, axes = plt.subplots(1, len(sorted_types), figsize=(6 * len(sorted_types), 6))
if len(sorted_types) == 1:
    axes = [axes]
fig.suptitle("Subgraphs by Entity Type", fontsize=16, fontweight="bold", y=1.02)

for ax, (etype, count) in zip(axes, sorted_types):
    nodes_of_type = [n for n in G.nodes() if node_types[n] == etype]
    sub = G.subgraph(nodes_of_type).copy()
    if sub.number_of_nodes() == 0:
        ax.axis("off")
        ax.set_title(f"{etype} (0)")
        continue

    pos_sub = nx.spring_layout(sub, seed=42, k=1.5) if sub.number_of_nodes() > 1 else {nodes_of_type[0]: (0.5, 0.5)}
    nx.draw_networkx_edges(sub, pos_sub, alpha=0.3, ax=ax)
    nx.draw_networkx_nodes(sub, pos_sub, node_color=colors_types[type_names.index(etype) % len(colors_types)], node_size=300, alpha=0.85, ax=ax)
    labels_sub = {n: node_names[n][:15] for n in sub.nodes()}
    nx.draw_networkx_labels(sub, pos_sub, labels_sub, font_size=7, ax=ax)
    ax.set_title(f"{etype} ({count})")
    ax.axis("off")

plt.tight_layout()
path = WORKING_DIR / "fig3_entity_type_subgraphs.png"
plt.savefig(path, bbox_inches="tight")
plt.close()
print(f"Saved {path}")

# %% --- Figure 4: Neighborhood of Top Entities ---
top_entities = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:6]

fig, axes = plt.subplots(2, 3, figsize=(24, 16))
fig.suptitle("Neighborhood of Top Entities (1-hop)", fontsize=16, fontweight="bold", y=0.98)
axes = axes.flatten()

for ax, (center_id, center_deg) in zip(axes, top_entities):
    neighbors = list(G.neighbors(center_id))
    ego_nodes = [center_id] + neighbors
    ego = G.subgraph(ego_nodes).copy()

    pos_ego = nx.spring_layout(ego, seed=42, k=1.2) if ego.number_of_nodes() > 1 else {center_id: (0.5, 0.5)}

    ego_colors = []
    ego_sizes = []
    for n in ego.nodes():
        if n == center_id:
            ego_colors.append("#E45756")
            ego_sizes.append(600)
        else:
            ego_colors.append(community_colors.get(node_to_l0.get(n), "#CCCCCC"))
            ego_sizes.append(250)

    nx.draw_networkx_edges(ego, pos_ego, alpha=0.3, ax=ax)
    edge_labels_ego = {}
    for u, v in ego.edges():
        desc = G.edges[u, v].get("description", "")
        if len(desc) > 25:
            desc = desc[:22] + "..."
        edge_labels_ego[(u, v)] = desc
    nx.draw_networkx_edge_labels(ego, pos_ego, edge_labels_ego, font_size=6, ax=ax)
    nx.draw_networkx_nodes(ego, pos_ego, node_color=ego_colors, node_size=ego_sizes, edgecolors="white", linewidths=0.5, ax=ax)
    labels_ego = {n: node_names[n][:18] for n in ego.nodes()}
    nx.draw_networkx_labels(ego, pos_ego, labels_ego, font_size=8, ax=ax)

    center_name = node_names[center_id]
    ax.set_title(f"{center_name} (degree={center_deg})", fontsize=12, fontweight="bold")
    ax.axis("off")

plt.tight_layout(rect=[0, 0, 1, 0.95])
path = WORKING_DIR / "fig4_entity_neighborhoods.png"
plt.savefig(path, bbox_inches="tight")
plt.close()
print(f"Saved {path}")

# %% --- Figure 5: Centrality Analysis ---
betweenness = nx.betweenness_centrality(G)
closeness = nx.closeness_centrality(G)
eigenvector = nx.eigenvector_centrality(G, max_iter=500)

fig, axes = plt.subplots(1, 3, figsize=(22, 8))
fig.suptitle("Centrality Analysis — Who Are the Key Entities?", fontsize=16, fontweight="bold", y=1.02)

centrality_measures = [
    ("Betweenness Centrality", betweenness, "Bridge nodes — control information flow"),
    ("Closeness Centrality", closeness, "Well-connected — reach others quickly"),
    ("Eigenvector Centrality", eigenvector, "Influential — connected to other important nodes"),
]

for ax, (title, measure, subtitle) in zip(axes, centrality_measures):
    top = sorted(measure.items(), key=lambda x: x[1], reverse=True)[:12]
    names = [node_names[nid][:25] for nid, _ in top]
    vals = [v for _, v in top]

    ax.barh(range(len(names)), vals, color="#4C78A8", edgecolor="white", alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Score")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.text(0.5, -0.08, subtitle, transform=ax.transAxes, ha="center", fontsize=9, style="italic", color="#666666")

plt.tight_layout()
path = WORKING_DIR / "fig5_centrality_analysis.png"
plt.savefig(path, bbox_inches="tight")
plt.close()
print(f"Saved {path}")

# %% --- Figure 6: Relationship Analysis ---
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
fig.suptitle("Relationship Analysis", fontsize=16, fontweight="bold", y=1.02)

# (a) Weighted top relationships — strongest connections
strongest_edges = sorted(G.edges(data=True), key=lambda x: float(x[2].get("weight", 1.0)), reverse=True)[:15]
edge_labels_list = []
edge_weights_top = []
for u, v, data in strongest_edges:
    name_u = node_names[u][:20]
    name_v = node_names[v][:20]
    w = float(data.get("weight", 1.0))
    desc = data.get("description", "")[:35]
    edge_labels_list.append(f"{name_u} — {name_v}")
    edge_weights_top.append(w)

axes[0].barh(range(len(edge_labels_list)), edge_weights_top, color="#F58518", edgecolor="white", alpha=0.85)
axes[0].set_yticks(range(len(edge_labels_list)))
axes[0].set_yticklabels(edge_labels_list, fontsize=8)
axes[0].set_xlabel("Weight (co-occurrence strength)")
axes[0].set_title("(a) Strongest Entity Relationships")
axes[0].invert_yaxis()

# (b) Community size distribution
comm_sizes = [len(members) for members in l0_communities.values()]
comm_names = [cname for cname in l0_communities.keys()]
axes[1].bar(range(len(comm_names)), comm_sizes, color=list(community_colors.values())[:len(comm_names)], edgecolor="white")
axes[1].set_xticks(range(len(comm_names)))
axes[1].set_xticklabels(comm_names, rotation=45, ha="right")
axes[1].set_ylabel("Number of entities")
axes[1].set_title("(b) Community Size (Level 0)")
for i, (_cn, sz) in enumerate(zip(comm_names, comm_sizes)):
    axes[1].text(i, sz + 0.2, str(sz), ha="center", fontweight="bold")

plt.tight_layout()
path = WORKING_DIR / "fig6_relationship_analysis.png"
plt.savefig(path, bbox_inches="tight")
plt.close()
print(f"Saved {path}")

# %% Print textual summaries
print("\n" + "=" * 70)
print("TEXTUAL ANALYSIS")
print("=" * 70)

print("\nTop 10 entities by degree:")
for nid, deg in sorted(G.degree(), key=lambda x: x[1], reverse=True)[:10]:
    name = node_names[nid]
    etype = node_types[nid]
    print(f"  {name:40s} ({etype:12s}) degree={deg}")

print("\nStrongest relationships:")
for u, v, data in strongest_edges[:10]:
    name_u = node_names[u]
    name_v = node_names[v]
    w = data.get("weight", 1.0)
    desc = data.get("description", "")
    print(f"  {name_u:25s} <-> {name_v:25s} weight={w}  ({desc[:50]})")

print("\nCommunities (Level 0):")
for cname in sorted(l0_communities.keys()):
    members = l0_communities[cname]
    member_names = [node_names[n] for n in members]
    print(f"  {cname} ({len(members)} entities): {', '.join(member_names[:8])}")

# Community reports
import sqlite3

db_path = WORKING_DIR / "kv_store_community_reports.db"
if db_path.exists():
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute("SELECT key, value FROM kv_store").fetchall()
    conn.close()

    print(f"\nCommunity reports: {len(rows)}")
    for key, val_str in rows[:5]:
        val = json.loads(val_str)
        title = val.get("title", "N/A")
        level = val.get("level", "N/A")
        report_str = val.get("report_string", "")
        print(f"  [{key}] \"{title}\" (level={level})")
        if report_str:
            preview = report_str[:120].replace("\n", " ")
            print(f"    {preview}...")
