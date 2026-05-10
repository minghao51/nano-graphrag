"""
Query Modes — Pick the right query mode
========================================
Objective: Compare global, local, naive, and entity-grounded query modes
with timing benchmarks, token/cost analysis, and quality comparison.

Run:  dotenvx run -- uv run notebooks/02_query_modes.py
"""

import json
import logging
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.WARNING)

WORKING_DIR = Path("./_cache/02_query_modes")
WORKING_DIR.mkdir(parents=True, exist_ok=True)

# %% Imports
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag.base import ResponseType

# ============================================================
# HELPERS — measure LLM cost by diffing the cache DB
# ============================================================


def _get_cache_keys(db_path):
    if not db_path.exists():
        return set(), 0.0, 0, 0, 0
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute("SELECT key, value FROM kv_store").fetchall()
    conn.close()
    keys = set()
    total_cost = 0.0
    total_prompt = 0
    total_completion = 0
    total_tokens = 0
    for key, val_str in rows:
        keys.add(key)
        data = json.loads(val_str)
        total_cost += float(data.get("cost_usd", 0))
        total_prompt += int(data.get("prompt_tokens", 0))
        total_completion += int(data.get("completion_tokens", 0))
        total_tokens += int(data.get("total_tokens", 0))
    return keys, total_cost, total_prompt, total_completion, total_tokens


def snapshot_cache(db_path):
    return _get_cache_keys(db_path)


def diff_cache(db_path, before):
    keys_after, cost_after, prompt_after, completion_after, tokens_after = _get_cache_keys(db_path)
    keys_before, cost_before, prompt_before, completion_before, tokens_before = before
    new_keys = keys_after - keys_before
    return {
        "llm_calls": len(new_keys),
        "prompt_tokens": prompt_after - prompt_before,
        "completion_tokens": completion_after - completion_before,
        "total_tokens": tokens_after - tokens_before,
        "cost_usd": cost_after - cost_before,
    }


# ============================================================
# BUILD INDEX
# ============================================================

# %% Build index (with all modes enabled)
rag = GraphRAG(
    working_dir=str(WORKING_DIR),
    enable_llm_cache=True,
    enable_local=True,
    enable_naive_rag=True,
)

with open("../tests/fixtures/mock_data.txt", encoding="utf-8-sig") as f:
    text = "".join(f.readlines()[:400])

print("Building index...")
start = time.time()
await rag.ainsert(text)
print(f"Done in {time.time() - start:.1f}s")

CACHE_DB = WORKING_DIR / "kv_store_llm_response_cache.db"

# ============================================================
# PER-MODE BENCHMARKS — timing, tokens, cost
# ============================================================

QUESTIONS = [
    "What are the main themes in A Christmas Carol?",
    "What is the relationship between Scrooge and Bob Cratchit?",
    "How does Ebenezer Scrooge transform throughout the story?",
    "Who is Jacob Marley and what role does he play?",
    "Describe the Cratchit family and their circumstances.",
]

MODES = ["global", "local", "naive", "entity_grounded"]

benchmark_results = defaultdict(lambda: {
    "times": [],
    "llm_calls": [],
    "prompt_tokens": [],
    "completion_tokens": [],
    "total_tokens": [],
    "cost_usd": [],
    "answers": [],
})

print(f"\nBenchmarking {len(QUESTIONS)} questions x {len(MODES)} modes...")
print("=" * 80)

for q_idx, question in enumerate(QUESTIONS):
    print(f"\nQ{q_idx + 1}: {question[:70]}...")
    for mode in MODES:
        try:
            before = snapshot_cache(CACHE_DB)
            t0 = time.time()
            result = await rag.aquery(question, param=QueryParam(mode=mode))
            elapsed = time.time() - t0
            after = diff_cache(CACHE_DB, before)

            benchmark_results[mode]["times"].append(elapsed)
            benchmark_results[mode]["llm_calls"].append(after["llm_calls"])
            benchmark_results[mode]["prompt_tokens"].append(after["prompt_tokens"])
            benchmark_results[mode]["completion_tokens"].append(after["completion_tokens"])
            benchmark_results[mode]["total_tokens"].append(after["total_tokens"])
            benchmark_results[mode]["cost_usd"].append(after["cost_usd"])
            benchmark_results[mode]["answers"].append(result)

            print(f"  [{mode:17s}] {elapsed:.1f}s | {after['llm_calls']} LLM calls | {after['total_tokens']} tokens | ${after['cost_usd']:.4f}")
        except Exception as e:
            print(f"  [{mode:17s}] ERROR: {e}")

# ============================================================
# BENCHMARK VISUALIZATION
# ============================================================

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})

fig, axes = plt.subplots(2, 3, figsize=(24, 14))
fig.suptitle(
    f"Query Mode Benchmark — {len(QUESTIONS)} Questions",
    fontsize=18, fontweight="bold", y=0.98,
)

mode_colors = {"global": "#4C78A8", "local": "#F58518", "naive": "#72B7B2", "entity_grounded": "#E45756"}
x = np.arange(len(QUESTIONS))
bar_width = 0.18

# (a) Latency per question
ax = axes[0, 0]
for i, mode in enumerate(MODES):
    vals = benchmark_results[mode]["times"]
    ax.bar(x + i * bar_width, vals, bar_width, label=mode, color=mode_colors[mode], alpha=0.85)
ax.set_xlabel("Question")
ax.set_ylabel("Latency (seconds)")
ax.set_title("(a) Query Latency per Question")
ax.set_xticks(x + bar_width * 1.5)
ax.set_xticklabels([f"Q{i + 1}" for i in range(len(QUESTIONS))])
ax.legend()

# (b) Average latency
ax = axes[0, 1]
avg_times = [np.mean(benchmark_results[m]["times"]) for m in MODES]
std_times = [np.std(benchmark_results[m]["times"]) for m in MODES]
bars = ax.bar(MODES, avg_times, color=[mode_colors[m] for m in MODES], alpha=0.85, edgecolor="white")
ax.errorbar(MODES, avg_times, yerr=std_times, fmt="none", ecolor="black", capsize=5)
for bar, val in zip(bars, avg_times):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, f"{val:.1f}s", ha="center", fontweight="bold")
ax.set_ylabel("Avg Latency (seconds)")
ax.set_title("(b) Average Query Latency")

# (c) LLM calls per query
ax = axes[0, 2]
avg_calls = [np.mean(benchmark_results[m]["llm_calls"]) for m in MODES]
bars = ax.bar(MODES, avg_calls, color=[mode_colors[m] for m in MODES], alpha=0.85, edgecolor="white")
for bar, val in zip(bars, avg_calls):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, f"{val:.1f}", ha="center", fontweight="bold")
ax.set_ylabel("Avg LLM Calls per Query")
ax.set_title("(c) LLM API Calls (More = Higher Cost)")

# (d) Total tokens per query
ax = axes[1, 0]
avg_tokens = [np.mean(benchmark_results[m]["total_tokens"]) for m in MODES]
bars = ax.bar(MODES, avg_tokens, color=[mode_colors[m] for m in MODES], alpha=0.85, edgecolor="white")
for bar, val in zip(bars, avg_tokens):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5, f"{val:.0f}", ha="center", fontweight="bold")
ax.set_ylabel("Avg Total Tokens per Query")
ax.set_title("(d) Token Usage (Input + Output)")

# (e) Token breakdown (prompt vs completion)
ax = axes[1, 1]
avg_prompt = [np.mean(benchmark_results[m]["prompt_tokens"]) for m in MODES]
avg_completion = [np.mean(benchmark_results[m]["completion_tokens"]) for m in MODES]
x_modes = np.arange(len(MODES))
ax.bar(x_modes, avg_prompt, 0.6, label="Prompt (input)", color="#4C78A8", alpha=0.85)
ax.bar(x_modes, avg_completion, 0.6, bottom=avg_prompt, label="Completion (output)", color="#F58518", alpha=0.85)
ax.set_xticks(x_modes)
ax.set_xticklabels(MODES)
ax.set_ylabel("Avg Tokens per Query")
ax.set_title("(e) Token Breakdown (Input vs Output)")
ax.legend()

# (f) Estimated cost
ax = axes[1, 2]
avg_cost = [np.mean(benchmark_results[m]["cost_usd"]) for m in MODES]
total_cost = [sum(benchmark_results[m]["cost_usd"]) for m in MODES]
bars = ax.bar(MODES, avg_cost, color=[mode_colors[m] for m in MODES], alpha=0.85, edgecolor="white")
for bar, val, _total in zip(bars, avg_cost, total_cost):
    label = f"${val:.4f}" if val >= 0.0001 else f"${val:.6f}"
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label, ha="center", va="bottom", fontweight="bold")
ax.set_ylabel("Avg Cost per Query ($)")
ax.set_title(f"(f) Estimated Cost (total: ${sum(total_cost):.4f})")

plt.tight_layout(rect=[0, 0, 1, 0.95])
path = WORKING_DIR / "fig_benchmark_modes.png"
plt.savefig(path, bbox_inches="tight")
plt.close()
print(f"\nSaved {path}")

# ============================================================
# QUALITY COMPARISON — same question, all modes
# ============================================================

print("\n" + "=" * 80)
print("QUALITY COMPARISON — Same Question, All Modes")
print("=" * 80)

for q_idx, question in enumerate(QUESTIONS[:3]):
    print(f"\nQ{q_idx + 1}: {question}")
    print("-" * 70)
    for mode in MODES:
        ans = benchmark_results[mode]["answers"][q_idx]
        preview = ans[:250].replace("\n", " ")
        print(f"\n  [{mode:17s}] ({len(ans)} chars)")
        print(f"  {preview}...")

# ============================================================
# DETAILED MODE DEMOS
# ============================================================

# %% Global mode tuning
print("\n" + "=" * 80)
print("GLOBAL MODE — Tuning Parameters")
print("=" * 80)

result = await rag.aquery(
    "What are the main themes in A Christmas Carol?",
    param=QueryParam(
        mode="global",
        level=0,
        global_max_consider_community=256,
        global_max_token_for_community_report=8192,
    ),
)
print(f"Global (level=0, 256 communities): {result[:200]}...")

# %% Local mode tuning
print("\n--- LOCAL MODE — Deep Retrieval ---")
result = await rag.aquery(
    "Describe Tiny Tim's condition and its significance to the story.",
    param=QueryParam(
        mode="local",
        top_k=30,
        local_max_token_for_text_unit=6000,
        local_max_token_for_local_context=8000,
        local_max_token_for_community_report=4000,
    ),
)
print(f"Local (top_k=30, expanded budget): {result[:200]}...")

# ============================================================
# STREAMING
# ============================================================

print("\n--- Streaming (global mode) ---")
chunks = []
async for chunk in rag.astream_query(
    "How does Scrooge change throughout the story?",
    param=QueryParam(mode="global"),
):
    chunks.append(chunk)
    print(chunk, end="", flush=True)
print(f"\n({len(''.join(chunks))} chars streamed)")

# ============================================================
# RESPONSE TYPES
# ============================================================

print("\n--- Bullet Points ---")
result = await rag.aquery(
    "What are the key events in A Christmas Carol?",
    param=QueryParam(mode="global", response_type=ResponseType.BULLET_POINTS),
)
print(result)

print("\n--- Concise Answer ---")
result = await rag.aquery(
    "Who wrote A Christmas Carol?",
    param=QueryParam(mode="local", response_type=ResponseType.CONCISE),
)
print(result)

# ============================================================
# CONTEXT-ONLY MODE
# ============================================================

context = await rag.aquery(
    "Scrooge",
    param=QueryParam(mode="local", only_need_context=True),
)
print(f"\nContext-only ({len(context)} chars):\n{context[:800]}...")

# ============================================================
# SUMMARY TABLE
# ============================================================

print("\n" + "=" * 80)
print("BENCHMARK SUMMARY")
print("=" * 80)

print(f"\n{'Mode':<20s} {'Avg Latency':>12s} {'Avg LLM Calls':>14s} {'Avg Tokens':>12s} {'Avg Cost':>12s}")
print("-" * 70)
for mode in MODES:
    avg_t = np.mean(benchmark_results[mode]["times"])
    avg_c = np.mean(benchmark_results[mode]["llm_calls"])
    avg_tok = np.mean(benchmark_results[mode]["total_tokens"])
    avg_co = np.mean(benchmark_results[mode]["cost_usd"])
    print(f"{mode:<20s} {avg_t:>10.1f}s {avg_c:>12.1f} {avg_tok:>10.0f} ${avg_co:>10.4f}")

print("""
+------------------+-------------------------------------------+-------------------------+
| Mode             | Best for                                  | Enabled by              |
+------------------+-------------------------------------------+-------------------------+
| global           | Broad thematic questions                  | Default                 |
| local            | Entity-specific questions                 | enable_local=True       |
| naive            | Simple fact lookup, baseline              | enable_naive_rag=True   |
| entity_grounded  | Precise answers with entity grounding     | enable_local=True       |
+------------------+-------------------------------------------+-------------------------+

Tuning parameters:
  Global: level, global_max_consider_community, global_max_token_for_community_report
  Local:  top_k, local_max_token_for_text_unit, local_max_token_for_local_context

Streaming:      astream_query() — tokens as they arrive
Response types: QueryParam(response_type=ResponseType.BULLET_POINTS)
Context only:   QueryParam(only_need_context=True) — skip LLM generation
""")
