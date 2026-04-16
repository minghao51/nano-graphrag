# Multi-Hop RAG Benchmarks

Canonical guide for running the benchmark configs in [`/experiments`](.) without guessing which script or provider path to use.

## Recommended Reading Order

1. This file for the benchmark workflow
2. [`SELECTING_MODELS.md`](./SELECTING_MODELS.md) if you want the generic configs to inherit models from `.env`
3. [`OPENROUTER_SETUP.md`](./OPENROUTER_SETUP.md) if you want the OpenRouter-specific configs/scripts

## LLM Provider Options

nano-graphrag uses **LiteLLM** which supports 100+ providers! Choose the best option for your needs:

### 🚀 OpenRouter (Recommended - Cost Effective)
**Best for:** Development and production (10x cheaper than OpenAI)

```bash
# 1. Get API key from https://openrouter.ai/keys
# 2. Add to .env file:
echo "OPENROUTER_API_KEY=sk-or-v1-..." >> .env

# 3. Run benchmarks:
./experiments/run_all_benchmarks_openrouter.sh --quick
```

**Cost:** ~$0.50-2.00 for quick test (vs $5-10 with OpenAI)
**Models:** Claude 3.5 Sonnet, Llama 3.1, GPT-4, and more

### 🔓 Ollama (Free, Local)
**Best for:** Development, testing, offline use

```bash
# 1. Install Ollama: https://ollama.ai
# 2. No API key needed!
# 3. Update config to use: llm_model: "ollama/llama3.2"
```

**Cost:** Free (runs locally)
**Models:** Llama 3.2, Qwen 2.5, Mistral, and more

### 🤖 Generic / OpenAI-Default Path
**Best for:** Using built-in defaults or your own `.env` model selection

```bash
# 1. Set an API key or provider settings your chosen model needs
echo "OPENAI_API_KEY=sk-..." >> .env

# 2. Run the generic configs
./experiments/run_all_benchmarks.sh --quick
```

The generic benchmark YAMLs do **not** pin `llm_model` or `embedding_model`, so they inherit:
- your `.env` values if set
- otherwise nano-graphrag defaults (`gpt-4o-mini` and `text-embedding-3-small`)

### 📚 Anthropic
**Best for:** Highest quality output

```bash
# 1. Set API key:
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env

# 2. Use model: "claude-3-5-sonnet-20241022"
```

**Cost:** ~$3-7 for quick test
**Models:** Claude 3.5 Sonnet, Claude 3 Haiku

---

## Datasets

| Dataset | Questions | Hop Depth | Download |
|---------|-----------|-----------|----------|
| **MultiHop-RAG** | 2,556 (dev) | 2 | [HuggingFace](https://huggingface.co/datasets/TIGER-Lab/MultiHop-RAG) |
| **MuSiQue** | 2,417 (dev) | 2-4 | [HuggingFace](https://huggingface.co/datasets/MuSiQue) |
| **HotpotQA** | 7,405 (dev) | 2 | [HuggingFace](https://huggingface.co/datasets/hotpotqa/hotpotqa) |
| **2WikiMHQA** | 12,576 (dev) | 2-5 | [HuggingFace](https://huggingface.co/datasets/2WikiMultiHopQA) |

## Quick Start

### 1. Install Dependencies

```bash
# Ensure all dependencies are installed
uv sync
```

### 2. Run Quick Test

```bash
# Runs the checked-in quick configs
./run_all_benchmarks.sh --quick
```

This will:
- Download datasets automatically (first run only)
- Run the sample sizes currently checked into each YAML
- Take ~15-30 minutes
- Save results to `./results/`

### 3. View Results

```bash
python experiments/compare_results.py
```

## Running Full Benchmarks

### Single Dataset

```bash
# MultiHop-RAG (2,556 samples)
python -m bench.run --config experiments/benchmark_multihop_rag.yaml

# MuSiQue (2,417 samples)
python -m bench.run --config experiments/benchmark_musique.yaml

# HotpotQA (7,405 samples)
python -m bench.run --config experiments/benchmark_hotpotqa.yaml

# 2WikiMHQA (12,576 samples)
python -m bench.run --config experiments/benchmark_2wiki.yaml
```

### All Datasets

```bash
# Run the checked-in configs for all datasets
./run_all_benchmarks.sh
```

To run a larger evaluation, edit `dataset.max_samples` and `dataset.max_corpus_samples` in the YAMLs first.

## Configuration Options

### Sample Size

Edit the `max_samples` and `max_corpus_samples` fields in each config:

```yaml
dataset:
  max_samples: 50  # Quick test
  max_corpus_samples: 500
  # Use -1 for full datasets/corpora
```

### Query Modes

All configs test three modes:
- **naive**: Baseline vector search
- **local**: GraphRAG local retrieval
- **multihop**: Multi-hop retrieval (Phase 3)
- **adaptive**: Automatic mode selection based on query patterns (Phase 4)
- **hipporag**: Personalized PageRank for multi-hop discovery (Phase 4)
- **hybrid**: Fusion of multiple retrieval strategies (Phase 4)
- **raptor**: Hierarchical clustering and summarization tree (Phase 4)

To test only specific modes:

```yaml
query:
  modes:
    - local
    - multihop
  # Remove 'naive' to skip baseline
```

#### Phase 4 Advanced Retrieval Modes

**Adaptive Router** (`benchmark_adaptive.yaml`)
- Automatically selects optimal retrieval mode per query
- Uses regex patterns to detect multi-hop and global queries
- Falls back to local mode for simple queries

```yaml
query:
  modes:
    - adaptive
  param_overrides:
    llm_fallback_threshold: 2    # Pattern matches to trigger mode
    default_mode: "local"
```

**HippoRAG PPR** (`benchmark_hipporag.yaml`)
- Single-operation multi-hop via Personalized PageRank
- Seed entities identified from query, then PageRank propagates relevance
- Effective for questions requiring multi-step reasoning

```yaml
query:
  modes:
    - hipporag
  param_overrides:
    alpha: 0.85              # PageRank damping factor
    top_k_seed: 5            # Number of seed entities
    top_k_result: 20         # Number of final results
```

**Hybrid Retrieval** (`benchmark_hybrid.yaml`)
- Fusion of multiple retrieval strategies (local + global)
- Supports reciprocal rank fusion, weighted averaging, and RRF
- Balances precision and recall

```yaml
query:
  modes:
    - hybrid
  param_overrides:
    fusion_strategy: "reciprocal_rank"  # weighted_avg, reciprocal_rank, rrf
    retriever_weights:
      local: 0.6
      global: 0.4
```

**RAPTOR** (`benchmark_raptor.yaml`)
- Hierarchical clustering and summarization tree
- Clusters similar passages and creates summaries at each level
- Enables multi-granular retrieval from leaf to root

```yaml
query:
  modes:
    - raptor
  param_overrides:
    max_tree_levels: 3           # Maximum depth of hierarchy
    cluster_method: "gmm"        # gmm, kmeans
    top_k_cluster: 5             # Clusters per level
    summary_token_limit: 500     # Token limit for summaries
```

### Multi-Hop Parameters

Adjust multi-hop retrieval behavior:

```yaml
query:
  param_overrides:
    max_hops: 4              # Maximum decomposition depth
    entities_per_hop: 10     # Entities to retrieve per hop
    context_token_budget: 8000  # Max context tokens
    top_k: 60               # Top-k for retrieval
```

### Cache Settings

LLM response caching saves money on repeated runs:

```yaml
cache:
  enabled: true    # Enable/disable caching
  backend: disk    # Storage backend
```

**Tip:** Always keep cache enabled during development. Disable only for fresh runs.

## Expected Results

Based on roadmap targets:

| Dataset | Naive F1 | Target Multihop F1 | Improvement |
|---------|----------|-------------------|-------------|
| MultiHop-RAG | ~0.40 | 0.57+ | +0.17 |
| MuSiQue | ~0.25 | 0.42+ | +0.17 |
| HotpotQA | ~0.45 | 0.62+ | +0.17 |
| 2WikiMHQA | ~0.38 | 0.55+ | +0.17 |

### Phase 4 Advanced Techniques

The following advanced retrieval techniques are now available:

| Technique | Purpose | Config |
|-----------|---------|--------|
| **Cross-Encoder Reranking** | Re-score passages using `sentence-transformers` | Use via `reranker: cross_encoder` in config |
| **Adaptive Router** | Auto-select local/global/multihop mode | `modes: [adaptive]` |
| **Edge Confidence** | Weighted graph edges by confidence | Post-insert hook |
| **HippoRAG PPR** | Single-operation multi-hop via PageRank | `modes: [hipporag]` |
| **Hybrid Retrieval** | Fusion of multiple retrievers | `modes: [hybrid]` |
| **DSPy Tuning** | Optimize entity extraction prompts | CLI: `uv run python -m bench.dspy_tune` |
| **RAPTOR** | Hierarchical clustering summarization | `modes: [raptor]` |

**Installation:**

```bash
# Install optional dependencies for Phase 4 features
uv sync --extra advanced-retrieval
```

**Example configs:**
- `experiments/benchmark_adaptive.yaml`
- `experiments/benchmark_hipporag.yaml`
- `experiments/benchmark_hybrid.yaml`
- `experiments/benchmark_raptor.yaml`

## Output Format

Results are saved as JSON in `./results/<dataset>/`:

```json
{
  "experiment_name": "multihop_rag_benchmark",
  "timestamp": "2026-03-24T12:00:00",
  "mode_results": {
    "naive": {
      "exact_match": 0.35,
      "token_f1": 0.40
    },
    "local": {
      "exact_match": 0.42,
      "token_f1": 0.48
    },
    "multihop": {
      "exact_match": 0.45,
      "token_f1": 0.52
    }
  },
  "duration_seconds": 1234,
  "cache_stats": {
    "hits": 450,
    "misses": 50,
    "hit_rate": 0.90
  }
}
```

## Cost Estimation

With GPT-4o-mini and caching enabled:

| Samples | Est. Cost (first run) | Est. Cost (cached) |
|---------|----------------------|-------------------|
| 50 | ~$0.50 | ~$0.05 |
| 500 | ~$5.00 | ~$0.50 |
| 2500 | ~$25.00 | ~$2.50 |

**Note:** Cache hit rates of 80-90% are typical after the first run.

## Notes On Repo Layout

- `experiments/*.yaml` are the runnable configs
- `experiments/run_all_benchmarks*.sh` are thin wrappers around those configs
- `experiments/compare_results.py` compares saved outputs under `./results/**`
- `docs/archive/roadmap/technical-roadmap.md` is planning/history, not the day-to-day runbook
- `docs/archive/verification/20260324-phase3-verification-report.md` tracks implementation status, not live benchmark scores

## Troubleshooting

### Dataset Download Fails

```bash
uv sync
uv run python experiments/validate_setup.py
```

### Out of Memory Errors

Reduce `max_samples` or decrease `chunk_token_size`:

```yaml
graphrag:
  chunk_token_size: 800  # Reduce from 1200
```

### Slow Performance

1. Enable caching (already enabled by default)
2. Reduce sample size for testing
3. Use cheaper model for development:

```yaml
graphrag:
  llm_model: gpt-4o-mini  # Cheaper than gpt-4o
```

## Next Steps

After running benchmarks:

1. **Compare results**: `python experiments/compare_results.py`
2. **Analyze failures**: Check `predictions.json` for error patterns
3. **Update verification report**: Add actual F1 scores to `docs/archive/verification/20260324-phase3-verification-report.md`
4. **Ablation studies**: Create configs testing different parameters

## Advanced Usage

### A/B Testing

Compare different configurations:

```yaml
# experiments/ablation_chunk_size.yaml
name: ablation_chunk_size
ab_test: true

shared:
  dataset:
    name: multihop_rag
    max_samples: 100

variant_a:
  label: "chunk_1200"
  graphrag:
    chunk_token_size: 1200

variant_b:
  label: "chunk_800"
  graphrag:
    chunk_token_size: 800
```

### Custom Datasets

Use your own dataset:

```yaml
dataset:
  name: multihop_rag  # Use existing loader
  path: /path/to/your/questions.json
  corpus_path: /path/to/your/corpus.json
  auto_download: false
```

Format:
- `questions.json`: Array of `{id, question, answer}` objects
- `corpus.json`: Array of `{id, content, title}` objects

---

**Questions?** See the main [README.md](../README.md), the docs index at [docs/README.md](../docs/README.md), or the historical roadmap at [docs/archive/roadmap/technical-roadmap.md](../docs/archive/roadmap/technical-roadmap.md)

## Cost Comparison

| Provider | Quick Test (50×4) | Full Benchmark | Quality |
|----------|-------------------|----------------|---------|
| **OpenRouter** | ~$0.50-2.00 | ~$5-15 | High |
| Ollama | Free | Free | Medium |
| OpenAI | ~$5-10 | ~$25-50 | High |
| Anthropic | ~$3-7 | ~$15-35 | Very High |

*Prices are estimates. Actual costs vary by model and usage.*

**Recommendation:** Use **OpenRouter** for the best balance of cost and quality!
