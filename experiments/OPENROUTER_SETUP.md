# OpenRouter Setup Guide

Quick setup for using OpenRouter with nano-graphrag benchmarks.

## Why OpenRouter?

- 💰 **10x cheaper** than OpenAI
- 🎯 Access to Claude 3.5, GPT-4, Llama 3, and more
- 🔄 Single API key for all providers
- 📊 Transparent pricing: https://openrouter.ai/docs#models

## Setup (2 minutes)

### 1. Get OpenRouter API Key

```bash
# Visit: https://openrouter.ai/keys
# Sign up (free) and create an API key
```

### 2. Configure nano-graphrag

```bash
# Add your API key to .env
echo "OPENROUTER_API_KEY=sk-or-v1-your-key-here" >> .env
```

### 3. Run Benchmarks

```bash
# Quick test (50 samples × 4 datasets)
./experiments/run_all_benchmarks_openrouter.sh --quick

# View results
uv run python experiments/compare_results.py
```

## Available Models

OpenRouter supports 100+ models. Popular choices:

### For LLM (Entity Extraction, Query Decomposition, Answer Generation)

```yaml
# Best quality (recommended)
llm_model: openrouter/anthropic/claude-3.5-sonnet

# Best value
llm_model: openrouter/meta-llama/llama-3.1-70b

# Fast & cheap
llm_model: openrouter/microsoft/phi-3-medium-128k-instruct
```

### For Embeddings

OpenRouter supports many embedding models. Recommended choices:

```yaml
# Qwen3 (good balance of quality and cost)
embedding_model: openrouter/qwen/qwen3-embedding-8b
embedding_dim: 4096

# OpenAI via OpenRouter (highest quality)
embedding_model: openrouter/openai/text-embedding-3-small
embedding_dim: 1536

# BGE-M3 (multilingual support, long context)
embedding_model: openrouter/baai/bge-m3
embedding_dim: 1024
```

**Note:** OpenRouter does not document specific concurrent call limits for embeddings. The models API returns `per_request_limits: null` for all embedding models. Rate limits primarily apply to free models (20 req/min, 50-1000/day depending on credits). For production use, implement proper backoff and retry logic.

## Configuration Examples

### Example 1: Cost-Optimized (Recommended)
```yaml
graphrag:
  llm_model: openrouter/meta-llama/llama-3.1-70b
  embedding_model: openrouter/qwen/qwen3-embedding-8b
  embedding_dim: 4096
```
**Cost:** ~$0.50 for quick test

### Example 2: Quality-Optimized
```yaml
graphrag:
  llm_model: openrouter/anthropic/claude-3.5-sonnet
  embedding_model: openrouter/openai/text-embedding-3-small
  embedding_dim: 1536
```
**Cost:** ~$1.50 for quick test

### Example 3: Ultra-Cheap
```yaml
graphrag:
  llm_model: openrouter/microsoft/phi-3-medium-128k-instruct
  embedding_model: openrouter/sentence-transformers/all-mpnet-base-v2
  embedding_dim: 768
```
**Cost:** ~$0.20 for quick test

## Troubleshooting

### Issue: "OPENROUTER_API_KEY not found"

**Solution:** Make sure you added it to `.env` file:
```bash
cat .env | grep OPENROUTER
```

### Issue: Model not found

**Solution:** Check available models at https://openrouter.ai/docs#models

Use the format: `openrouter/<provider>/<model>`

### Issue: Slow performance

**Solution:** Enable caching (already enabled by default):
```yaml
cache:
  enabled: true
```

## Pricing Details

Current prices (per 1M tokens):

| Model | Input | Output |
|-------|-------|--------|
| Claude 3.5 Sonnet | $3.00 | $15.00 |
| Llama 3.1 70B | $0.59 | $0.79 |
| Phi 3 Medium | $0.10 | $0.10 |

For more details: https://openrouter.ai/docs#models

## Next Steps

1. ✅ Run quick test: `./experiments/run_all_benchmarks_openrouter.sh --quick`
2. 📊 Compare results: `uv run python experiments/compare_results.py`
3. 🚀 Run full benchmark: `./experiments/run_all_benchmarks_openrouter.sh`
4. 📈 Update verification report with results

---

**Need help?** See [experiments/README.md](./README.md) or main [README.md](../README.md)
