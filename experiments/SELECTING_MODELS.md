# How to Select LLM and Embedding Models

## The Short Version

**Edit your `.env` file and uncomment ONE option:**

### Option 1: OpenRouter (Recommended - 10x Cheaper)
```bash
LLM_MODEL=openrouter/anthropic/claude-3.5-sonnet
EMBEDDING_MODEL=openrouter/microsoft/wizardlm-2-7b
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

### Option 2: Ollama (Free)
```bash
LLM_MODEL=ollama/llama3.2
EMBEDDING_MODEL=ollama/nomic-embed-text
# No API key needed!
```

### Option 3: OpenAI (Default)
```bash
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_KEY=sk-your-key-here
```

---

## How It Works

### 1. Set Default Models in `.env`

```bash
# .env file
LLM_MODEL=openrouter/anthropic/claude-3.5-sonnet
EMBEDDING_MODEL=openrouter/microsoft/wizardlm-2-7b
OPENROUTER_API_KEY=sk-or-v1-...
```

### 2. Use in Python Code

```python
from nano_graphrag import GraphRAG

# Will use models from .env
rag = GraphRAG(working_dir="./dickens")

# Or override specific models
rag = GraphRAG(
    working_dir="./dickens",
    llm_model="ollama/llama3.2",  # Override LLM
    embedding_model="ollama/nomic-embed-text"  # Override embedding
)
```

### 3. Use in YAML Configs

```yaml
# experiments/benchmark_multihop_rag.yaml
graphrag:
  # Add these only if you want to override .env defaults
  llm_model: openrouter/anthropic/claude-3.5-sonnet
  embedding_model: openrouter/microsoft/wizardlm-2-7b
  enable_llm_cache: true
```

---

## Model Selection Priority

**Highest priority** → **Lowest priority:**

1. **Explicit in code** - `GraphRAG(llm_model="...")`
2. **YAML config** - `graphrag.llm_model:`
3. **Environment variable** - `LLM_MODEL` in `.env`
4. **System default** - Built-in default

The generic benchmark configs in `/experiments` now rely on steps 3 or 4 unless you add explicit model overrides.

---

## Examples

### Example 1: Use .env Defaults

```bash
# .env
LLM_MODEL=openrouter/anthropic/claude-3.5-sonnet
EMBEDDING_MODEL=openrouter/microsoft/wizardlm-2-7b
OPENROUTER_API_KEY=sk-or-v1-...
```

```python
# Python
rag = GraphRAG(working_dir="./dickens")
# Uses: openrouter/anthropic/claude-3.5-sonnet
```

### Example 2: Override in YAML

```yaml
# experiments/benchmark_musique.yaml
graphrag:
  llm_model: ollama/llama3.2  # Override .env
  embedding_model: ollama/nomic-embed-text
```

```bash
# Run
uv run python -m bench.run --config experiments/benchmark_musique.yaml
# Uses: ollama/llama3.2 (not .env default)
```

### Example 3: Override in Code

```python
# .env has: LLM_MODEL=gpt-4o-mini

rag = GraphRAG(
    working_dir="./test",
    llm_model="ollama/llama3.2",  # Override .env
)
# Uses: ollama/llama3.2 (not gpt-4o-mini)
```

### Example 4: Mix Providers

```bash
# .env
LLM_MODEL=openrouter/anthropic/claude-3.5-sonnet
EMBEDDING_MODEL=ollama/nomic-embed-text
OPENROUTER_API_KEY=sk-or-v1-...
```

```python
# Uses OpenRouter for LLM, Ollama for embeddings
rag = GraphRAG(working_dir="./dickens")
```

---

## Popular Model Combinations

### Cost-Optimized (OpenRouter)
```bash
LLM_MODEL=openrouter/meta-llama/llama-3.1-70b
EMBEDDING_MODEL=openrouter/microsoft/wizardlm-2-7b
```
**Cost:** ~$0.50 for quick test

### Quality-Optimized (OpenRouter)
```bash
LLM_MODEL=openrouter/anthropic/claude-3.5-sonnet
EMBEDDING_MODEL=openrouter/microsoft/wizardlm-2-7b
```
**Cost:** ~$1.50 for quick test

### Local & Free (Ollama)
```bash
LLM_MODEL=ollama/llama3.2
EMBEDDING_MODEL=ollama/nomic-embed-text
```
**Cost:** FREE

### Best Quality (Anthropic)
```bash
LLM_MODEL=claude-3-5-sonnet-20241022
EMBEDDING_MODEL=voyage-large-2
ANTHROPIC_API_KEY=sk-ant-...
```
**Cost:** ~$3.00 for quick test

---

## Quick Setup Commands

### OpenRouter Setup
```bash
# 1. Get API key from https://openrouter.ai/keys

# 2. Create .env
cat > .env << 'ENVEOF'
LLM_MODEL=openrouter/anthropic/claude-3.5-sonnet
EMBEDDING_MODEL=openrouter/microsoft/wizardlm-2-7b
OPENROUTER_API_KEY=sk-or-v1-your-key-here
ENVEOF

# 3. Test
uv run python experiments/test_openrouter.py

# 4. Run benchmarks
./experiments/run_all_benchmarks_openrouter.sh --quick
```

### Ollama Setup
```bash
# 1. Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 2. Pull models
ollama pull llama3.2
ollama pull nomic-embed-text

# 3. Create .env
cat > .env << 'ENVEOF'
LLM_MODEL=ollama/llama3.2
EMBEDDING_MODEL=ollama/nomic-embed-text
ENVEOF

# 4. Run benchmarks (free!)
./experiments/run_all_benchmarks.sh --quick
```

---

## Summary

**To select models:**

1. **Edit `.env`** - Set `LLM_MODEL` and `EMBEDDING_MODEL`
2. **Add API key** - Set the corresponding `*_API_KEY`
3. **Done!** - All GraphRAG instances will use these defaults

**To override:**

- **In YAML:** Add `llm_model:` and `embedding_model:` under `graphrag:`
- **In code:** Pass `llm_model=` and `embedding_model=` to `GraphRAG()`

**That's it!** 🎉

---

*See .env.example for complete configuration options*
