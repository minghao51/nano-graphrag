# Experiments Quick Start

Use the benchmark docs in this order:

1. [`/Users/minghao/Desktop/personal/nano-graphrag/experiments/README.md`](./experiments/README.md)
2. [`/Users/minghao/Desktop/personal/nano-graphrag/experiments/SELECTING_MODELS.md`](./experiments/SELECTING_MODELS.md)
3. [`/Users/minghao/Desktop/personal/nano-graphrag/experiments/OPENROUTER_SETUP.md`](./experiments/OPENROUTER_SETUP.md) if you want the OpenRouter-specific path

Minimal setup:

```bash
cp .env.example .env
uv sync
uv run python experiments/validate_setup.py
./experiments/run_all_benchmarks.sh --quick
```

The generic benchmark YAMLs inherit `LLM_MODEL` and `EMBEDDING_MODEL` from `.env` when you set them there. If you want to pin provider-specific models inside the YAMLs instead, use the dedicated OpenRouter configs in `/experiments`.
