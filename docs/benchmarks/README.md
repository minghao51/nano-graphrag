# Benchmark Docs

The canonical runnable benchmark documentation lives under [`../../experiments/`](../../experiments).

## Use These Files

- Main workflow: [`../../experiments/docs/README.md`](../../experiments/docs/README.md)
- Model selection and `.env` defaults: [`../../experiments/docs/SELECTING_MODELS.md`](../../experiments/docs/SELECTING_MODELS.md)
- OpenRouter-specific setup: [`../../experiments/docs/OPENROUTER_SETUP.md`](../../experiments/docs/OPENROUTER_SETUP.md)

## Minimal Flow

```bash
cp .env.example .env
uv sync
dotenvx run -- ./experiments/scripts/run_all_benchmarks.sh --quick
dotenvx run -- uv run python experiments/scripts/compare_results.py
```

## What Lives Where

- `experiments/configs/`: YAML benchmark configs (`core/` and `techniques/`)
- `experiments/scripts/`: Shell scripts and Python utilities
- `experiments/docs/`: Benchmark documentation
- `docs/benchmarks/`: Navigation only
- `docs/archive/benchmarks/`: Older benchmark notes kept for history
