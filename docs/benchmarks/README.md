# Benchmark Docs

The canonical runnable benchmark documentation lives under [`../../experiments/`](../../experiments).

## Use These Files

- Main workflow: [`../../experiments/README.md`](../../experiments/README.md)
- Model selection and `.env` defaults: [`../../experiments/SELECTING_MODELS.md`](../../experiments/SELECTING_MODELS.md)
- OpenRouter-specific setup: [`../../experiments/OPENROUTER_SETUP.md`](../../experiments/OPENROUTER_SETUP.md)

## Minimal Flow

```bash
cp .env.example .env
uv sync
uv run python experiments/validate_setup.py
./experiments/run_all_benchmarks.sh --quick
uv run python experiments/compare_results.py
```

## What Lives Here vs There

- `experiments/`: runnable configs, provider setup, and scripts
- `docs/benchmarks/`: navigation only
- `docs/archive/benchmarks/`: older benchmark notes kept for history
