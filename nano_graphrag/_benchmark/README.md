# Benchmark Infrastructure for nano-graphrag

A minimal benchmark & experimentation framework for evaluating GraphRAG on multi-hop RAG benchmarks.

## Features

- **Dataset loaders**: MultiHop-RAG, HotpotQA, MuSiQue, 2WikiMultiHopQA
- **Metrics**: Exact Match (EM), Token F1, optional Ragas LLM-as-judge
- **LLM caching**: Persistent cache with statistics using existing storage patterns
- **YAML configs**: Human-readable, reproducible experiment configuration
- **CLI**: Simple command-line interface for running experiments

## Quick Start

### 1. Install dependencies

```bash
# Core dependencies (already installed)
uv sync

# Optional: For LLM-as-judge metrics
uv add ragas
```

### 2. Create a config file

```yaml
# my_experiment.yaml
dataset_name: "multihop_rag"
dataset_path: "./fixtures/MultiHopRAG.json"
corpus_path: "./fixtures/MultiHopRAG_corpus.json"
max_samples: 100

graphrag_config:
  working_dir: "./nano_graphrag_cache_benchmark"
  llm_model: "gpt-4o-mini"
  embedding_model: "text-embedding-3-small"

query_modes: ["local", "global"]
metrics: ["exact_match", "token_f1"]
output_dir: "./benchmark_results"
experiment_name: "my_experiment"
```

### 3. Run experiment

```bash
uv run python examples/benchmarks/run_experiment.py --config my_experiment.yaml
```

## Usage Examples

### Python API

```python
import asyncio
from nano_graphrag._benchmark import BenchmarkConfig, ExperimentRunner

async def main():
    # Load config
    config = BenchmarkConfig.from_yaml("my_experiment.yaml")

    # Run experiment
    runner = ExperimentRunner(config)
    result = await runner.run()

    # Print results
    for mode, scores in result.mode_results.items():
        print(f"{mode}: {scores}")

asyncio.run(main())
```

### Custom datasets

```python
from nano_graphrag._benchmark import BenchmarkDataset
from typing import Dict, List, Any

class MyDataset(BenchmarkDataset):
    def questions(self, split: str = "test") -> List[Dict[str, Any]]:
        return [
            {"question": "What is...?", "answer": "The answer"},
            # ... more questions
        ]

    def corpus(self) -> List[str]:
        return [
            "Document 1 content...",
            "Document 2 content...",
            # ... more documents
        ]
```

### Custom metrics

```python
from nano_graphrag._benchmark import Metric, MetricSuite

class MyMetric(Metric):
    async def compute(self, prediction: str, gold: str, **kwargs) -> float:
        # Your metric logic here
        return 0.5

suite = MetricSuite()
suite.add_metric("my_metric", MyMetric())
```

## Dataset Format

### MultiHop-RAG

**Questions file:**
```json
[
    {
        "question": "What is the capital of France?",
        "answer": "Paris"
    }
]
```

**Corpus file:**
```json
[
    {
        "content": "France is a country in Europe. Its capital is Paris."
    }
]
```

### HotpotQA / MuSiQue / 2WikiMultiHopQA

Standard format from the respective datasets. See the original dataset sources for details.

## Metrics

- **Exact Match (EM)**: Normalized string comparison (case-insensitive, articles removed)
- **Token F1**: Token-level F1 score based on word overlap
- **Ragas (optional)**: LLM-as-judge metrics (faithfulness, answer relevance)

## Configuration

See `examples/benchmarks/configs/example.yaml` for a complete example.

### Config options

| Option | Description | Default |
|--------|-------------|---------|
| `dataset_name` | Dataset type | Required |
| `dataset_path` | Path to questions JSON | Required |
| `corpus_path` | Path to corpus JSON (for multihop_rag) | None |
| `max_samples` | Number of samples (-1 for all) | -1 |
| `graphrag_config` | GraphRAG configuration dict | {} |
| `query_modes` | List of modes to test | ["local", "global"] |
| `query_params` | QueryParam overrides | {} |
| `metrics` | List of metrics to compute | ["exact_match", "token_f1"] |
| `output_dir` | Results directory | "./benchmark_results" |
| `experiment_name` | Experiment identifier | "experiment" |

## Output

Results are saved as JSON with the following structure:

```json
{
  "experiment_name": "my_experiment",
  "timestamp": "2026-03-23T12:00:00",
  "config": { ... },
  "mode_results": {
    "local": {"exact_match": 0.85, "token_f1": 0.82},
    "global": {"exact_match": 0.78, "token_f1": 0.75}
  },
  "predictions": {
    "local": [
      {"question": "...", "prediction": "...", "gold": "..."}
    ]
  },
  "duration_seconds": 123.45
}
```

## CLI Options

```bash
python examples/benchmarks/run_experiment.py --help

# Run with config
python run_experiment.py --config config.yaml

# Override experiment name
python run_experiment.py --config config.yaml --name my_test

# Limit samples
python run_experiment.py --config config.yaml --max-samples 10

# Specific query modes
python run_experiment.py --config config.yaml --modes local global

# Dry run (validate config only)
python run_experiment.py --config config.yaml --dry-run
```

## Design Decisions

1. **Separate `_benchmark` package**: Keeps benchmark code isolated from core
2. **Protocol for datasets**: Users can add custom datasets without modifying core
3. **YAML configs**: Human-readable, version-control friendly, enables reproducibility
4. **Ragas optional**: EM and F1 sufficient for basic benchmarks; LLM-as-judge adds cost
5. **Async throughout**: Consistent with core GraphRAG architecture

## Files

```
nano_graphrag/_benchmark/
├── __init__.py          # Public API exports
├── datasets.py          # Dataset loaders (~120 lines)
├── metrics.py           # Metrics framework (~150 lines)
├── runner.py            # Experiment runner (~180 lines)
└── cache.py             # LLM cache formalization (~80 lines)

examples/benchmarks/
├── run_experiment.py    # CLI entry point
└── configs/
    └── example.yaml     # Example configuration
```

## License

MIT
