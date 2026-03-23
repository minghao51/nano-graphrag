# Benchmark Usage Guide

Complete guide to running and analyzing GraphRAG benchmarks.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Configuration](#configuration)
4. [Datasets](#datasets)
5. [Metrics](#metrics)
6. [Caching](#caching)
7. [A/B Testing](#ab-testing)
8. [Python API](#python-api)

## Installation

```bash
# Core dependencies
uv sync

# Optional: For dataset auto-download
uv add --optional datasets
```

## Quick Start

### 1. Create a config file

```yaml
# my_experiment.yaml
name: my_experiment
dataset:
  name: musique
  split: validation
  max_samples: 100
  auto_download: true

graphrag:
  working_dir: ./workdirs/my_experiment
  llm_model: gpt-4o-mini

query:
  modes:
    - local
    - global

cache:
  enabled: true

metrics:
  exact_match: true
  token_f1: true

output:
  results_dir: ./results
```

### 2. Run the experiment

```bash
uv run python -m bench.run --config my_experiment.yaml
```

### 3. View results

Results are saved to `./results/my_experiment_<timestamp>.json`:

```json
{
  "experiment_name": "my_experiment",
  "mode_results": {
    "local": {"exact_match": 0.42, "token_f1": 0.55},
    "global": {"exact_match": 0.38, "token_f1": 0.51}
  },
  "cache_stats": {
    "hits": 150,
    "misses": 50,
    "hit_rate": 0.75
  },
  "duration_seconds": 123.45
}
```

## Configuration

### Nested Schema (Recommended)

```yaml
name: experiment_name
version: "1.0"
description: "Experiment description"

dataset:
  name: musique | hotpotqa | 2wiki | multihop_rag
  path: /path/to/data  # or use auto_download
  corpus_path: /path/to/corpus  # for multihop_rag
  split: validation | dev | test
  max_samples: 100
  auto_download: true

graphrag:
  working_dir: ./workdirs/exp
  llm_model: gpt-4o-mini
  embedding_model: text-embedding-3-small
  chunk_token_size: 1200

query:
  modes:
    - local
    - global
    - naive
  param_overrides:
    top_k: 20

cache:
  enabled: true
  backend: disk

metrics:
  exact_match: true
  token_f1: true
  llm_judge:
    enabled: false
    model: gpt-4o-mini

output:
  results_dir: ./results
  save_predictions: true
```

### CLI Options

```bash
uv run python -m bench.run --config config.yaml [OPTIONS]

Options:
  --name, -n         Override experiment name
  --max-samples N    Limit number of samples
  --modes MODES      Query modes to run (space-separated)
  --output-dir, -o   Override output directory
  --dry-run          Validate config without running
```

## Datasets

### Supported Datasets

| Dataset | Split Sizes | Hop Depth | Auto-Download |
|---------|-------------|-----------|---------------|
| MultiHop-RAG | 2556 dev | 2 | Yes |
| MuSiQue | 2417 dev | 2-4 | Yes |
| HotpotQA | 7405 dev | 2 | Yes |
| 2WikiMultiHopQA | 12576 dev | 2-5 | Yes |

### Using Custom Datasets

```python
from bench import BenchmarkDataset, QAPair, Passage
from typing import Iterator

class MyDataset(BenchmarkDataset):
    name = "my_dataset"

    def questions(self, split: str = "test") -> Iterator[QAPair]:
        # Load your questions
        yield QAPair(
            id="q1",
            question="What is...?",
            answer="The answer",
            supporting_facts=["fact1", "fact2"],
        )

    def corpus(self) -> Iterator[Passage]:
        # Load your corpus
        yield Passage(
            id="doc1",
            title="Document 1",
            text="Document content...",
        )

    def download(self, cache_dir: str = "~/.cache/nano-bench") -> None:
        # Optional: Implement auto-download
        pass
```

## Metrics

### Available Metrics

- **Exact Match (EM)**: Normalized string comparison
- **Token F1**: Token overlap F1 score
- **Context Recall** (Native): Are supporting facts in context?

### Custom Metrics

```python
from bench import Metric

class MyMetric(Metric):
    async def compute(self, prediction, gold, question="", context=""):
        # Your metric logic
        return 0.5

# Add to suite
suite = MetricSuite()
suite.add_metric("my_metric", MyMetric())
```

## Caching

LLM caching is enabled by default to reduce API costs.

### Cache Statistics

After running an experiment, cache stats are printed:

```
[Cache] Hit rate: 75.00%
[Cache] 150 hits, 50 misses
```

### Disable Cache

```yaml
cache:
  enabled: false
```

## A/B Testing

### Compare Two Experiments

```bash
uv run python -m bench.compare results/exp1.json results/exp2.json
```

Output:

```
## Benchmark Comparison

**Baseline:** `results/exp1.json`
**Challenger:** `results/exp2.json`

### Results

| Mode | Metric | Baseline | Challenger | Delta |
|------|--------|----------|-------------|-------|
| local | exact_match | 0.500 | 0.600 | +0.100 ✓ |
| local | token_f1 | 0.600 | 0.550 | -0.050 ✗ |
```

### Save Comparison

```bash
uv run python -m bench.compare exp1.json exp2.json --output comparison.md
```

## Python API

### Basic Usage

```python
import asyncio
from bench import BenchmarkConfig, ExperimentRunner

async def main():
    # Load config
    config = BenchmarkConfig.from_yaml("config.yaml")

    # Run experiment
    runner = ExperimentRunner(config)
    result = await runner.run()

    # Access results
    print(result.mode_results)
    print(result.cache_stats)

asyncio.run(main())
```

### Custom Workflow

```python
from bench import BenchmarkConfig, ExperimentRunner, create_benchmark_cache

# Create cache
cache = create_benchmark_cache("./cache", enabled=True)

# Create config
config = BenchmarkConfig(
    dataset_name="musique",
    dataset_path="",
    auto_download=True,
    graphrag_config={"working_dir": "./workdirs"},
)

# Run with cache
runner = ExperimentRunner(config)
runner._cache = cache  # Inject custom cache

result = await runner.run()
```

## Tips

1. **Start small**: Use `max_samples: 10` to test your config
2. **Enable cache**: Saves costs on repeated runs
3. **Use auto-download**: Don't manually download datasets
4. **Compare often**: Use `bench.compare` to track improvements
5. **Check predictions**: Set `save_predictions: true` to debug

## Troubleshooting

### Import Error

```
ImportError: cannot import name 'BenchmarkConfig' from 'nano_graphrag._benchmark'
```

**Solution**: Use `from bench import BenchmarkConfig` instead.

### Cache Not Working

**Symptoms**: Hit rate is 0%

**Solution**: Ensure `cache.enabled: true` in config.

### Download Fails

**Symptoms**: `NotImplementedError: Auto-download not yet implemented`

**Solution**: Install datasets library: `uv add --optional datasets`
