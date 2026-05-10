# Notebooks

Interactive tutorials demonstrating nano-graphrag features.

| Notebook | Description |
|----------|-------------|
| [Quick Start](quick-start.md) | Get running in 5 minutes — insert a document, query it, and explore the resulting knowledge graph. |
| [Query Modes](query-modes.md) | Compare global, local, naive, and entity-grounded query modes with timing benchmarks, token/cost analysis, and quality comparison. |
| [Configuration](configuration.md) | Learn the three main config paths (kwargs, YAML, env vars) and see a practical Ollama example. |
| [Storage & Extraction](storage-extraction.md) | Swap storage backends for your scale, do incremental inserts, tune extraction quality vs speed, and enable entity linking. |

## Running Locally

```bash
uv sync --extra docs
uv run quarto render notebooks/
```
