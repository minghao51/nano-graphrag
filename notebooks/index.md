# Notebooks

Interactive tutorials demonstrating nano-graphrag features.

| Notebook | Description |
|----------|-------------|
| [Quick Start](quick-start.md) | Minimal end-to-end: insert documents, build a knowledge graph, and query in global & local modes. |
| [Configuration](configuration.md) | All config paths: direct kwargs, environment variables, YAML files, and dictionaries — nested vs flat format. |
| [Query Modes](query-modes.md) | Deep dive into global, local, naive, entity-grounded, and streaming query modes — when to use each and how to tune them. |

## Running Locally

```bash
uv sync --extra docs
uv run quarto render notebooks/
```
