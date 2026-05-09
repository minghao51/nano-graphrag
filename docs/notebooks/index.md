# Notebooks

Interactive tutorials demonstrating nano-graphrag features.

| Notebook | Description |
|----------|-------------|
| [Quick Start](quick-start.md) | Minimal end-to-end: insert documents, build a knowledge graph, and query in global & local modes. |
| [Configuration](configuration.md) | All config paths: direct kwargs, environment variables, YAML files, and dictionaries — nested vs flat format. |
| [Query Modes](query-modes.md) | Deep dive into global, local, naive, entity-grounded, and streaming query modes — when to use each and how to tune them. |
| [Storage Backends](storage-backends.md) | Swap graph, vector, and KV storage backends — NetworkX vs SQLite vs Neo4j for graphs, HNSWLib vs NanoVectorDB for vectors. |
| [Extraction Pipeline](extraction-pipeline.md) | Explore entity extraction backends — structured (batched), legacy (text parsing), and GLiNER — plus quality modes and fallback behavior. |
| [Advanced Features](advanced-features.md) | Incremental insertion, entity linking, temporal extraction, community reports, and graph rebuild — production features for real-world usage. |

## Running Locally

```bash
uv sync --extra docs
uv run quarto render notebooks/
```
