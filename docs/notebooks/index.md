# Notebooks

Interactive tutorials demonstrating nano-graphrag features.

| Notebook | Description |
|----------|-------------|
| [Quick Start](01_quick_start.ipynb) | Insert a document, query it, and inspect the resulting knowledge graph — in 5 minutes. |
| [Query Modes](02_query_modes.ipynb) | Compare global, local, naive, and entity-grounded query modes with timing benchmarks. |
| [Configuration](03_configuration.ipynb) | Inline kwargs, YAML configs, env vars, and a practical Ollama example. |
| [Storage & Extraction](04_storage_and_extraction.ipynb) | Swap storage backends (NetworkX/SQLite/Neo4j), incremental inserts, extraction tuning. |
| [Refinement & Vault](05_refinement_and_vault.ipynb) | Refinement pipeline (merge, enrich, infer), typed relations, and Obsidian vault export. |

## Viewing docs (no API keys needed)

The docs are pre-built from executed notebooks. No API keys or `.env` file needed:

```bash
uv run mkdocs serve
```

## Re-executing notebooks (requires API keys)

To re-run a notebook (e.g., after changing the code):

```bash
dotenvx run -- uv run jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=600 \
  --inplace notebooks/01_quick_start.ipynb
```

To re-run all notebooks:

```bash
for nb in notebooks/*.ipynb; do
  dotenvx run -- uv run jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=600 \
    --inplace "$nb"
done
```
