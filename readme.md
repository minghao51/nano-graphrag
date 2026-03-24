<div align="center">
  <a href="https://github.com/gusye1234/nano-graphrag">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://assets.memodb.io/nano-graphrag-dark.png">
      <img alt="Shows the MemoDB logo" src="https://assets.memodb.io/nano-graphrag.png" width="512">
    </picture>
  </a>
  <p><strong>A simple, easy-to-hack GraphRAG implementation</strong></p>
  <p>
    <img src="https://img.shields.io/badge/python->=3.9.11-blue">
    <a href="https://pypi.org/project/nano-graphrag/">
      <img src="https://img.shields.io/pypi/v/nano-graphrag.svg">
    </a>
    <a href="https://codecov.io/github/gusye1234/nano-graphrag">
      <img src="https://codecov.io/github/gusye1234/nano-graphrag/graph/badge.svg?token=YFPMj9uQo7"/>
    </a>
    <a href="https://pepy.tech/project/nano-graphrag">
      <img src="https://static.pepy.tech/badge/nano-graphrag/month">
    </a>
  </p>
</div>

`nano-graphrag` keeps the GraphRAG runtime small, hackable, and practical. The repo is organized so current docs, benchmark runbooks, and historical implementation notes are easy to tell apart.

## Quick Start

Install with `uv`:

```bash
git clone https://github.com/gusye1234/nano-graphrag.git
cd nano-graphrag
uv sync
```

Download a sample corpus:

```bash
curl https://raw.githubusercontent.com/gusye1234/nano-graphrag/main/tests/mock_data.txt > book.txt
```

Run a minimal example:

```python
from nano_graphrag import GraphRAG, GraphRAGConfig, QueryParam

config = GraphRAGConfig(working_dir="./dickens")
graph = GraphRAG.from_config(config)

with open("./book.txt") as f:
    graph.insert(f.read())

print(graph.query("What are the top themes in this story?"))
print(graph.query("What are the top themes in this story?", param=QueryParam(mode="local")))
```

## Common Setup Paths

- Default hosted path: set `OPENAI_API_KEY` and use `GraphRAG(working_dir="./dickens")`
- Local Ollama path: set `llm_model="ollama/llama3.2"` and `embedding_model="ollama/nomic-embed-text"`
- OpenAI-compatible endpoint: set `llm_api_base` and `embedding_api_base`

Example Ollama configuration:

```python
graph = GraphRAG(
    working_dir="./dickens",
    llm_model="ollama/llama3.2",
    llm_api_base="http://localhost:11434",
    embedding_model="ollama/nomic-embed-text",
    embedding_api_base="http://localhost:11434",
    embedding_dim=768,
)
```

## Repo Guide

- Docs index: [`docs/README.md`](./docs/README.md)
- Architecture: [`docs/architecture/architecture.md`](./docs/architecture/architecture.md)
- Guides and troubleshooting: [`docs/guides/`](./docs/guides)
- Benchmark workflow: [`experiments/README.md`](./experiments/README.md)
- Benchmark docs map: [`docs/benchmarks/README.md`](./docs/benchmarks/README.md)
- Contributor guide: [`CONTRIBUTING.md`](./CONTRIBUTING.md)

## Runtime Notes

- `GraphRAGConfig` is the canonical configuration surface.
- `GraphRAG(...)` keyword arguments remain available as compatibility aliases.
- Reusing the same `working_dir` lets the runtime reload persisted state automatically.
- `insert` accepts either a single string or a list of strings.
- Async variants are available as `ainsert(...)` and `aquery(...)`.

## Components

| Type | Default | Alternatives |
|------|---------|--------------|
| LLM | OpenAI via LiteLLM | Ollama, Anthropic, OpenAI-compatible endpoints |
| Embedding | OpenAI | Ollama, custom embedding functions |
| Vector store | `nano-vectordb` | `hnswlib`, Milvus, Qdrant, Faiss |
| Graph store | `networkx` | Neo4j |

For Neo4j setup, see [`docs/guides/neo4j.md`](./docs/guides/neo4j.md).

## Benchmarks

Benchmark configs, scripts, and provider-specific setup live under [`experiments/`](./experiments).

Typical flow:

```bash
cp .env.example .env
uv sync
uv run python experiments/validate_setup.py
./experiments/run_all_benchmarks.sh --quick
uv run python experiments/compare_results.py
```

## Historical Material

Older planning notes, verification reports, and superseded benchmark docs live under [`docs/archive/`](./docs/archive).
