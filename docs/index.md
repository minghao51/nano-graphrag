# nano-graphrag

A simple, easy-to-hack GraphRAG implementation.

## Notebooks

Interactive tutorials are available in the [Notebooks](notebooks/index.md) section.

## Quick Start

```python
from nano_graphrag import GraphRAG, QueryParam

rag = GraphRAG(working_dir="./my_graphrag")
rag.insert("Your text content here...")

result = rag.query("What is this about?")
print(result)
```
