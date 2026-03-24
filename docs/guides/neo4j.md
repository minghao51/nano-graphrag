# Neo4j Guide

Use this guide when you want Neo4j as the graph storage backend instead of the built-in `networkx` storage.

## Setup

1. Install Neo4j 5.x.
2. Install the Neo4j Graph Data Science plugin.
3. Start the Neo4j server.
4. Collect `NEO4J_URL`, `NEO4J_USER`, and `NEO4J_PASSWORD`.

Default local values are usually:

- `NEO4J_URL=neo4j://localhost:7687`
- `NEO4J_USER=neo4j`
- `NEO4J_PASSWORD=neo4j`

## Example

```python
import os

from nano_graphrag import GraphRAG
from nano_graphrag._storage import Neo4jStorage

neo4j_config = {
    "neo4j_url": os.environ.get("NEO4J_URL", "neo4j://localhost:7687"),
    "neo4j_auth": (
        os.environ.get("NEO4J_USER", "neo4j"),
        os.environ.get("NEO4J_PASSWORD", "neo4j"),
    ),
}

graph = GraphRAG(
    graph_storage_cls=Neo4jStorage,
    addon_params=neo4j_config,
)
```
