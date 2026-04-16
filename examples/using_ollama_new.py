"""
Ollama example using the new config system.

This example demonstrates using Ollama for both LLM and embedding with the new simplified config:
- llm_model: Ollama model name (e.g., "ollama/llama3.2")
- llm_api_base: Ollama server URL
- embedding_model: Ollama embedding model
- embedding_api_base: Same as llm_api_base

Usage:
    # Start Ollama first
    ollama serve
    ollama pull llama3.2
    ollama pull nomic-embed-text

    # Run this example
    python examples/using_ollama_new.py
"""

import os
import logging

from nano_graphrag import GraphRAG, QueryParam

logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.INFO)

WORKING_DIR = "./nano_graphrag_cache_ollama"


def remove_if_exist(file):
    if os.path.exists(file):
        os.remove(file)


def insert():
    from time import time

    with open("./tests/mock_data.txt", encoding="utf-8-sig") as f:
        FAKE_TEXT = f.read()

    # Clean up previous data
    remove_if_exist(f"{WORKING_DIR}/vdb_entities.json")
    remove_if_exist(f"{WORKING_DIR}/kv_store_full_docs.json")
    remove_if_exist(f"{WORKING_DIR}/kv_store_text_chunks.json")
    remove_if_exist(f"{WORKING_DIR}/kv_store_community_reports.json")
    remove_if_exist(f"{WORKING_DIR}/graph_chunk_entity_relation.graphml")

    # Using new config system - simple and clean!
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        enable_llm_cache=True,

        # LLM config - use Ollama
        llm_model="ollama/llama3.2",
        llm_api_base="http://localhost:11434",

        # Embedding config - use Ollama
        embedding_model="ollama/nomic-embed-text",
        embedding_api_base="http://localhost:11434",
        embedding_dim=768,  # nomic-embed-text uses 768 dimensions

        # Compute settings for local/Limited compute
        llm_max_async=4,
        extraction_max_async=4,
        embedding_func_max_async=4,
    )

    start = time()
    rag.insert(FAKE_TEXT)
    print(f"Indexing time: {time() - start:.2f}s")


def query():
    # Same config for querying
    rag = GraphRAG(
        working_dir=WORKING_DIR,

        # LLM config
        llm_model="ollama/llama3.2",
        llm_api_base="http://localhost:11434",

        # Embedding config
        embedding_model="ollama/nomic-embed-text",
        embedding_api_base="http://localhost:11434",
        embedding_dim=768,
    )

    print("\n=== Global Search ===")
    print(rag.query("What are the top themes in this story?", param=QueryParam(mode="global")))

    print("\n=== Local Search ===")
    print(rag.query("What are the top themes in this story?", param=QueryParam(mode="local")))


if __name__ == "__main__":
    insert()
    query()
