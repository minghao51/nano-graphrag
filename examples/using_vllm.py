"""
vLLM / Custom API example using the new config system.

This example demonstrates using any OpenAI-compatible API (vLLM, custom endpoints, etc.):
- llm_model: Use "openai/<model-name>" format
- llm_api_base: Your API base URL
- embedding_model: Similarly use "openai/<model-name>"

Usage:
    # Start vLLM server
    vllm serve <model> --host 0.0.0.0 --port 8000

    # Or use any other OpenAI-compatible API
    # Examples: LM Studio, LocalAI, Text Generation Webui, etc.

    python examples/using_vllm.py
"""

import os
import logging

from nano_graphrag import GraphRAG, QueryParam

logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.INFO)

WORKING_DIR = "./nano_graphrag_cache_vllm"

# vLLM server configuration
VLLM_BASE_URL = "http://localhost:8000/v1"
LLM_MODEL = "meta-llama/Llama-3.2-1B-Instruct"  # or your deployed model
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # or your deployed embedding model


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

    # Using new config system - use "openai/" prefix for OpenAI-compatible APIs
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        enable_llm_cache=True,

        # LLM config - use OpenAI-compatible format
        llm_model=f"openai/{LLM_MODEL}",
        llm_api_base=VLLM_BASE_URL,

        # Embedding config
        embedding_model=f"openai/{EMBEDDING_MODEL}",
        embedding_api_base=VLLM_BASE_URL,
        embedding_dim=384,  # bge-small uses 384 dimensions

        # Compute settings for limited compute
        llm_max_async=4,
        extraction_max_async=4,
        embedding_func_max_async=4,

        # Note: Some local models may not support structured output
        structured_output=False,
    )

    start = time()
    rag.insert(FAKE_TEXT)
    print(f"Indexing time: {time() - start:.2f}s")


def query():
    rag = GraphRAG(
        working_dir=WORKING_DIR,

        # LLM config
        llm_model=f"openai/{LLM_MODEL}",
        llm_api_base=VLLM_BASE_URL,

        # Embedding config
        embedding_model=f"openai/{EMBEDDING_MODEL}",
        embedding_api_base=VLLM_BASE_URL,
        embedding_dim=384,

        structured_output=False,
    )

    print("\n=== Global Search ===")
    print(rag.query("What are the top themes in this story?", param=QueryParam(mode="global")))

    print("\n=== Local Search ===")
    print(rag.query("What are the top themes in this story?", param=QueryParam(mode="local")))


if __name__ == "__main__":
    insert()
    query()
