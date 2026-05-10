"""
Configuration — Configure for your environment
================================================
Objective: Learn the three main config paths (kwargs, YAML, env vars)
and see a practical example of switching to local Ollama models.

Run:  dotenvx run -- uv run notebooks/03_configuration.py
"""

import logging
from pathlib import Path

logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.WARNING)

WORKING_DIR = Path("./_cache/03_configuration")
WORKING_DIR.mkdir(parents=True, exist_ok=True)

# %% Imports
from nano_graphrag import GraphRAG
from nano_graphrag.base import GraphRAGConfig

# %% Method 1: Direct kwargs — simplest, best for scripts and notebooks
rag = GraphRAG(
    working_dir=str(WORKING_DIR / "kwargs"),
    enable_llm_cache=True,
    chunk_token_size=1200,
    chunk_overlap_token_size=100,
)
print(f"LLM model:      {rag.llm_model}")
print(f"Embedding dim:  {rag.embedding_dim}")
print(f"Chunk size:     {rag.chunk_token_size}")

# %% Method 2: YAML file — best for team sharing and reproducibility
yaml_content = """
working_dir: ./_cache/03_configuration/yaml
llm:
  model: openrouter/google/gemma-4-31b-it
  max_async: 16
embedding:
  model: openrouter/qwen/qwen3-embedding-8b
  dim: 4096
features:
  local_search: true
  llm_cache: true
"""

yaml_path = WORKING_DIR / "settings.yaml"
yaml_path.write_text(yaml_content)

config = GraphRAGConfig.from_yaml(str(yaml_path))
rag = GraphRAG.from_config(config)
print(f"LLM model:      {rag.llm_model}")
print(f"Embedding dim:  {rag.embedding_dim}")
print(f"Local search:   {rag.enable_local}")

# %% Method 3: Environment variables — best for Docker, CI/CD
# Set via dotenvx: GRAPH_API_KEY, LLM_MODEL, EMBEDDING_MODEL, etc.
config = GraphRAGConfig.from_env()
print(f"LLM model:      {config.llm_model}")
print(f"Embedding model: {config.embedding_model}")
print(f"Working dir:     {config.working_dir}")

# %% Practical example: switch to local Ollama models
# No API key needed — runs everything locally
ollama_config = GraphRAGConfig(
    working_dir=str(WORKING_DIR / "ollama"),
    llm_model="ollama/llama3.2",
    llm_api_base="http://localhost:11434",
    embedding_model="ollama/nomic-embed-text",
    embedding_api_base="http://localhost:11434",
    embedding_dim=768,
)
print("\nOllama config:")
print(f"  LLM:       {ollama_config.llm_model}")
print(f"  Embedding: {ollama_config.embedding_model}")
print(f"  API base:  {ollama_config.llm_api_base}")
print(f"  Dim:       {ollama_config.embedding_dim}")

# Uncomment to use:
# rag = GraphRAG.from_config(ollama_config)
# await rag.ainsert("Your text here")

# %% Key config parameters reference
print("""
Common parameters:
  working_dir              — where all data is stored (default: ./nano_graphrag)
  llm_model                — LLM for extraction + queries (default: from env)
  llm_cheap_model          — cheaper LLM for fast mode
  embedding_model          — embedding model name
  embedding_dim            — embedding vector dimension
  chunk_token_size         — tokens per chunk (default: 1200)
  chunk_overlap_token_size — overlap between chunks (default: 100)
  enable_local             — enable local query mode (default: True)
  enable_naive_rag         — enable naive RAG mode (default: False)
  enable_llm_cache         — cache LLM responses (default: True)
  enable_entity_linking    — merge duplicate entities (default: False)
  extraction_batch_size    — chunks per LLM call (default: 5)
  entity_extraction_quality — 'balanced' or 'fast' (default: balanced)
""")
