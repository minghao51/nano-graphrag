#!/usr/bin/env python3
"""Test that YOUR .env configuration works."""

import os
import sys

# Load .env file
from pathlib import Path

env_file = Path(".env")
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key] = value
    print("✅ Loaded .env file")
else:
    print("❌ No .env file found")
    sys.exit(1)

# Check what models you configured
llm_model = os.environ.get("LLM_MODEL", "not set")
embedding_model = os.environ.get("EMBEDDING_MODEL", "not set")

print("\n📋 Your configuration:")
print(f"   LLM: {llm_model}")
print(f"   Embedding: {embedding_model}")

# Test imports
try:
    from nano_graphrag import GraphRAGConfig

    print("\n✅ nano-graphrag imported")
except ImportError as e:
    print(f"\n❌ Import error: {e}")
    sys.exit(1)

# Test configuration
try:
    config = GraphRAGConfig(
        working_dir="./test_env_setup",
        llm_model=llm_model,
        embedding_model=embedding_model,
    )
    print("\n✅ GraphRAGConfig created successfully")
    print(f"   Working dir: {config.working_dir}")
    print(f"   LLM model: {config.llm_model}")
    print(f"   Embedding model: {config.embedding_model}")
except Exception as e:
    print(f"\n❌ Config error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ Your .env configuration is valid!")
print("=" * 60)
print("\nYou can now run benchmarks:")
print("  ./experiments/run_all_benchmarks.sh --quick")
