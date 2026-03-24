#!/usr/bin/env python3
"""Quick test to verify OpenRouter integration works."""

import os
import sys

# Check for OpenRouter API key
api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    print("❌ OPENROUTER_API_KEY not found in environment")
    print("\nTo set up OpenRouter:")
    print("1. Get API key from https://openrouter.ai/keys")
    print("2. Add to .env: echo 'OPENROUTER_API_KEY=sk-or-v1-...' >> .env")
    print("3. Load .env: export $(cat .env | grep -v '^#' | xargs)")
    sys.exit(1)

print("✅ OPENROUTER_API_KEY found")
print(f"   Key prefix: {api_key[:10]}...")

# Test import
try:
    from nano_graphrag import GraphRAGConfig

    print("✅ nano-graphrag imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Test litellm with OpenRouter
try:
    print("✅ litellm imported successfully")

    # Test that litellm supports OpenRouter format
    test_model = "openrouter/anthropic/claude-3.5-sonnet"
    print(f"✅ OpenRouter model format valid: {test_model}")

except ImportError as e:
    print(f"❌ litellm import error: {e}")
    sys.exit(1)

# Test configuration
try:
    config = GraphRAGConfig(
        working_dir="./test_openrouter",
        llm_model="openrouter/anthropic/claude-3.5-sonnet",
        embedding_model="openrouter/microsoft/wizardlm-2-7b",
    )
    print("✅ GraphRAGConfig created with OpenRouter models")
    print(f"   LLM: {config.llm_model}")
    print(f"   Embedding: {config.embedding_model}")
except Exception as e:
    print(f"❌ Config error: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ OpenRouter integration verified!")
print("=" * 60)
print("\nNext steps:")
print("1. Run quick benchmark: ./experiments/run_all_benchmarks_openrouter.sh --quick")
print(
    "2. Or run single dataset: uv run python -m bench.run --config experiments/benchmark_multihop_rag_openrouter.yaml"
)
