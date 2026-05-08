#!/bin/bash
# Run all 4 multi-hop benchmarks using OpenRouter (cost-effective)
# Usage: dotenvx run -- ./experiments/scripts/run_all_benchmarks_openrouter.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_DIR="$SCRIPT_DIR/../configs/core"

if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "❌ OPENROUTER_API_KEY not found!"
    echo ""
    echo "To use OpenRouter:"
    echo "1. Get API key from https://openrouter.ai/keys"
    echo "2. Add to .env file:"
    echo "   dotenvx set OPENROUTER_API_KEY 'sk-or-v1-...'"
    echo "3. Run with: dotenvx run -- $0"
    echo ""
    exit 1
fi

echo "================================"
echo "Multi-Hop RAG Benchmark Suite"
echo "Provider: OpenRouter"
echo "================================"
echo ""

mkdir -p ./results

START_TIME=$(date +%s)

echo "📊 [1/4] MultiHop-RAG..."
uv run python -m bench --config "$CONFIG_DIR/benchmark_multihop_rag_openrouter.yaml"

echo ""
echo "📊 [2/4] MuSiQue..."
uv run python -m bench --config "$CONFIG_DIR/benchmark_musique_openrouter.yaml"

echo ""
echo "📊 [3/4] HotpotQA..."
uv run python -m bench --config "$CONFIG_DIR/benchmark_hotpotqa_openrouter.yaml"

echo ""
echo "📊 [4/4] 2WikiMultiHopQA..."
uv run python -m bench --config "$CONFIG_DIR/benchmark_2wiki_openrouter.yaml"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "================================"
echo "✅ All benchmarks complete!"
echo "================================"
echo "Provider: OpenRouter"
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo "💰 Estimated cost: ~$0.50-2.00 (quick test)"
echo "📁 Results saved to: ./results/"
echo ""
echo "📈 To compare results, run:"
echo "   dotenvx run -- uv run python experiments/scripts/compare_results.py"
