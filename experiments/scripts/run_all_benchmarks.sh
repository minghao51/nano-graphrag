#!/bin/bash
# Run all 4 multi-hop benchmarks sequentially
# Usage: dotenvx run -- ./experiments/scripts/run_all_benchmarks.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_DIR="$SCRIPT_DIR/../configs/core"

echo "================================"
echo "Multi-Hop RAG Benchmark Suite"
echo "================================"
echo ""

mkdir -p ./results

START_TIME=$(date +%s)

echo "📊 [1/4] MultiHop-RAG..."
uv run python -m bench --config "$CONFIG_DIR/benchmark_multihop_rag.yaml"

echo "📊 [2/4] MuSiQue..."
uv run python -m bench --config "$CONFIG_DIR/benchmark_musique.yaml"

echo "📊 [3/4] HotpotQA..."
uv run python -m bench --config "$CONFIG_DIR/benchmark_hotpotqa.yaml"

echo "📊 [4/4] 2WikiMultiHopQA..."
uv run python -m bench --config "$CONFIG_DIR/benchmark_2wiki.yaml"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "================================"
echo "✅ All benchmarks complete!"
echo "================================"
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo "📁 Results saved to: ./results/"
echo ""
echo "📈 To compare results, run:"
echo "   dotenvx run -- uv run python experiments/scripts/compare_results.py"
