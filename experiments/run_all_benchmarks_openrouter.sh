#!/bin/bash
# Run all 4 multi-hop benchmarks using OpenRouter (cost-effective)
# Usage: ./run_all_benchmarks_openrouter.sh [--quick]

set -e

# Check for OpenRouter API key
if [ -z "$OPENROUTER_API_KEY" ] && [ ! -f .env ]; then
    echo "❌ OPENROUTER_API_KEY not found!"
    echo ""
    echo "To use OpenRouter:"
    echo "1. Get API key from https://openrouter.ai/keys"
    echo "2. Add to .env file:"
    echo "   echo 'OPENROUTER_API_KEY=sk-or-v1-...' >> .env"
    echo ""
    exit 1
fi

# Load .env if exists
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo "✅ Loaded .env file"
fi

QUICK_MODE=""
if [ "$1" == "--quick" ]; then
  QUICK_MODE="--quick"
  echo "🚀 Running the checked-in OpenRouter quick configs"
  echo "💰 Using OpenRouter (cost-effective)"
else
  echo "🔄 Running the checked-in OpenRouter configs"
  echo "ℹ️  Sample sizes come from each YAML file. Edit max_samples/max_corpus_samples for full runs."
  echo "💰 Using OpenRouter (cost-effective)"
fi

echo "================================"
echo "Multi-Hop RAG Benchmark Suite"
echo "Provider: OpenRouter"
echo "================================"
echo ""

# Create results directory
mkdir -p ./results

# Track start time
START_TIME=$(date +%s)

# Run benchmarks
echo "📊 [1/4] MultiHop-RAG..."
uv run python -m bench.run --config experiments/benchmark_multihop_rag_openrouter.yaml

echo ""
echo "📊 [2/4] MuSiQue..."
# Use OpenRouter config for other datasets too
uv run python -m bench.run --config experiments/benchmark_musique_openrouter.yaml

echo ""
echo "📊 [3/4] HotpotQA..."
uv run python -m bench.run --config experiments/benchmark_hotpotqa_openrouter.yaml

echo ""
echo "📊 [4/4] 2WikiMultiHopQA..."
uv run python -m bench.run --config experiments/benchmark_2wiki_openrouter.yaml

# Calculate duration
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
echo "   ./experiments/compare_results.sh"
