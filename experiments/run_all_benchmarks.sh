#!/bin/bash
# Run all 4 multi-hop benchmarks sequentially
# Usage: ./run_all_benchmarks.sh [--quick]

set -e

# Load .env file if it exists
if [ -f .env ]; then
  export $(cat .env | grep -v '^#' | xargs)
  echo "✅ Loaded .env file"
fi

QUICK_MODE=""
if [ "$1" == "--quick" ]; then
  QUICK_MODE="--quick"
  echo "🚀 Running the checked-in quick configs"
else
  echo "🔄 Running the checked-in configs"
  echo "ℹ️  Sample sizes come from each YAML file. Edit max_samples/max_corpus_samples for full runs."
fi

echo "================================"
echo "Multi-Hop RAG Benchmark Suite"
echo "================================"
echo ""

# Create results directory
mkdir -p ./results

# Track start time
START_TIME=$(date +%s)

# Run benchmarks
echo "📊 [1/4] MultiHop-RAG..."
uv run python -m bench.run --config experiments/benchmark_multihop_rag.yaml

echo "📊 [2/4] MuSiQue..."
uv run python -m bench.run --config experiments/benchmark_musique.yaml

echo "📊 [3/4] HotpotQA..."
uv run python -m bench.run --config experiments/benchmark_hotpotqa.yaml

echo "📊 [4/4] 2WikiMultiHopQA..."
uv run python -m bench.run --config experiments/benchmark_2wiki.yaml

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
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo "📁 Results saved to: ./results/"
echo ""
echo "📈 To compare results, run:"
echo "   uv run python experiments/compare_results.py"
