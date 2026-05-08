#!/bin/bash
# Compare benchmark results
# Usage: dotenvx run -- ./experiments/scripts/compare_results.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
uv run python "$SCRIPT_DIR/compare_results.py" "$@"
