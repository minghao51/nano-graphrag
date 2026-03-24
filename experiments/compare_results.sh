#!/bin/bash
# Compare benchmark results

# Load .env file if it exists
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

uv run python experiments/compare_results.py "$@"
