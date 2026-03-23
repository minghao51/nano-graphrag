"""CLI for running GraphRAG benchmark experiments."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from examples.benchmarks.run_experiment import main

if __name__ == "__main__":
    sys.exit(main())
