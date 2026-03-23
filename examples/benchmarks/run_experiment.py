#!/usr/bin/env python3
"""Simple CLI for running GraphRAG benchmark experiments."""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from bench import BenchmarkConfig, ExperimentRunner


def main():
    parser = argparse.ArgumentParser(
        description="Run GraphRAG benchmark experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with config file
  python run_experiment.py --config configs/example.yaml

  # Override experiment name
  python run_experiment.py --config configs/example.yaml --name my_experiment

  # Limit to 10 samples
  python run_experiment.py --config configs/example.yaml --max-samples 10

  # Only run local mode
  python run_experiment.py --config configs/example.yaml --modes local
        """,
    )

    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to benchmark config YAML file",
    )

    parser.add_argument(
        "--name",
        "-n",
        help="Override experiment name",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        help="Limit number of samples to process",
    )

    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["local", "global", "naive"],
        help="Query modes to run (default: from config)",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        help="Override output directory",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config without running experiment",
    )

    args = parser.parse_args()

    # Load config
    print(f"[Config] Loading from {args.config}")
    config = BenchmarkConfig.from_yaml(args.config)

    # Apply overrides
    if args.name:
        config.experiment_name = args.name
    if args.max_samples is not None:
        config.max_samples = args.max_samples
    if args.modes:
        config.query_modes = args.modes
    if args.output_dir:
        config.output_dir = args.output_dir

    print(f"[Config] Experiment: {config.experiment_name}")
    print(f"[Config] Dataset: {config.dataset_name}")
    print(f"[Config] Max samples: {config.max_samples}")
    print(f"[Config] Query modes: {config.query_modes}")
    print(f"[Config] Output dir: {config.output_dir}")

    if args.dry_run:
        print("\n[Config] Configuration is valid (dry run)")
        return

    # Run experiment
    print("\n[Starting] Running benchmark experiment...")
    runner = ExperimentRunner(config)

    try:
        result = runner.run_sync()

        # Print summary
        print("\n" + "=" * 50)
        print("RESULTS SUMMARY")
        print("=" * 50)
        print(f"Experiment: {result.experiment_name}")
        print(f"Duration: {result.duration_seconds:.2f} seconds")
        print()

        for mode, scores in result.mode_results.items():
            print(f"{mode.upper()} Mode:")
            for metric, score in scores.items():
                print(f"  {metric}: {score:.4f}")

        print("\n" + "=" * 50)

    except Exception as e:
        print(f"\n[Error] Experiment failed: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
