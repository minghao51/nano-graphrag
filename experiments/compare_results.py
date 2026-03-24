#!/usr/bin/env python3
"""Compare benchmark results across all datasets and modes."""

import json
from pathlib import Path
from typing import Any


def _result_files(results_dir: Path) -> list[Path]:
    """Return saved benchmark result files, including dataset subdirectories."""
    return sorted(
        [
            path
            for path in results_dir.rglob("*.json")
            if path.is_file() and path.parent != results_dir.parent
        ]
    )


def _dataset_label(file: Path, data: dict[str, Any]) -> str:
    """Infer a stable dataset label from the output directory or experiment name."""
    parent = file.parent.name
    if parent != "results":
        return parent
    experiment_name = data.get("experiment_name", file.stem)
    return experiment_name.replace("_benchmark", "").replace("_openrouter", "")


def load_results(results_dir):
    """Load the latest benchmark result for each dataset directory."""
    latest_by_dataset = {}
    for file in _result_files(Path(results_dir)):
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
        dataset = _dataset_label(file, data)
        existing = latest_by_dataset.get(dataset)
        if existing is None or file.stat().st_mtime > existing["path"].stat().st_mtime:
            latest_by_dataset[dataset] = {"path": file, "data": data}
    return latest_by_dataset


def print_comparison_table(results):
    """Print comparison table of all results."""
    if not results:
        print("No results found in ./results/")
        return

    print("\n" + "=" * 80)
    print("MULTI-HOP RAG BENCHMARK RESULTS")
    print("=" * 80)
    print()

    # Print results for each dataset
    for dataset, payload in sorted(results.items()):
        data = payload["data"]
        print(f"\n{'-' * 80}")
        print(f"DATASET: {dataset}")
        print(f"{'─' * 80}")
        print(f"Source: {payload['path']}")

        mode_results = data.get("mode_results", {})
        if not mode_results:
            print("  No mode results available")
            continue

        # Print table header
        print(f"\n{'Mode':<15} {'Exact Match':<15} {'Token F1':<15} {'Delta vs Naive'}")
        print(f"{'-' * 15} {'-' * 15} {'-' * 15} {'-' * 15}")

        # Get naive baseline for comparison
        naive_f1 = mode_results.get("naive", {}).get("token_f1", 0)

        # Print each mode
        for mode in ["naive", "local", "multihop"]:
            if mode not in mode_results:
                continue

            metrics = mode_results[mode]
            em = metrics.get("exact_match", 0)
            f1 = metrics.get("token_f1", 0)

            # Calculate delta
            if mode == "naive":
                delta = "baseline"
            else:
                delta_f1 = f1 - naive_f1
                delta = (
                    f"{'+' if delta_f1 > 0 else ''}{delta_f1:.3f} {'✓' if delta_f1 > 0 else '✗'}"
                )

            print(f"{mode:<15} {em:<15.3f} {f1:<15.3f} {delta}")

        # Print timing info
        duration = data.get("duration_seconds", 0)
        hours = int(duration // 3600)
        mins = int((duration % 3600) // 60)
        secs = int(duration % 60)
        print(f"\n⏱️  Duration: {hours}h {mins}m {secs}s")

        # Print cache stats if available
        cache_stats = data.get("cache_stats")
        if cache_stats:
            hit_rate = cache_stats.get("hit_rate", 0) * 100
            print(f"💾 Cache hit rate: {hit_rate:.1f}%")

    # Print overall summary
    print("\n" + "=" * 80)
    print("SUMMARY ACROSS ALL DATASETS")
    print("=" * 80)

    # Calculate average improvement of multihop vs naive
    multihop_improvements = []
    for dataset, payload in results.items():
        data = payload["data"]
        mode_results = data.get("mode_results", {})
        naive_f1 = mode_results.get("naive", {}).get("token_f1")
        multihop_f1 = mode_results.get("multihop", {}).get("token_f1")
        if naive_f1 is not None and multihop_f1 is not None:
            improvement = multihop_f1 - naive_f1
            multihop_improvements.append(improvement)

    if multihop_improvements:
        avg_improvement = sum(multihop_improvements) / len(multihop_improvements)
        print(f"\n📈 Average multihop improvement over naive: {avg_improvement:+.3f} F1")

        wins = sum(1 for x in multihop_improvements if x > 0)
        print(f"🏆 Multihop wins on {wins}/{len(multihop_improvements)} datasets")

    print("\n" + "=" * 80 + "\n")


def main():
    results_dir = Path("./results")
    if not results_dir.exists():
        print("❌ No results directory found. Run benchmarks first.")
        return

    results = load_results(results_dir)
    print_comparison_table(results)


if __name__ == "__main__":
    main()
