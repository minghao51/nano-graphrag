#!/usr/bin/env python3
"""Compare benchmark results across all datasets and modes."""

import json
from pathlib import Path
from typing import Any

ALL_MODES = ["naive", "local", "multihop", "global", "adaptive", "hipporag", "hybrid", "raptor"]
BASELINE_MODE = "naive"


def _result_files(results_dir: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in results_dir.rglob("*.json")
            if path.is_file() and path.parent != results_dir.parent
        ]
    )


def _dataset_label(file: Path, data: dict[str, Any]) -> str:
    parent = file.parent.name
    if parent != "results":
        return parent
    experiment_name = data.get("experiment_name", file.stem)
    return experiment_name.replace("_benchmark", "").replace("_openrouter", "")


def load_results(results_dir):
    latest_by_dataset = {}
    for file in _result_files(Path(results_dir)):
        with open(file, encoding="utf-8") as f:
            data = json.load(f)
        dataset = _dataset_label(file, data)
        existing = latest_by_dataset.get(dataset)
        if existing is None or file.stat().st_mtime > existing["path"].stat().st_mtime:
            latest_by_dataset[dataset] = {"path": file, "data": data}
    return latest_by_dataset


def print_comparison_table(results):
    if not results:
        print("No results found in ./results/")
        return

    print("\n" + "=" * 80)
    print("MULTI-HOP RAG BENCHMARK RESULTS")
    print("=" * 80)
    print()

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

        present_modes = [m for m in ALL_MODES if m in mode_results]

        print(f"\n{'Mode':<15} {'Exact Match':<15} {'Token F1':<15} {'Delta vs Naive'}")
        print(f"{'-' * 15} {'-' * 15} {'-' * 15} {'-' * 15}")

        baseline_f1 = mode_results.get(BASELINE_MODE, {}).get("token_f1", 0)

        for mode in present_modes:
            metrics = mode_results[mode]
            em = metrics.get("exact_match", 0)
            f1 = metrics.get("token_f1", 0)

            if mode == BASELINE_MODE:
                delta = "baseline"
            else:
                delta_f1 = f1 - baseline_f1
                delta = (
                    f"{'+' if delta_f1 > 0 else ''}{delta_f1:.3f} {'✓' if delta_f1 > 0 else '✗'}"
                )

            print(f"{mode:<15} {em:<15.3f} {f1:<15.3f} {delta}")

        duration = data.get("duration_seconds", 0)
        hours = int(duration // 3600)
        mins = int((duration % 3600) // 60)
        secs = int(duration % 60)
        print(f"\n  Duration: {hours}h {mins}m {secs}s")

        cache_stats = data.get("cache_stats")
        if cache_stats:
            hit_rate = cache_stats.get("hit_rate", 0) * 100
            print(f"  Cache hit rate: {hit_rate:.1f}%")

    print("\n" + "=" * 80)
    print("SUMMARY ACROSS ALL DATASETS")
    print("=" * 80)

    all_improvements = {}
    for _dataset, payload in results.items():
        data = payload["data"]
        mode_results = data.get("mode_results", {})
        baseline_f1 = mode_results.get(BASELINE_MODE, {}).get("token_f1")
        for mode in present_modes:
            if mode == BASELINE_MODE:
                continue
            mode_f1 = mode_results.get(mode, {}).get("token_f1")
            if baseline_f1 is not None and mode_f1 is not None:
                all_improvements.setdefault(mode, []).append(mode_f1 - baseline_f1)

    for mode, improvements in sorted(all_improvements.items()):
        if not improvements:
            continue
        avg = sum(improvements) / len(improvements)
        wins = sum(1 for x in improvements if x > 0)
        print(f"  {mode:<12} avg improvement: {avg:+.3f} F1  ({wins}/{len(improvements)} wins)")

    print("\n" + "=" * 80 + "\n")


def main():
    results_dir = Path("./results")
    if not results_dir.exists():
        print("No results directory found. Run benchmarks first.")
        return

    results = load_results(results_dir)
    print_comparison_table(results)


if __name__ == "__main__":
    main()
