"""Compare benchmark experiment results."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class ComparisonResult:
    """Result of comparing two experiments."""

    baseline: str
    challenger: str
    deltas: Dict[str, Dict[str, Dict[str, float]]]


def load_result(result_path: str) -> Dict[str, Any]:
    """Load experiment result from JSON file."""
    with open(result_path, "r") as f:
        return json.load(f)


def compare_results(baseline_path: str, challenger_path: str) -> ComparisonResult:
    """Compare two experiment results and compute deltas.

    Args:
        baseline_path: Path to baseline experiment result JSON
        challenger_path: Path to challenger experiment result JSON

    Returns:
        ComparisonResult with deltas for each mode and metric
    """
    baseline = load_result(baseline_path)
    challenger = load_result(challenger_path)

    deltas = {}

    for mode in baseline["mode_results"]:
        if mode not in challenger["mode_results"]:
            continue

        baseline_scores = baseline["mode_results"][mode]
        challenger_scores = challenger["mode_results"][mode]

        deltas[mode] = {}

        for metric in baseline_scores:
            if metric not in challenger_scores:
                continue

            baseline_val = baseline_scores[metric]
            challenger_val = challenger_scores[metric]

            deltas[mode][metric] = {
                "baseline": baseline_val,
                "challenger": challenger_val,
                "delta": challenger_val - baseline_val,
            }

    return ComparisonResult(
        baseline=baseline_path,
        challenger=challenger_path,
        deltas=deltas,
    )


def print_diff_table(comparison: ComparisonResult) -> str:
    """Print comparison as markdown table.

    Returns:
        Markdown table string
    """
    lines = []
    lines.append("## Benchmark Comparison")
    lines.append("")
    lines.append(f"**Baseline:** `{comparison.baseline}`")
    lines.append(f"**Challenger:** `{comparison.challenger}`")
    lines.append("")
    lines.append("### Results")
    lines.append("")

    lines.append("| Mode | Metric | Baseline | Challenger | Delta |")
    lines.append("|------|--------|----------|-------------|-------|")

    for mode in comparison.deltas:
        for metric in comparison.deltas[mode]:
            data = comparison.deltas[mode][metric]
            delta = data["delta"]

            delta_str = f"{delta:+.3f}"
            if delta > 0:
                delta_str += " \u2713"
            elif delta < 0:
                delta_str += " \u2717"

            lines.append(
                f"| {mode} | {metric} | {data['baseline']:.3f} | "
                f"{data['challenger']:.3f} | {delta_str} |"
            )

    return "\n".join(lines)


def main():
    """CLI entry point for compare command."""
    parser = argparse.ArgumentParser(
        description="Compare two benchmark experiment results",
    )
    parser.add_argument(
        "baseline",
        help="Path to baseline experiment result JSON",
    )
    parser.add_argument(
        "challenger",
        help="Path to challenger experiment result JSON",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Save comparison to file instead of printing",
    )

    args = parser.parse_args()

    comparison = compare_results(args.baseline, args.challenger)

    output = print_diff_table(comparison)

    if args.output:
        Path(args.output).write_text(output)
        print(f"[Compare] Saved to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
