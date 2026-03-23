"""Tests for benchmark result comparison."""

import json
import tempfile
from pathlib import Path

import pytest

from bench.compare import compare_results, print_diff_table, ComparisonResult


def test_compare_two_experiments():
    """Compare should compute deltas between two experiment results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result1 = {
            "experiment_name": "exp1",
            "mode_results": {
                "local": {"exact_match": 0.5, "token_f1": 0.6}
            }
        }
        result2 = {
            "experiment_name": "exp2",
            "mode_results": {
                "local": {"exact_match": 0.6, "token_f1": 0.55}
            }
        }

        path1 = Path(tmpdir) / "result1.json"
        path2 = Path(tmpdir) / "result2.json"
        path1.write_text(json.dumps(result1))
        path2.write_text(json.dumps(result2))

        diff = compare_results(str(path1), str(path2))

        assert diff.deltas["local"]["exact_match"]["delta"] == pytest.approx(0.1)
        assert diff.deltas["local"]["token_f1"]["delta"] == pytest.approx(-0.05)


def test_compare_missing_mode():
    """Compare should skip modes present in only one experiment."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result1 = {
            "experiment_name": "exp1",
            "mode_results": {
                "local": {"exact_match": 0.5},
                "global": {"exact_match": 0.4},
            }
        }
        result2 = {
            "experiment_name": "exp2",
            "mode_results": {
                "local": {"exact_match": 0.6},
            }
        }

        path1 = Path(tmpdir) / "result1.json"
        path2 = Path(tmpdir) / "result2.json"
        path1.write_text(json.dumps(result1))
        path2.write_text(json.dumps(result2))

        diff = compare_results(str(path1), str(path2))

        assert "local" in diff.deltas
        assert "global" not in diff.deltas


def test_print_diff_table():
    """print_diff_table should generate markdown table."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result1 = {
            "experiment_name": "exp1",
            "mode_results": {
                "local": {"exact_match": 0.5}
            }
        }
        result2 = {
            "experiment_name": "exp2",
            "mode_results": {
                "local": {"exact_match": 0.6}
            }
        }

        path1 = Path(tmpdir) / "result1.json"
        path2 = Path(tmpdir) / "result2.json"
        path1.write_text(json.dumps(result1))
        path2.write_text(json.dumps(result2))

        comparison = compare_results(str(path1), str(path2))
        output = print_diff_table(comparison)

        assert "## Benchmark Comparison" in output
        assert "| Mode | Metric | Baseline | Challenger | Delta |" in output
        assert "local" in output
        assert "exact_match" in output
