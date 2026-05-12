"""Repository consistency checks for runtime/version metadata."""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _extract_requires_python() -> str:
    with (ROOT / "pyproject.toml").open("rb") as f:
        data = tomllib.load(f)
    return data["project"]["requires-python"]


def _extract_min_python(requires_python: str) -> tuple[int, int]:
    match = re.search(r">=\s*([0-9]+)\.([0-9]+)", requires_python)
    if not match:
        raise ValueError(f"Unable to parse requires-python: {requires_python}")
    return int(match.group(1)), int(match.group(2))


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def main() -> int:
    errors = []

    requires_python = _extract_requires_python()
    min_major, min_minor = _extract_min_python(requires_python)
    min_python_short = f"{min_major}.{min_minor}"

    readme_text = _read_text(ROOT / "README.md")
    badge_match = re.search(r"python->=([0-9]+\.[0-9]+)", readme_text)
    if badge_match is None:
        errors.append("README Python badge not found")
    elif badge_match.group(1) != min_python_short:
        errors.append(
            f"README Python badge ({badge_match.group(1)}) != pyproject requires-python ({min_python_short})"
        )

    workflow_text = _read_text(ROOT / ".github" / "workflows" / "test.yml")
    matrix_versions = re.findall(r'"([0-9]+\.[0-9]+)"', workflow_text)
    for version in matrix_versions:
        try:
            major, minor = version.split(".")
            major_i = int(major)
            minor_i = int(minor)
        except ValueError:
            continue
        if (major_i, minor_i) < (min_major, min_minor):
            errors.append(
                f"CI matrix version {version} is below requires-python {min_python_short}"
            )

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print("Consistency checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
