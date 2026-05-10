"""Generate MkDocs stubs from Quarto .qmd notebooks.

Reads frontmatter from notebooks/*.qmd and generates:
  - docs/notebooks/<slug>.md  (iframe embed stub)
  - docs/notebooks/index.md   (notebook listing)

Usage:
    uv run python scripts/generate_notebook_docs.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = ROOT / "notebooks"
DOCS_NB_DIR = ROOT / "docs" / "notebooks"
MKDOCS_YML = ROOT / "mkdocs.yml"


def get_base_path() -> str:
    try:
        with open(MKDOCS_YML) as f:
            cfg = yaml.safe_load(f)
        site_url = cfg.get("site_url", "").rstrip("/")
        if not site_url:
            return ""
        from urllib.parse import urlparse

        parsed = urlparse(site_url)
        path = parsed.path.strip("/")
        return f"/{path}" if path else ""
    except Exception:
        return ""


def parse_frontmatter(qmd_path: Path) -> dict:
    text = qmd_path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return {}
    end = text.find("---", 3)
    if end == -1:
        return {}
    fm = yaml.safe_load(text[3:end])
    return fm if isinstance(fm, dict) else {}


def slugify(title: str) -> str:
    slug = title.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    return slug


def generate_stub(qmd_path: Path, base_path: str) -> tuple[str, str]:
    fm = parse_frontmatter(qmd_path)
    title = fm.get("title", qmd_path.stem)
    description = fm.get("description", "")
    html_name = qmd_path.stem + ".html"
    slug = slugify(title)
    iframe_src = f"{base_path}" + f"/notebooks/html/{html_name}"

    stub = f"""---
hide:
  - navigation
  - toc
---

# {title}

{description}

<div class="iframe-container" id="iframe-wrapper-{slug}">
  <div class="iframe-controls">
    <button onclick="toggleNotebookFullscreen(this)" class="md-button">Expand</button>
    <a href="{iframe_src}" target="_blank" class="md-button">Open in New Tab</a>
  </div>
  <iframe src="{iframe_src}" allowfullscreen loading="lazy"></iframe>
</div>

## Run Locally

```bash
dotenvx run -- uv run quarto render notebooks/{qmd_path.name}
```
"""
    return slug, stub


def generate_index(notebooks: list[tuple[str, str, str]]) -> str:
    rows = []
    for slug, title, description in notebooks:
        rows.append(f"| [{title}]({slug}.md) | {description} |")
    table = "\n".join(rows)
    return f"""# Notebooks

Interactive tutorials demonstrating nano-graphrag features.

| Notebook | Description |
|----------|-------------|
{table}

## Running Locally

```bash
uv sync --extra docs
uv run quarto render notebooks/
```
"""


def main():
    base_path = get_base_path()
    print(f"Base path: {base_path!r}")

    qmd_files = sorted(NOTEBOOKS_DIR.glob("*.qmd"))
    if not qmd_files:
        print("No .qmd files found in notebooks/")
        sys.exit(1)

    notebooks_meta = []
    for qmd_path in qmd_files:
        slug, stub = generate_stub(qmd_path, base_path)
        stub_path = DOCS_NB_DIR / f"{slug}.md"
        stub_path.write_text(stub, encoding="utf-8")
        print(f"  Generated: {stub_path}")

        fm = parse_frontmatter(qmd_path)
        title = fm.get("title", qmd_path.stem)
        description = fm.get("description", "")
        notebooks_meta.append((slug, title, description))

    index_path = DOCS_NB_DIR / "index.md"
    index_path.write_text(generate_index(notebooks_meta), encoding="utf-8")
    print(f"  Generated: {index_path}")

    print(f"\nDone. Generated {len(qmd_files)} stubs + index.")


if __name__ == "__main__":
    main()
