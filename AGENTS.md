# Project Guidelines

## Core Principles

1. **Think before acting** — Read the codebase and understand context before making changes.
2. **Verify plans** — Check in before major changes; get user approval first.
3. **Keep it simple** — Every change should be minimal and impact as little code as possible. Avoid complexity.
4. **Stay grounded** — Never speculate about code you haven't read. Investigate before answering.
5. **Document progress** — Provide high-level explanations at each step.

## Code Investigation Rules

- **Read before claiming** — If a user references a file, read it first. Never make claims about unopened code.
- **No hallucinations** — Give grounded answers only. If uncertain, say so and investigate.
- **package documentation** - Always use Context7 MCP when I need library/API documentation, code generation, setup or configuration steps without me having to explicitly ask.

## Environment & Tools

| Tool | Usage |
|------|-------|
| **Python** | Use `uv` for all package management. Always run `uv run <command>` — never plain `python` or `python3`. |
| **Docker** | Use `docker compose build` and `docker compose up` instead of npm/bun for test/exploration. |
| **Playwright MCP** | Use for testing app/frontend/features when required. |

### Common uv Commands
```bash
uv sync              # Install/sync dependencies
uv run <command>     # Execute within managed environment (e.g., uv run pytest)
uv add <package>     # Add dependency to pyproject.toml
```

## File Naming Convention

Markdown files must follow: **`YYYYMMDD-filename.md`**

## Documentation

Maintain an architecture documentation file that describes how the app works inside and out.
