# Contributing to nano-graphrag

## Pull Requests

To contribute:

1. Fork and clone the repository.
2. Make the smallest change that solves the problem cleanly.
3. Add or update tests if you modify core behavior in `nano_graphrag/`.
4. Update docs, examples, or docstrings when behavior changes.
5. Run `uv run pytest` before opening the PR.
6. Submit the pull request.

For GitHub-specific PR mechanics, see the [GitHub pull request guide](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request).

## Dependency Policy

`nano-graphrag` should stay small and lightweight. Add a dependency only when it is clearly necessary and the value is worth the maintenance cost.
