# Project conventions

> Single source of truth for AI coding agents. `CLAUDE.md` points here.

## Project

Product Insight Agent — a multi-tenant agentic RAG system built as a uv workspace
with namespace packages under `poc.*`.

## Key conventions

- Python 3.12+, uv workspace, namespace package `poc` (no `__init__.py` in `src/poc/`)
- Ruff for linting (line length 100, target py312)
- Pytest with asyncio_mode = "auto"
- One feature per branch; conventional commit messages
- English only — code, docs, comments, commit messages

## Commands

```bash
uv sync                  # install all deps
uv run pytest            # run tests
uv run ruff check .      # lint
uv run ruff format .     # format
```

## Package layout

Each package lives at `packages/<name>/src/poc/<name>/`.
The API app lives at `apps/api/src/api/`.

## Do NOT

- Add `__init__.py` to `src/poc/` (breaks namespace package)
- Commit `.env` or any secrets
- Use `print()` for logging — use `structlog`
