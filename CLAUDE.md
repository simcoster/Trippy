# Trippy — rules for Claude

Repo-wide rules live in `.cursor/rules/` so Cursor and Claude Code share one
source. Each file is imported here; edit the rule there, not in this file.

@.cursor/rules/llm-decides-semantics.mdc
@.cursor/rules/plan-is-an-append-only-log.mdc
@.cursor/rules/log-experiments-and-decisions.mdc
@.cursor/rules/no-incidental-reformatting.mdc
@.cursor/rules/existing-tests-permission.mdc
@.cursor/rules/python-imports.mdc
@.cursor/rules/ruff.mdc
@.cursor/rules/agent-temp-files.mdc

## Orientation

- `just --list` is the command surface. Data-loading recipes are all
  `scrape-<thing>`; `update-tables` applies Alembic migrations.
- Local Postgres (pgvector) runs via `docker compose up -d`; `.env` holds
  secrets and is git-ignored. `/app/.venv` in the `api` container is a named
  volume, so after a dependency change run `docker compose down`,
  `docker volume rm trippy_trippy_venv`, then `docker compose up -d --build`.
  Never `docker compose down -v` — it destroys the database volume too.
- `pytest -m "not llm"` is the no-token test run; `llm`-marked tests call
  Nebius.
- Every `scrape-*` run prints its LLM cost by role and model and appends one
  JSON line to `reports/scrape_costs.jsonl`. A new LLM call site must pass
  `role=` and `model=` to `LlmUsage.add_chat` / `add_embed`, or the report
  misprices it.
