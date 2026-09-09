---
name: experiments-schema
description: >-
  Copy production Postgres into the experiments schema, empty selected
  tables, and run scrapes or the planner against that copy via
  TRIPPY_SCHEMA=experiments. Use when the user wants an isolated scrape,
  planner replay, or any write that must not touch public.
---

# Experiments schema

Do not write one-off `temp/` scripts that open their own connection and
clone a handful of tables. Use this harness.

## Setup

```text
just setup-experiments copy
just setup-experiments copy --skip availability
just setup-experiments copy --empty campsite_rules,subject_vectors,conflict_cases
just setup-experiments empty campsite_rules
just setup-experiments freeze-availability
just setup-experiments status
```

`copy` rebuilds `experiments` from `public` (DDL + rows + views). No FK
crosses into `public`. `--empty` is `TRUNCATE … CASCADE` after the copy.
`--skip availability` clones that table empty so FKs survive, and does
not copy live occupancy. `freeze-availability` snapshots
`public.availability` into `experiments.availability_frozen` for the
planner benchmark (`evals/planner_v1.json`). `copy` does not drop the
frozen table. `just run-eval` runs this copy (skip availability) first.

## Run production code against the copy

```text
just on-experiments scrape-info -- --site 2
just on-experiments scrape-rules -- --site 2
just run-eval
```

`TRIPPY_SCHEMA=experiments` makes `db.connect.connect` set
`search_path=experiments,extensions`. Scrapes, search, and the planner
already go through that helper. Unset the env var to hit production again.

A connection that passes `options=` keeps them (pytest fixtures).

Pytest's `experiments_conn` fixture still drops and reclones an
**empty** subset of tables. Do not run those tests between `copy` and
the scrape you are measuring.

## Do not

- `INSERT`/`UPDATE`/`DELETE` `public.*`
- Seed by `INSERT INTO experiments.x SELECT * FROM public.x` in a new
  script — `copy` already does that
- Put `public` on the experiments `search_path`
