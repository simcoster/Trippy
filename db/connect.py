"""Postgres connections that can be pointed at the `experiments` schema.

Unqualified table names follow `search_path`. Production is
`"$user", public, extensions`. Set `TRIPPY_SCHEMA=experiments` and every
`connect()` here uses `experiments, extensions` instead, so scrapes, search
and the planner write and read the copy. `public` is not on that path.

Callers that pass `options=` keep them; tests that already pin
`search_path=experiments` are unchanged.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import psycopg
from dotenv import load_dotenv

load_dotenv()

SCHEMA_ENV = "TRIPPY_SCHEMA"
EXPERIMENTS_SCHEMA = "experiments"
EXPERIMENTS_OPTIONS = "-csearch_path=experiments,extensions"

_announced = False


def database_url(config: dict | None = None) -> str:
    """`DATABASE_URL`, else `config['database_url']`. Host runs use localhost."""
    url = os.environ.get("DATABASE_URL") or (config or {}).get("database_url")
    if not url:
        raise RuntimeError("No database_url in config or DATABASE_URL env")
    return str(url).replace("@db:", "@localhost:")


def connect_options() -> str | None:
    """Libpq `options=` when `TRIPPY_SCHEMA=experiments`, else None (production)."""
    value = (os.environ.get(SCHEMA_ENV) or "").strip().casefold()
    if value == EXPERIMENTS_SCHEMA:
        return EXPERIMENTS_OPTIONS
    if value:
        raise RuntimeError(
            f"{SCHEMA_ENV}={value!r} is not supported; "
            f"use {EXPERIMENTS_SCHEMA!r} or unset"
        )
    return None


def _announce() -> None:
    global _announced
    if _announced:
        return
    _announced = True
    print(
        f"{SCHEMA_ENV}={EXPERIMENTS_SCHEMA} - unqualified tables are "
        "experiments, not public",
        file=sys.stderr,
    )


def connect(
    conninfo: str | None = None,
    *,
    config: dict | None = None,
    **kwargs: Any,
) -> psycopg.Connection:
    """`psycopg.connect` with `TRIPPY_SCHEMA` applied unless `options=` is set."""
    url = database_url(config) if conninfo is None else conninfo.replace(
        "@db:", "@localhost:"
    )
    extras = connect_options()
    if extras and "options" not in kwargs:
        kwargs["options"] = extras
        _announce()
    return psycopg.connect(url, **kwargs)
