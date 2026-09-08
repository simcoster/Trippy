"""Test tables in the `experiments` schema, cloned from production's own DDL.

See `.cursor/rules/no-test-data-in-prod.mdc`: a test may create, delete,
truncate and generally do whatever it likes here and must never touch a
table in `public`. The connection is opened with
`options="-csearch_path=experiments,extensions"`.

DDL cloning lives in `db.experiments.clone_tables` so the same helper
fills a full production copy for scrape/planner experiments.
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

from db.experiments import SEARCH_PATH
from db.experiments import clone_tables as _clone_tables

load_dotenv()

__all__ = ["SEARCH_PATH", "SITE_ID", "TABLES", "clone_tables", "db_url"]

# Cloned in this order: a foreign key can only be replayed once both ends exist.
TABLES = (
    "campsites",
    "subject_vectors",
    "info_website_names",
    "accommodation_types",
    "campsite_rules",
    "conflict_cases",
    "list_prices",
)

# Far from any real campsite id, so a query that somehow escaped to `public`
# would match nothing rather than something.
SITE_ID = 900001


def db_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL is required")
    return url.replace("@db:", "@localhost:")


def clone_tables(cur, tables: tuple[str, ...] = TABLES) -> None:
    """Rebuild the fixture's tables in `experiments` from production's DDL, empty."""
    _clone_tables(cur, tables)
