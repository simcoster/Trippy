"""Fixtures shared by the database-backed tests.

Only one, and it is the important one: a connection that cannot see production.
See `.cursor/rules/no-test-data-in-prod.mdc` -- a test may do whatever it likes
in the `experiments` schema and must never touch a table in `public`.
"""

from __future__ import annotations

import psycopg
import pytest

from source.test.test_cases.experiments_schema import (
    SEARCH_PATH,
    SITE_ID,
    clone_tables,
    db_url,
)


@pytest.fixture
def experiments_conn():
    """A connection scoped to `experiments`, its tables freshly cloned and empty.

    `search_path` excludes `public`, so an unqualified table name in the test --
    or in the production code it calls -- resolves here or raises.
    """
    with psycopg.connect(db_url(), options=SEARCH_PATH) as conn:
        with conn.cursor() as cur:
            clone_tables(cur)
        conn.commit()
        yield conn
        conn.rollback()


@pytest.fixture
def experiments_site(experiments_conn):
    """`experiments_conn` with one campsite already in it. Returns (conn, id)."""
    with experiments_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO campsites (id, name, url) VALUES (%s, %s, %s)",
            (SITE_ID, "test campsite", "https://example.invalid/test"),
        )
    return experiments_conn, SITE_ID
