"""Discovery seeds subcamp rows from config, and the schema keeps them honest.

A subcamp is a `campsites` row with a `parent_id` and no page of its own. That
shape is what lets `campsite_rules` stay unchanged (one campsite id per subcamp)
and what lets three of the four scrapers ignore subcamps entirely, by filtering
`WHERE url IS NOT NULL`.

Both halves are tested: the seeding, against a mocked cursor, and the
constraints, against the real database inside a transaction that is rolled back.
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import psycopg
import pytest
from dotenv import load_dotenv

from source.scraper import discover_sites

load_dotenv()

AKHZIV = {"id": 2, "name": "חניון לילה גן לאומי אכזיב", "url": "https://x/akhziv/"}
NORTH = {"heading": "חניון צפוני", "aliases": ["אכזיב צפון"]}
SOUTH = {"heading": "חניון דרומי", "aliases": ["אכזיב דרום"], "default_units": True}


# --- seeding -----------------------------------------------------------------


def mock_connection(rows=((99, "a"), (100, "b"))):
    """A psycopg connection whose cursor returns one RETURNING row per call."""
    cursor = MagicMock()
    cursor.fetchone.side_effect = list(rows) + [None] * 10
    cursor.__enter__ = lambda self: cursor
    cursor.__exit__ = lambda *a: False
    conn = MagicMock()
    conn.cursor.return_value = cursor
    conn.__enter__ = lambda self: conn
    conn.__exit__ = lambda *a: False
    return conn, cursor


def run_upsert(config, saved):
    conn, cursor = mock_connection()
    with (
        patch.object(discover_sites, "load_subcamp_config", return_value=config),
        patch.object(discover_sites.psycopg, "connect", return_value=conn),
    ):
        written = discover_sites.upsert_subcamps(saved)
    params = [c.args[1] for c in cursor.execute.call_args_list]
    return written, params


def test_each_configured_subcamp_becomes_a_child_row():
    written, params = run_upsert({AKHZIV["url"]: [NORTH, SOUTH]}, [AKHZIV])

    assert written == 2
    assert [p["parent_id"] for p in params] == [2, 2]
    assert [p["name"] for p in params] == [
        "חניון לילה גן לאומי אכזיב – חניון צפוני",
        "חניון לילה גן לאומי אכזיב – חניון דרומי",
    ]


def test_the_subcamp_json_is_stored_whole_so_no_name_lives_in_code():
    _written, params = run_upsert({AKHZIV["url"]: [SOUTH]}, [AKHZIV])

    stored = json.loads(params[0]["subcamp"])
    assert stored == SOUTH
    assert stored["aliases"] == ["אכזיב דרום"]
    assert stored["default_units"] is True


def test_a_config_url_matching_no_campsite_is_skipped_not_guessed():
    # A renamed page would otherwise attach subcamps to whatever sorted first.
    written, params = run_upsert({"https://x/gone/": [NORTH]}, [AKHZIV])

    assert written == 0
    assert params == []


def test_a_site_absent_from_the_config_gets_no_children():
    written, params = run_upsert({}, [AKHZIV])

    assert written == 0
    assert params == []


def test_the_shipped_config_names_exactly_one_default_units_subcamp():
    """Units naming no subcamp fall to one of them; two claimants is ambiguous."""
    config = discover_sites.load_subcamp_config()
    assert config, "expected a subcamps block in config.json"
    for url, areas in config.items():
        defaults = [a for a in areas if a.get("default_units")]
        assert len(defaults) == 1, f"{url}: {len(defaults)} default_units subcamps"
        for area in areas:
            assert area.get("heading"), f"{url}: a subcamp with no heading"


# --- schema ------------------------------------------------------------------


def _db_url() -> str:
    url = os.environ.get("DATABASE_URL")
    assert url, "DATABASE_URL is required"
    return url.replace("@db:", "@localhost:")


@pytest.fixture
def scratch():
    """A throwaway parent campsite; every test rolls back."""
    with psycopg.connect(_db_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO campsites (name, url) VALUES "
                "('test parent', 'https://example.invalid/test-parent/') RETURNING id"
            )
            yield cur.fetchone()[0], cur
        conn.rollback()


def add_child(cur, parent_id, name="test parent – north", **overrides):
    values = {
        "name": name,
        "parent_id": parent_id,
        "subcamp": json.dumps({"heading": "north"}),
        "url": None,
        **overrides,
    }
    cur.execute(
        "INSERT INTO campsites (name, parent_id, subcamp, url) "
        "VALUES (%(name)s, %(parent_id)s, %(subcamp)s::jsonb, %(url)s) RETURNING id",
        values,
    )
    return cur.fetchone()[0]


def test_a_subcamp_may_not_own_a_page(scratch):
    """`WHERE url IS NOT NULL` is how scrapers skip children — so it must hold."""
    parent_id, cur = scratch
    with pytest.raises(psycopg.errors.CheckViolation):
        add_child(cur, parent_id, url="https://example.invalid/child/")


def test_a_parent_link_without_a_subcamp_is_rejected(scratch):
    parent_id, cur = scratch
    with pytest.raises(psycopg.errors.CheckViolation):
        cur.execute(
            "INSERT INTO campsites (name, parent_id) VALUES ('half a subcamp', %s)",
            (parent_id,),
        )


def test_a_subcamp_without_a_parent_is_rejected(scratch):
    _parent_id, cur = scratch
    with pytest.raises(psycopg.errors.CheckViolation):
        cur.execute(
            "INSERT INTO campsites (name, url, subcamp) "
            "VALUES ('orphan', 'https://example.invalid/orphan/', '{}'::jsonb)"
        )


def test_many_subcamps_coexist_under_the_unique_url_index(scratch):
    """The whole shape rests on NULLs being distinct under UNIQUE(url)."""
    parent_id, cur = scratch
    add_child(cur, parent_id, name="test parent – north")
    add_child(cur, parent_id, name="test parent – south")
    cur.execute("SELECT count(*) FROM campsites WHERE parent_id = %s", (parent_id,))
    assert cur.fetchone()[0] == 2


def test_reseeding_the_same_subcamp_updates_it_rather_than_duplicating(scratch):
    """Discovery runs repeatedly; a second run must not fork the children."""
    parent_id, cur = scratch
    add_child(cur, parent_id)
    cur.execute(
        discover_sites.UPSERT_SUBCAMP_SQL,
        {
            "name": "test parent – north",
            "parent_id": parent_id,
            "subcamp": json.dumps({"heading": "north", "aliases": ["n"]}),
        },
    )
    cur.execute(
        "SELECT count(*), max(subcamp->'aliases'->>0) FROM campsites "
        "WHERE parent_id = %s",
        (parent_id,),
    )
    assert cur.fetchone() == (1, "n")


def test_deleting_a_parent_takes_its_subcamps_with_it(scratch):
    parent_id, cur = scratch
    add_child(cur, parent_id)
    cur.execute("DELETE FROM campsites WHERE id = %s", (parent_id,))
    cur.execute("SELECT count(*) FROM campsites WHERE parent_id = %s", (parent_id,))
    assert cur.fetchone()[0] == 0
