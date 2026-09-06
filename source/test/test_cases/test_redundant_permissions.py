"""`X` and `X_allowed` from one sentence are one fact; the amenity keeps it.

`ניתן להזמין מזרנים בתוספת תשלום` yields `mattress_rental` (amenity, true) and
`mattress_rental_allowed` (boolean_rule, true) from the same words. Measured
over 18 sites: 5 rows of 438, in exactly two subjects.

The deletion is deterministic, so it runs before the conflict resolver asks the
model anything. What it must NOT touch is the pair the extractor prompt
deliberately produces -- `barbecue_allowed` true AND `barbecue_equipment` false
are two different facts, and their polarities differing is what says so.

Every test rolls back; the dev database is left untouched.
"""

from __future__ import annotations

import os

import psycopg
import pytest
from dotenv import load_dotenv

from source.scraper.rules_ingest.db import ResolvedRule
from source.scraper.rules_ingest.resolve_conflicts import (
    drop_redundant_permissions,
    redundant_permissions,
)

load_dotenv()


def _db_url() -> str:
    url = os.environ.get("DATABASE_URL")
    assert url, "DATABASE_URL is required"
    return url.replace("@db:", "@localhost:")


@pytest.fixture
def conn():
    with psycopg.connect(_db_url()) as connection:
        yield connection
        connection.rollback()


@pytest.fixture
def scope(conn):
    """A campsite, and a factory for throwaway subject + rule pairs."""
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM campsites ORDER BY id LIMIT 1")
        row = cur.fetchone()
        if row is None:
            pytest.skip("no campsites in the database")
        campsite_id = row[0]

        def add(name: str, category: int, polarity: bool | None, qualifier=None) -> int:
            cur.execute(
                "INSERT INTO subject_vectors (name, category, aliases) "
                "VALUES (%s, %s, ARRAY[%s]) RETURNING id",
                (name, category, name),
            )
            subject_id = cur.fetchone()[0]
            cur.execute(
                "INSERT INTO campsite_rules "
                "(campsite_id, subject_id, polarity, qualifier) VALUES (%s,%s,%s,%s)",
                (campsite_id, subject_id, polarity, qualifier),
            )
            return subject_id

        yield campsite_id, add, cur


def surviving(cur, campsite_id: int, names: list[str]) -> set[str]:
    cur.execute(
        "SELECT s.name FROM campsite_rules r JOIN subject_vectors s ON s.id = r.subject_id "
        "WHERE r.campsite_id = %s AND s.name = ANY(%s)",
        (campsite_id, names),
    )
    return {r[0] for r in cur.fetchall()}


PAIR = ["zz_probe_rental", "zz_probe_rental_allowed"]
NAMES = {1: "zz_probe_rental", 2: "zz_probe_rental_allowed", 3: "zz_probe_other"}


def rule(subject_id: int, polarity=None, qualifier=None) -> ResolvedRule:
    return ResolvedRule(
        subject_id=subject_id, polarity=polarity, qualifier=qualifier
    )


# --- detection, from the pass's own statements: no database ---------------
def test_the_permission_is_named_and_the_amenity_is_not():
    doomed = redundant_permissions(
        [rule(1, polarity=True), rule(2, polarity=True)], NAMES
    )
    assert doomed == [(2, "zz_probe_rental_allowed")]


def test_opposite_polarities_are_two_facts_and_neither_is_named():
    """`barbecue_allowed` true + `barbecue_equipment` false: you may barbecue,
    the gear is not provided. Deleting either would lose a fact."""
    assert redundant_permissions(
        [rule(1, polarity=False), rule(2, polarity=True)], NAMES
    ) == []


def test_a_permission_carrying_a_number_is_never_named():
    assert redundant_permissions(
        [rule(1, polarity=True), rule(2, polarity=True, qualifier=2)], NAMES
    ) == []


def test_a_permission_whose_amenity_this_pass_did_not_write_is_left_alone():
    """The behaviour the in-memory form pins: detection sees only what this pass
    wrote, so an amenity from an earlier run does not condemn a permission
    written now. A query over the campsite would have deleted it."""
    assert redundant_permissions([rule(2, polarity=True)], NAMES) == []


def test_an_unrelated_amenity_is_left_alone():
    assert redundant_permissions(
        [rule(1, polarity=True), rule(3, polarity=True)], NAMES
    ) == []


def test_nothing_written_names_nothing():
    assert redundant_permissions([], NAMES) == []


# --- the delete itself, against a live database ---------------------------
def test_the_permission_row_goes_and_the_amenity_row_stays(scope):
    campsite_id, add, cur = scope
    amenity = add("zz_probe_rental", 1, True)
    permission = add("zz_probe_rental_allowed", 2, True)

    dropped = drop_redundant_permissions(
        cur.connection,
        campsite_id=campsite_id,
        rules=[rule(amenity, polarity=True), rule(permission, polarity=True)],
    )

    assert dropped == ["zz_probe_rental_allowed"]
    assert surviving(cur, campsite_id, PAIR) == {"zz_probe_rental"}


def test_a_per_unit_permission_never_cancels_a_site_level_amenity(scope):
    """Scope is part of the identity: the site providing showers says nothing
    about whether one unit permits them."""
    campsite_id, add, cur = scope
    amenity = add("zz_probe_rental", 1, True)
    permission = add("zz_probe_rental_allowed", 2, True)

    dropped = drop_redundant_permissions(
        cur.connection,
        campsite_id=campsite_id,
        rules=[rule(amenity, polarity=True), rule(permission, polarity=True)],
        accommodation_type_id=-1,  # a scope holding neither row
    )

    assert dropped == ["zz_probe_rental_allowed"]
    assert surviving(cur, campsite_id, PAIR) == set(PAIR)
