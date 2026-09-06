"""`X` and `X_allowed` from one sentence are one fact; the amenity keeps it.

`ניתן להזמין מזרנים בתוספת תשלום` yields `mattress_rental` (amenity, true) and
`mattress_rental_allowed` (boolean_rule, true) from the same words. Measured
over 18 sites: 5 rows of 438, in exactly two subjects.

The deletion is deterministic, so it runs before the conflict resolver asks the
model anything. What it must NOT touch is the pair the extractor prompt
deliberately produces -- `barbecue_allowed` true AND `barbecue_equipment` false
are two different facts, and their polarities differing is what says so.

Everything runs in the `experiments` schema: the code under test DELETEs rows,
and it must never be pointed at a real one.
"""

from __future__ import annotations

import pytest

from source.scraper.rules_ingest.db import ResolvedRule
from source.scraper.rules_ingest.resolve_conflicts import (
    drop_redundant_permissions,
    redundant_permissions,
)


@pytest.fixture
def scope(experiments_site):
    """The test campsite, and a factory for subject + rule pairs.

    In `experiments`, so the deletes under test cannot reach a real rule.
    """
    conn, campsite_id = experiments_site
    with conn.cursor() as cur:

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


PAIR = ["probe_rental", "probe_rental_allowed"]
NAMES = {1: "probe_rental", 2: "probe_rental_allowed", 3: "probe_other"}


def rule(subject_id: int, polarity=None, qualifier=None) -> ResolvedRule:
    return ResolvedRule(
        subject_id=subject_id, polarity=polarity, qualifier=qualifier
    )


# --- detection, from the pass's own statements: no database ---------------
def test_the_permission_is_named_and_the_amenity_is_not():
    doomed = redundant_permissions(
        [rule(1, polarity=True), rule(2, polarity=True)], NAMES
    )
    assert doomed == [(2, "probe_rental_allowed")]


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
    amenity = add("probe_rental", 1, True)
    permission = add("probe_rental_allowed", 2, True)

    dropped = drop_redundant_permissions(
        cur.connection,
        campsite_id=campsite_id,
        rules=[rule(amenity, polarity=True), rule(permission, polarity=True)],
    )

    assert dropped == ["probe_rental_allowed"]
    assert surviving(cur, campsite_id, PAIR) == {"probe_rental"}


def test_a_per_unit_permission_never_cancels_a_site_level_amenity(scope):
    """Scope is part of the identity: the site providing showers says nothing
    about whether one unit permits them."""
    campsite_id, add, cur = scope
    amenity = add("probe_rental", 1, True)
    permission = add("probe_rental_allowed", 2, True)

    dropped = drop_redundant_permissions(
        cur.connection,
        campsite_id=campsite_id,
        rules=[rule(amenity, polarity=True), rule(permission, polarity=True)],
        accommodation_type_id=-1,  # a scope holding neither row
    )

    assert dropped == ["probe_rental_allowed"]
    assert surviving(cur, campsite_id, PAIR) == set(PAIR)


def test_a_dropped_permission_reaches_the_run_report(scope):
    """It is not a loss, but it says the extractor split one sentence into two
    statements where the prompt asks for one -- so it has to be readable
    afterwards, not just printed as the run scrolls past."""
    campsite_id, add, cur = scope
    amenity = add("probe_rental", 1, True)
    permission = add("probe_rental_allowed", 2, True)
    sink: list[tuple[str, str]] = []

    drop_redundant_permissions(
        cur.connection,
        campsite_id=campsite_id,
        rules=[rule(amenity, polarity=True), rule(permission, polarity=True)],
        sink=sink,
        scope="חושה",
    )

    assert sink == [("probe_rental_allowed", "חושה")]


def test_the_report_renders_what_was_dropped():
    from source.scraper.rules_ingest.ingest import SiteReport

    report = SiteReport()
    report.redundant.append(("probe_rental_allowed", "חושה"))
    rendered = report.render()

    assert "Redundant permissions dropped" in rendered
    assert "probe_rental_allowed" in rendered
    assert "חושה" in rendered
