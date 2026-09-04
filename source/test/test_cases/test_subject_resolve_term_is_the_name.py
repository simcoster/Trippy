"""The extractor's term is the subject's name. Nothing downstream renames it.

On a live run `late_check_out_fee_applies` was an alias of #38
(`late_check_out_fee_percent`). The judge rightly rejected #38 for the term
`late_check_out_in_accommodation_units_fee_applies`, then the classifier
proposed `late_check_out_fee_applies` as its canonical name and a second
subject (#64) was inserted under that exact string -- unreachable by its own
name from then on, because alias lookup runs first.

That path no longer exists: the term itself is inserted, so the stored name is
the string that just missed the alias lookup and cannot collide with anything.
"""

from __future__ import annotations

from source.scraper.subjects.resolve import ResolutionTrace, resolve_subject
from source.test.test_cases.test_subject_resolve import (
    CLOSE,
    FakeCursor,
    make_adjudicator,
    make_conn,
    make_embedder,
)


def test_the_term_is_inserted_verbatim_whatever_the_classifier_proposes():
    cursor = FakeCursor(
        nearest=[(38, "late_check_out_fee_percent", 2, CLOSE)],
        inserted=(64, "late_check_out_in_accommodation_units_fee_applies", 2),
    )
    # The classifier would have renamed this onto #38's alias. It is ignored.
    adjudicator = make_adjudicator(
        match=None, category=2, canonical_name="late_check_out_fee_applies"
    )
    sink: list[ResolutionTrace] = []

    ref = resolve_subject(
        make_conn(cursor),
        "late_check_out_in_accommodation_units_fee_applies",
        embedder=make_embedder(),
        adjudicator=adjudicator,
        category=2,
        trace_sink=sink,
        verbose=False,
    )

    assert ref is not None and ref.id == 64
    inserts = cursor.sql_matching("INSERT")
    assert len(inserts) == 1
    assert inserts[0][1]["name"] == "late_check_out_in_accommodation_units_fee_applies"
    assert inserts[0][1]["aliases"] == [
        "late_check_out_in_accommodation_units_fee_applies"
    ]
    assert not cursor.sql_matching("array_append")
    assert sink[0].kind == "inserted"


def test_the_classifier_is_not_consulted_when_the_extractor_gave_a_category():
    cursor = FakeCursor(nearest=[], inserted=(70, "quiet_hours_start_time", 2))
    adjudicator = make_adjudicator(match=None, category=1, canonical_name="renamed")

    resolve_subject(
        make_conn(cursor),
        "quiet_hours_start_time",
        embedder=make_embedder(),
        adjudicator=adjudicator,
        category=2,
        verbose=False,
    )

    adjudicator.classify.assert_not_called()
    params = cursor.sql_matching("INSERT")[0][1]
    assert params["name"] == "quiet_hours_start_time"
    assert params["category"] == 2


def test_the_classifier_supplies_only_a_missing_category():
    cursor = FakeCursor(nearest=[], inserted=(71, "quiet_hours_start_time", 2))
    adjudicator = make_adjudicator(match=None, category=2, canonical_name="renamed")

    resolve_subject(
        make_conn(cursor),
        "quiet_hours_start_time",
        embedder=make_embedder(),
        adjudicator=adjudicator,
        verbose=False,
    )

    adjudicator.classify.assert_called_once()
    params = cursor.sql_matching("INSERT")[0][1]
    assert params["name"] == "quiet_hours_start_time"  # not "renamed"
    assert params["category"] == 2
