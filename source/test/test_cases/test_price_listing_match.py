"""A rate-card price always lands somewhere.

The rate card and the lodging panel are two descriptions of one campsite, so a
price belongs to a product even when the wording is far apart -- `חדר צוות קטן
אמצע שבוע` against a panel that calls the room `חדר צוות קטן אלון ורקפת`, or
`השכרת אוהל קמפינג זוגי כולל מזרנים (עד 2 לנים)` against a panel that mentions
neither the mattresses nor the occupancy. Both were dropped on the floor before
the matcher was told never to refuse.

So the matcher reports a poor match as a low `confidence` instead, and this
covers what `snapshot_list_prices` does with it: store the price either way,
and record the ones worth a person's eye.

Everything runs in the `experiments` schema, which these tests own outright --
`snapshot_list_prices` deletes a site's whole rate card before writing, and
pointed at production that is real data gone.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from source.scraper.info_site.db import UNCERTAIN_BELOW, snapshot_list_prices
from source.scraper.info_site.match_listing import InfoWebsiteNameMatcher
from source.scraper.info_site.schemas import ClassifiedPriceRow

PANEL_NAME = "חדר צוות קטן אלון ורקפת"
OTHER_NAME = "מאהל גדול קבוע"


@pytest.fixture
def site(experiments_site):
    """The test campsite with two lodging products, in `experiments`.

    `snapshot_list_prices` opens with `DELETE FROM list_prices WHERE site_id`,
    so this must never point at a real campsite: aimed at production it would
    delete that campsite's whole rate card.
    """
    conn, site_id = experiments_site
    with conn.cursor() as cur:
        for name in (PANEL_NAME, OTHER_NAME):
            cur.execute(
                "INSERT INTO info_website_names (site_id, name) VALUES (%s, %s)",
                (site_id, name),
            )
    return conn, site_id


def price_row(label: str, accommodation_type: str) -> ClassifiedPriceRow:
    return ClassifiedPriceRow(
        raw_label=label,
        price=100.0,
        accommodation_type=accommodation_type,
        guest_type="any",
        rate_period="any",
        kind="lodging",
    )


def stored_names(conn, site_id: int) -> list[str]:
    """Which lodging product each stored price landed on.

    No filtering needed: `snapshot_list_prices` replaces the site's whole
    regular rate card, so after the call these are exactly the rows under test.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT n.name FROM list_prices p "
            "JOIN info_website_names n ON n.id = p.info_website_name_id "
            "WHERE p.site_id = %s",
            (site_id,),
        )
        return [r[0] for r in cur.fetchall()]


def test_an_exact_name_is_stored_and_never_questioned(site):
    conn, site_id = site
    sink: list[str] = []
    matcher = MagicMock(spec=InfoWebsiteNameMatcher)

    stored = snapshot_list_prices(
        conn,
        site_id=site_id,
        rows=[price_row("לינה", PANEL_NAME)],
        matcher=matcher,
        unmatched_sink=sink,
    )

    assert len(stored) == 1
    assert stored_names(conn, site_id) == [PANEL_NAME]
    assert sink == []
    matcher.pick_name.assert_not_called()


def test_a_confident_near_match_is_stored_without_a_note(site):
    """`חדר צוות קטן אמצע שבוע` -> the room the panel names. 0.90 in the real
    run, which is well clear of the threshold."""
    conn, site_id = site
    sink: list[str] = []
    matcher = MagicMock(spec=InfoWebsiteNameMatcher)
    matcher.pick_name.return_value = (PANEL_NAME, 0.9)

    stored = snapshot_list_prices(
        conn,
        site_id=site_id,
        rows=[price_row("חדר צוות קטן אמצע שבוע", "חדר צוות קטן")],
        matcher=matcher,
        unmatched_sink=sink,
    )

    assert len(stored) == 1
    assert stored_names(conn, site_id) == [PANEL_NAME]
    assert sink == []


def test_an_unsure_match_is_stored_and_recorded(site):
    conn, site_id = site
    sink: list[str] = []
    matcher = MagicMock(spec=InfoWebsiteNameMatcher)
    matcher.pick_name.return_value = (PANEL_NAME, UNCERTAIN_BELOW - 0.2)
    # Below the threshold, so the rescue pass runs; it confirms the one product.
    matcher.pick_names.return_value = ([PANEL_NAME], UNCERTAIN_BELOW - 0.2)

    stored = snapshot_list_prices(
        conn,
        site_id=site_id,
        # A name no lodging product here carries, so the exact-match path is
        # not taken and the matcher is actually asked.
        rows=[price_row("חדר צוות עץ אמצע שבוע", "חדר צוות עץ")],
        matcher=matcher,
        unmatched_sink=sink,
    )

    # Stored, because a price belongs somewhere -- and flagged, because this one
    # is worth checking.
    assert len(stored) == 1
    assert stored_names(conn, site_id) == [PANEL_NAME]
    assert len(sink) == 1
    assert PANEL_NAME in sink[0]


def test_a_refusal_forces_a_match_rather_than_dropping_the_price(site):
    """The prompt forbids a null, so a null is the model failing to follow it.
    A price filed against the wrong unit is visible and fixable; a price dropped
    on the floor is neither."""
    conn, site_id = site
    sink: list[str] = []
    matcher = MagicMock(spec=InfoWebsiteNameMatcher)
    matcher.pick_name.return_value = (None, None)
    # The rescue finds nothing either, which is what leaves the forced path.
    matcher.pick_names.return_value = ([], None)

    stored = snapshot_list_prices(
        conn,
        site_id=site_id,
        rows=[price_row("משהו אחר לגמרי", "משהו אחר לגמרי")],
        matcher=matcher,
        unmatched_sink=sink,
    )

    assert len(stored) == 1, "the price must not be dropped"
    # It lands on whichever product came first: arbitrary, and deliberately so.
    # This path only fires when the model ignores an instruction, and a price
    # filed against the wrong unit is visible where a dropped one is not.
    assert len(stored_names(conn, site_id)) == 1
    assert len(sink) == 1, "a forced match has to be recorded"


def test_every_row_is_stored_even_when_none_match_cleanly(site):
    """The count that matters: rows in equals rows stored."""
    conn, site_id = site
    matcher = MagicMock(spec=InfoWebsiteNameMatcher)
    matcher.pick_name.return_value = (OTHER_NAME, 0.2)
    matcher.pick_names.return_value = ([OTHER_NAME], 0.2)
    rows = [
        price_row("a", "x"),
        price_row("b", "y"),
        price_row("c", "z"),
    ]

    stored = snapshot_list_prices(
        conn, site_id=site_id, rows=rows, matcher=matcher
    )

    assert len(stored) == len(rows)

