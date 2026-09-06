"""One rate line can price two products, so a doubted match gets a second look.

Khan Be'erot's rate card has `חדר צוות גדול אמצע שבוע (חדרים 5 ו-6)` while its
lodging panel lists room 5 and room 6 separately. Any single pick is wrong about
one of them, and the two competing picks then collided on
`list_prices_unique_rate` and cost the run a row.

So a first answer below `UNCERTAIN_BELOW`, or a refusal, is asked again with a
prompt that permits several names -- and only then. A confident match still
costs one call and still lands on one product.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from source.scraper.info_site.db import UNCERTAIN_BELOW, snapshot_list_prices
from source.scraper.info_site.match_listing import InfoWebsiteNameMatcher
from source.scraper.info_site.schemas import ClassifiedPriceRow
from source.scraper.info_site.scrape import match_verdict

ROOM_5 = "חדר צוות מאובזר ומונגש חדר מספר 5"
ROOM_6 = "חדר צוות מאובזר חדר מספר 6"
TENTS = "לינת שטח באוהלים פרטיים"
CANDIDATES = [TENTS, ROOM_5, ROOM_6]


def matcher_returning(*replies: str) -> InfoWebsiteNameMatcher:
    """A real matcher over a stubbed client: one canned reply per call, in order."""
    client = MagicMock()

    def respond(**_kwargs):
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = replies[respond.calls]
        response.usage = None
        respond.calls += 1
        return response

    respond.calls = 0
    client.chat.completions.create.side_effect = respond
    return InfoWebsiteNameMatcher(client=client)


def price_row(label: str, accommodation_type: str) -> ClassifiedPriceRow:
    return ClassifiedPriceRow(
        raw_label=label,
        price=680.0,
        accommodation_type=accommodation_type,
        guest_type="any",
        rate_period="weekday",
        kind="lodging",
    )


@pytest.fixture
def site(experiments_site):
    """The test campsite with three lodging products, in `experiments`."""
    conn, site_id = experiments_site
    with conn.cursor() as cur:
        for name in CANDIDATES:
            cur.execute(
                "INSERT INTO info_website_names (site_id, name) VALUES (%s, %s)",
                (site_id, name),
            )
    return conn, site_id


def stored_names(conn, site_id: int) -> list[str]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT n.name FROM list_prices p "
            "JOIN info_website_names n ON n.id = p.info_website_name_id "
            "WHERE p.site_id = %s ORDER BY n.name",
            (site_id,),
        )
        return sorted(r[0] for r in cur.fetchall())


# --- the second call itself --------------------------------------------------


def test_pick_names_reads_every_name_it_is_given():
    matcher = matcher_returning(
        '{"names": ["%s", "%s"], "confidence": 0.9}' % (ROOM_5, ROOM_6)
    )

    names, confidence = matcher.pick_names("חדר צוות גדול חדרים 5 ו-6", CANDIDATES)

    assert names == [ROOM_5, ROOM_6]
    assert confidence == 0.9


def test_a_name_that_is_not_a_candidate_is_dropped_not_repaired():
    matcher = matcher_returning(
        '{"names": ["%s", "בונגלו עם מזגן"], "confidence": 0.8}' % ROOM_5
    )

    names, _ = matcher.pick_names("חדר צוות גדול", CANDIDATES)

    assert names == [ROOM_5], "an invented name must not reach the database"


def test_one_name_is_a_normal_answer():
    """Most labels price one product; the rescue must not invent a split."""
    matcher = matcher_returning('{"names": ["%s"], "confidence": 1.0}' % ROOM_5)

    names, _ = matcher.pick_names("חדר צוות מספר 5", CANDIDATES)

    assert names == [ROOM_5]


def test_a_split_is_visible_in_the_report():
    matcher = matcher_returning(
        '{"names": ["%s", "%s"], "confidence": 0.9}' % (ROOM_5, ROOM_6)
    )
    matcher.pick_names("חדר צוות גדול חדרים 5 ו-6", CANDIDATES)

    # Confident, so not "uncertain" -- but it writes two rows for one rate line,
    # which is exactly the thing a person should be able to see.
    assert match_verdict(matcher.calls[0]) == "split"


# --- and what it does to the stored prices -----------------------------------


def test_a_doubted_label_is_priced_against_every_product_it_names(site):
    conn, site_id = site
    matcher = matcher_returning(
        '{"name": "%s", "confidence": 0.4}' % ROOM_5,
        '{"names": ["%s", "%s"], "confidence": 0.9}' % (ROOM_5, ROOM_6),
    )

    stored = snapshot_list_prices(
        conn,
        site_id=site_id,
        rows=[price_row("חדר צוות גדול אמצע שבוע (חדרים 5 ו-6)", "חדר צוות גדול")],
        matcher=matcher,
    )

    assert len(stored) == 1, "one rate-card line"
    assert stored_names(conn, site_id) == sorted([ROOM_5, ROOM_6]), "two products"
    assert len(matcher.calls) == 2, "one pick, then one rescue"


def test_a_confident_match_is_never_asked_twice(site):
    conn, site_id = site
    matcher = matcher_returning('{"name": "%s", "confidence": 0.95}' % ROOM_5)

    snapshot_list_prices(
        conn,
        site_id=site_id,
        rows=[price_row("חדר צוות מספר 5 אמצע שבוע", "חדר צוות מאובזר")],
        matcher=matcher,
    )

    assert len(matcher.calls) == 1, "the rescue pass costs a call; earn it"
    assert stored_names(conn, site_id) == [ROOM_5]


def test_a_refusal_is_rescued_before_it_is_forced(site):
    """A null is doubt too, and the rescue may still find the right product."""
    conn, site_id = site
    matcher = matcher_returning(
        '{"name": null, "confidence": null}',
        '{"names": ["%s"], "confidence": 0.8}' % ROOM_6,
    )

    snapshot_list_prices(
        conn,
        site_id=site_id,
        rows=[price_row("חדר צוות כלשהו", "חדר צוות כלשהו")],
        matcher=matcher,
    )

    assert stored_names(conn, site_id) == [ROOM_6]


def test_a_rescue_that_finds_nothing_still_files_the_price(site):
    """The price is never dropped, whatever both passes come back with."""
    conn, site_id = site
    matcher = matcher_returning(
        '{"name": null, "confidence": null}',
        '{"names": [], "confidence": 0.0}',
    )

    stored = snapshot_list_prices(
        conn,
        site_id=site_id,
        rows=[price_row("משהו אחר לגמרי", "משהו אחר לגמרי")],
        matcher=matcher,
    )

    assert len(stored) == 1
    assert len(stored_names(conn, site_id)) == 1


def test_an_uncertain_single_answer_stays_single(site):
    """The rescue may confirm one product; it must not multiply rows for fun."""
    conn, site_id = site
    matcher = matcher_returning(
        '{"name": "%s", "confidence": 0.5}' % ROOM_5,
        '{"names": ["%s"], "confidence": 0.5}' % ROOM_5,
    )

    snapshot_list_prices(
        conn,
        site_id=site_id,
        rows=[price_row("חדר צוות גדול אמצע שבוע", "חדר צוות גדול")],
        matcher=matcher,
    )

    assert stored_names(conn, site_id) == [ROOM_5]
    assert UNCERTAIN_BELOW == 0.7, "the threshold this file's fixtures straddle"
