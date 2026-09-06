"""Two rate lines cannot share one product, and the code can see that itself.

`list_prices_unique_rate` is (product, guest type, rate period, class), so two
rate-card lines resolving to the same product do not both survive the insert --
the second overwrites the first, and nothing is printed. Tel Arad lost a row
that way: `מאהל גדול קבוע מבנה כנעני (עד 10 לנים)` and
`... כפול (עד 36)` both matched `מאהל גדול קבוע (מבנה כנעני)` at 1.00 and 0.80
while `מתחם כפול בתוך מבנה החאן הכנעני` got no price at all.

Confidence cannot catch that -- both answers were confident. The clash itself
can: the rate card priced them on separate lines, so they are separate products
and one of the two matches is wrong whatever the model says.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from source.scraper.info_site.db import (
    colliding_rows,
    snapshot_list_prices,
)
from source.scraper.info_site.match_listing import InfoWebsiteNameMatcher
from source.scraper.info_site.schemas import ClassifiedPriceRow
from source.scraper.info_site.scrape import match_verdict

SINGLE = "מאהל גדול קבוע (מבנה כנעני)"
DOUBLE = "מתחם כפול בתוך מבנה החאן הכנעני"
TENTS = "לינת שטח באוהלים פרטיים"
CANDIDATES = [TENTS, SINGLE, DOUBLE]

LABEL_A = "מאהל גדול קבוע מבנה כנעני (עד 10 לנים)"
LABEL_B = "מאהל גדול קבוע מבנה כנעני כפול (עד 36)"


def price_row(label: str, price: float, guest_type: str = "any") -> ClassifiedPriceRow:
    return ClassifiedPriceRow(
        raw_label=label,
        price=price,
        accommodation_type="מאהל גדול קבוע",
        guest_type=guest_type,
        rate_period="any",
        kind="lodging",
    )


def matcher_returning(*replies: str) -> InfoWebsiteNameMatcher:
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


@pytest.fixture
def site(experiments_site):
    conn, site_id = experiments_site
    with conn.cursor() as cur:
        for name in CANDIDATES:
            cur.execute(
                "INSERT INTO info_website_names (site_id, name) VALUES (%s, %s)",
                (site_id, name),
            )
    return conn, site_id


def stored(conn, site_id: int) -> dict[str, float]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT n.name, p.price FROM list_prices p "
            "JOIN info_website_names n ON n.id = p.info_website_name_id "
            "WHERE p.site_id = %s",
            (site_id,),
        )
        return {name: float(price) for name, price in cur.fetchall()}


# --- detection, no model involved --------------------------------------------


def test_two_rows_on_one_product_at_one_rate_are_a_collision():
    rows = [(price_row(LABEL_A, 860.0), [7], 1.0), (price_row(LABEL_B, 3080.0), [7], 0.8)]

    assert colliding_rows(rows, "regular") == [[0, 1]]


def test_the_same_product_at_different_rates_is_not_a_collision():
    """Adult and child prices for one product are two legitimate rows."""
    rows = [
        (price_row(LABEL_A, 64.0, guest_type="adult"), [7], 1.0),
        (price_row(LABEL_A, 47.0, guest_type="child"), [7], 1.0),
    ]

    assert colliding_rows(rows, "regular") == []


def test_different_products_do_not_collide():
    rows = [(price_row(LABEL_A, 860.0), [7], 1.0), (price_row(LABEL_B, 3080.0), [8], 0.9)]

    assert colliding_rows(rows, "regular") == []


def test_a_split_row_collides_on_either_of_its_products():
    """A rescued row holds two products; a clash on one of them still counts."""
    rows = [(price_row(LABEL_A, 860.0), [7, 8], 0.9), (price_row(LABEL_B, 3080.0), [8], 1.0)]

    assert colliding_rows(rows, "regular") == [[0, 1]]


# --- and what the extra call does with it ------------------------------------


def test_a_collision_gives_each_label_its_own_product(site):
    conn, site_id = site
    matcher = matcher_returning(
        '{"name": "%s", "confidence": 1.0}' % SINGLE,
        '{"name": "%s", "confidence": 0.8}' % SINGLE,
        '{"first": "%s", "second": "%s", "confidence": 0.95}' % (SINGLE, DOUBLE),
    )

    snapshot_list_prices(
        conn,
        site_id=site_id,
        rows=[price_row(LABEL_A, 860.0), price_row(LABEL_B, 3080.0)],
        matcher=matcher,
    )

    # Both prices survive, on the two products the rate card was pricing.
    assert stored(conn, site_id) == {SINGLE: 860.0, DOUBLE: 3080.0}
    assert matcher.calls[-1].kind == "collision"


def test_an_answer_naming_one_product_twice_is_refused(site):
    """The whole point is that they differ; an equal pair settles nothing."""
    conn, site_id = site
    matcher = matcher_returning(
        '{"name": "%s", "confidence": 1.0}' % SINGLE,
        '{"name": "%s", "confidence": 0.8}' % SINGLE,
        '{"first": "%s", "second": "%s", "confidence": 0.9}' % (SINGLE, SINGLE),
    )

    snapshot_list_prices(
        conn,
        site_id=site_id,
        rows=[price_row(LABEL_A, 860.0), price_row(LABEL_B, 3080.0)],
        matcher=matcher,
    )

    call = matcher.calls[-1]
    assert call.picked is None, "an equal pair is not an answer"
    assert match_verdict(call) == "collision unresolved"
    # Left exactly as they were: one product, one surviving price.
    assert list(stored(conn, site_id)) == [SINGLE]


def test_a_name_off_the_list_is_refused_too(site):
    conn, site_id = site
    matcher = matcher_returning(
        '{"name": "%s", "confidence": 1.0}' % SINGLE,
        '{"name": "%s", "confidence": 0.8}' % SINGLE,
        '{"first": "%s", "second": "בונגלו עם מזגן", "confidence": 0.9}' % SINGLE,
    )

    snapshot_list_prices(
        conn,
        site_id=site_id,
        rows=[price_row(LABEL_A, 860.0), price_row(LABEL_B, 3080.0)],
        matcher=matcher,
    )

    assert matcher.calls[-1].picked is None


def test_a_collision_is_always_in_the_report(site):
    """It has no confidence of its own to be judged against, so it always shows."""
    matcher = matcher_returning(
        '{"first": "%s", "second": "%s", "confidence": 0.95}' % (SINGLE, DOUBLE)
    )
    matcher.pick_pair(LABEL_A, LABEL_B, SINGLE, CANDIDATES)

    call = matcher.calls[0]
    assert match_verdict(call) == "collision"
    assert "Label A:" in call.user and "Label B:" in call.user
    assert f"Both were matched to: {SINGLE}" in call.user


def test_three_labels_on_one_product_are_reported_not_guessed(site, capsys):
    """Three is a different shape -- a catalog missing an entry, most likely."""
    conn, site_id = site
    matcher = matcher_returning(
        *['{"name": "%s", "confidence": 0.9}' % SINGLE] * 3
    )

    snapshot_list_prices(
        conn,
        site_id=site_id,
        rows=[
            price_row(LABEL_A, 860.0),
            price_row(LABEL_B, 3080.0),
            price_row("מאהל גדול קבוע משהו שלישי", 500.0),
        ],
        matcher=matcher,
    )

    assert "3 labels on one product" in capsys.readouterr().out
    assert all(call.kind == "pick" for call in matcher.calls), "no pair was asked"
