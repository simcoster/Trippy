"""Booking units of a split site land on the right subcamp.

The INPA booking engine has one hotel id per *site*, so every unit type it
returns for Akhziv arrives under the parent — six types, one flat list, no hint
of the split beyond what the operator wrote into two of the names.

The rule is asymmetric, and that is the whole design: the northern subcamp has
only tents, and its two tent pitches say so
(`לינת שטח באוהלים פרטיים - חניון צפוני`). The four `חושה` types name no
subcamp at all, so `default_units` places them south. Anything unnamed is
southern, and these tests are what holds that in place — a routing bug here is
silent, it just quietly gives one subcamp everything.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

from source.scraper import populate_availability as pa
from source.scraper.discover_sites import load_subcamp_config
from source.scraper.rules_ingest.subcamps import (
    Subcamp,
    group_by_owner,
    owned_site_ids,
    unit_owner,
)

PARENT = 2
NORTH = Subcamp(
    campsite_id=37,
    name="אכזיב – חניון צפוני",
    heading="חניון צפוני",
    unit_name_contains=("חניון צפוני",),
)
SOUTH = Subcamp(
    campsite_id=38,
    name="אכזיב – חניון דרומי",
    heading="חניון דרומי",
    unit_name_contains=("חניון דרומי",),
    default_units=True,
)
SUBCAMPS = [NORTH, SOUTH]

NORTH_TENTS = "לינת שטח באוהלים פרטיים - חניון צפוני"
SOUTH_TENTS = "לינת שטח באוהלים פרטיים - חניון דרומי"
HUT = "חושה"
HUT_AC = "חושה עם מזגן שירותים ומקלחת"


# --- the rule ------------------------------------------------------------------


def test_a_unit_naming_the_northern_subcamp_goes_north():
    assert unit_owner(NORTH_TENTS, PARENT, SUBCAMPS) == 37


def test_a_unit_naming_the_southern_subcamp_goes_south():
    assert unit_owner(SOUTH_TENTS, PARENT, SUBCAMPS) == 38


def test_a_unit_naming_no_subcamp_goes_south():
    """The northern camp has only tents, so everything else is southern."""
    assert unit_owner(HUT, PARENT, SUBCAMPS) == 38
    assert unit_owner(HUT_AC, PARENT, SUBCAMPS) == 38


def test_the_whole_akhziv_unit_list_splits_two_north_four_south():
    units = [NORTH_TENTS, SOUTH_TENTS, HUT, HUT_AC, "חושה כפולה", "חושה כפולה עם מזגן"]
    grouped = group_by_owner(units, PARENT, SUBCAMPS)

    assert grouped[37] == [NORTH_TENTS]
    assert grouped[38] == [SOUTH_TENTS, HUT, HUT_AC, "חושה כפולה", "חושה כפולה עם מזגן"]


def test_an_ordinary_site_keeps_every_unit_itself():
    """The other 17 sites must not notice this code exists."""
    assert unit_owner(HUT, PARENT, []) == PARENT
    assert group_by_owner([HUT, HUT_AC], PARENT, []) == {PARENT: [HUT, HUT_AC]}
    assert owned_site_ids(PARENT, []) == [PARENT]


def test_with_no_default_configured_an_unnamed_unit_stays_on_the_parent():
    """Better an unrouted unit than one silently given to an arbitrary subcamp."""
    assert unit_owner(HUT, PARENT, [NORTH]) == PARENT


def test_owned_site_ids_covers_the_parent_and_every_subcamp():
    assert owned_site_ids(PARENT, SUBCAMPS) == [2, 37, 38]


def test_the_shipped_config_routes_akhzivs_tents_north_and_the_rest_south():
    """Against the real config.json, not the fixtures above."""
    areas = next(iter(load_subcamp_config().values()))
    subcamps = [
        Subcamp.from_row(37 + i, f"c{i}", area) for i, area in enumerate(areas)
    ]

    assert unit_owner(NORTH_TENTS, PARENT, subcamps) == 37
    assert unit_owner(SOUTH_TENTS, PARENT, subcamps) == 38
    assert unit_owner(HUT, PARENT, subcamps) == 38


# --- what actually gets written ------------------------------------------------


def fake_conn():
    cursor = MagicMock()
    cursor.rowcount = 0
    cursor.fetchone.return_value = (101, None)
    cursor.__enter__ = lambda self: cursor
    cursor.__exit__ = lambda *a: False
    conn = MagicMock()
    conn.cursor.return_value = cursor
    return conn, cursor


def upserted(cursor) -> list[dict]:
    return [
        call.args[1]
        for call in cursor.execute.call_args_list
        if call.args[0] is pa.UPSERT_AVAILABILITY_SQL
    ]


def created_types(cursor) -> list[dict]:
    return [
        call.args[1]
        for call in cursor.execute.call_args_list
        if call.args[0] is pa.GET_OR_CREATE_ACCOMMODATION_TYPE_SQL
    ]


def run_upsert(subcamps):
    conn, cursor = fake_conn()
    saved = pa.upsert_availability_rows(
        conn,
        site_id=PARENT,
        start=date(2026, 9, 10),
        end=date(2026, 9, 11),
        adults_no=1,
        offerings=[
            {"room_type": NORTH_TENTS},
            {"room_type": SOUTH_TENTS},
            {"room_type": HUT},
        ],
        listings=[],
        subcamps=subcamps,
    )
    return saved, cursor


def test_availability_rows_carry_the_owning_subcamps_id():
    saved, cursor = run_upsert(SUBCAMPS)

    assert saved == 3
    assert [row["site_id"] for row in upserted(cursor)] == [37, 38, 38]


def test_the_accommodation_type_is_created_under_the_owning_subcamp():
    """Types are keyed (hotel_id, name), so this is what keeps the six apart."""
    _saved, cursor = run_upsert(SUBCAMPS)

    assert [row["hotel_id"] for row in created_types(cursor)] == [37, 38, 38]


def test_without_subcamps_everything_still_lands_on_the_site():
    _saved, cursor = run_upsert([])

    assert {row["site_id"] for row in upserted(cursor)} == {PARENT}
    assert {row["hotel_id"] for row in created_types(cursor)} == {PARENT}


def test_re_scraping_a_night_clears_the_parent_and_both_subcamps():
    """Clearing only the parent would leave last week's northern tents behind."""
    conn, cursor = fake_conn()
    pa.clear_availability_for_night(
        conn,
        site_ids=owned_site_ids(PARENT, SUBCAMPS),
        start=date(2026, 9, 10),
        end=date(2026, 9, 11),
        adults_no=1,
    )

    params = cursor.execute.call_args.args[1]
    assert params["site_ids"] == [2, 37, 38]
