"""The 30B availability match folds Latin case so pitch == PITCH."""

from __future__ import annotations

from unittest.mock import MagicMock

from source.scraper.populate_availability import match_accommodation_type

PITCH_ID = 12
CANDIDATES = [
    (11, "לינת שטח באוהלים פרטיים"),
    (PITCH_ID, "מתחם PITCH"),
    (13, "עמדת חניה לקרוואן פרטי"),
]


def test_match_accommodation_type_folds_latin_case_for_the_30b():
    """Booking `מתחם pitch` is catalog `מתחם PITCH`; the pick copies a candidate."""
    matcher = MagicMock()
    matcher.pick_name.return_value = ("מתחם pitch", 0.9)
    cur = MagicMock()
    cur.fetchone.return_value = None

    type_id = match_accommodation_type(
        cur,
        hotel_id=15,
        booking_name="מתחם pitch",
        candidates=CANDIDATES,
        matcher=matcher,
    )

    assert type_id == PITCH_ID
    needle, names = matcher.pick_name.call_args.args[:2]
    assert needle == "מתחם pitch"
    assert names == [
        "לינת שטח באוהלים פרטיים",
        "מתחם pitch",
        "עמדת חניה לקרוואן פרטי",
    ]
    alias = cur.execute.call_args.args[1]
    assert alias["alias"] == "מתחם pitch"
    assert alias["id"] == PITCH_ID
