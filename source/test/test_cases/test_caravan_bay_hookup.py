"""Caravan-bay power and water are vehicle hookups, not campsite electricity.

Retrieve matches the subject name, not the unit type on the row, so a
private-caravan listing that says חיבור חשמל ומים must not emit bare
`electric_hookup` / `water_hookup`. Measured on the unit extractor
(experiments.md 2026-09-08 §1).
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

from source.scraper.amenity_enrichment.schemas import ALLOWED_CATEGORIES
from source.scraper.rules_ingest.units import UNIT_PROMPT, unit_section
from source.scraper.rules_ingest.units import unit_extractor as make_unit_extractor

load_dotenv()

CARAVAN_NAME = "עמדת חניה לקרוואן פרטי"
CARAVAN_TOOLTIP = (
    "כניסה לחניון לילה עם קרוואן פרטי\n"
    "קיים חיבור חשמל ומים\n"
    "הרכב ההזמנה: עד 6 לנים בהרכב."
)

PITCH_NAME = "מתחם PITCH"
PITCH_TOOLTIP = "אוהל בשטח. חיבור חשמל. שירותים ומקלחות משותפים."

BUNGALOW_NAME = "בונגלו עם מזגן"
BUNGALOW_TOOLTIP = (
    "בכל בונגלו: 4 מיטות, מזרנים, מיני מקרר, ארונית, שקע חשמל, מזגן. "
    "עד 4 לנים בכל בונגלו. "
    "אין מקור מים, ללא מקלחת ושירותים."
)

BARE_HOOKUPS = frozenset(
    {"electric_hookup", "electricity", "water_hookup", "water"}
)


def test_unit_prompt_teaches_the_caravan_bay_hookup_names():
    assert "caravan_bay_electric_hookup" in UNIT_PROMPT
    assert "caravan_bay_water_hookup" in UNIT_PROMPT
    assert "vehicle hookup" in UNIT_PROMPT


def test_unit_prompt_still_keeps_guest_power_as_electric_hookup():
    """PITCH / bungalow sockets stay the generic names; only the vehicle bay
    takes the exception."""
    assert "electric_hookup / electric_outlet" in UNIT_PROMPT


@pytest.fixture(scope="module")
def extractor():
    if not os.environ.get("NEBIUS_API_KEY"):
        pytest.skip("NEBIUS_API_KEY required")
    return make_unit_extractor()


def _extract(extractor, type_name: str, tooltip: str):
    section = unit_section(type_name, tooltip, source_url=None)
    return extractor.extract(section.text, section_title=section.title)


def _names(extract) -> set[str]:
    return {s.subject for s in extract.statements}


@pytest.mark.llm
def test_caravan_bay_hookups_are_not_generic_electricity(extractor):
    extract = _extract(extractor, CARAVAN_NAME, CARAVAN_TOOLTIP)
    names = _names(extract)
    print("caravan subjects:", " ".join(sorted(names)))
    assert "caravan_bay_electric_hookup" in names
    assert "caravan_bay_water_hookup" in names
    assert names.isdisjoint(BARE_HOOKUPS)
    assert any(s in ALLOWED_CATEGORIES for s in names)
    occupancy = [
        s.subject
        for s in extract.statements
        if "occupancy" in s.subject or "guest" in s.subject or "people" in s.subject
    ]
    assert not occupancy, occupancy


@pytest.mark.llm
def test_tent_pitch_power_stays_electric_hookup(extractor):
    extract = _extract(extractor, PITCH_NAME, PITCH_TOOLTIP)
    names = _names(extract)
    print("pitch subjects:", " ".join(sorted(names)))
    assert "electric_hookup" in names
    assert not any(n.startswith("caravan_bay_") for n in names)


@pytest.mark.llm
def test_bungalow_socket_is_not_a_caravan_bay_hookup(extractor):
    """שקע חשמל in a bungalow is a guest outlet, not a vehicle hookup."""
    extract = _extract(extractor, BUNGALOW_NAME, BUNGALOW_TOOLTIP)
    names = _names(extract)
    print("bungalow subjects:", " ".join(sorted(names)))
    assert not any(n.startswith("caravan_bay_") for n in names)
    assert "electric_outlet" in names
