"""Compile prompt: מעל X is X and up; per-person vs per-unit keys."""

from source.scraper.info_site.compile_price import (
    OCCUPANCY_RETRY_PREAMBLE,
    SYSTEM_PROMPT,
)


def test_compile_prompt_meals_over_x_as_group_min_x():
    assert "מעל X לנים" in SYSTEM_PROMPT
    assert "GROUP_MIN` is X, not X+1" in SYSTEM_PROMPT
    assert "מעל X לנים" in OCCUPANCY_RETRY_PREAMBLE


def test_compile_prompt_separates_per_person_and_per_unit_keys():
    assert "per-person (לינת שטח, pitch, group)" in SYSTEM_PROMPT
    assert "per-unit (חושה, bungalow, room, family tent, mahal)" in SYSTEM_PROMPT
    assert 'Do not read `rates["adult"]` on a unit dict.' in SYSTEM_PROMPT
