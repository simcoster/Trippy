"""One-night quotes from published list_prices (per-person vs per-unit)."""

from __future__ import annotations

from dataclasses import dataclass

from source.scraper.info_site.quote import quote_night
from source.scraper.info_site.schemas import GuestType, RatePeriod


@dataclass(frozen=True)
class Rate:
    guest_type: GuestType
    rate_period: RatePeriod
    price: float


# Horashat Tal רגיל (fixture table): tent is per person, rooms are per unit.
TENT = [
    Rate("adult", "any", 76.0),
    Rate("child", "any", 58.0),
]
BUNGALOW = [
    Rate("any", "weekday", 430.0),
    Rate("any", "weekend_holiday", 530.0),
]
STAFF_ROOM = [
    Rate("any", "weekday", 480.0),
    Rate("any", "weekend_holiday", 680.0),
]


def test_tent_one_adult_weekday():
    assert quote_night(TENT, adults=1, rate_period="weekday") == 76.0


def test_tent_two_adults_weekday():
    assert quote_night(TENT, adults=2, rate_period="weekday") == 152.0


def test_tent_two_adults_one_child_weekend():
    assert quote_night(
        TENT, adults=2, children=1, rate_period="weekend_holiday"
    ) == 210.0


def test_tent_two_adults_two_children_weekday():
    assert quote_night(TENT, adults=2, children=2, rate_period="weekday") == 268.0


def test_bungalow_one_adult_weekday():
    assert quote_night(BUNGALOW, adults=1, rate_period="weekday") == 430.0


def test_bungalow_two_adults_weekend():
    assert quote_night(
        BUNGALOW, adults=2, rate_period="weekend_holiday"
    ) == 530.0


def test_staff_room_two_adults_one_child_weekday():
    assert quote_night(
        STAFF_ROOM, adults=2, children=1, rate_period="weekday"
    ) == 480.0


def test_staff_room_two_adults_two_children_weekend():
    assert quote_night(
        STAFF_ROOM, adults=2, children=2, rate_period="weekend_holiday"
    ) == 680.0
