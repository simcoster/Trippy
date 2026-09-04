"""Subject names must never be negatively phrased; negation lives in polarity."""

import pytest

from source.scraper.subjects.naming import (
    normalize_alias,
    opposed,
    to_positive_subject,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Air Conditioning", "air_conditioning"),
        ("  hot   showers  ", "hot_showers"),
        ("electric-hookup", "electric_hookup"),
        ("Drinking Water Fountain (6)", "drinking_water_fountain_6"),
        ("__shower__", "shower"),
        ("", ""),
    ],
)
def test_normalize_alias(raw, expected):
    assert normalize_alias(raw) == expected


@pytest.mark.parametrize(
    ("raw", "name", "polarity"),
    [
        ("dogs_not_allowed", "dogs_allowed", False),
        ("pets_not_permitted", "pets_permitted", False),
        ("towels_not_included", "towels_included", False),
        ("linens_not_provided", "linens_provided", False),
        ("pets_forbidden", "pets_allowed", False),
        ("smoking_prohibited", "smoking_allowed", False),
        ("campfires_banned", "campfires_allowed", False),
        ("wifi_unavailable", "wifi_available", False),
        ("no_pets", "pets_allowed", False),
        ("without_electricity", "electricity_allowed", False),
    ],
)
def test_negation_moves_out_of_the_name(raw, name, polarity):
    assert to_positive_subject(raw) == (name, polarity)


@pytest.mark.parametrize(
    "raw",
    [
        "dogs_allowed",
        "dogs_must_wear_a_muzzle",
        "last_dogs_entry_time",
        "min_weekend_nights",
        "max_occupancy",
        "pool_min_age",
        "drinking_water_fountain",
    ],
)
def test_positive_names_pass_through_untouched(raw):
    """A polarity of None means the name said nothing — not that it is unknown."""
    assert to_positive_subject(raw) == (raw, None)


def test_rewriting_is_idempotent():
    name, _ = to_positive_subject("dogs_not_allowed")
    assert to_positive_subject(name) == (name, None)


@pytest.mark.parametrize(
    "raw",
    [
        "cant_be_without_muzzle",
        "cannot_enter_the_pool",
        "pets_not_on_leash",
        "not_allowed",
        "dogs_may_not_be_left_alone",
    ],
)
def test_unrewritable_double_negatives_are_dropped(raw):
    """Better to lose a statement than to store a name nobody can query."""
    name, polarity = to_positive_subject(raw)
    assert name is None
    assert polarity is False


def test_empty_term_yields_nothing():
    assert to_positive_subject("   ") == (None, None)


# Whether two names state the same predicate is the judge LLM's decision now
# (ADJUDICATE_SYSTEM_PROMPT); the suffix-list gate that lived here fragmented
# `late_check_out_*` into nine subjects and was removed.


# --- antonyms: opposite facts wearing near-identical names ---------------------


@pytest.mark.parametrize(
    ("left", "right"),
    [
        # Every one of these was merged by the judge on a live production run.
        ("child_min_age", "child_max_age"),
        ("mattress_pickup_start_time", "mattress_pickup_end_time"),
        ("mattress_pickup_start_time", "mattress_return_start_time"),
        ("mattress_pickup_end_time", "mattress_return_end_time"),
        ("check_in_time", "check_out_time"),
        ("early_arrival_allowed", "late_check_out_available"),
        ("min_weekend_nights", "max_weekend_nights"),
        ("first_entry_time", "last_entry_time"),
    ],
)
def test_antonyms_are_never_the_same_subject(left, right):
    assert opposed(left, right)
    assert opposed(right, left)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("late_check_out_available", "late_check_out_fee"),
        ("towels_included", "towels"),
        ("refrigerators", "refrigerator"),
        ("air_conditioner", "air_conditioning"),
        ("latest_arrival_time", "check_in_time"),
        ("dogs_allowed", "pets_allowed"),
    ],
)
def test_ordinary_pairs_are_not_treated_as_opposites(left, right):
    assert not opposed(left, right)
