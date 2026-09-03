"""Subject names must never be negatively phrased; negation lives in polarity."""

import pytest

from source.scraper.subjects.naming import (
    normalize_alias,
    opposed,
    predicate_suffix,
    same_predicate,
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


@pytest.mark.parametrize(
    ("name", "suffix"),
    [
        ("barbecue_allowed", "allowed"),
        ("late_check_out_fee", "fee"),
        # A trailing provision word is not a predicate of its own.
        ("barbecue_equipment_included", None),
        ("late_check_out_available", None),
        ("towels_included", None),
        ("check_in_time", "time"),
        ("adult_min_age", "age"),
        ("min_weekend_nights", "nights"),
        ("rental_deposit_required", "required"),
        ("barbecue", None),
        ("refrigerator", None),
        ("ash_collection_stations", None),
        ("drinking_water_fountain", None),
    ],
)
def test_predicate_suffix(name, suffix):
    assert predicate_suffix(name) == suffix


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("air_conditioner", "air_conditioning"),  # both bare nouns
        ("refrigerators", "refrigerator"),
        ("check_in_time", "check_out_time"),  # same predicate, nouns differ
        ("dogs_allowed", "pets_allowed"),
    ],
)
def test_same_predicate_lets_real_candidates_through(left, right):
    assert same_predicate(left, right)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        # The over-merges seen on the first live Hurshat Tal run.
        ("barbecue_allowed", "barbecue"),
        ("late_check_out_fee", "late_check_out_available"),
        ("equipment_rental_deposit_required", "rental_equipment_available"),
        ("last_dogs_entry_time", "dogs_allowed"),
    ],
)
def test_different_predicates_are_never_the_same_subject(left, right):
    assert not same_predicate(left, right)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        # "is it provided" is what polarity records, so these are one subject
        # and the pair must reach the judge rather than be blocked in code.
        ("towels_included", "towels"),
        ("electric_hookup_included", "electric_hookup"),
        ("picnic_tables_provided", "picnic_tables"),
        ("wifi_available", "wifi"),
        # Word order is not a new subject either.
        ("child_min_age", "min_child_age"),
        # Same noun, so the judge gets to decide; it rejects this one on the
        # nouns (equipment is not the activity), not on the suffix.
        ("barbecue_equipment_included", "barbecue"),
    ],
)
def test_provision_suffixes_and_word_order_reach_the_judge(left, right):
    assert same_predicate(left, right)


# --- a suffix can mean different things on a rule and on an amenity -----------

AMENITY, RULE = 1, 2


@pytest.mark.parametrize(
    ("left", "right"),
    [
        # On a rule, every way of asking "may I?" is the same predicate.
        ("late_checkout_allowed", "late_check_out_available"),
        ("dogs_allowed", "dogs_permitted"),
        ("early_arrival_available", "early_arrival_allowed"),
    ],
)
def test_permission_words_are_one_predicate_on_a_rule(left, right):
    assert same_predicate(left, right, category=RULE)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        # On an amenity the same word means "supplied", which polarity records.
        ("wifi_available", "wifi"),
        ("towels_included", "towels"),
        ("electric_hookup_included", "electric_hookup"),
    ],
)
def test_provision_words_are_no_predicate_on_an_amenity(left, right):
    assert same_predicate(left, right, category=AMENITY)


@pytest.mark.parametrize(
    ("left", "right", "category"),
    [
        ("late_check_out_available", "late_check_out_fee", RULE),
        ("barbecue_allowed", "barbecue", RULE),
        ("last_dogs_entry_time", "dogs_allowed", RULE),
    ],
)
def test_a_real_predicate_difference_survives_either_way(left, right, category):
    assert not same_predicate(left, right, category=category)


def test_available_is_read_differently_by_category():
    """The whole reason the comparison needs to know what it is looking at."""
    assert same_predicate("x_available", "x", category=AMENITY)
    assert not same_predicate("x_available", "x", category=RULE)


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
