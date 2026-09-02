"""Live adjudication: which near-neighbours are actually the same subject.

The whole alias mechanism turns on this judgement. Merging too eagerly is the
expensive failure — two facts collapse onto one subject and the second one is
dropped at upsert — so the pairs below are pinned.
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

from source.scraper.subjects.llm import SubjectAdjudicatorLLMClient

pytestmark = pytest.mark.llm

load_dotenv()


@pytest.fixture(scope="module")
def adjudicator() -> SubjectAdjudicatorLLMClient:
    if not os.environ.get("NEBIUS_API_KEY"):
        pytest.skip("NEBIUS_API_KEY required")
    return SubjectAdjudicatorLLMClient()


@pytest.mark.parametrize(
    ("term", "candidates", "expected"),
    [
        # Spelling and pluralisation variants of one subject.
        ("air_conoditioning", ["air_conditioning", "heating", "fan"], "air_conditioning"),
        ("refrigerators", ["refrigerator", "freezer", "cooler"], "refrigerator"),
        ("hot_showers", ["hot_shower", "shower", "toilets"], "hot_shower"),
        (
            "mattress_pickup_time",
            ["mattress_rental_pickup_time", "check_in_time"],
            "mattress_rental_pickup_time",
        ),
    ],
)
def test_true_variants_merge(adjudicator, term, candidates, expected):
    assert adjudicator.pick_match(term, candidates) == expected


@pytest.mark.parametrize(
    ("term", "candidates"),
    [
        # Two facts about one noun are two subjects.
        ("last_dogs_entry_time", ["dogs_allowed", "check_in_time"]),
        # A qualifier is usually the thing being searched for.
        ("accessible_toilets", ["toilets", "shower", "field_kitchen"]),
        ("mattresses_for_rent", ["mattress", "bed", "pillow"]),
        ("hot_water_shower", ["shower", "toilets"]),
    ],
)
def test_different_subjects_stay_apart(adjudicator, term, candidates):
    assert adjudicator.pick_match(term, candidates) is None


def test_a_name_outside_the_candidate_list_is_never_returned(adjudicator):
    """Whatever the model says, the answer has to be one of the options."""
    match = adjudicator.pick_match("air_conditioning_unit", ["shower", "toilets"])
    assert match in (None, "shower", "toilets")
    assert match is None


@pytest.mark.parametrize(
    ("term", "category"),
    [
        ("dogs_allowed", 2),
        ("min_weekend_nights", 2),
        ("check_out_time", 2),
        ("refrigerator", 1),
        ("hot_shower", 1),
    ],
)
def test_classify_assigns_the_right_category(adjudicator, term, category):
    assert adjudicator.classify(term).category == category


@pytest.mark.parametrize(
    ("term", "expected"),
    [
        ("no dogs allowed", "dogs_allowed"),
        ("bring your own towels", "towels_included"),
    ],
)
def test_classify_names_negatives_positively(adjudicator, term, expected):
    """Polarity is a column; it must never end up in the name."""
    assert adjudicator.classify(term).canonical_name == expected


@pytest.mark.parametrize(
    ("term", "candidates", "expected"),
    [
        # "is it provided" lives in polarity, so the suffix names no new subject.
        ("towels_included", ["towels", "bed_linens", "blankets"], "towels"),
        (
            "electric_hookup_included",
            ["electric_hookup", "power_outlet"],
            "electric_hookup",
        ),
        # Word order is not a new subject either.
        ("min_child_age", ["child_min_age", "adult_min_age"], "child_min_age"),
    ],
)
def test_provision_suffixes_and_word_order_merge(adjudicator, term, candidates, expected):
    assert adjudicator.pick_match(term, candidates) == expected


def test_a_provision_suffix_does_not_merge_across_different_nouns(adjudicator):
    """barbecue_equipment is what is supplied; barbecue is the activity."""
    assert (
        adjudicator.pick_match("barbecue_equipment_included", ["barbecue", "barbecue_pit"])
        is None
    )


@pytest.mark.parametrize(
    ("term", "candidates"),
    [
        # Narrowing by a word anywhere in the name, each seen collapsing a real
        # fact on a live Hurshat Tal run.
        ("gas_in_field_kitchen", ["field_kitchen", "kitchenette"]),
        (
            "early_arrival_parking_allowed",
            ["early_arrival_allowed", "check_in_time"],
        ),
        (
            "late_check_out_saturday_allowed",
            ["late_check_out_allowed", "late_check_out_fee"],
        ),
        (
            "late_check_out_accommodation_units_allowed",
            ["late_check_out_allowed", "check_out_time"],
        ),
    ],
)
def test_a_qualifier_anywhere_in_the_name_blocks_the_merge(
    adjudicator, term, candidates
):
    assert adjudicator.pick_match(term, candidates) is None


@pytest.mark.parametrize(
    "term",
    [
        # A well-formed name must survive the classifier untouched, or the same
        # word yields a different canonical name on the next run and the
        # vocabulary forks instead of aliasing.
        "child_max_age",
        "suitable_for_shabbat_observers",
        "service_center_regular_hours",
        "min_weekend_nights",
        "dogs_allowed",
        "electric_hookup_included",
    ],
)
def test_a_well_formed_name_is_returned_unchanged(adjudicator, term):
    assert adjudicator.classify(term).canonical_name == term


def test_the_classifier_still_drops_noise(adjudicator):
    """Simplifying is allowed; adding or reordering words is not."""
    assert adjudicator.classify("picnic_tables_and_benches").canonical_name == (
        "picnic_tables"
    )


@pytest.mark.parametrize(
    "term", ["late_checkout_fee", "coolers", "child_max_age", "mattresses_for_rent"]
)
def test_classification_is_stable_across_calls(adjudicator, term):
    """The property that matters: one term, one canonical name, every time.

    `coolers` came back as `cooler` on one run and `cooler_included` on another,
    which forks the vocabulary — the next campsite makes a duplicate subject
    instead of aliasing onto the first.
    """
    first = adjudicator.classify(term).canonical_name
    assert adjudicator.classify(term).canonical_name == first


@pytest.mark.parametrize(
    "term", ["towels_included", "electric_hookup_included", "barbecue_equipment_included"]
)
def test_a_provision_suffix_names_an_amenity(adjudicator, term):
    """It says the site supplies something; whether it does is polarity."""
    assert adjudicator.classify(term).category == 1
