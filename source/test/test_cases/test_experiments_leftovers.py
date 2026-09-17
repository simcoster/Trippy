"""copy drops leftover experiments tables that are not in public."""

from db.experiments import KEEP_FROZEN, leftover_relation_names


def test_leftover_drops_unknown_tables_and_keeps_frozen():
    existing = (
        "campsites",
        "pytest_listing_match",
        KEEP_FROZEN,
        "ad_hoc_probe",
    )
    keep = ("campsites", "accommodation_types")
    assert leftover_relation_names(existing, keep) == (
        "pytest_listing_match",
        "ad_hoc_probe",
    )


def test_leftover_empty_when_schema_matches_public():
    keep = ("campsites", "accommodation_types")
    existing = keep + (KEEP_FROZEN,)
    assert leftover_relation_names(existing, keep) == ()
