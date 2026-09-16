"""Gold catalog must cover identities, group occupancy, and each lodging."""

from source.price_sandbox.gold.cases import (
    BUNGALOW,
    CARAVAN,
    CATALOG,
    HUSHA,
    HUSHA_AC,
    HUSHA_DOUBLE,
    HUSHA_DOUBLE_AC,
    IDENTITY_GUEST_TYPES,
    MATMON,
    REGULAR,
    SENIOR,
    SOLDIER,
    STAFF,
    STAFF_WOOD,
    TENT,
)

REQUIRED_LODGINGS = {
    "אכזיב": {
        TENT,
        HUSHA,
        HUSHA_DOUBLE,
        HUSHA_AC,
        HUSHA_DOUBLE_AC,
    },
    "חורשת-טל": {TENT, BUNGALOW, STAFF, STAFF_WOOD, CARAVAN},
}

SITES_WITH_GROUP = frozenset({"אכזיב"})
EVERY_SITE_GUEST_TYPES = frozenset({REGULAR, MATMON, SOLDIER, SENIOR})
SITES_WITH_ALL_IDENTITIES = frozenset({"אכזיב"})


def _occupancy(params: dict) -> int:
    ages = params.get("child_ages") or ()
    child_num = params.get("child_num")
    if child_num is None:
        child_num = len(ages)
    return int(params.get("adults_num") or 0) + int(child_num)


def test_every_site_has_core_identity_guest_types():
    for row in CATALOG:
        types = {case["params"].get("guest_type", REGULAR) for case in row["cases"]}
        missing = EVERY_SITE_GUEST_TYPES - types
        assert not missing, f"{row['match']}: missing {missing}"


def test_achziv_has_every_identity_guest_type():
    for row in CATALOG:
        if row["match"] not in SITES_WITH_ALL_IDENTITIES:
            continue
        types = {case["params"].get("guest_type", REGULAR) for case in row["cases"]}
        missing = set(IDENTITY_GUEST_TYPES) - types
        assert not missing, f"{row['match']}: missing {missing}"


def test_published_group_sites_have_an_occupancy_override_case():
    for row in CATALOG:
        if row["match"] not in SITES_WITH_GROUP:
            continue
        assert any(
            _occupancy(case["params"]) >= 30 for case in row["cases"]
        ), row["match"]


def test_required_lodgings_each_have_a_case():
    by_match = {row["match"]: row for row in CATALOG}
    for match, lodgings in REQUIRED_LODGINGS.items():
        seen = {case["params"]["lodging"] for case in by_match[match]["cases"]}
        missing = lodgings - seen
        assert not missing, f"{match}: missing {missing}"
