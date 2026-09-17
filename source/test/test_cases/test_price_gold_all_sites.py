"""Gold JSON catalog: identities, group occupancy, and explicit prices."""

from source.price_sandbox.gold.runner import (
    SITE_FILES,
    SITES_DIR,
    load_gold_catalog,
    load_site,
    site_paths,
)

TENT = "לינת שטח באוהלים פרטיים"
IDENTITY_GUEST_TYPES = frozenset(
    {
        "רגיל",
        "מנוי",
        "חייל בשירות חובה + שירות לאומי",
        "משרת מילואים פעיל",
        "אזרח ותיק",
        "סטודנט",
        "נכה צהל ומלווה",
    }
)
GROUP_MIN = 30


def _occupancy(params: dict) -> int:
    ages = params.get("child_ages") or ()
    child_num = params.get("child_num")
    if child_num is None:
        child_num = len(ages)
    return int(params.get("adults_num") or 0) + int(child_num)


def test_catalog_is_one_json_per_campsite():
    paths = site_paths()
    assert [path.name for path in paths] == list(SITE_FILES)
    catalog = load_gold_catalog()
    assert len(catalog) == 18
    matches = [row["match"] for row in catalog]
    assert len(matches) == len(set(matches))


def test_every_site_has_every_identity_guest_type():
    for path in site_paths():
        doc = load_site(path)
        types = {case["params"].get("guest_type", "רגיל") for case in doc["cases"]}
        missing = IDENTITY_GUEST_TYPES - types
        assert not missing, f"{path.name}: missing {missing}"


def test_every_site_has_group_occupancy_override():
    for path in site_paths():
        doc = load_site(path)
        assert any(
            _occupancy(case["params"]) >= GROUP_MIN for case in doc["cases"]
        ), path.name


def test_every_site_has_tent_prices():
    for path in site_paths():
        doc = load_site(path)
        seen = {case["params"]["lodging"] for case in doc["cases"]}
        assert TENT in seen, path.name


def test_beerot_student_follows_the_live_card_not_a_shared_band():
    doc = load_site(SITES_DIR / "beerot.json")
    prices = {
        case["params"].get("guest_type"): case["expected_price"]
        for case in doc["cases"]
        if case["params"]["lodging"] == TENT
        and case["params"].get("adults_num") == 1
    }
    assert prices["סטודנט"] == 53.0
