"""Compiled quote() must bake rate-card labels into named fields."""

from source.scraper.info_site.compile_price import runtime_string_scan_hits

NAMED_RATES = """
from enum import Enum

class Lodging(Enum):
    TENT = "לינת שטח באוהלים פרטיים"

class GuestType(Enum):
    REGULAR = "רגיל"

def quote(
    lodging,
    adults_num,
    child_num=0,
    child_ages=(),
    guest_type="רגיל",
    is_weekend_or_holiday=False,
    planned_entry_time=None,
    planned_exit_time=None,
):
    lodging = Lodging(lodging)
    guest_type = GuestType(guest_type)
    rates = {
        Lodging.TENT: {
            GuestType.REGULAR: {"adult": 76.0, "child": 58.0},
        }
    }[lodging][guest_type]
    adult_p = rates["adult"]
    child_p = rates.get("child", adult_p)
    return adult_p * adults_num + child_p * child_num, "ok"
"""

LABEL_SCAN = """
def quote(lodging, adults_num, child_num=0, child_ages=(), guest_type="רגיל", is_weekend_or_holiday=False, planned_entry_time=None, planned_exit_time=None):
    for rate in [{"label": "מבוגר", "price": 76.0}]:
        if "מבוגר" in rate["label"]:
            return rate["price"], rate["label"]
    return 0.0, ""
"""


def test_named_rate_fields_are_not_string_scans():
    assert runtime_string_scan_hits(NAMED_RATES) == []


def test_label_membership_is_a_compile_hit():
    hits = runtime_string_scan_hits(LABEL_SCAN)
    assert hits
    assert "מבוגר" in hits[0]
