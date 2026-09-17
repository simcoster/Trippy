"""קבוצה is an occupancy override, never a GuestType member."""

from source.scraper.info_site.compile_price import (
    GROUP_TAB,
    CompileRateRow,
    compile_user_prompt,
    guest_type_must_not_be_group_hits,
    partition_compile_rows,
)

IDENTITY = CompileRateRow(
    lodging="לינת שטח באוהלים פרטיים",
    guest_type="רגיל",
    label="לינת שטח באוהלים פרטיים - מבוגר",
    price=76.0,
    notes=None,
)
GROUP_ROW = CompileRateRow(
    lodging="לינת שטח באוהלים פרטיים",
    guest_type=GROUP_TAB,
    label="לינת שטח באוהלים פרטיים - מבוגר בקבוצה",
    price=65.0,
    notes="קבוצה המגיעה יחד מעל 30 לנים",
)

GROUP_ON_ENUM = """
from enum import Enum

class Lodging(Enum):
    TENT = "לינת שטח באוהלים פרטיים"

class GuestType(Enum):
    REGULAR = "רגיל"
    GROUP = "קבוצה"

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
    if lodging is Lodging.TENT and guest_type is not GuestType.GROUP:
        if adults_num + child_num >= 30:
            guest_type = GuestType.GROUP
    return 1.0, "x"
"""


def test_partition_keeps_group_off_identity():
    buckets = partition_compile_rows(
        [IDENTITY, GROUP_ROW], ["רגיל", GROUP_TAB]
    )
    assert [row.guest_type for row in buckets.identity_rows] == ["רגיל"]
    assert buckets.group_rows == [GROUP_ROW]
    assert buckets.identity_guest_types == ["רגיל"]


def test_user_prompt_omits_group_from_guest_type_enum():
    text = compile_user_prompt(
        site_name="אכזיב",
        lodgings=["לינת שטח באוהלים פרטיים"],
        guest_types=["רגיל", GROUP_TAB],
        rows=[IDENTITY, GROUP_ROW],
        visitor_info="",
    )
    guest_block = text.split("class GuestType")[1].split("Rate-card")[0]
    members = [
        line[2:] for line in guest_block.splitlines() if line.startswith("- ")
    ]
    assert members == ["רגיל"]
    assert "Occupancy override" in text
    assert GROUP_ROW.notes in text
    assert "ignore identity" in text


def test_guest_type_group_member_is_a_compile_hit():
    hits = guest_type_must_not_be_group_hits(GROUP_ON_ENUM)
    assert hits
    assert any("GROUP" in hit for hit in hits)
