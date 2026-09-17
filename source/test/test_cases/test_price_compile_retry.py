"""Compile retry classifies AST vs occupancy without leaking gold prices."""

from source.price_sandbox.gold import GoldCase
from source.price_sandbox.params import QuoteParams
from source.scraper.info_site.compile_price import (
    GoldMiss,
    assess_compiled_source,
    occupancy_retry_suffix,
)

_QUOTE = """
from enum import Enum

class Lodging(Enum):
    TENT = "לינת שטח באוהלים פרטיים"
    FAMILY = "השכרת אוהל קמפינג משפחתי"

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
    if lodging is Lodging.FAMILY:
        return 350.0, "unit"
    return 64.0 * adults_num, "tent"
"""

_NAME_ERROR = _QUOTE.replace(
    'return 64.0 * adults_num, "tent"',
    'return 64.0 * adults_num, str(todder_count)',
)

_MEMBERSHIP = _QUOTE.replace(
    "if lodging is Lodging.FAMILY:",
    'rates = {"late_exit": 1}\n    if "late_exit" in rates:\n        pass\n    if lodging is Lodging.FAMILY:',
)


def _case(note: str, lodging: str, adults: int, price: float) -> GoldCase:
    return GoldCase(
        params=QuoteParams(lodging=lodging, adults_num=adults),
        expected_price=price,
        note=note,
        explanation="2 x 64",
    )


def test_syntax_error_is_a_fix_retry():
    verdict = assess_compiled_source("def quote(\n", [])
    assert verdict.ok is False
    assert verdict.retry == "fix"
    assert verdict.stage == "allowlist"


def test_string_membership_is_a_fix_retry():
    verdict = assess_compiled_source(_MEMBERSHIP, [])
    assert verdict.retry == "fix"
    assert verdict.stage == "static"


def test_name_error_gold_is_a_fix_retry():
    cases = [
        _case("tent two adults regular", "לינת שטח באוהלים פרטיים", 2, 128.0),
    ]
    verdict = assess_compiled_source(_NAME_ERROR, cases)
    assert verdict.retry == "fix"
    assert verdict.stage == "gold"
    joined = " ".join(verdict.retry_errors)
    assert "todder_count" in joined
    assert "128" not in joined
    assert "expected" not in joined.lower()


def test_price_miss_is_an_occupancy_regenerate():
    cases = [
        _case(
            "השכרת אוהל קמפינג משפחתי plus one extra",
            "השכרת אוהל קמפינג משפחתי",
            5,
            438.0,
        ),
    ]
    verdict = assess_compiled_source(_QUOTE, cases)
    assert verdict.retry == "regen"
    assert verdict.stage == "gold"
    hint = verdict.retry_errors[0]
    assert "438" not in hint
    assert "350" not in hint
    assert "expected" not in hint.lower()
    assert "השכרת אוהל קמפינג משפחתי" in hint
    assert "plus one extra" in hint


def test_occupancy_suffix_has_no_prices():
    misses = [
        GoldMiss(
            note="mahal included 36",
            lodging="מאהל גדול קבוע (מבנה כנעני)",
            kind="price",
            message="",
        ),
    ]
    text = occupancy_retry_suffix(misses)
    assert "3080" not in text
    assert "expected" not in text.lower()
    assert "מאהל גדול קבוע (מבנה כנעני)" in text


def test_passing_source_needs_no_retry():
    cases = [
        _case("tent two adults regular", "לינת שטח באוהלים פרטיים", 2, 128.0),
    ]
    verdict = assess_compiled_source(_QUOTE, cases)
    assert verdict.ok is True
    assert verdict.retry is None
