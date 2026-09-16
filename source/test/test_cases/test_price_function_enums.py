"""AST allowlist accepts Lodging / GuestType enums and rejects other classes."""

from __future__ import annotations

import pytest

from source.price_sandbox.ast_check import PriceFunctionError, compile_quote
from source.price_sandbox.execute import eval_quote_inprocess
from source.price_sandbox.params import QuoteParams

ENUM_QUOTE = """
from enum import Enum

class Lodging(Enum):
    TENT = "לינת שטח באוהלים פרטיים"
    HUSHA = "חושה"

class GuestType(Enum):
    REGULAR = "רגיל"
    MATMON = "מנוי"

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
    adult_p = 76.0
    if guest_type is GuestType.MATMON:
        adult_p = 57.0
    if lodging is Lodging.TENT:
        return adult_p * adults_num, "tent"
    if lodging is Lodging.HUSHA:
        return 350.0, "husha"
    raise ValueError("unknown lodging")
"""


def test_enum_quote_parses_strings_then_compares_members():
    tent = eval_quote_inprocess(
        ENUM_QUOTE,
        QuoteParams(lodging="לינת שטח באוהלים פרטיים", adults_num=2),
    )
    assert tent.price == 152.0
    matmon = eval_quote_inprocess(
        ENUM_QUOTE,
        QuoteParams(
            lodging="לינת שטח באוהלים פרטיים",
            adults_num=2,
            guest_type="מנוי",
        ),
    )
    assert matmon.price == 114.0


def test_unknown_lodging_string_raises():
    with pytest.raises(ValueError):
        eval_quote_inprocess(
            ENUM_QUOTE,
            QuoteParams(lodging="tent", adults_num=1),
        )


def test_plain_class_is_rejected():
    source = """
class Helper:
    x = 1
def quote(lodging, adults_num, child_num=0, child_ages=(), guest_type="רגיל", is_weekend_or_holiday=False, planned_entry_time=None, planned_exit_time=None):
    return 1, ""
"""
    with pytest.raises(PriceFunctionError):
        compile_quote(source)
