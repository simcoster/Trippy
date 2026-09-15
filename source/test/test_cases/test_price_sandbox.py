"""AST-checked price functions and gold cases (new coverage only)."""

from __future__ import annotations

from pathlib import Path

import pytest

from source.price_sandbox.ast_check import PriceFunctionError, compile_quote
from source.price_sandbox.execute import eval_quote_inprocess, run_quote
from source.price_sandbox.gold import gold_for_url
from source.price_sandbox.gold.cases import CATALOG
from source.price_sandbox.params import QuoteParams
from source.price_sandbox.server import load_functions, quote_batch
from source.scraper.info_site.parse import parse_rate_table, parse_rate_tables

HORASHAT_URL = (
    "https://www.parks.org.il/camping/"
    "חניון-לילה-גן-לאומי-חורשת-טל/"
)

HORASHAT_QUOTE = """
def quote(
    lodging,
    adults_num,
    child_ages=(),
    is_matmon_sub=False,
    is_soldier=False,
    is_active_reserve=False,
    is_senior=False,
    is_student=False,
    is_disabled_idf=False,
    is_group=False,
    is_weekend_or_holiday=False,
    planned_entry_time=None,
    planned_exit_time=None,
):
    ages = tuple(child_ages)
    children = 0
    for age in ages:
        if age < 5:
            pass
        elif age < 14:
            children = children + 1
        else:
            adults_num = adults_num + 1
    name = str(lodging)
    tent = ("שטח" in name) or ("אוהל" in name)
    bungalow = "בונגלו" in name
    if tent:
        adult_p = 76.0
        child_p = 58.0
        if is_matmon_sub:
            adult_p = 57.0
            child_p = 44.0
        price = adult_p * adults_num + child_p * children
        bits = str(adults_num) + " adults [" + str(int(adult_p)) + "]"
        if children:
            bits = bits + " + " + str(children) + " children [" + str(int(child_p)) + "]"
        return price, bits
    if bungalow:
        price = 530.0 if is_weekend_or_holiday else 430.0
        if is_weekend_or_holiday and planned_exit_time is not None:
            if planned_exit_time > "12:00":
                price = price + 265.0
        return price, "bungalow"
    raise ValueError("unknown lodging")
"""

_LAZY_HTML = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "info_site"
    / "hurshat_tal_sales_lazy.html"
).read_text(encoding="utf-8")


def test_compile_rejects_os_import():
    with pytest.raises(PriceFunctionError):
        compile_quote("import os\ndef quote():\n    return 1, ''\n")


def test_compile_rejects_nested_helper():
    source = """
def quote():
    return 1, 'x'
def other():
    return 2
"""
    with pytest.raises(PriceFunctionError):
        compile_quote(source)


def test_hurashat_tent_two_adults():
    result = eval_quote_inprocess(
        HORASHAT_QUOTE,
        QuoteParams(lodging="לינת שטח באוהלים פרטיים", adults_num=2),
    )
    assert result.price == 152.0


def test_gold_matches_percent_encoded_hurshat_url():
    url = (
        "https://www.parks.org.il/camping/"
        "%d7%97%d7%a0%d7%99%d7%95%d7%9f-%d7%9c%d7%99%d7%9c%d7%94-%d7%92%d7%9f-"
        "%d7%9c%d7%90%d7%95%d7%9e%d7%99-%d7%97%d7%95%d7%a8%d7%a9%d7%aa-%d7%98%d7%9c/"
    )
    cases = gold_for_url(url)
    assert cases is not None
    assert len(cases) == 5


def test_hurashat_gold_cases_pass_handwritten_function():
    cases = gold_for_url(HORASHAT_URL)
    assert cases is not None
    assert len(cases) == 5
    for case in cases:
        got = eval_quote_inprocess(HORASHAT_QUOTE, case.params)
        assert round(got.price, 2) == round(case.expected_price, 2), case.note
    cases = gold_for_url(HORASHAT_URL)
    assert cases is not None
    assert len(cases) == 5
    for case in cases:
        got = eval_quote_inprocess(HORASHAT_QUOTE, case.params)
        assert round(got.price, 2) == round(case.expected_price, 2), case.note


def test_catalog_has_five_cases_per_site():
    assert len(CATALOG) == 18
    for row in CATALOG:
        assert len(row["cases"]) == 5, row.get("match")


def test_parse_sales_lazy_all_tabs():
    rows = parse_rate_tables(_LAZY_HTML)
    assert len(rows) == 3
    tabs = {row.rate_class for row in rows}
    assert "רגיל" in tabs
    assert "מנוי" in tabs
    regular = parse_rate_table(_LAZY_HTML)
    assert len(regular) == 2
    assert regular[0]["notes"] == "גיל 14 ומעלה"


def test_child_process_times_out():
    source = """
def quote(lodging, adults_num, child_ages=(), is_matmon_sub=False, is_soldier=False, is_active_reserve=False, is_senior=False, is_student=False, is_disabled_idf=False, is_group=False, is_weekend_or_holiday=False, planned_entry_time=None, planned_exit_time=None):
    total = 0
    for i in range(10 ** 9):
        total = total + 1
    return total, 'slow'
"""
    with pytest.raises(TimeoutError):
        run_quote(
            source,
            QuoteParams(lodging="x", adults_num=1),
            timeout_s=0.2,
        )


def test_server_load_and_quote():
    loaded = load_functions(
        [{"site_id": 1, "source": HORASHAT_QUOTE, "sha256": "abc"}]
    )
    assert loaded["ok"] is True
    assert loaded["loaded"] == [1]
    payload = quote_batch(
        [
            {
                "id": "a",
                "site_id": 1,
                "params": {
                    "lodging": "לינת שטח באוהלים פרטיים",
                    "adults_num": 2,
                },
            }
        ]
    )
    assert payload["ok"] is True
    assert payload["results"][0]["ok"] is True
    assert payload["results"][0]["price"] == 152.0
