"""Quotation marks are stripped from lodging and guest_type identifiers."""

from source.price_sandbox.params import QuoteParams, strip_type_quotes
from source.scraper.info_site.compile_price import CompileRateRow, compile_user_prompt


def test_strip_type_quotes_drops_gershayim():
    assert strip_type_quotes('נכה צה"ל ומלווה') == "נכה צהל ומלווה"
    assert strip_type_quotes("נכה צה״ל ומלווה") == "נכה צהל ומלווה"


def test_quote_params_strips_lodging_and_guest_type():
    params = QuoteParams(lodging='חושה "כפולה"', guest_type='נכה צה"ל ומלווה')
    assert params.lodging == "חושה כפולה"
    assert params.guest_type == "נכה צהל ומלווה"


def test_from_mapping_strips_quotes():
    params = QuoteParams.from_mapping(
        {"lodging": 'אוהל "משפחתי"', "adults_num": 1, "guest_type": 'צה"ל'}
    )
    assert params.lodging == "אוהל משפחתי"
    assert params.guest_type == "צהל"


def test_compile_prompt_enum_values_have_no_quotes():
    row = CompileRateRow(
        lodging='לינת שטח באוהלים "פרטיים"',
        guest_type='נכה צה"ל ומלווה',
        label="x",
        price=1.0,
        notes=None,
    )
    text = compile_user_prompt(
        site_name="x",
        lodgings=['לינת שטח באוהלים "פרטיים"'],
        guest_types=['נכה צה"ל ומלווה'],
        rows=[row],
        visitor_info="",
    )
    assert 'צה"ל' not in text
    assert "נכה צהל ומלווה" in text
    assert "לינת שטח באוהלים פרטיים" in text
