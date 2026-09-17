"""Compile system prompt must not name a real park or copy a live tariff."""

from source.scraper.info_site.compile_price import (
    FIX_SYSTEM_PROMPT,
    OCCUPANCY_RETRY_PREAMBLE,
    SYSTEM_PROMPT,
)

_PARKS = (
    "תל ערד",
    "חורשת",
    "אכזיב",
    "בארות",
    "יחיעם",
    "מצדה",
    "הבשור",
    "יוטבתה",
    "יהודיה",
    "משמר",
    "ממשית",
    "Tel Arad",
    "Yehiam",
    "Achziv",
    "Beerot",
    "Horashat",
    "Masada",
)


def test_compile_prompts_name_no_real_campsite():
    blob = SYSTEM_PROMPT + FIX_SYSTEM_PROMPT + OCCUPANCY_RETRY_PREAMBLE
    for name in _PARKS:
        assert name not in blob, name
    assert "76.0" not in SYSTEM_PROMPT
    assert "58.0" not in SYSTEM_PROMPT
    assert "3080" not in blob
    assert "GROUP_MIN = 30" not in SYSTEM_PROMPT
    assert "עד 4 לנים" not in SYSTEM_PROMPT
    assert "עד 5 לנים" not in SYSTEM_PROMPT
