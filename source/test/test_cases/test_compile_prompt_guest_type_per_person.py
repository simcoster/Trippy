"""GuestType identity tabs apply to per-person rates, not units."""

from source.scraper.info_site.compile_price import SYSTEM_PROMPT


def test_compile_prompt_guest_type_is_per_person_only():
    assert "change **per-person** rates only" in SYSTEM_PROMPT
    assert "Per-unit lodging" in SYSTEM_PROMPT
    assert "ignores `guest_type`" in SYSTEM_PROMPT
