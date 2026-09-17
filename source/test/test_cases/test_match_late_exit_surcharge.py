"""Late-exit / extra-person labels are surcharges on a unit, not extras."""

from source.scraper.info_site.match_listing import SYSTEM_PROMPT


def test_match_prompt_treats_late_exit_as_a_rate_word():
    assert "תוספת יציאה מאוחרת" in SYSTEM_PROMPT
    assert "תוספת יציאה מאוחרת בונגלו סופי שבוע" in SYSTEM_PROMPT
