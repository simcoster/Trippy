"""Syntactic unreachable code is rejected without executing quote()."""

from source.scraper.info_site.compile_price import unreachable_code_hits

_QUOTE_HEAD = """
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
"""


def test_statement_after_return_is_unreachable():
    source = _QUOTE_HEAD + "    return 1.0, 'x'\n    total = 2\n"
    hits = unreachable_code_hits(source)
    assert hits
    assert any("unreachable" in hit for hit in hits)


def test_if_false_body_is_unreachable():
    source = _QUOTE_HEAD + "    if False:\n        total = 1\n    return 1.0, 'x'\n"
    hits = unreachable_code_hits(source)
    assert hits
    assert any("unreachable" in hit for hit in hits)


def test_both_branches_return_then_following_is_unreachable():
    source = (
        _QUOTE_HEAD
        + "    if lodging:\n"
        + "        return 1.0, 'a'\n"
        + "    else:\n"
        + "        return 2.0, 'b'\n"
        + "    leftover = 3\n"
    )
    hits = unreachable_code_hits(source)
    assert any("leftover" in hit or "unreachable" in hit for hit in hits)


def test_straight_line_quote_is_reachable():
    source = _QUOTE_HEAD + "    return 1.0, 'x'\n"
    assert unreachable_code_hits(source) == []
