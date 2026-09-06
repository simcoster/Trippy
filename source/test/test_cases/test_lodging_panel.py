"""The `אפשרויות לינה` panel: finding it, and reading units out of it.

No LLM and no database. The panel is AJAX-loaded, so `panel_request` reads the
three things the page's own JavaScript sends, and `parse_lodging_blocks` turns
the response into units and rule paragraphs.

Both panel fixtures are real captures; the two anchor fixtures are trimmed live
pages kept down to what `panel_request` reads.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from source.scraper.rules_ingest.lodging import (
    PANEL_TITLE,
    fold,
    panel_request,
    parse_lodging_blocks,
    parse_lodging_panel,
)

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "info_site"
HURSHAT_TAL = (_FIXTURES / "hurshat_tal_lodging_panel.html").read_text(encoding="utf-8")
METSADA = (_FIXTURES / "metsada_lodging_panel.html").read_text(encoding="utf-8")


def units(panel: str) -> dict:
    found, _ = parse_lodging_panel(panel)
    return {u.name: u for u in found}


# ------------------------------------------------------------ finding it
def test_the_request_triple_is_read_off_the_page():
    page = (_FIXTURES / "panel_anchors_plain.html").read_text(encoding="utf-8")
    request = panel_request(page)
    assert request is not None
    assert request.post_id == "14874"
    # Read from the anchor's own data-cnt, never assumed: panel order differs
    # between sites.
    assert request.offset == "1"
    assert request.nonce == "64d6671b73"


def test_a_zero_width_space_in_the_title_still_finds_the_panel():
    """Tel Ashkelon writes `אפשרויות לי<U+200B>נה`. A plain substring test for the
    title misses it, and the site silently yields no panel at all."""
    page = (_FIXTURES / "panel_anchors_zwsp.html").read_text(encoding="utf-8")
    assert PANEL_TITLE not in page, "fixture no longer carries the zero-width space"
    request = panel_request(page)
    assert request is not None
    assert request.offset == "1"


def test_a_page_without_the_panel_asks_for_nothing():
    assert panel_request("<html><body><p>hello</p></body></html>") is None


def test_fold_removes_zero_width_characters():
    assert fold("אפשרויות לי​נה") == PANEL_TITLE
    assert fold("  a  b ") == "a b"


# --------------------------------------------------------- reading units
def test_hurshat_tal_has_its_seven_units():
    found, _ = parse_lodging_panel(HURSHAT_TAL)
    assert len(found) == 7
    assert "בונגלו עם מזגן" in {u.name for u in found}


def test_the_inventory_count_comes_off_the_heading():
    found = units(HURSHAT_TAL)
    assert found["בונגלו עם מזגן"].unit_count == 48
    assert found["עמדת חניה לקרוואן פרטי"].unit_count == 7
    # The panel does not always say how many there are.
    assert found["לינת שטח באוהלים פרטיים"].unit_count is None


def test_the_count_is_not_left_in_the_name():
    assert all("(" not in name for name in units(HURSHAT_TAL))


def test_an_empty_heading_is_a_separator_not_a_unit():
    """The panel ends with `<h4> </h4>`, and Hurshat Tal has one mid-list too."""
    assert all(u.name.strip() for u in parse_lodging_panel(HURSHAT_TAL)[0])


def test_the_site_wide_paragraph_is_a_rule_not_the_bungalow_description():
    """It sits *after* the bungalow's own paragraph, so "a paragraph belongs to
    the heading above it" would file a site-wide minimum-nights rule against
    bungalows. This is the assertion the parser exists to pass."""
    found, rules = parse_lodging_panel(HURSHAT_TAL)
    minimum = "מותנה במינימום 2 לילות"
    assert any(minimum in rule for rule in rules)
    assert all(minimum not in unit.text for unit in found)


def test_every_unit_keeps_its_own_description():
    found = units(HURSHAT_TAL)
    assert "עד 4 לנים בכל בונגלו" in found["בונגלו עם מזגן"].text
    assert "עד 5 לנים בכל חדר" in found["חדר צוות"].text


# ------------------------------------------------------------ block model
def test_blocks_are_indexed_in_document_order():
    blocks = parse_lodging_blocks(HURSHAT_TAL)
    assert [b.index for b in blocks] == list(range(len(blocks)))
    assert {b.kind for b in blocks} <= {"heading", "para", "scope"}


def test_the_panel_title_is_not_a_block():
    assert all(b.text != PANEL_TITLE for b in parse_lodging_blocks(HURSHAT_TAL))


@pytest.mark.parametrize(
    ("name", "count"),
    [
        ("חדר צוות מאובזר", 5),
        ("חדר צוות מאובזר ומונגש חדר מספר 6", 1),
        ("חדר צוות מאובזר ומונגש חדר מספר 1", 1),
    ],
)
def test_metsada_headings_survive_the_b_tag(name, count):
    """`<b>` is editorial bolding, not a count marker: at Metsada it wraps
    `ומונגש (1) חדר מספר 6`, so reading the count out of it and dropping the rest
    left three staff rooms sharing one name."""
    found = units(METSADA)
    assert name in found, sorted(found)
    assert found[name].unit_count == count
