"""The AJAX `מידע למבקר` panel: finding it, and reading it as a section.

No LLM and no database. The panel is not in the static page; `panel_request`
reads the accordion tab's `data-cnt`, and `parse_visitor_info_panel` turns the
loadmore body into one section the same way `מה בחניון?` is read.
"""

from __future__ import annotations

from pathlib import Path

from source.scraper.rules_ingest.lodging import panel_request
from source.scraper.rules_ingest.sections import (
    VISITOR_INFO_TITLE,
    parse_visitor_info_panel,
)

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "info_site"
PANEL = (_FIXTURES / "akhziv_visitor_info.html").read_text(encoding="utf-8")
ANCHORS = (_FIXTURES / "panel_anchors_plain.html").read_text(encoding="utf-8")


def test_the_visitor_info_tab_has_its_own_offset():
    request = panel_request(ANCHORS, title=VISITOR_INFO_TITLE)
    assert request is not None
    assert request.post_id == "14874"
    assert request.offset == "3"
    assert request.nonce == "64d6671b73"


def test_the_default_title_is_still_the_lodging_panel():
    request = panel_request(ANCHORS)
    assert request is not None
    assert request.offset == "1"


def test_a_page_without_the_tab_asks_for_nothing():
    assert panel_request("<html><body><p>hello</p></body></html>", title=VISITOR_INFO_TITLE) is None


def test_akhziv_panel_is_one_section_titled_visitor_info():
    sections = parse_visitor_info_panel(PANEL, source_url="https://x")
    assert len(sections) == 1
    section = sections[0]
    assert section.title == VISITOR_INFO_TITLE
    assert section.source_url == "https://x"
    assert not section.text.startswith(VISITOR_INFO_TITLE)


def test_each_list_item_is_its_own_line():
    text = parse_visitor_info_panel(PANEL)[0].text
    assert "אין כניסה לכלבים לחניון הלילה." in text
    assert "אסור להפעיל גנרטורים." in text
    assert "הדלקת מדורות אסורה." in text
    assert "לא יותר מ- 6 לילות" in text
    assert "בלוני גז העולים על 10" in text
    assert "איננו יכולים להפעיל את\nחניון הקרוואנים" in text or "חניון הקרוואנים" in text
    # One line per <li>, so the extractor sees them as separate facts.
    assert "אין כניסה לכלבים לחניון הלילה.\nאסור להפעיל גנרטורים." in text


def test_an_empty_panel_yields_nothing():
    assert parse_visitor_info_panel("<div class='infoContent'></div>") == []
