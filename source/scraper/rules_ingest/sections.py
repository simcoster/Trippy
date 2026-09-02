"""Split a parks.org.il camping page into rule-bearing sections.

Pure BS4 — no HTTP, no LLM, no DB — so it can be tested against a saved page.

Four shapes carry site-level rules and amenities:

  div.infoArea      accordion panels rendered server-side. Only `מה בחניון?` is;
                    `אפשרויות לינה` and the rest arrive over AJAX and are
                    deliberately not fetched — per-unit data comes from the
                    availability scrape (see docs/design.md).
  div.wrapUseInfo   the visitor-info columns: arrival/departure hours, booking,
                    directions. Each child column is its own titled section.
  div.rp_info_icons the icon strip: `כניסת כלבים` / `הכניסה לכלבים אסורה`.
  div.tableMain     rate-card rows whose `data-content` tooltip carries a rule
                    ("מותנה במינימום 2 לילות", "גיל 14 ומעלה").
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup

from source.scraper.info_site.parse import normalize_label, parse_rate_table

# Accordion panels whose content is per-accommodation-type; the availability
# scrape owns that data, so drop them even if a page renders them inline.
UNIT_SECTION_TITLES = ("אפשרויות לינה",)

# Visitor-info columns that are wayfinding, not policy. Skipped so each page
# costs one fewer extraction call; nothing in them states a rule.
NAVIGATION_SECTION_TITLES = ("איך להגיע", "יצירת קשר")

_BLANK_LINES_RE = re.compile(r"\n{2,}")


@dataclass(frozen=True)
class Section:
    """A titled block of page text to hand to the rule extractor."""

    title: str
    text: str
    source_url: str | None = None


def parse_sections(html: str, *, source_url: str | None = None) -> list[Section]:
    """Every rule-bearing site-level section on the page, in document order."""
    soup = BeautifulSoup(html, "html.parser")
    sections: list[Section] = []
    sections.extend(_info_area_sections(soup))
    # Icons first: they sit inside a visitor-info column, so pulling them out of
    # the tree keeps the same statement from being extracted twice.
    sections.extend(_icon_sections(soup))
    sections.extend(_visitor_info_sections(soup))
    sections.extend(_rate_note_sections(html))
    return [
        Section(s.title, s.text, source_url) for s in sections if s.text
    ]


def _block_text(node) -> str:
    """Readable text with one line per block element, blank lines collapsed."""
    text = node.get_text("\n", strip=True)
    return _BLANK_LINES_RE.sub("\n", text).strip()


def _info_area_sections(soup: BeautifulSoup) -> list[Section]:
    out: list[Section] = []
    for area in soup.select("div.infoArea"):
        content = area.select_one("div.infoContent")
        if content is None:
            continue
        heading = content.find(["h2", "h3", "h4"])
        title = normalize_label(heading.get_text(" ", strip=True)) if heading else ""
        if any(skip in title for skip in UNIT_SECTION_TITLES):
            continue
        body = _block_text(content)
        if heading is not None and body.startswith(title):
            body = body[len(title) :].strip()
        out.append(Section(title or "מידע כללי", body))
    return out


def _visitor_info_sections(soup: BeautifulSoup) -> list[Section]:
    out: list[Section] = []
    for wrap in soup.select("div.wrapUseInfo"):
        for column in wrap.find_all("div", recursive=False):
            heading = column.select_one(".hourTitle, h2, h3, h4")
            title = normalize_label(heading.get_text(" ", strip=True)) if heading else ""
            if any(skip in title for skip in NAVIGATION_SECTION_TITLES):
                continue
            body = _block_text(column)
            if title and body.startswith(title):
                body = body[len(title) :].strip()
            out.append(Section(title or "מידע למבקר", body))
    return out


def _icon_sections(soup: BeautifulSoup) -> list[Section]:
    """The icon strip states a policy in two halves: label, then verdict."""
    out: list[Section] = []
    for row in soup.select("div.rp_info_icons"):
        label_el = row.select_one("div.useIcon")
        value_el = row.select_one("div.popoverUseIcon")
        label = normalize_label(label_el.get_text(" ", strip=True)) if label_el else ""
        value = normalize_label(value_el.get_text(" ", strip=True)) if value_el else ""
        if not label and not value:
            continue
        # Label alone ("כניסת כלבים") is a heading, not a statement; the verdict
        # is what carries the rule, so keep both and let the LLM read them together.
        out.append(Section(label or "מידע", f"{label}: {value}".strip(": ")))
        row.extract()
    return out


def _rate_note_sections(html: str) -> list[Section]:
    """Rate-card tooltips: one section for all of them, deduped."""
    notes: list[str] = []
    seen: set[str] = set()
    for row in parse_rate_table(html):
        note = (row.get("notes") or "").strip()
        if not note or note in seen:
            continue
        seen.add(note)
        notes.append(f"{row['raw_label']}: {note}")
    if not notes:
        return []
    return [Section("הערות למחירון", "\n".join(notes))]
